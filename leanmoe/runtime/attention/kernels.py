"""Attention 的 Triton kernel 与 KV 写入（图安全、无 host 逻辑）。"""

from __future__ import annotations

import torch

from leanmoe.logger import get_logger, once

logger = get_logger("attention")

try:  # Triton 可选
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _reshape_and_cache_kernel(
        KEY, VALUE, CACHE, SLOTS,
        KS0, KS1, H: tl.constexpr, D: tl.constexpr,
    ):
        """把 K/V 直接写进分页 KV cache（布局 [blocks, block, H, 2D]）。"""
        t = tl.program_id(0)
        h = tl.program_id(1)
        slot = tl.load(SLOTS + t)
        # 负槽位是补齐 bucket 的 dummy 行（与 vLLM 的 slot=-1 语义一致），跳过写入
        if slot < 0:
            return
        offs = tl.arange(0, D)
        src = t * KS0 + h * KS1 + offs
        dst = (slot * H + h) * (2 * D) + offs
        tl.store(CACHE + dst, tl.load(KEY + src))
        tl.store(CACHE + dst + D, tl.load(VALUE + src))



def write_to_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor | None,
) -> None:
    """把本步新算的 K/V 写进分页缓存。"""
    if kv_cache is None or slot_mapping is None:
        return
    num_tokens, num_kv_heads, head_dim = key.shape
    flat = kv_cache.view(-1, num_kv_heads, 2 * head_dim)
    if _HAS_TRITON and key.is_cuda and num_tokens > 0:
        try:
            _reshape_and_cache_kernel[(num_tokens, num_kv_heads)](
                key, value, flat, slot_mapping,
                key.stride(0), key.stride(1),
                H=num_kv_heads, D=head_dim,
            )
            return
        except Exception as exc:  # 编译/运行时失败都回退
            once.warn(logger, "cache_fuse_fail", f"融合 KV 写入失败，回退 cat+copy: {exc}")
    kv = torch.cat([key, value], dim=-1)
    keep = slot_mapping >= 0
    if bool(keep.all()):
        flat.index_copy_(0, slot_mapping, kv)
    elif bool(keep.any()):
        flat.index_copy_(0, slot_mapping[keep], kv[keep])



if _HAS_TRITON:

    @triton.jit
    def _paged_decode_kernel(
        Q, KVC, BT, SEQ, OUT,
        sq_b, sq_h,
        so_b, so_h,
        sc_b, sc_n, sc_h,
        st_b, st_n,
        Hq, Hkv,
        D: tl.constexpr, BS: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        scale,
    ):
        """paged decode 注意力（GQA 支持），图安全：不含 host 逻辑。"""
        b = tl.program_id(0)
        h = tl.program_id(1)
        kvh = h // (Hq // Hkv)
        seq_len = tl.load(SEQ + b)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        q = tl.load(Q + b * sq_b + h * sq_h + offs_d, mask=mask_d, other=0.0).to(tl.float32)

        m_i = -1e30
        l_i = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start in range(0, seq_len, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len
            blk = tl.load(BT + b * st_b + (offs_n // BS) * st_n, mask=mask_n, other=0)
            kv_base = blk * sc_b + (offs_n % BS) * sc_n + kvh * sc_h
            k = tl.load(
                KVC + kv_base[:, None] + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :], other=0.0,
            ).to(tl.float32)
            v = tl.load(
                KVC + kv_base[:, None] + D + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :], other=0.0,
            ).to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1) * scale
            s = tl.where(mask_n, s, -1e30)
            m_new = tl.maximum(m_i, tl.max(s, 0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + tl.sum(p, 0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new
        tl.store(
            OUT + b * so_b + h * so_h + offs_d,
            (acc / l_i).to(OUT.dtype.element_ty), mask=mask_d,
        )

    @triton.jit
    def _paged_decode_gqa_split_kernel(
        Q, KVC, BT, SEQ, PACC, PM, PL,
        sq_b, sq_h,
        sc_b, sc_n, sc_h,
        st_b, st_n,
        Hkv,
        D: tl.constexpr, BS: tl.constexpr, S: tl.constexpr,
        GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        scale,
    ):
        """GQA packing 的 split decode：一个 program 处理同一 KV head 下的 GROUP 个 q head。"""
        b = tl.program_id(0)
        kh = tl.program_id(1)
        sp = tl.program_id(2)
        seq_len = tl.load(SEQ + b)
        chunk = tl.cdiv(seq_len, S)
        n_start = sp * chunk
        n_end = tl.minimum(seq_len, n_start + chunk)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        offs_g = tl.arange(0, GROUP)
        h = kh * GROUP + offs_g
        base = ((b * Hkv + kh) * S + sp) * GROUP

        if n_start >= n_end:
            tl.store(PM + base + offs_g, tl.full((GROUP,), -1e30, tl.float32))
            tl.store(PL + base + offs_g, tl.zeros((GROUP,), tl.float32))
            tl.store(
                PACC + (base + offs_g)[:, None] * D + offs_d[None, :],
                tl.zeros((GROUP, BLOCK_D), tl.float32), mask=mask_d[None, :],
            )
            return

        q = tl.load(
            Q + b * sq_b + h[:, None] * sq_h + offs_d[None, :],
            mask=mask_d[None, :], other=0.0,
        )
        m_i = tl.full((GROUP,), -1e30, tl.float32)
        l_i = tl.zeros((GROUP,), tl.float32)
        acc = tl.zeros((GROUP, BLOCK_D), dtype=tl.float32)

        for n0 in range(n_start, n_end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_end
            blk = tl.load(BT + b * st_b + (offs_n // BS) * st_n, mask=mask_n, other=0)
            kv_base = blk * sc_b + (offs_n % BS) * sc_n + kh * sc_h
            k = tl.load(
                KVC + kv_base[:, None] + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :], other=0.0,
            )
            v = tl.load(
                KVC + kv_base[:, None] + D + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :], other=0.0,
            )
            s_ij = tl.dot(q, tl.trans(k)) * scale
            s_ij = tl.where(mask_n[None, :], s_ij, -1e30)
            m_new = tl.maximum(m_i, tl.max(s_ij, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s_ij - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        tl.store(PM + base + offs_g, m_i)
        tl.store(PL + base + offs_g, l_i)
        tl.store(
            PACC + (base + offs_g)[:, None] * D + offs_d[None, :],
            acc, mask=mask_d[None, :],
        )

    @triton.jit
    def _gqa_combine_kernel(
        PACC, PM, PL, OUT,
        so_b, so_h, Hkv,
        S: tl.constexpr, GROUP: tl.constexpr,
        D: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        b = tl.program_id(0)
        kh = tl.program_id(1)
        offs_g = tl.arange(0, GROUP)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        h = kh * GROUP + offs_g
        base = (b * Hkv + kh) * S
        m = tl.full((GROUP,), -1e30, tl.float32)
        for s in range(S):
            m = tl.maximum(m, tl.load(PM + (base + s) * GROUP + offs_g))
        l_i = tl.zeros((GROUP,), tl.float32)
        acc = tl.zeros((GROUP, BLOCK_D), dtype=tl.float32)
        for s in range(S):
            w = tl.exp(tl.load(PM + (base + s) * GROUP + offs_g) - m)
            l_i += w * tl.load(PL + (base + s) * GROUP + offs_g)
            acc += w[:, None] * tl.load(
                PACC + ((base + s) * GROUP + offs_g)[:, None] * D + offs_d[None, :],
                mask=mask_d[None, :], other=0.0,
            )
        tl.store(
            OUT + b * so_b + h[:, None] * so_h + offs_d[None, :],
            (acc / l_i[:, None]).to(OUT.dtype.element_ty), mask=mask_d[None, :],
        )

    @triton.jit
    def _paged_decode_split_kernel(
        Q, KVC, BT, SEQ, PACC, PM, PL,
        sq_b, sq_h,
        sc_b, sc_n, sc_h,
        st_b, st_n,
        Hq, Hkv,
        D: tl.constexpr, BS: tl.constexpr, S: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        scale,
    ):
        """split-KV 的 paged decode：每个 (seq, head, split) 算局部 softmax 状态。"""
        b = tl.program_id(0)
        h = tl.program_id(1)
        s = tl.program_id(2)
        kvh = h // (Hq // Hkv)
        seq_len = tl.load(SEQ + b)
        chunk = tl.cdiv(seq_len, S)
        n_start = s * chunk
        n_end = tl.minimum(seq_len, n_start + chunk)

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        base = (b * Hq + h) * S + s
        if n_start >= n_end:
            tl.store(PM + base, -1e30)
            tl.store(PL + base, 0.0)
            tl.store(PACC + base * D + offs_d, tl.zeros((BLOCK_D,), tl.float32), mask=mask_d)
            return

        q = tl.load(Q + b * sq_b + h * sq_h + offs_d, mask=mask_d, other=0.0)
        m_i = tl.full((), -1e30, tl.float32)
        l_i = tl.zeros((), tl.float32)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for n0 in range(n_start, n_end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_end
            blk = tl.load(BT + b * st_b + (offs_n // BS) * st_n, mask=mask_n, other=0)
            kv_base = blk * sc_b + (offs_n % BS) * sc_n + kvh * sc_h
            k = tl.load(
                KVC + kv_base[:, None] + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :], other=0.0,
            )
            v = tl.load(
                KVC + kv_base[:, None] + D + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :], other=0.0,
            )
            s_ij = tl.sum(k * q[None, :], axis=1) * scale
            s_ij = tl.where(mask_n, s_ij, -1e30)
            m_new = tl.maximum(m_i, tl.max(s_ij, 0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s_ij - m_new)
            l_i = l_i * alpha + tl.sum(p, 0)
            acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
            m_i = m_new

        tl.store(PM + base, m_i)
        tl.store(PL + base, l_i)
        tl.store(PACC + base * D + offs_d, acc, mask=mask_d)

    @triton.jit
    def _split_combine_kernel(
        PACC, PM, PL, OUT,
        so_b, so_h, Hq,
        S: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        base = (b * Hq + h) * S
        m = tl.full((), -1e30, tl.float32)
        for s in range(S):
            m = tl.maximum(m, tl.load(PM + base + s))
        l_i = tl.zeros((), tl.float32)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for s in range(S):
            w = tl.exp(tl.load(PM + base + s) - m)
            l_i += w * tl.load(PL + base + s)
            acc += w * tl.load(PACC + (base + s) * D + offs_d, mask=mask_d, other=0.0)
        tl.store(
            OUT + b * so_b + h * so_h + offs_d,
            (acc / l_i).to(OUT.dtype.element_ty), mask=mask_d,
        )

    @triton.jit
    def _paged_prefill_kernel(
        Q, KVC, BT, QS, SEQ, OUT,
        sq_b, sq_h,
        so_b, so_h,
        sc_b, sc_n, sc_h,
        st_b, st_n,
        Hq, Hkv,
        D: tl.constexpr, BS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        scale,
    ):
        """paged prefill（支持分块 prefill 与混合批，带因果掩码）。"""
        b = tl.program_id(0)
        h = tl.program_id(1)
        mblk = tl.program_id(2)
        qs = tl.load(QS + b)
        qe = tl.load(QS + b + 1)
        qlen = qe - qs
        # 混合批里 decode 序列 q_len=1，但 grid 按 max_q 分配了多个 query block；
        # 超出本序列的 block 直接退出，否则每条 decode 序列都要白跑一遍 KV 循环。
        if mblk * BLOCK_M >= qlen:
            return
        off_m = mblk * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = off_m < qlen
        seq_len = tl.load(SEQ + b)
        start_pos = seq_len - qlen          # 本 chunk 之前已有的 KV 长度
        kvh = h // (Hq // Hkv)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        q = tl.load(
            Q + (qs + off_m)[:, None] * sq_b + h * sq_h + offs_d[None, :],
            mask=mask_m[:, None] & mask_d[None, :], other=0.0,
        )
        m_i = tl.full((BLOCK_M,), -1e30, tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        kv_end = start_pos + qlen
        for n0 in range(0, kv_end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < kv_end
            blk = tl.load(BT + b * st_b + (offs_n // BS) * st_n, mask=mask_n, other=0)
            kv_base = blk * sc_b + (offs_n % BS) * sc_n + kvh * sc_h
            k = tl.load(
                KVC + kv_base[:, None] + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :], other=0.0,
            )
            v = tl.load(
                KVC + kv_base[:, None] + D + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :], other=0.0,
            )
            s = tl.dot(q, tl.trans(k)) * scale
            causal = offs_n[None, :] <= (start_pos + off_m)[:, None]
            s = tl.where(mask_n[None, :] & causal, s, -1e30)
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        out = acc / l_i[:, None]
        tl.store(
            OUT + (qs + off_m)[:, None] * so_b + h * so_h + offs_d[None, :],
            out.to(OUT.dtype.element_ty),
            mask=mask_m[:, None] & mask_d[None, :],
        )


