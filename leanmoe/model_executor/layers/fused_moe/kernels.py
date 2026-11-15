"""图安全的 grouped GEMM。"""

from __future__ import annotations

import torch

from leanmoe.logger import get_logger, once

logger = get_logger("kernels.grouped_gemm")

try:  # Triton 是可选依赖
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - 环境相关
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _scatter_pairs_kernel(
        TOPK, SORTED, FLATIDX, INV, CURSOR, n_pairs,
        K: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """按专家做计数排序的 scatter 部分（grid-strided）。"""
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n_pairs
        e = tl.load(TOPK + idx, mask=mask, other=0)
        tok = idx // K
        pos = tl.atomic_add(CURSOR + e, 1, mask=mask)
        tl.store(SORTED + pos, tok, mask=mask)
        tl.store(FLATIDX + pos, idx, mask=mask)
        tl.store(INV + idx, pos, mask=mask)

    @triton.jit
    def _moe_combine_kernel(
        EXPERT_OUT, INV, W, OUT,
        K: tl.constexpr, H: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        """decode 非原子加权求和：每个 (token, H 分块) 读 top-k 行后一次写出。"""
        t = tl.program_id(0)
        offs_h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for k in range(K):
            r = tl.load(INV + t * K + k)
            w = tl.load(W + t * K + k).to(tl.float32)
            v = tl.load(
                EXPERT_OUT + r * H + offs_h, mask=mask_h, other=0.0
            ).to(tl.float32)
            acc = (acc + v * w).to(OUT.dtype.element_ty).to(tl.float32)
        tl.store(OUT + t * H + offs_h, acc.to(OUT.dtype.element_ty), mask=mask_h)

    @triton.jit
    def _moe_weighted_scatter_kernel(
        EXPERT_OUT, SORTED, FLATIDX, FLATW, OUT,
        n_rows,
        H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        """把专家输出按路由权重原子累加回 token 行（OUT 为 fp32）。"""
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n_rows
        tok = tl.load(SORTED + offs_m, mask=mask_m, other=0)
        fidx = tl.load(FLATIDX + offs_m, mask=mask_m, other=0)
        w = tl.load(FLATW + fidx, mask=mask_m, other=0.0).to(tl.float32)
        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            mask_h = offs_h < H
            mask = mask_m[:, None] & mask_h[None, :]
            v = tl.load(
                EXPERT_OUT + offs_m[:, None] * H + offs_h[None, :],
                mask=mask, other=0.0,
            ).to(tl.float32)
            tl.atomic_add(
                OUT + tok[:, None] * H + offs_h[None, :],
                v * w[:, None],
                mask=mask,
            )

    @triton.jit
    def _grouped_gemm_silu_kernel(
        A, B, C, OFFS, ROW_IDX, STRIDE_AM,
        K: tl.constexpr, N: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GATHER: tl.constexpr,
    ):
        """w13 GEMM + SiLU×up 融合；B 为 [E, K, 2N]，输出 C 为 [M, N]。"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        g = tl.program_id(2)

        end = tl.load(OFFS + g)
        start = tl.load(OFFS + g - 1, mask=g > 0, other=0)
        m_begin = start + pid_m * BLOCK_M
        if m_begin >= end:
            return

        offs_m = m_begin + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < end
        mask_n = offs_n < N

        if GATHER:
            a_row = tl.load(ROW_IDX + offs_m, mask=mask_m, other=0).to(tl.int64)
        else:
            a_row = offs_m
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            a = tl.load(
                A + a_row[:, None] * STRIDE_AM + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            base = g * K * 2 * N + offs_k[:, None] * 2 * N
            bg = tl.load(
                B + base + offs_n[None, :],
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            bu = tl.load(
                B + base + N + offs_n[None, :],
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            acc_g += tl.dot(a, bg, input_precision="ieee")
            acc_u += tl.dot(a, bu, input_precision="ieee")

        # 与"先写 bf16 gate_up 再算 silu"的语义对齐：激活前先舍入到输出 dtype，
        # 否则逐 token 输出会与 HF/vLLM 的参考实现分叉
        g = acc_g.to(C.dtype.element_ty).to(tl.float32)
        u = acc_u.to(C.dtype.element_ty).to(tl.float32)
        out = g * tl.sigmoid(g) * u
        tl.store(
            C + offs_m[:, None] * N + offs_n[None, :],
            out.to(C.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _grouped_gemm_kernel(
        A, B, C, OFFS,
        K: tl.constexpr, N: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        g = tl.program_id(2)

        end = tl.load(OFFS + g)
        start = tl.load(OFFS + g - 1, mask=g > 0, other=0)

        m_begin = start + pid_m * BLOCK_M
        if m_begin >= end:
            return

        offs_m = m_begin + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < end
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            a = tl.load(
                A + offs_m[:, None] * K + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            b = tl.load(
                B + g * K * N + offs_k[:, None] * N + offs_n[None, :],
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            # fp32 输入时 Triton 的 tl.dot 默认走 TF32；用 ieee 保证与参考实现可对拍。
            # bf16/fp16 输入不受该参数影响。
            acc += tl.dot(a, b, input_precision="ieee")

        tl.store(
            C + offs_m[:, None] * N + offs_n[None, :],
            acc.to(C.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def moe_align_block_size_fast(
    topk_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """计数排序版 align：bincount + cumsum + 一个原子 scatter kernel。"""
    num_tokens, top_k = topk_ids.shape
    n = num_tokens * top_k
    device = topk_ids.device
    flat = topk_ids.reshape(-1)
    # bincount 在捕获期会做 host 同步；用 scatter_add_ 完全留在设备端
    counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    offsets = torch.cumsum(counts, dim=0).to(torch.int32)
    if _HAS_TRITON and topk_ids.is_cuda and n > 0:
        try:
            cursor = (offsets - counts.to(torch.int32)).contiguous()
            # 三块输出都被 _scatter_pairs_kernel 完整覆盖（每个 idx 恰好写一次），
            # 用 empty 省掉每层 3 次 memset
            sorted_ids = torch.empty(n, dtype=torch.long, device=device)
            flat_indices = torch.empty(n, dtype=torch.long, device=device)
            inv = torch.empty(n, dtype=torch.int32, device=device)
            block = 256
            grid = (triton.cdiv(n, block),)
            _scatter_pairs_kernel[grid](
                flat.to(torch.int32), sorted_ids, flat_indices, inv, cursor, n,
                K=top_k, BLOCK=block,
            )
            return sorted_ids, offsets, flat_indices, inv
        except Exception as exc:
            once.warn(logger, "align_fast_fail", f"计数排序 align 失败，回退 argsort: {exc}")

    order = flat.argsort(stable=True)
    token_idx = torch.arange(num_tokens, device=device).repeat_interleave(top_k)
    sorted_ids = token_idx.index_select(0, order)
    flat_indices = torch.arange(n, device=device).index_select(0, order)
    return sorted_ids, offsets, flat_indices, None


def moe_combine(
    expert_out: torch.Tensor,
    inv: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor | None:
    """decode 融合加权求和：out[t] = Σ_k expert_out[inv[t*K+k]] * w[t,k]。"""
    if not (_HAS_TRITON and expert_out.is_cuda and num_tokens > 0):
        return None
    hidden = expert_out.shape[1]
    out = torch.empty(
        (num_tokens, hidden), dtype=expert_out.dtype, device=expert_out.device
    )
    top_k = topk_weights.shape[1]
    try:
        _moe_combine_kernel[(num_tokens, triton.cdiv(hidden, 512))](
            expert_out, inv, topk_weights.reshape(-1), out,
            K=top_k, H=hidden, BLOCK_H=512,
        )
    except Exception as exc:
        once.warn(logger, "combine_fail", f"融合加权求和失败，回退 index_add: {exc}")
        return None
    return out


def moe_weighted_scatter(
    expert_out: torch.Tensor,
    sorted_ids: torch.Tensor,
    flat_indices: torch.Tensor,
    flat_weights: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """把 [n_rows, H] 的专家输出按权重累加到 [num_tokens, H]。"""
    n_rows, hidden = expert_out.shape
    if _HAS_TRITON and expert_out.is_cuda and n_rows > 0:
        try:
            out_f32 = torch.zeros(
                (num_tokens, hidden), dtype=torch.float32, device=expert_out.device
            )
            block_m, block_h = 16, 256
            _moe_weighted_scatter_kernel[(triton.cdiv(n_rows, block_m),)](
                expert_out, sorted_ids, flat_indices, flat_weights, out_f32,
                n_rows, H=hidden, BLOCK_M=block_m, BLOCK_H=block_h,
            )
            return out_f32.to(expert_out.dtype)
        except Exception as exc:
            once.warn(logger, "scatter_fail", f"融合加权求和失败，回退 index_add: {exc}")
    weights = torch.where(
        flat_indices >= 0,
        flat_weights[flat_indices.clamp(min=0)],
        torch.zeros_like(flat_weights[:1]).expand_as(flat_indices),
    )
    out = expert_out.new_zeros((num_tokens, hidden))
    out.index_add_(0, sorted_ids, expert_out * weights.unsqueeze(-1))
    return out


def grouped_gemm_silu(
    a: torch.Tensor,
    b: torch.Tensor,
    offsets: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    row_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """w13 GEMM + SiLU×up 融合：a [M,K]，b [G,K,2N] -> [M,N]。"""
    num_groups, k, two_n = b.shape
    n = two_n // 2
    m = row_idx.numel() if row_idx is not None else a.shape[0]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    if _HAS_TRITON and a.is_cuda and m > 0:
        try:
            grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), num_groups)
            _grouped_gemm_silu_kernel[grid](
                a, b, out, offsets,
                row_idx if row_idx is not None else offsets,
                a.stride(0),
                K=k, N=n,
                BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                GATHER=row_idx is not None,
                num_warps=num_warps,
            )
            return out
        except Exception as exc:
            once.warn(logger, "silu_fuse_fail", f"融合 SiLU 失败，回退两段式: {exc}")
    if row_idx is not None:
        a = a.index_select(0, row_idx)
    gate_up = grouped_gemm(a, b, offsets, block_m, block_n, block_k, num_warps)
    gate, up = gate_up.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    offsets: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    """a: [M, K]，b: [G, K, N]，offsets: [G] 累计上界。返回 [M, N]。"""
    num_groups, k, n = b.shape
    m = a.shape[0]
    # 当前唯一的调用方（experts.py）用 block_size=1 的 align，offsets[-1] == m，
    # 每行都会被 Triton kernel 写入，因此用 empty 省掉每层 2 次 memset。
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)

    if _HAS_TRITON and a.is_cuda and m > 0:
        try:
            grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), num_groups)
            _grouped_gemm_kernel[grid](
                a, b, out, offsets,
                K=k, N=n,
                BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                num_warps=num_warps,
            )
            return out
        except Exception as exc:  # 编译或运行时失败都回退
            once.warn(logger, "triton_fail", f"Triton grouped GEMM 失败，回退 torch 路径: {exc}")

    # 回退 1：torch._grouped_mm
    if hasattr(torch, "_grouped_mm") and not torch.cuda.is_current_stream_capturing():
        try:
            # 对齐后的行数可能大于 offsets[-1]，torch 路径要求两者相等
            total = int(offsets[-1].item())
            return torch._grouped_mm(a[:total], b, offsets)
        except Exception as exc:
            once.warn(logger, "grouped_mm_fail", f"torch._grouped_mm 失败，回退逐组: {exc}")

    # 回退 2：逐组循环（捕获时若走到这里说明 Triton 不可用，图会退化）
    parts = []
    start = 0
    for g in range(num_groups):
        end = int(offsets[g].item())
        if end > start:
            parts.append(a[start:end] @ b[g])
        start = end
    if not parts:
        return torch.zeros_like(out)
    return torch.cat(parts, dim=0)
