"""Attention 后端实现：SDPA / FlashAttention（可选外部包）/ 自研 Triton paged。"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from leanmoe.logger import get_logger, once
from leanmoe.runtime.attention.kernels import (
    _HAS_TRITON,
    _gqa_combine_kernel,
    _paged_decode_gqa_split_kernel,
    _paged_decode_kernel,
    _paged_decode_split_kernel,
    _paged_prefill_kernel,
    _split_combine_kernel,
    write_to_kv_cache,
)
from leanmoe.runtime.attention.metadata import AttentionMetadata

try:
    import triton
except Exception:  # pragma: no cover
    triton = None

logger = get_logger("attention")


def gather_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把一条序列的 KV 从分页缓存 gather 成 dense（eager 路径）。"""
    block_size = kv_cache.shape[1]
    num_blocks = (seq_len + block_size - 1) // block_size
    blocks = block_table[:num_blocks]
    slots = (
        blocks.unsqueeze(1) * block_size
        + torch.arange(block_size, device=kv_cache.device)
    ).reshape(-1)[:seq_len]
    flat = kv_cache.view(-1, num_kv_heads, 2 * head_dim)
    kv = flat.index_select(0, slots)
    return kv[..., :head_dim], kv[..., head_dim:]



class SDPABackend:
    """通用后端：分词 prefill + 批量 decode，全部基于 torch SDPA。"""

    name = "sdpa"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor | None,
        metadata: AttentionMetadata,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        write_to_kv_cache(key, value, kv_cache, metadata.slot_mapping)
        num_heads = query.shape[1]
        gqa = num_heads != num_kv_heads

        # 批量 prefill：所有序列都是首块且等长时，一次 SDPA 取代逐序列循环
        if (
            kv_cache is not None
            and metadata.uniform_prefill_len
            and metadata.num_prefill_seqs == metadata.num_seqs
            and not metadata.is_pure_decode
        ):
            num_seqs = metadata.num_seqs
            length = metadata.uniform_prefill_len
            q = query.view(num_seqs, length, num_heads, head_dim).transpose(1, 2)
            k = key.view(num_seqs, length, num_kv_heads, head_dim).transpose(1, 2)
            v = value.view(num_seqs, length, num_kv_heads, head_dim).transpose(1, 2)
            # 与 vLLM 的 FA 路径对齐：因果用 is_causal，不用显式 bool mask——
            # 显式 mask + enable_gqa 会把 SDPA 逼到 math 后端，物化 O(L^2) 分数矩阵。
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=gqa
            )
            return out.transpose(1, 2).reshape(-1, num_heads, head_dim)

        # 纯 decode：gather + 批量 SDPA
        if metadata.is_pure_decode and kv_cache is not None:
            return self._decode(query, kv_cache, metadata, num_kv_heads, head_dim, gqa)

        # prefill / 混合批：逐序列处理
        outputs: list[torch.Tensor] = []
        cu = metadata.query_start_loc.tolist()
        for i in range(metadata.num_seqs):
            qs, qe = cu[i], cu[i + 1]
            if qe == qs:
                continue
            q_len = qe - qs
            k_len = int(metadata.seq_lens[i].item())
            if kv_cache is None:
                k_i, v_i = key[qs:qe], value[qs:qe]
                start = 0
            else:
                k_i, v_i = gather_kv(
                    kv_cache, metadata.block_tables[i], k_len, num_kv_heads, head_dim
                )
                start = k_len - q_len
            q_i = query[qs:qe].transpose(0, 1).unsqueeze(0)
            k_i = k_i.transpose(0, 1).unsqueeze(0)
            v_i = v_i.transpose(0, 1).unsqueeze(0)
            if q_len == 1:
                out = F.scaled_dot_product_attention(q_i, k_i, v_i, enable_gqa=gqa)
            elif start == 0:
                # 首块 prefill：q/k 等长，is_causal 与下三角 mask 等价，
                # 且能让 SDPA 选中 flash/efficient 后端而不是 math。
                out = F.scaled_dot_product_attention(
                    q_i, k_i, v_i, is_causal=True, enable_gqa=gqa
                )
            else:
                q_pos = start + torch.arange(q_len, device=query.device)
                k_pos = torch.arange(k_len, device=query.device)
                mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
                out = F.scaled_dot_product_attention(
                    q_i, k_i, v_i, attn_mask=mask, enable_gqa=gqa
                )
            outputs.append(out.squeeze(0).transpose(0, 1))
        if not outputs:
            return query.new_zeros(query.shape)
        return torch.cat(outputs, dim=0)

    def _decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
        num_kv_heads: int,
        head_dim: int,
        gqa: bool,
    ) -> torch.Tensor:
        num_seqs = metadata.num_seqs
        block_size = kv_cache.shape[1]
        device = query.device
        extra = metadata.extra or {}

        if extra.get("kv_indices") is not None:
            # CUDA Graph 路径：索引与掩码是捕获前预分配的持久张量
            indices = extra["kv_indices"]
            mask = extra["kv_mask"]
            max_len = indices.shape[1]
        else:
            max_len = int(metadata.seq_lens.max().item())
            indices = torch.zeros((num_seqs, max_len), dtype=torch.long, device=device)
            mask = torch.zeros((num_seqs, 1, 1, max_len), dtype=torch.bool, device=device)
            for i in range(num_seqs):
                length = int(metadata.seq_lens[i].item())
                blocks = metadata.block_tables[i, : (length + block_size - 1) // block_size]
                slots = (blocks.unsqueeze(1) * block_size + torch.arange(block_size, device=device)).reshape(-1)
                indices[i, :length] = slots[:length]
                mask[i, 0, 0, :length] = True

        # 扁平 slot 索引一次 index_select 取出整批 KV
        flat = kv_cache.view(-1, num_kv_heads, 2 * head_dim)
        kv = flat.index_select(0, indices.reshape(-1)).view(
            num_seqs, max_len, num_kv_heads, 2 * head_dim
        )
        k_all, v_all = kv[..., :head_dim], kv[..., head_dim:]
        q_b = query.view(num_seqs, 1, query.shape[1], head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_b,
            k_all.transpose(1, 2),
            v_all.transpose(1, 2),
            attn_mask=mask,
            enable_gqa=gqa,
        )
        return out.squeeze(2)



class FlashAttnBackend:
    """FA2：prefill 用 varlen，decode 用 paged block_table（与 vLLM 同路径）。"""

    name = "flash_attn"

    def __init__(self) -> None:
        self._fa = None
        try:
            # 只尝试独立包，不再 import vllm（引擎零 vllm 依赖）
            from vllm_flash_attn import flash_attn_varlen_func  # type: ignore

            self._fa = flash_attn_varlen_func
            once.info(logger, "fa2_pkg", "使用独立 vllm_flash_attn 包的 paged varlen")
        except Exception:
            once.info(logger, "fa2_missing", "未安装独立 flash-attn 包，FA 后端不可用")

    @property
    def available(self) -> bool:
        return self._fa is not None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor | None,
        metadata: AttentionMetadata,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        write_to_kv_cache(key, value, kv_cache, metadata.slot_mapping)
        if self._fa is None or kv_cache is None:
            return _SDPA_FALLBACK.forward(
                query, key, value, kv_cache, metadata, num_kv_heads, head_dim
            )

        # kv_cache: [blocks, block_size, kv_heads, 2D]；FA2 要 [blocks, block_size, kv_heads, D]
        k_cache = kv_cache[..., :head_dim]
        v_cache = kv_cache[..., head_dim:]

        if metadata.is_pure_decode and metadata.decode_cu_seqlens is not None:
            return self._fa(
                query,
                k_cache,
                v_cache,
                max_seqlen_q=1,
                cu_seqlens_q=metadata.decode_cu_seqlens,
                max_seqlen_k=metadata.max_seqlen_k,
                seqused_k=metadata.seq_lens.int(),
                block_table=metadata.block_tables.int(),
                causal=False,
            )

        # prefill 首块：本步 K/V 就是全部历史
        if metadata.uniform_prefill_len and metadata.num_prefill_seqs == metadata.num_seqs:
            return self._fa(
                query,
                key,
                value,
                max_seqlen_q=metadata.uniform_prefill_len,
                cu_seqlens_q=metadata.query_start_loc,
                max_seqlen_k=metadata.uniform_prefill_len,
                cu_seqlens_k=metadata.query_start_loc,
                causal=True,
            )

        # chunked prefill 的后续块需要历史 KV，先回退 SDPA
        return _SDPA_FALLBACK.forward(
            query, key, value, kv_cache, metadata, num_kv_heads, head_dim
        )



class TritonPagedDecodeBackend:
    """自研 paged decode 注意力（Triton），prefill 走 SDPA 回退。"""

    name = "triton_paged"

    def __init__(self) -> None:
        self._fi_prefill = None

    @property
    def available(self) -> bool:
        return _HAS_TRITON

    def _prefill_triton(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        num_seqs = metadata.num_seqs
        num_heads = query.shape[1]
        max_q = max(1, metadata.max_query_len)
        out = torch.empty_like(query)
        # tile 按批形态选：纯 prefill 用大 BLOCK_M（batch256 实测 +10.9%），
        # 混合批用小块（短 prefill 下 BLOCK_M=128 的因果掩码浪费太大）。
        pure_prefill = metadata.num_prefill_seqs == num_seqs
        default_bm = "128" if pure_prefill else "16"
        default_w = "8" if pure_prefill else "4"
        bm = int(os.environ.get("LEANMOE_PREFILL_BM", default_bm))
        bn = int(os.environ.get("LEANMOE_PREFILL_BN", "64"))
        warps = int(os.environ.get("LEANMOE_PREFILL_WARPS", default_w))
        _paged_prefill_kernel[(num_seqs, num_heads, triton.cdiv(max_q, bm))](
            query, kv_cache, metadata.block_tables,
            metadata.query_start_loc, metadata.seq_lens, out,
            query.stride(0), query.stride(1),
            out.stride(0), out.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            metadata.block_tables.stride(0), metadata.block_tables.stride(1),
            num_heads, num_kv_heads,
            D=head_dim, BS=kv_cache.shape[1],
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=128,
            scale=float(head_dim) ** -0.5, num_warps=warps,
        )
        return out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor | None,
        metadata: AttentionMetadata,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        write_to_kv_cache(key, value, kv_cache, metadata.slot_mapping)
        # prefill（非纯 decode）：优先自研 Triton paged prefill
        if not metadata.is_pure_decode and kv_cache is not None and _HAS_TRITON:
            try:
                return self._prefill_triton(
                    query, kv_cache, metadata, num_kv_heads, head_dim
                )
            except Exception as exc:
                once.warn(logger, "tp_prefill_fail", f"自研 prefill 失败，回退: {exc}")
            return _SDPA_FALLBACK.forward(
                query, key, value, kv_cache, metadata, num_kv_heads, head_dim
            )
        if not (metadata.is_pure_decode and kv_cache is not None and _HAS_TRITON):
            return _SDPA_FALLBACK.forward(
                query, key, value, kv_cache, metadata, num_kv_heads, head_dim
            )
        num_seqs = metadata.num_seqs
        num_heads = query.shape[1]
        out = torch.empty_like(query)
        group = (
            num_heads // num_kv_heads
            if num_kv_heads and num_heads % num_kv_heads == 0 else 0
        )
        if (
            os.environ.get("LEANMOE_ATTN_GQA", "1") == "1"
            and 0 < group <= 16
            and head_dim <= 128
        ):
            # GQA packing：同一 KV head 下的 group 个 q head 共享一次 K/V 读取
            gqa_splits = int(os.environ.get("LEANMOE_GQA_SPLITS", "2"))
            pacc = torch.empty(
                (num_seqs, num_kv_heads, gqa_splits, group, head_dim),
                dtype=torch.float32, device=query.device,
            )
            pm = torch.empty(
                (num_seqs, num_kv_heads, gqa_splits, group),
                dtype=torch.float32, device=query.device,
            )
            pl = torch.empty_like(pm)
            _paged_decode_gqa_split_kernel[(num_seqs, num_kv_heads, gqa_splits)](
                query, kv_cache, metadata.block_tables, metadata.seq_lens, pacc, pm, pl,
                query.stride(0), query.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                metadata.block_tables.stride(0), metadata.block_tables.stride(1),
                num_kv_heads,
                D=head_dim, BS=kv_cache.shape[1], S=gqa_splits, GROUP=group,
                BLOCK_N=128, BLOCK_D=128,
                scale=float(head_dim) ** -0.5, num_warps=4,
            )
            _gqa_combine_kernel[(num_seqs, num_kv_heads)](
                pacc, pm, pl, out,
                out.stride(0), out.stride(1), num_kv_heads,
                S=gqa_splits, GROUP=group, D=head_dim, BLOCK_D=128, num_warps=4,
            )
            return out
        num_splits = int(os.environ.get("LEANMOE_AR_SPLITS", "1"))
        block_n = int(os.environ.get("LEANMOE_ATTN_BLOCK_N", "128"))
        if num_splits <= 1:
            # 与 vLLM 的 num_splits 启发式一致：batch×heads 已超过 SM 数，无需切分
            _paged_decode_kernel[(num_seqs, num_heads)](
                query, kv_cache, metadata.block_tables, metadata.seq_lens, out,
                query.stride(0), query.stride(1),
                out.stride(0), out.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                metadata.block_tables.stride(0), metadata.block_tables.stride(1),
                num_heads, num_kv_heads,
                D=head_dim, BS=kv_cache.shape[1], BLOCK_N=block_n, BLOCK_D=128,
                scale=float(head_dim) ** -0.5, num_warps=4,
            )
            return out
        pacc = torch.empty(
            (num_seqs, num_heads, num_splits, head_dim),
            dtype=torch.float32, device=query.device,
        )
        pm = torch.empty(
            (num_seqs, num_heads, num_splits), dtype=torch.float32, device=query.device
        )
        pl = torch.empty_like(pm)
        _paged_decode_split_kernel[(num_seqs, num_heads, num_splits)](
            query, kv_cache, metadata.block_tables, metadata.seq_lens, pacc, pm, pl,
            query.stride(0), query.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            metadata.block_tables.stride(0), metadata.block_tables.stride(1),
            num_heads, num_kv_heads,
            D=head_dim, BS=kv_cache.shape[1], S=num_splits,
            BLOCK_N=block_n, BLOCK_D=128,
            scale=float(head_dim) ** -0.5, num_warps=4,
        )
        _split_combine_kernel[(num_seqs, num_heads)](
            pacc, pm, pl, out,
            out.stride(0), out.stride(1), num_heads,
            S=num_splits, D=head_dim, BLOCK_D=128, num_warps=4,
        )
        return out




_SDPA_FALLBACK = SDPABackend()
