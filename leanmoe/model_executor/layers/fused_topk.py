"""融合路由：softmax + top-k + 可选 renormalize，一个 kernel。"""

from __future__ import annotations

import torch

from leanmoe.logger import get_logger, once

logger = get_logger("fused_topk")

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _topk_softmax_kernel(
        LOGITS, IDS, WTS, num_tokens,
        E: tl.constexpr, K: tl.constexpr,
        NORM: tl.constexpr, BLOCK_E: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= num_tokens:
            return
        cols = tl.arange(0, BLOCK_E)
        mask = cols < E
        x = tl.load(LOGITS + row * E + cols, mask=mask, other=float("-inf")).to(tl.float32)
        m = tl.max(x, axis=0)
        e = tl.exp(x - m)
        s = tl.sum(tl.where(mask, e, 0.0), axis=0)
        probs = e / s

        for j in range(0, K):
            best = tl.max(probs, axis=0)
            idx = tl.argmax(probs, axis=0)
            tl.store(IDS + row * K + j, idx.to(tl.int64))
            tl.store(WTS + row * K + j, best.to(WTS.dtype.element_ty))
            probs = tl.where(cols == idx, float("-inf"), probs)

        if NORM:
            wsum = 0.0
            for j in range(0, K):
                wsum += tl.load(WTS + row * K + j).to(tl.float32)
            for j in range(0, K):
                w = tl.load(WTS + row * K + j).to(tl.float32)
                tl.store(WTS + row * K + j, (w / wsum).to(WTS.dtype.element_ty))


def topk_softmax(
    logits: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """logits [T, E] -> (ids [T,K] int64, weights [T,K])。"""
    num_tokens, num_experts = logits.shape
    ids = torch.empty((num_tokens, top_k), dtype=torch.long, device=logits.device)
    weights = torch.empty((num_tokens, top_k), dtype=logits.dtype, device=logits.device)
    if _HAS_TRITON and logits.is_cuda and num_tokens > 0:
        try:
            block_e = triton.next_power_of_2(num_experts)
            _topk_softmax_kernel[(num_tokens,)](
                logits, ids, weights, num_tokens,
                E=num_experts, K=top_k,
                NORM=norm_topk_prob, BLOCK_E=block_e,
            )
            return ids, weights
        except Exception as exc:
            once.warn(logger, "topk_fail", f"融合 topk_softmax 失败，回退 torch: {exc}")
    probs = torch.softmax(logits.float(), dim=-1)
    w, idx = torch.topk(probs, top_k, dim=-1)
    if norm_topk_prob:
        w = w / w.sum(dim=-1, keepdim=True)
    return idx, w.to(logits.dtype)
