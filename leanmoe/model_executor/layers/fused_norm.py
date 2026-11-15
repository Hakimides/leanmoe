"""融合残差加法 + RMSNorm（一个 Triton kernel）。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _fused_add_rms_norm_kernel(
        X, R, W, eps, N: tl.constexpr, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        s = x + r
        tl.store(R + row * N + cols, s.to(R.dtype.element_ty), mask=mask)
        var = tl.sum(s * s, axis=0) / N
        normed = s * tl.rsqrt(var + eps)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            X + row * N + cols,
            (normed * w).to(X.dtype.element_ty),
            mask=mask,
        )


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (normed, new_residual)；new_residual 原地写入 residual。"""
    hidden = x.shape[-1]
    if _HAS_TRITON and x.is_cuda and x.dim() == 2:
        block = triton.next_power_of_2(hidden)
        try:
            _fused_add_rms_norm_kernel[(x.shape[0],)](
                x, residual, weight, eps, hidden, block
            )
            return x, residual
        except Exception:
            pass
    new_residual = x + residual
    normed = F.rms_norm(new_residual, (hidden,), weight, eps)
    return normed, new_residual
