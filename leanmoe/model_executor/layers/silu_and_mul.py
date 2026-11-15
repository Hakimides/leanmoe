"""融合 SiLU×mul：一个 Triton kernel 处理 SwiGLU。"""

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
    def _silu_and_mul_kernel(X, Y, I: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        pid = tl.program_id(1)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < I
        base = row * 2 * I
        gate = tl.load(X + base + cols, mask=mask, other=0.0)
        up = tl.load(X + base + I + cols, mask=mask, other=0.0)
        out = gate * tl.sigmoid(gate.to(tl.float32)) * up
        tl.store(Y + row * I + cols, out, mask=mask)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """x: [T, 2I] -> [T, I]。优先 Triton，失败回退 PyTorch。"""
    two_i = x.shape[-1]
    intermediate = two_i // 2
    if _HAS_TRITON and x.is_cuda and x.dim() == 2:
        out = torch.empty(
            (x.shape[0], intermediate), dtype=x.dtype, device=x.device
        )
        block = 128
        grid = (x.shape[0], triton.cdiv(intermediate, block))
        try:
            _silu_and_mul_kernel[grid](x, out, intermediate, block)
            return out
        except Exception:
            pass
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up
