"""融合 QK-Norm + RoPE：一个 Triton kernel 完成 per-head RMSNorm 与旋转。"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _rmsnorm_rope_kernel(
        X, W, COS, SIN, POS, eps,
        ROW_STRIDE: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
    ):
        pid = tl.program_id(0)  # token * H + head
        row = pid // H
        head = pid % H
        lo = tl.arange(0, HALF)
        hi = lo + HALF
        # q/k 是融合 qkv 的切片，token 间 stride 不一定是 H*D
        base = row * ROW_STRIDE + head * D
        x_lo = tl.load(X + base + lo).to(tl.float32)
        x_hi = tl.load(X + base + hi).to(tl.float32)
        var = (tl.sum(x_lo * x_lo) + tl.sum(x_hi * x_hi)) / D
        scale = tl.rsqrt(var + eps)
        w_lo = tl.load(W + lo).to(tl.float32)
        w_hi = tl.load(W + hi).to(tl.float32)
        x_lo = x_lo * scale * w_lo
        x_hi = x_hi * scale * w_hi
        # cos/sin 表按 position 在 kernel 内索引（省掉每层一次 gather）
        pos = tl.load(POS + row).to(tl.int64)
        c = tl.load(COS + pos * D + lo).to(tl.float32)
        s = tl.load(SIN + pos * D + lo).to(tl.float32)
        out_lo = x_lo * c - x_hi * s
        out_hi = x_hi * c + x_lo * s
        tl.store(X + base + lo, out_lo.to(X.dtype.element_ty))
        tl.store(X + base + hi, out_hi.to(X.dtype.element_ty))


def fused_qk_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_heads: int,
    head_dim: int,
) -> None:
    """原地对 x [T, H, D] 做 RMSNorm 与 RoPE（cos/sin 传整表，按 positions 索引）。"""
    if _HAS_TRITON and x.is_cuda and head_dim % 2 == 0:
        try:
            _rmsnorm_rope_kernel[(x.shape[0] * num_heads,)](
                x, weight, cos, sin, positions, eps,
                ROW_STRIDE=x.stride(0),
                H=num_heads, D=head_dim, HALF=head_dim // 2,
            )
            return
        except Exception:
            pass
    cos = cos.index_select(0, positions.to(torch.long))
    sin = sin.index_select(0, positions.to(torch.long))
    # 回退：norm + 手写 rope
    normed = torch.nn.functional.rms_norm(x, (head_dim,), weight, eps)
    half = head_dim // 2
    x1, x2 = normed[..., :half], normed[..., half:]
    c = cos[:, None, :half]
    s = sin[:, None, :half]
    x[..., :half] = x1 * c - x2 * s
    x[..., half:] = x2 * c + x1 * s
