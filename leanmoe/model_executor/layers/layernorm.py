"""归一化层。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm，优先使用 PyTorch 的融合实现。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype), requires_grad=False)
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self._use_fused = hasattr(F, "rms_norm")

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_fused:
            # 单 kernel 的 RMSNorm，避免 pow/mean/rsqrt/mul 多次 elementwise
            return F.rms_norm(
                x, (self.hidden_size,), self.weight, self.variance_epsilon
            )
        input_dtype = x.dtype
        xf = x.float()
        variance = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * xf.to(input_dtype)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is not None:
            from leanmoe.model_executor.layers.fused_norm import fused_add_rms_norm

            return fused_add_rms_norm(
                x, residual, self.weight, self.variance_epsilon
            )
        return self._norm(x)

    def extra_repr(self) -> str:
        return f"hidden={self.weight.numel()}, eps={self.variance_epsilon}"
