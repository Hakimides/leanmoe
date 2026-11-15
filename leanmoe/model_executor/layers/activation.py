"""激活函数。"""

from __future__ import annotations

import torch
from torch import nn

from leanmoe.model_executor.layers.silu_and_mul import silu_and_mul


class SiluAndMul(nn.Module):
    """SwiGLU：把 [..., 2*I] 切成 gate/up，返回 silu(gate) * up。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x)
