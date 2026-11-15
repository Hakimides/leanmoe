"""FP8（W8A8）参考实现。"""

from __future__ import annotations

import torch
from torch import nn

from leanmoe.model_executor.quantization.base import QuantizeMethodBase


class FP8LinearMethod(QuantizeMethodBase):
    def __init__(self, dynamic_scale: bool = True) -> None:
        self.dynamic_scale = dynamic_scale

    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        has_bias: bool,
    ) -> None:
        layer.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        # per-output-channel scale，float32
        layer.weight_scale = nn.Parameter(
            torch.ones(output_size, dtype=torch.float32), requires_grad=False
        )
        if has_bias:
            layer.bias = nn.Parameter(
                torch.empty(output_size, dtype=params_dtype), requires_grad=False
            )
        else:
            layer.bias = None
        layer.compute_dtype = params_dtype

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # 反量化一次，换回 bf16 权重，避免每步做逐通道乘法
        scale = layer.weight_scale.to(torch.float32).unsqueeze(-1)
        layer.weight = nn.Parameter(
            (layer.weight.to(torch.float32) * scale).to(layer.compute_dtype),
            requires_grad=False,
        )
        del layer.weight_scale

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight, layer.bias)
