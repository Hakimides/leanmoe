"""AWQ（4bit 权重量化）参考实现。"""

from __future__ import annotations

import torch
from torch import nn

from leanmoe.model_executor.quantization.base import QuantizeMethodBase


class AWQLinearMethod(QuantizeMethodBase):
    def __init__(self, group_size: int = 128) -> None:
        self.group_size = group_size

    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        has_bias: bool,
    ) -> None:
        # AWQ 的 qweight 是 int32，每 8 个 4bit 权重打包成一个 int32
        layer.qweight = nn.Parameter(
            torch.empty(input_size // 8, output_size, dtype=torch.int32),
            requires_grad=False,
        )
        num_groups = (input_size + self.group_size - 1) // self.group_size
        layer.qzeros = nn.Parameter(
            torch.empty(num_groups, output_size // 8, dtype=torch.int32),
            requires_grad=False,
        )
        layer.scales = nn.Parameter(
            torch.empty(num_groups, output_size, dtype=params_dtype),
            requires_grad=False,
        )
        if has_bias:
            layer.bias = nn.Parameter(
                torch.empty(output_size, dtype=params_dtype), requires_grad=False
            )
        else:
            layer.bias = None
        layer.compute_dtype = params_dtype

    @staticmethod
    def _unpack_int4(packed: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """把 int32 里 8 个 4bit 解包到 dim 维。"""
        shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
        unpacked = (packed.unsqueeze(-1) >> shifts) & 0xF
        return unpacked.reshape(*packed.shape, -1).to(torch.float32)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # qweight: [in/8, out] -> [in, out]
        w = self._unpack_int4(layer.qweight, dim=0)
        w = w.reshape(-1, w.shape[-1])
        zeros = self._unpack_int4(layer.qzeros, dim=0).reshape(-1, w.shape[-1])
        scales = layer.scales.to(torch.float32)
        if zeros.shape[0] != scales.shape[0]:
            # group 数与 zeros 行数不一致时按最后一组补齐
            zeros = zeros[: scales.shape[0]]
        dequant = (w - zeros.repeat_interleave(self.group_size, dim=0)[: w.shape[0]]) * \
                  scales.repeat_interleave(self.group_size, dim=0)[: w.shape[0]]
        layer.weight = nn.Parameter(dequant.t().contiguous().to(layer.compute_dtype), requires_grad=False)
        del layer.qweight, layer.qzeros, layer.scales

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight, layer.bias)
