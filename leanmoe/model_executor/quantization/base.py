"""量化方法基类。"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class QuantizeMethodBase(Protocol):
    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        has_bias: bool,
    ) -> None:
        ...

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        ...

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        ...


class UnquantizedLinearMethod:
    """BF16/FP16 参考路径。"""

    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        has_bias: bool,
    ) -> None:
        layer.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=params_dtype), requires_grad=False
        )
        if has_bias:
            layer.bias = nn.Parameter(
                torch.empty(output_size, dtype=params_dtype), requires_grad=False
            )
        else:
            layer.bias = None

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        return None

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight, layer.bias)


class UnquantizedEmbeddingMethod:
    """embedding 与 lm_head 共用的参考路径。"""

    def create_weights(
        self,
        layer: nn.Module,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype,
    ) -> None:
        layer.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=params_dtype),
            requires_grad=False,
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        return None

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(x, layer.weight)
