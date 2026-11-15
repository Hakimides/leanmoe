"""线性层：Replicated / ColumnParallel / RowParallel。"""

from __future__ import annotations

import torch
from torch import nn

from leanmoe.distributed.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from leanmoe.distributed.parallel_state import get_tp_group


class LinearBase(nn.Module):
    """所有线性层的基类，持有量化方法。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_method=None,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.quant_method = quant_method
        self.params_dtype = params_dtype
        if quant_method is not None:
            # 量化方法自行决定参数名与 dtype（fp8 / qweight / scales 等）
            quant_method.create_weights(self, input_size, output_size, params_dtype, bias)
        else:
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, dtype=params_dtype), requires_grad=False
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(output_size, dtype=params_dtype), requires_grad=False
                )
            else:
                self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_method is not None:
            return self.quant_method.apply(self, x)
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.input_size}, out={self.output_size}, "
            f"bias={self.has_bias}, dtype={self.params_dtype}"
        )


class ReplicatedLinear(LinearBase):
    """每个 rank 保存完整权重，不做切分。"""

    def __init__(self, input_size: int, output_size: int, bias: bool = False, **kwargs) -> None:
        super().__init__(input_size, output_size, bias, **kwargs)


class ColumnParallelLinear(LinearBase):
    """按输出维切分：每个 rank 持有 output_size / tp_size 行。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        gather_output: bool = False,
        **kwargs,
    ) -> None:
        self.tp_size = get_tp_group().world_size
        self.tp_rank = get_tp_group().rank_in_group
        if output_size % self.tp_size != 0:
            raise ValueError(f"output_size={output_size} 不能被 tp_size={self.tp_size} 整除")
        self.output_size_per_partition = output_size // self.tp_size
        self.gather_output = gather_output
        super().__init__(input_size, self.output_size_per_partition, bias, **kwargs)
        self.full_output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self.gather_output and self.tp_size > 1:
            out = tensor_model_parallel_all_gather(out, dim=-1)
        return out


class RowParallelLinear(LinearBase):
    """按输入维切分：每个 rank 持有 input_size / tp_size 列，输出后 all-reduce。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        reduce_results: bool = True,
        **kwargs,
    ) -> None:
        self.tp_size = get_tp_group().world_size
        self.tp_rank = get_tp_group().rank_in_group
        if input_size % self.tp_size != 0:
            raise ValueError(f"input_size={input_size} 不能被 tp_size={self.tp_size} 整除")
        self.input_size_per_partition = input_size // self.tp_size
        self.reduce_results = reduce_results
        super().__init__(self.input_size_per_partition, output_size, bias, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self.reduce_results and self.tp_size > 1:
            # 每卡只算了部分和，跨卡求和后才是完整输出
            out = tensor_model_parallel_all_reduce(out)
        return out


def create_linear(
    kind: str,
    input_size: int,
    output_size: int,
    bias: bool = False,
    quant_method=None,
    params_dtype: torch.dtype = torch.bfloat16,
    gather_output: bool = False,
) -> LinearBase:
    """按名字创建线性层，供模型代码统一调用。"""
    kwargs = dict(
        bias=bias,
        quant_method=quant_method,
        params_dtype=params_dtype,
    )
    if kind == "replicated":
        return ReplicatedLinear(input_size, output_size, **kwargs)
    if kind == "column":
        return ColumnParallelLinear(
            input_size, output_size, gather_output=gather_output, **kwargs
        )
    if kind == "row":
        return RowParallelLinear(input_size, output_size, **kwargs)
    raise ValueError(f"未知的线性层类型: {kind}")
