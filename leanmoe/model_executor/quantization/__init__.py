"""量化方法注册表。"""

from __future__ import annotations

from leanmoe.config import QuantizationConfig
from leanmoe.model_executor.quantization.awq import AWQLinearMethod
from leanmoe.model_executor.quantization.base import (
    QuantizeMethodBase,
    UnquantizedEmbeddingMethod,
    UnquantizedLinearMethod,
)
from leanmoe.model_executor.quantization.fp8 import FP8LinearMethod
from leanmoe.model_executor.quantization.gptq import GPTQLinearMethod

__all__ = [
    "AWQLinearMethod",
    "FP8LinearMethod",
    "GPTQLinearMethod",
    "QuantizeMethodBase",
    "UnquantizedEmbeddingMethod",
    "UnquantizedLinearMethod",
    "get_linear_quant_method",
    "get_embedding_quant_method",
]


def get_linear_quant_method(config: QuantizationConfig) -> QuantizeMethodBase:
    if config.method == "none":
        return UnquantizedLinearMethod()
    if config.method == "fp8":
        return FP8LinearMethod(dynamic_scale=config.fp8_dynamic_scale)
    if config.method == "awq":
        return AWQLinearMethod()
    if config.method == "gptq":
        return GPTQLinearMethod()
    raise ValueError(f"未知的量化方法: {config.method}")


def get_embedding_quant_method(config: QuantizationConfig) -> UnquantizedEmbeddingMethod:
    # embedding 与 lm_head 暂不量化
    return UnquantizedEmbeddingMethod()
