"""模型注册表。"""

from leanmoe.model_executor.models.qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    normalize_moe_state_dict,
)

_REGISTRY = {
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
}


def get_model_class(architecture: str):
    if architecture not in _REGISTRY:
        raise ValueError(f"不支持的架构: {architecture}；当前只注册了 {list(_REGISTRY)}")
    return _REGISTRY[architecture]


__all__ = [
    "Qwen3MoeConfig",
    "Qwen3MoeForCausalLM",
    "get_model_class",
    "normalize_moe_state_dict",
]
