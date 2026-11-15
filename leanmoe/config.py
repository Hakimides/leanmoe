"""运行配置：模型、缓存、调度、并行、投机解码、量化、KV 传输、编译。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# 默认块大小（vLLM config/cache.py 同为 16）
DEFAULT_BLOCK_SIZE = 16
# 默认单步 token 预算
DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
# 默认 CUDA Graph 最大捕获 token 数
DEFAULT_MAX_CUDAGRAPH_CAPTURE_SIZE = 512


@dataclass
class ModelConfig:
    model_path: str
    dtype: str = "bfloat16"
    max_model_len: int = 32768
    tokenizer_path: str | None = None
    trust_remote_code: bool = False
    # 预转换 checkpoint 标记文件；存在时走快速加载路径
    layout_file: str = "leanmoe_layout.json"
    seed: int = 0
    # 是否在 CPU 上先加载再搬 GPU（30B 上更省峰值显存）
    load_on_cpu_first: bool = True

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("model_path 不能为空")
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"不支持的 dtype: {self.dtype}")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len 必须为正")

    @property
    def torch_dtype(self):
        import torch

        return getattr(torch, self.dtype)


@dataclass
class CacheConfig:
    block_size: int = DEFAULT_BLOCK_SIZE
    gpu_memory_utilization: float = 0.9
    # 默认开启，与 vLLM 一致
    enable_prefix_caching: bool = True
    num_gpu_blocks_override: int | None = None
    # 块哈希函数：sha256 / xxhash（未安装 xxhash 时自动回退）
    prefix_caching_hash_algo: Literal["sha256", "xxhash"] = "sha256"
    # 每个请求预留的 lookahead slot 数（投机解码用）
    num_lookahead_slots: int = 0

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size 必须为正")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization 必须在 (0, 1]")
        if self.num_gpu_blocks_override is not None and self.num_gpu_blocks_override <= 0:
            raise ValueError("num_gpu_blocks_override 必须为正")


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS
    # 与 max_num_batched_tokens 分离的输入预算；None 时等于 max_num_batched_tokens
    max_num_scheduled_tokens: int | None = None
    enable_chunked_prefill: bool = True
    # 长 prefill 单次最多调度多少 token；0 表示不限制
    long_prefill_token_threshold: int = 0
    policy: Literal["fcfs", "priority"] = "fcfs"
    # 异步调度：调度时给 decode 请求预加占位符，不等上一步采样。
    # 注意：当前 ModelRunner 尚未实现占位 token 的输入填充，默认关闭；
    # 打开前必须先让 _prepare_inputs 支持占位位置（否则 token 序列与 query 长度错位）。
    async_scheduling: bool = False
    # 单请求最多被抢占次数，超过则直接报错（防止重算风暴掩盖问题）
    max_num_preemptions: int = 64

    def __post_init__(self) -> None:
        if self.max_num_seqs <= 0 or self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_seqs / max_num_batched_tokens 必须为正")
        if self.max_num_scheduled_tokens is not None and self.max_num_scheduled_tokens <= 0:
            raise ValueError("max_num_scheduled_tokens 必须为正")


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    # 开 EP 后 MoE 层内部 tp_size 强制为 1，EP 组 = tp × dp × pp
    enable_expert_parallel: bool = False
    # 当前只实现 allgather_reducescatter（A100 上 vLLM 的默认且唯一可用后端）。
    # nccl_p2p 的 pair 级直投在 distributed/all2all.py 中保留接口，尚未接入 MoERunner。
    all2all_backend: Literal["allgather_reducescatter"] = "allgather_reducescatter"

    def __post_init__(self) -> None:
        for name in ("tensor_parallel_size", "data_parallel_size", "pipeline_parallel_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正")
        if self.enable_expert_parallel and self.tensor_parallel_size == 1 \
                and self.data_parallel_size == 1 and self.pipeline_parallel_size == 1:
            raise ValueError(
                "enable_expert_parallel 需要 tp×dp×pp > 1；单卡开 EP 无意义"
            )

    @property
    def world_size(self) -> int:
        return (
            self.tensor_parallel_size
            * self.data_parallel_size
            * self.pipeline_parallel_size
        )

    @property
    def expert_parallel_size(self) -> int:
        if not self.enable_expert_parallel:
            return 1
        return self.tensor_parallel_size * self.data_parallel_size * self.pipeline_parallel_size

    @property
    def moe_tensor_parallel_size(self) -> int:
        """EP 开启后 MoE 内部 tp_size 恒为 1。"""
        return 1 if self.enable_expert_parallel else self.tensor_parallel_size


@dataclass
class SpeculativeConfig:
    """投机解码。method=None 表示关闭。"""

    method: Literal["ngram", "draft_model", "mtp"] | None = None
    num_speculative_tokens: int = 0
    draft_model: str | None = None
    # ngram 提案器参数
    ngram_prompt_lookup_max: int = 4
    ngram_prompt_lookup_min: int = 1
    # 草稿模型张量并行度；None 时沿用目标模型
    draft_tensor_parallel_size: int | None = None

    def __post_init__(self) -> None:
        if self.method is None:
            if self.num_speculative_tokens != 0:
                raise ValueError("未开启投机解码时 num_speculative_tokens 必须为 0")
            return
        if self.num_speculative_tokens <= 0:
            raise ValueError("开启投机解码时 num_speculative_tokens 必须为正")
        if self.method == "draft_model" and not self.draft_model:
            raise ValueError("draft_model 方法必须提供 draft_model 路径")
        if self.method == "ngram" and self.ngram_prompt_lookup_max <= 0:
            raise ValueError("ngram_prompt_lookup_max 必须为正")


@dataclass
class KVTransferConfig:
    """PD 分离：prefill 实例把 KV 传给 decode 实例。"""

    kv_connector: Literal["none", "simple"] = "none"
    role: Literal["prefill", "decode", "both"] = "both"
    host: str = "127.0.0.1"
    port: int = 29500
    # 传输层级：layer（逐层）或 request（请求结束）
    transfer_mode: Literal["layer", "request"] = "layer"

    def __post_init__(self) -> None:
        if self.kv_connector == "none" and self.role != "both":
            raise ValueError("未启用 KV 连接器时 role 必须为 both")
        if self.kv_connector != "none" and self.role == "both":
            raise ValueError("启用 PD 分离时必须指定 prefill 或 decode")


@dataclass
class QuantizationConfig:
    """量化方法。none 表示 BF16/FP16。"""

    method: Literal["none", "fp8", "awq", "gptq"] = "none"
    # FP8 的动态/静态缩放；awq/gptq 使用 checkpoint 内的 scale
    fp8_dynamic_scale: bool = True
    # 反量化后的计算精度
    compute_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.method in ("awq", "gptq") and self.compute_dtype not in ("bfloat16", "float16"):
            raise ValueError("awq/gptq 的计算精度必须是 bf16 或 fp16")


@dataclass
class CompilationConfig:
    """CUDA Graph 与 torch.compile 相关。"""

    level: Literal["none", "piecewise", "full", "full_and_piecewise"] = "full_and_piecewise"
    cudagraph_capture_sizes: list[int] | None = None
    max_cudagraph_capture_size: int = DEFAULT_MAX_CUDAGRAPH_CAPTURE_SIZE
    # 是否允许超出桶的步退回 eager（退回会打日志并计数）
    allow_eager_fallback: bool = True
    # 每张图独立的持久输出缓冲，避免共享池覆写（旧引擎踩过的坑）
    separate_output_buffer: bool = True

    def __post_init__(self) -> None:
        if self.max_cudagraph_capture_size <= 0:
            raise ValueError("max_cudagraph_capture_size 必须为正")
        if self.cudagraph_capture_sizes is not None:
            if not self.cudagraph_capture_sizes:
                raise ValueError("cudagraph_capture_sizes 不能为空列表")
            if any(s <= 0 for s in self.cudagraph_capture_sizes):
                raise ValueError("cudagraph_capture_sizes 必须全为正")

    def resolved_capture_sizes(self, max_num_seqs: int) -> list[int]:
        """与 vLLM 相同的默认桶：1,2,4 + 8..248 步长 8 + 256..max 步长 16。"""
        if self.cudagraph_capture_sizes is not None:
            return sorted({s for s in self.cudagraph_capture_sizes})
        upper = min(max_num_seqs * 2, self.max_cudagraph_capture_size)
        sizes = [1, 2, 4]
        sizes.extend(range(8, min(256, upper + 1), 8))
        sizes.extend(range(256, upper + 1, 16))
        return sorted({s for s in sizes if s <= self.max_cudagraph_capture_size})


@dataclass
class VllmConfig:
    """聚合配置。所有子配置在这里完成跨模块校验。"""

    model: ModelConfig
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    kv_transfer: KVTransferConfig = field(default_factory=KVTransferConfig)
    compilation: CompilationConfig = field(default_factory=CompilationConfig)
    # 是否输出每个 token 的路由专家（调试/EP 验证）
    collect_routed_experts: bool = False
    # 关闭自定义 all-reduce，强制走 NCCL（排查用）
    disable_custom_all_reduce: bool = False

    def __post_init__(self) -> None:
        if self.speculative.method is not None and self.parallel.pipeline_parallel_size > 1:
            raise ValueError("投机解码暂不支持与 PP 组合")
        if self.quantization.method == "fp8" and not self._supports_fp8():
            raise ValueError(
                "FP8 W8A8 需要 SM89+；当前设备不支持。请改用 bf16 或 awq/gptq"
            )
        if self.cache.num_lookahead_slots and self.speculative.method is None:
            raise ValueError("num_lookahead_slots 只应在投机解码时设置")

    @staticmethod
    def _supports_fp8() -> bool:
        import torch

        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        return (major, minor) >= (8, 9)

    @classmethod
    def from_json(cls, path: str | Path) -> "VllmConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            cache=CacheConfig(**raw.get("cache", {})),
            scheduler=SchedulerConfig(**raw.get("scheduler", {})),
            parallel=ParallelConfig(**raw.get("parallel", {})),
            speculative=SpeculativeConfig(**raw.get("speculative", {})),
            quantization=QuantizationConfig(**raw.get("quantization", {})),
            kv_transfer=KVTransferConfig(**raw.get("kv_transfer", {})),
            compilation=CompilationConfig(**raw.get("compilation", {})),
            collect_routed_experts=raw.get("collect_routed_experts", False),
            disable_custom_all_reduce=raw.get("disable_custom_all_reduce", False),
        )
