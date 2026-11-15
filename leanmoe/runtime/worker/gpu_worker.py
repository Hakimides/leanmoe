"""GPU Worker：显存测算、KV 初始化、图捕获、前向委托。"""

from __future__ import annotations

import torch

from leanmoe.config import VllmConfig
from leanmoe.logger import get_logger
from leanmoe.runtime.core.sched.output import SchedulerOutput
from leanmoe.runtime.worker.gpu_model_runner import GPUModelRunner, ModelRunnerOutput

logger = get_logger("gpu_worker")


class Worker:
    """单 rank 门面：设备、权重、KV、图与执行。"""
    def __init__(self, config: VllmConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = device
        self.runner = GPUModelRunner(config, device)
        self.available_kv_bytes = 0

    # ---------- 启动 ----------

    def init_device(self) -> None:
        """绑定 LOCAL_RANK 对应的 GPU。"""
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到 CUDA 设备")
        index = torch.cuda.current_device()
        torch.cuda.set_device(index)
        torch.cuda.empty_cache()
        props = torch.cuda.get_device_properties(index)
        logger.info(
            "设备 %d: %s, %.1f GiB, SM%d%d",
            index,
            props.name,
            props.total_memory / (1024 ** 3),
            props.major,
            props.minor,
        )

    def load_model(self, state: dict[str, torch.Tensor]) -> None:
        """构建模型并加载权重。"""
        self.runner.load_model()
        self.runner.load_weights(state)
        self.runner.post_load_quantize()
        # 推理引擎不需要梯度；不冻结会让 index_select(out=...) 等图安全路径报错
        self.runner.model.requires_grad_(False)

    def determine_available_memory(self) -> int:
        """与 vLLM 相同的顺序：预算 → profile 实测激活 → 扣图池。"""
        free, total = torch.cuda.mem_get_info()
        requested = int(total * self.config.cache.gpu_memory_utilization)
        used = total - free

        profile_peak = self.runner.profile_run()
        graph_reserve = self._estimate_graph_memory()
        available = min(requested - used, free) - profile_peak - graph_reserve
        if available <= 0:
            raise RuntimeError(
                f"可用 KV 显存为负：requested={requested}, used={used}, "
                f"activation_peak={profile_peak}, graph_reserve={graph_reserve}, free={free}。"
                "请降低 gpu_memory_utilization / max_num_seqs / max_cudagraph_capture_size"
            )
        self.available_kv_bytes = available
        logger.info(
            "显存预算：total=%.2f GiB, requested=%.2f, used=%.2f, "
            "activation_peak=%.2f, graph=%.2f, KV=%.2f",
            total / 1024 ** 3,
            requested / 1024 ** 3,
            used / 1024 ** 3,
            profile_peak / 1024 ** 3,
            graph_reserve / 1024 ** 3,
            available / 1024 ** 3,
        )
        return available

    def _estimate_graph_memory(self) -> int:
        """按 token 桶 × 长度桶估算所有图的 KV gather 缓冲。"""
        if self.runner.dispatcher.mode.name == "NONE":
            return 0
        sizes = self.runner.dispatcher.capture_sizes
        len_buckets = self.runner.dispatcher.len_buckets
        if not sizes or not len_buckets:
            return 0
        cfg = self.runner.model_cfg
        if cfg is None:
            return 0
        kv_heads = cfg.num_key_value_heads // self.config.parallel.tensor_parallel_size
        per_elem = self.config.model.torch_dtype.itemsize
        # 与 vLLM 相同的口径：共享池只占最大图，其余按每图独有部分累加。
        # 额外 ×2 为捕获期的中间激活留余量。
        shared = (
            max(sizes) * max(len_buckets) * kv_heads * 2 * cfg.head_dim * per_elem
        ) * 2
        per_graph = (
            min(max(sizes), 4) * max(len_buckets) * kv_heads * 2 * cfg.head_dim * per_elem
        )
        return int(shared + per_graph * len(sizes) * len(len_buckets))

    def initialize_kv_cache(self) -> None:
        """按显存预算分配 KV 块。"""
        self.runner.initialize_kv_cache(self.available_kv_bytes)

    def capture_cuda_graphs(self) -> None:
        """捕获 decode CUDA Graph。"""
        self.runner.capture_cuda_graphs()

    # ---------- 执行 ----------

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """交给 ModelRunner 执行一步。"""
        return self.runner.execute_model(scheduler_output)

    @property
    def num_gpu_blocks(self) -> int:
        return self.runner.num_gpu_blocks
