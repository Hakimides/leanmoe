"""离线入口：加载权重、初始化引擎、批量生成。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from leanmoe.config import VllmConfig
from leanmoe.distributed.parallel_state import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from leanmoe.logger import get_logger
from leanmoe.sequence import Request, RequestOutput, SamplingParams
from leanmoe.utils import now
from leanmoe.runtime.core.block_pool import BlockPool
from leanmoe.runtime.core.kv_cache import KVCacheManager
from leanmoe.runtime.core.sched.scheduler import AsyncScheduler, Scheduler
from leanmoe.runtime.engine.core import EngineCore
from leanmoe.runtime.engine.output_processor import OutputProcessor
from leanmoe.runtime.kv_transfer.connector import create_kv_connector
from leanmoe.runtime.spec_decode.proposer import create_proposer
from leanmoe.runtime.worker.gpu_worker import Worker

logger = get_logger("llm")


def load_state_dict_from_dir(model_path: str | Path) -> dict[str, torch.Tensor]:
    """读取 safetensors 或 pytorch bin 分片，合并成 state_dict。"""
    model_path = Path(model_path)
    index_file = model_path / "model.safetensors.index.json"
    state: dict[str, torch.Tensor] = {}

    if index_file.exists():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = sorted(p.name for p in model_path.glob("*.safetensors"))
    if shards:
        from safetensors.torch import load_file

        for shard in shards:
            logger.info("加载分片 %s", shard)
            state.update(load_file(model_path / shard))
        return state

    # 回退：pytorch_model*.bin
    bins = sorted(model_path.glob("pytorch_model*.bin"))
    if not bins:
        raise FileNotFoundError(f"{model_path} 下没有找到权重文件")
    for path in bins:
        logger.info("加载分片 %s", path)
        state.update(torch.load(path, map_location="cpu", weights_only=True))
    return state


class LLM:
    def __init__(self, config: VllmConfig) -> None:
        self.config = config
        self.requests: dict[str, Request] = {}
        self._next_id = 0
        self._build_engine()

    # ---------- 初始化 ----------

    def _build_engine(self) -> None:
        state = self.config.parallel
        world_size = state.world_size
        init_distributed_environment(
            world_size=world_size,
            rank=int(os.environ.get("RANK", 0)),
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        )
        ps = initialize_model_parallel(
            tensor_parallel_size=state.tensor_parallel_size,
            data_parallel_size=state.data_parallel_size,
            pipeline_parallel_size=state.pipeline_parallel_size,
        )

        self.worker = Worker(self.config)
        self.worker.init_device()
        weight_state = load_state_dict_from_dir(self.config.model.model_path)
        self.worker.load_model(weight_state)
        del weight_state

        self.worker.determine_available_memory()
        self.worker.initialize_kv_cache()
        self.worker.capture_cuda_graphs()

        block_pool = BlockPool(
            self.worker.num_gpu_blocks,
            enable_caching=self.config.cache.enable_prefix_caching,
        )
        kv_manager = KVCacheManager(
            block_pool,
            self.config.cache.block_size,
            enable_caching=self.config.cache.enable_prefix_caching,
        )
        scheduler_cls = AsyncScheduler if self.config.scheduler.async_scheduling else Scheduler
        scheduler = scheduler_cls(self.config.scheduler, kv_manager)
        scheduler.set_max_model_len(self.config.model.max_model_len)

        proposer = create_proposer(self.config.speculative)
        kv_connector = create_kv_connector(self.config.kv_transfer)
        self.engine = EngineCore(
            self.config, self.worker, kv_manager, scheduler, proposer, kv_connector
        )
        self.output_processor = OutputProcessor()
        if self.config.collect_routed_experts and self.worker.runner.model is not None:
            self.worker.runner.model.set_collect_routed_experts(True)
        self.ps = ps

    # ---------- 生成 ----------

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[RequestOutput]:
        if isinstance(sampling_params, SamplingParams):
            params_list = [sampling_params] * len(prompts)
        elif sampling_params is None:
            params_list = [SamplingParams()] * len(prompts)
        else:
            params_list = sampling_params
        if len(params_list) != len(prompts):
            raise ValueError("prompts 与 sampling_params 数量不一致")

        request_ids: list[str] = []
        for prompt, params in zip(prompts, params_list):
            request_id = f"req-{self._next_id}"
            self._next_id += 1
            request = Request(
                request_id=request_id,
                prompt_token_ids=list(prompt),
                sampling_params=params,
                arrival_time=now(),
            )
            self.requests[request_id] = request
            self.output_processor.add_request(request)
            self.engine.add_request(request)
            request_ids.append(request_id)

        results: dict[str, RequestOutput] = {}
        idle_steps = 0
        while self.engine.has_unfinished:
            outputs = self.engine.step()
            for output in outputs.outputs:
                request = self.requests[output.request_id]
                results[output.request_id] = self.output_processor.process(
                    request, output.new_token_ids
                )
            if not outputs.outputs:
                idle_steps += 1
                if idle_steps > 32:
                    raise RuntimeError(
                        "连续 32 步没有推进，检查 KV 容量、max_model_len 或调度配置"
                    )
            else:
                idle_steps = 0

        ordered = [results[rid] for rid in request_ids if rid in results]
        for rid in request_ids:
            self.requests.pop(rid, None)
        return ordered

    # ---------- 清理 ----------

    def shutdown(self) -> None:
        # 显式断链，确保下一个实例能拿回显存（尤其在图捕获之后）
        if getattr(self, "worker", None) is not None:
            runner = self.worker.runner
            runner.kv_caches = []
            runner.model = None
            runner._graphs.clear()
        self.requests.clear()
        destroy_model_parallel()
        import gc

        gc.collect()
        torch.cuda.empty_cache()
