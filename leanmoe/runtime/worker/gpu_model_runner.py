"""GPU ModelRunner：输入组装、前向、采样、CUDA Graph 回放。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

from leanmoe.config import VllmConfig
from leanmoe.distributed.parallel_state import get_parallel_state, get_pp_group, get_tp_group
from leanmoe.logger import get_logger, once
from leanmoe.model_executor.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from leanmoe.sequence import Request
from leanmoe.utils import cdiv
from leanmoe.runtime.attention.backend import AttentionMetadata
from leanmoe.runtime.core.kv_cache import (
    allocate_kv_cache_tensors,
    compute_kv_cache_bytes,
    get_num_gpu_blocks,
)
from leanmoe.runtime.core.sched.output import SchedulerOutput
from leanmoe.runtime.sample.sampler import Sampler
from leanmoe.runtime.worker.cudagraph_dispatcher import (
    CUDAGraphDispatcher,
    CUDAGraphMode,
)

logger = get_logger("model_runner")


@dataclass
class InputBatch:
    """一步前向的扁平批张量与索引表。"""
    token_ids: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor
    logits_indices: torch.Tensor
    num_seqs: int
    uniform_decode: bool
    # 由 Python 侧已知值传入，避免每步 .item() 的 D2H 同步
    max_seq_len: int = 0
    max_query_len: int = 1
    # 所有请求都是首块且等长时的长度，用于批量 prefill
    uniform_prefill_len: int | None = None
    # 与 logits_indices 一一对应的请求采样区间
    sampling_spans: list[tuple[str, int, int]] = field(default_factory=list)
    sampling_request_ids: list[str] = field(default_factory=list)
    # 每步由模型写入的路由记录（可选）
    collect_routed_experts: bool = False


@dataclass
class ModelRunnerOutput:
    """本步采样结果与时间戳。"""
    sampled_token_ids: dict[str, int] = field(default_factory=dict)
    sampled_token_times: dict[str, float] = field(default_factory=dict)
    accepted_drafts: dict[str, list[int]] = field(default_factory=dict)
    routed_experts: dict[str, torch.Tensor] = field(default_factory=dict)
    # 本步实际使用的图模式，供指标统计
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE


class GPUModelRunner:
    """模型执行器：组批、图分发/回放、前向、采样。"""
    def __init__(self, config: VllmConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = device
        self.model: Qwen3MoeForCausalLM | None = None
        self.model_cfg: Qwen3MoeConfig | None = None
        self.kv_caches: list[torch.Tensor] = []
        self.num_gpu_blocks = 0
        self.block_size = config.cache.block_size
        self.dispatcher = CUDAGraphDispatcher(
            config.compilation,
            config.scheduler.max_num_seqs,
            config.model.max_model_len,
        )
        self.sampler = Sampler()
        self._graphs: dict[tuple[int, int, int], dict] = {}
        self._graph_mode_count = {m: 0 for m in CUDAGraphMode}
        self._max_blocks_per_seq = 0
        self._debug_steps = 0
        # lm_head 分块大小：0 表示不分块（0.6B/16 batch 实测最快）
        self._vocab_chunk = int(os.environ.get("LEANMOE_VOCAB_CHUNK", "0"))

    # ---------- 模型 ----------

    def load_model(self) -> None:
        """构建模型结构并移到本卡。"""
        self.model_cfg = Qwen3MoeConfig.from_pretrained(self.config.model.model_path)
        self.model = Qwen3MoeForCausalLM(
            cfg=self.model_cfg,
            parallel=self.config.parallel,
            quant=self.config.quantization,
            params_dtype=self.config.model.torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.attn_backend_name = "sdpa"
        # 图回放是否需要重建 kv_indices/kv_mask（只有 SDPA 路径需要）
        self._graph_need_kv_fill = True

    def load_weights(self, state: dict[str, torch.Tensor]) -> None:
        """填充按 TP 切分后的权重。"""
        assert self.model is not None
        missing = self.model.load_weights(state, self.config.model.model_path)
        if missing:
            raise RuntimeError(
                f"权重缺失 {len(missing)} 个，前几个: {missing[:4]}；"
                "检查 checkpoint 布局与 routing 键名映射"
            )

    def post_load_quantize(self) -> None:
        """量化层在加载后做一次反量化/重排。"""
        assert self.model is not None
        for module in self.model.modules():
            method = getattr(module, "quant_method", None)
            if method is not None and hasattr(method, "process_weights_after_loading"):
                method.process_weights_after_loading(module)

    # ---------- 显存与 KV ----------

    @torch.inference_mode()
    def profile_run(self) -> int:
        """用最大 batch 跑一次 eager，返回激活峰值字节数。"""
        assert self.model is not None and self.model_cfg is not None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        max_tokens = min(
            self.config.scheduler.max_num_batched_tokens,
            self.config.model.max_model_len * self.config.scheduler.max_num_seqs,
        )
        # 与 vLLM 的 profile_run 一致：把 token 均分到 max_num_seqs 条请求，
        # 避免"一条超长序列"把 SDPA 逼到 O(L^2) 的最坏显存画像。
        num_reqs = max(1, min(max_tokens, self.config.scheduler.max_num_seqs))
        per_req = max_tokens // num_reqs
        lens = [per_req] * num_reqs
        lens[-1] += max_tokens % num_reqs
        dummy = torch.zeros(max_tokens, dtype=torch.long, device=self.device)
        positions = torch.arange(max_tokens, device=self.device)
        seq_lens = torch.tensor(lens, dtype=torch.int32, device=self.device)
        metadata = self._metadata_no_cache(num_reqs, max_tokens, dummy, positions, seq_lens)
        # 空 KV 列表会让 forward 跳过所有层；这里用 None 占位
        self.model(dummy, positions, [None] * self.model_cfg.num_hidden_layers, metadata)
        peak = torch.cuda.max_memory_allocated()
        return max(0, peak - base)

    def initialize_kv_cache(self, available_bytes: int) -> None:
        """按块数分配 KV buffer 与每层视图。"""
        assert self.model_cfg is not None
        cfg = self.model_cfg
        state = get_parallel_state()
        # KV 头数按 TP 切分；EP 时 attention 复制，不切
        kv_heads = cfg.num_key_value_heads // self.config.parallel.tensor_parallel_size
        self.num_gpu_blocks = get_num_gpu_blocks(
            available_bytes,
            cfg.num_hidden_layers,
            kv_heads,
            cfg.head_dim,
            self.block_size,
            self.config.model.torch_dtype.itemsize,
            self.config.cache.num_gpu_blocks_override,
        )
        self.kv_caches = allocate_kv_cache_tensors(
            cfg.num_hidden_layers,
            self.num_gpu_blocks,
            self.block_size,
            kv_heads,
            cfg.head_dim,
            self.config.model.torch_dtype,
            self.device,
        )
        self._max_blocks_per_seq = cdiv(self.config.model.max_model_len, self.block_size)
        logger.info(
            "KV cache 分配完成：blocks=%d block_size=%d kv_heads=%d，约 %.2f GiB",
            self.num_gpu_blocks,
            self.block_size,
            kv_heads,
            self.num_gpu_blocks
            * compute_kv_cache_bytes(
                1, kv_heads, cfg.head_dim, self.block_size,
                self.config.model.torch_dtype.itemsize,
            )
            * cfg.num_hidden_layers
            / (1024 ** 3),
        )

    # ---------- 前向 ----------

    @torch.inference_mode()
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """调度输出 → 组批 → 图/eager 前向 → 采样。"""
        requests = (
            scheduler_output.scheduled_new_reqs + scheduler_output.scheduled_cached_reqs
        )
        self._request_map = {r.request_id: r for r in requests}
        batch = self._prepare_inputs(scheduler_output)
        max_seq_len = batch.max_seq_len if batch.num_seqs else 1
        if self._debug_steps < 12:
            import os

            if os.environ.get("LEANMOE_DEBUG_STEP") == "1":
                print(
                    f"[step {self._debug_steps}] seqs={batch.num_seqs} "
                    f"tokens={batch.token_ids.numel()} "
                    f"uniform_prefill={batch.uniform_prefill_len} "
                    f"uniform_decode={batch.uniform_decode} max_len={max_seq_len}",
                    flush=True,
                )
                self._debug_steps += 1
        requests_all = [self._request_map[rid] for rid in batch.sampling_request_ids]
        graph_safe = all(
            r.sampling_params.is_greedy and not r.spec_token_ids for r in requests_all
        )
        if graph_safe:
            mode, descriptor = self.dispatcher.dispatch(
                num_tokens=batch.token_ids.numel(),
                num_reqs=batch.num_seqs,
                uniform_decode=batch.uniform_decode,
                max_seq_len=max_seq_len,
            )
        else:
            mode, descriptor = CUDAGraphMode.NONE, None
        self._graph_mode_count[mode] += 1

        if mode == CUDAGraphMode.FULL and descriptor is not None:
            tokens = self._replay_graph(batch, descriptor)
            if tokens is not None:
                result = ModelRunnerOutput(cudagraph_mode=mode)
                import time as _time

                now = _time.perf_counter()
                for req, token in zip(requests_all, tokens.tolist()):
                    result.sampled_token_ids[req.request_id] = int(token)
                    result.sampled_token_times[req.request_id] = now
                if batch.collect_routed_experts and self.model is not None:
                    experts = self.model.collect_routed_experts(batch.token_ids.numel())
                    if experts is not None:
                        self._split_routed_experts(scheduler_output, experts, result)
                return result
            # 图未命中：继续走下面的 eager 路径

        metadata = self._build_attn_metadata(batch)
        hidden_states = self._forward(batch, metadata)

        result = ModelRunnerOutput(cudagraph_mode=mode)
        state = get_parallel_state()
        if not state.is_last_pp_stage:
            # 非末段只负责把 hidden 传给下一段，不采样
            return result
        self._sample(batch, hidden_states, result)
        if batch.collect_routed_experts and self.model is not None:
            experts = self.model.collect_routed_experts(batch.token_ids.numel())
            if experts is not None:
                self._split_routed_experts(scheduler_output, experts, result)
        return result

    def _forward(self, batch: InputBatch, metadata: AttentionMetadata) -> torch.Tensor:
        assert self.model is not None
        state = get_parallel_state()
        if not state.is_first_pp_stage:
            hidden = self._pp_recv(batch.token_ids.numel())
        else:
            hidden = None
        out = self.model(
            input_ids=batch.token_ids if state.is_first_pp_stage else None,
            positions=batch.positions,
            kv_caches=self.kv_caches,
            attn_metadata=metadata,
            hidden_states=hidden,
        )
        if not state.is_last_pp_stage:
            self._pp_send(out)
            return out
        return out

    # ---------- 输入 ----------

    def _prepare_inputs(self, scheduler_output: SchedulerOutput) -> InputBatch:
        requests: list[Request] = (
            scheduler_output.scheduled_new_reqs + scheduler_output.scheduled_cached_reqs
        )
        # 纯 decode 快速路径：全部请求各 1 token、无草稿，一次性构建
        if requests and all(
            scheduler_output.num_scheduled_tokens.get(r.request_id, 0) == 1
            and r.num_computed_tokens >= r.num_prompt_tokens
            and not r.spec_token_ids
            for r in requests
        ):
            return self._prepare_decode_fast(requests)
        token_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        seq_lens: list[int] = []
        start_positions: list[int] = []
        num_news: list[int] = []
        query_start = [0]
        logits_indices: list[int] = []
        sampling_spans: list[tuple[str, int, int]] = []
        sampling_request_ids: list[str] = []
        block_tables: list[list[int]] = []
        uniform = True

        for req in requests:
            num_new = scheduler_output.num_scheduled_tokens[req.request_id]
            start = req.num_computed_tokens
            num_news.append(num_new)
            start_positions.append(start)
            if num_new == 1 and start >= req.num_prompt_tokens:
                # decode 快速路径：不要拼接整个 prompt（旧实现每步复制 512 token）
                ids = [req.output_token_ids[-1]]
            else:
                ids = self._tokens_with_spec(req)[start : start + num_new]
            if len(ids) < num_new:
                # 防御：async scheduling 的占位位置还没有真实 token 时用 0 填充。
                # 这些位置的 logits 会在采样阶段被丢弃（当前默认关闭 async）。
                ids = ids + [0] * (num_new - len(ids))
            token_ids.extend(ids)
            positions.extend(range(start, start + num_new))
            slots.extend(self._slot_mapping(req.block_ids, start, num_new))
            end = start + num_new
            seq_lens.append(end)
            query_start.append(query_start[-1] + num_new)
            # 只有算到序列末尾的请求才需要采样（O(1)，不拼接 token 列表）
            if end == req.num_tokens + len(req.spec_token_ids):
                span_start = len(logits_indices)
                # logits 行号必须是 batch 内的全局行号，不是请求内绝对位置
                if req.spec_token_ids:
                    logits_indices.extend(range(query_start[-1] - num_new, query_start[-1]))
                else:
                    logits_indices.append(query_start[-1] - 1)
                sampling_spans.append((req.request_id, span_start, len(logits_indices)))
                sampling_request_ids.append(req.request_id)
            else:
                uniform = False
            block_tables.append(list(req.block_ids))
            if not (num_new == 1 and start >= req.num_prompt_tokens):
                # 只有"已过 prefill 且本步 1 token"才是 uniform decode
                uniform = False

        max_blocks = max((len(b) for b in block_tables), default=1)
        # 块表补 0 到同一宽度；attention 只按 seq_lens 读有效范围
        padded = [b + [0] * (max_blocks - len(b)) for b in block_tables]
        # 所有请求都是首块且等长 -> 可走批量 prefill
        uniform_prefill_len = None
        if (
            start_positions
            and all(s == 0 for s in start_positions)
            and len(set(num_news)) == 1
        ):
            uniform_prefill_len = num_news[0]
        return InputBatch(
            token_ids=torch.tensor(token_ids, dtype=torch.long, device=self.device),
            positions=torch.tensor(positions, dtype=torch.long, device=self.device),
            query_start_loc=torch.tensor(query_start, dtype=torch.int32, device=self.device),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            block_tables=torch.tensor(padded, dtype=torch.int32, device=self.device),
            slot_mapping=torch.tensor(slots, dtype=torch.long, device=self.device),
            logits_indices=torch.tensor(logits_indices, dtype=torch.long, device=self.device),
            num_seqs=len(requests),
            uniform_decode=uniform and len(requests) > 0,
            max_seq_len=max(seq_lens) if seq_lens else 0,
            max_query_len=max(num_news) if num_news else 1,
            uniform_prefill_len=uniform_prefill_len,
            sampling_spans=sampling_spans,
            sampling_request_ids=sampling_request_ids,
            collect_routed_experts=self.config.collect_routed_experts,
        )

    def _prepare_decode_fast(self, requests: list[Request]) -> InputBatch:
        """decode 专用：避免逐请求函数调用与多次 list append。"""
        bs = self.block_size
        n = len(requests)
        token_ids = [r.output_token_ids[-1] for r in requests]
        positions = [r.num_computed_tokens for r in requests]
        slots = [
            r.block_ids[p // bs] * bs + p % bs
            for r, p in zip(requests, positions)
        ]
        seq_lens = [p + 1 for p in positions]
        block_tables = [r.block_ids for r in requests]
        max_blocks = max((len(b) for b in block_tables), default=1)
        # 块表补 0 到同一宽度；attention 只按 seq_lens 读有效范围
        padded = [b + [0] * (max_blocks - len(b)) for b in block_tables]
        return InputBatch(
            token_ids=torch.tensor(token_ids, dtype=torch.long, device=self.device),
            positions=torch.tensor(positions, dtype=torch.long, device=self.device),
            query_start_loc=torch.arange(n + 1, dtype=torch.int32, device=self.device),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            block_tables=torch.tensor(padded, dtype=torch.int32, device=self.device),
            slot_mapping=torch.tensor(slots, dtype=torch.long, device=self.device),
            logits_indices=torch.arange(n, dtype=torch.long, device=self.device),
            num_seqs=n,
            uniform_decode=True,
            max_seq_len=max(seq_lens),
            max_query_len=1,
            sampling_spans=[(r.request_id, i, i + 1) for i, r in enumerate(requests)],
            sampling_request_ids=[r.request_id for r in requests],
            collect_routed_experts=self.config.collect_routed_experts,
        )

    @staticmethod
    def _tokens_with_spec(req: Request) -> list[int]:
        return req.prompt_token_ids + req.output_token_ids + req.spec_token_ids

    def _slot_mapping(self, block_ids: list[int], start: int, num_new: int) -> list[int]:
        slots = []
        for pos in range(start, start + num_new):
            block = block_ids[pos // self.block_size]
            slots.append(block * self.block_size + pos % self.block_size)
        return slots

    def _build_attn_metadata(self, batch: InputBatch) -> AttentionMetadata:
        num_prefill_seqs = 0
        num_decode_seqs = 0
        for i in range(batch.num_seqs):
            qlen = int(batch.query_start_loc[i + 1] - batch.query_start_loc[i])
            if qlen == 1:
                num_decode_seqs += 1
            else:
                num_prefill_seqs += 1
        max_q = batch.max_query_len if batch.num_seqs else 1
        max_seqlen_k = batch.max_seq_len if batch.num_seqs else 0
        decode_cu = None
        if batch.uniform_decode:
            decode_cu = torch.arange(
                batch.num_seqs + 1, dtype=torch.int32, device=self.device
            )
        return AttentionMetadata(
            num_seqs=batch.num_seqs,
            num_prefill_seqs=num_prefill_seqs,
            num_decode_seqs=num_decode_seqs,
            num_prefill_tokens=int(batch.token_ids.numel()) - num_decode_seqs,
            num_decode_tokens=num_decode_seqs,
            query_start_loc=batch.query_start_loc,
            seq_lens=batch.seq_lens,
            block_tables=batch.block_tables,
            slot_mapping=batch.slot_mapping,
            is_pure_decode=batch.uniform_decode,
            max_query_len=max_q,
            uniform_prefill_len=batch.uniform_prefill_len,
            max_seqlen_k=max_seqlen_k,
            decode_cu_seqlens=decode_cu,
            is_capturing=False,
        )

    def _metadata_no_cache(
        self,
        num_seqs: int,
        total_tokens: int,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> AttentionMetadata:
        query_start_loc = torch.zeros(
            num_seqs + 1, dtype=torch.int32, device=self.device
        )
        torch.cumsum(seq_lens, dim=0, out=query_start_loc[1:])
        return AttentionMetadata(
            num_seqs=num_seqs,
            num_prefill_seqs=num_seqs,
            num_decode_seqs=0,
            num_prefill_tokens=total_tokens,
            num_decode_tokens=0,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_tables=torch.zeros((num_seqs, 1), dtype=torch.int32, device=self.device),
            slot_mapping=torch.arange(total_tokens, dtype=torch.long, device=self.device),
            is_pure_decode=False,
            max_query_len=int(seq_lens.max().item()),
            is_capturing=False,
        )

    # ---------- 采样 ----------

    def _sample(
        self, batch: InputBatch, hidden_states: torch.Tensor, result: ModelRunnerOutput
    ) -> None:
        if batch.logits_indices.numel() == 0 or self.model is None:
            return
        import time

        hidden = hidden_states[batch.logits_indices]
        now = time.perf_counter()

        has_spec = any(
            self._request_map[rid].spec_token_ids for rid in batch.sampling_request_ids
        )
        if not has_spec:
            requests = [self._request_map[rid] for rid in batch.sampling_request_ids]
            params = [r.sampling_params for r in requests]
            if all(p.is_greedy for p in params):
                # 分块 argmax，避免物化 [B, vocab]
                tokens = self.model.compute_greedy_tokens(
                    hidden, self._vocab_chunk
                )
                for req, token in zip(requests, tokens.tolist()):
                    result.sampled_token_ids[req.request_id] = int(token)
                    result.sampled_token_times[req.request_id] = now
                return
            sampled = self.sampler.sample(
                self.model.compute_logits(hidden), params
            )
            for req, token in zip(requests, sampled.tolist()):
                result.sampled_token_ids[req.request_id] = int(token)
                result.sampled_token_times[req.request_id] = now
            return

        # 混合批：逐请求处理（投机验证需要每个草稿位置的 logits）
        logits = self.model.compute_logits(hidden)
        for request_id, start, end in batch.sampling_spans:
            request = self._request_map[request_id]
            rows = logits[start:end]
            if request.spec_token_ids:
                accepted = 0
                # 可接受草稿上限还要受 max_tokens 剩余额度约束：
                # 本步最多再生成 remaining 个 token，其中最后一个留给修正 token。
                remaining = (
                    request.sampling_params.max_tokens - request.num_output_tokens
                )
                limit = min(
                    len(request.spec_token_ids),
                    rows.shape[0] - 1,
                    max(0, remaining - 1),
                )
                for j in range(limit):
                    if int(rows[j].argmax().item()) == request.spec_token_ids[j]:
                        accepted += 1
                    else:
                        break
                pos = min(accepted, rows.shape[0] - 1)
                token = self.sampler.sample(
                    rows[pos : pos + 1], [request.sampling_params]
                )[0]
                result.accepted_drafts[request.request_id] = request.spec_token_ids[:accepted]
            else:
                token = self.sampler.sample(
                    rows[-1:], [request.sampling_params]
                )[0]
            result.sampled_token_ids[request.request_id] = int(token)
            result.sampled_token_times[request.request_id] = now

    def _split_routed_experts(
        self,
        scheduler_output: SchedulerOutput,
        experts: torch.Tensor,
        result: ModelRunnerOutput,
    ) -> None:
        """按每个请求本步的 token 数切分路由记录。"""
        offset = 0
        requests = (
            scheduler_output.scheduled_new_reqs + scheduler_output.scheduled_cached_reqs
        )
        for req in requests:
            num_new = scheduler_output.num_scheduled_tokens[req.request_id]
            result.routed_experts[req.request_id] = experts[offset : offset + num_new]
            offset += num_new

    # ---------- CUDA Graph ----------

    def capture_cuda_graphs(self) -> None:
        """按 token×长度桶捕获 decode 图。"""
        if self.dispatcher.mode == CUDAGraphMode.NONE:
            logger.info("CUDA Graph 已关闭")
            return
        assert self.model is not None and self.model_cfg is not None
        sizes = self.dispatcher.capture_sizes
        len_buckets = self.dispatcher.len_buckets
        logger.info(
            "开始捕获 CUDA Graph：%d 个 token 桶 × %d 个长度桶",
            len(sizes),
            len(len_buckets),
        )
        # 所有图共享同一个显存池，按最大图分配；否则每张图独立池会迅速 OOM
        pool = torch.cuda.graph_pool_handle()
        # CustomAllreduce 需要在整个捕获期间标记，结束后统一注册图内缓冲地址
        with get_tp_group().capture_custom_all_reduce():
            for size in sorted(sizes, reverse=True):
                for len_bucket in reversed(len_buckets):
                    try:
                        self._capture_one(size, len_bucket, pool)
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        torch.cuda.empty_cache()
                        once.warn(
                            logger,
                            f"graph_oom_{size}_{len_bucket}",
                            f"桶 ({size}, {len_bucket}) 捕获 OOM，跳过",
                        )
        from leanmoe.runtime.attention.selector import get_attn_backend

        self._graph_need_kv_fill = (
            get_attn_backend("auto", capturing=True).name == "sdpa"
        )

    def _capture_one(self, num_tokens: int, len_bucket: int, pool) -> None:
        assert self.model is not None and self.model_cfg is not None
        device = self.device
        num_reqs = num_tokens
        max_len = len_bucket
        seq_lens = torch.ones(num_reqs, dtype=torch.int32, device=device)
        block_tables = torch.zeros((num_reqs, self._max_blocks_per_seq), dtype=torch.int32, device=device)
        slot_mapping = torch.zeros(num_tokens, dtype=torch.long, device=device)
        kv_indices = torch.zeros((num_reqs, max_len), dtype=torch.long, device=device)
        kv_mask = torch.zeros((num_reqs, 1, 1, max_len), dtype=torch.bool, device=device)
        kv_mask[:, :, :, 0] = True
        batch = InputBatch(
            token_ids=torch.zeros(num_tokens, dtype=torch.long, device=device),
            positions=torch.zeros(num_tokens, dtype=torch.long, device=device),
            query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32, device=device),
            seq_lens=seq_lens,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            logits_indices=torch.arange(num_reqs, dtype=torch.long, device=device),
            num_seqs=num_reqs,
            uniform_decode=True,
            max_seq_len=1,
            max_query_len=1,
        )
        metadata = self._build_attn_metadata(batch)
        # 图内 paged decode 的 KV 上界用长度桶，实际长度由 seqused_k 控制
        metadata.max_seqlen_k = len_bucket
        metadata.extra = {
            "kv_indices": kv_indices,
            "kv_mask": kv_mask,
        }
        metadata.is_capturing = True

        # warmup：让 kernel 完成懒初始化
        self._forward(batch, metadata)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            output = self._forward(batch, metadata)
            # 图内采样：lm_head + argmax 也进图，图外只读 token 索引
            logits = self.model.compute_logits(output[batch.logits_indices])
            tokens = logits.argmax(dim=-1)
        self._graphs[(num_tokens, num_reqs, len_bucket)] = {
            "graph": graph,
            "batch": batch,
            "metadata": metadata,
            "output": output,
            "tokens": tokens,
            "buffers": metadata.extra,
        }
        metadata.is_capturing = False

    def _replay_graph(self, batch: InputBatch, descriptor) -> torch.Tensor | None:
        entry = self._graphs.get(
            (descriptor.num_tokens, descriptor.num_reqs, descriptor.len_bucket)
        )
        if entry is None:
            # 返回 None 表示未命中，由调用方走 eager 路径
            once.warn(logger, "graph_miss", "未找到对应图，退回 eager")
            return None

        g_batch: InputBatch = entry["batch"]
        g_meta: AttentionMetadata = entry["metadata"]
        bufs = entry["buffers"]

        n = batch.token_ids.numel()
        g_batch.token_ids[:n].copy_(batch.token_ids)
        g_batch.positions[:n].copy_(batch.positions)
        g_batch.slot_mapping[:n].copy_(batch.slot_mapping)
        # 图输入的 KV 索引按当前请求重建（host 侧，图外）；仅 SDPA 回退路径需要，
        # Triton paged decode 直接读 block_tables/seq_lens，省掉每步 ~6ms 的物化。
        if self._graph_need_kv_fill:
            self._fill_graph_kv(batch, bufs, g_meta)
        g_meta.seq_lens[: batch.num_seqs].copy_(batch.seq_lens)
        # 捕获时的 block_table 宽度是 max_blocks_per_seq，实际 batch 可能更窄
        n = batch.num_seqs
        m = batch.block_tables.shape[1]
        g_meta.block_tables[:n, :m].copy_(batch.block_tables)
        # 请求数不足桶大小时补 dummy 行：slot=-1 跳过 KV 写入，
        # seq_len=1 避免 decode kernel 除零，输出行会被丢弃。
        if descriptor.num_reqs > n:
            g_batch.token_ids[n : descriptor.num_reqs].zero_()
            g_batch.positions[n : descriptor.num_reqs].zero_()
            g_batch.slot_mapping[n : descriptor.num_reqs].fill_(-1)
            g_meta.seq_lens[n : descriptor.num_reqs].fill_(1)
            g_meta.block_tables[n : descriptor.num_reqs].zero_()
        entry["graph"].replay()
        return entry["tokens"][: batch.num_seqs]

    def _fill_graph_kv(self, batch: InputBatch, bufs: dict, metadata: AttentionMetadata) -> None:
        """向量化填充 KV 索引与掩码，全程设备端，不含 .item()。"""
        kv_indices = bufs["kv_indices"]
        kv_mask = bufs["kv_mask"]
        num_seqs = batch.num_seqs
        max_len = kv_indices.shape[1]
        block_size = self.block_size

        device = kv_indices.device
        positions = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
        seq_lens = metadata.seq_lens[:num_seqs].to(torch.long).unsqueeze(1)  # [B,1]
        valid = positions < seq_lens
        # padding 位置重复最后一个有效位置，避免越界
        safe_pos = torch.minimum(positions, (seq_lens - 1).clamp(min=0))
        blocks = metadata.block_tables[:num_seqs].to(torch.long).gather(
            1, safe_pos // block_size
        )
        kv_indices[:num_seqs] = blocks * block_size + safe_pos % block_size
        kv_mask[:num_seqs, 0, 0] = valid
        # 补齐到桶的 dummy 行必须清空，否则会读到上一轮的陈旧索引
        if kv_indices.shape[0] > num_seqs:
            kv_indices[num_seqs:].zero_()
            kv_mask[num_seqs:].fill_(False)

    # ---------- 投机解码辅助 ----------

    def verify_and_sample(
        self,
        logits: torch.Tensor,
        requests: list[Request],
        draft_tokens: list[list[int]],
    ) -> list[int]:
        """标准拒绝采样：返回每个请求接受的 token 数（含 1 个修正 token）。"""
        accepted: list[int] = []
        for i, req in enumerate(requests):
            base = req.num_tokens - len(draft_tokens[i])
            # 目标模型对草稿位置的 argmax 与草稿一致则接受
            count = 0
            for j, token in enumerate(draft_tokens[i]):
                target = int(logits[base + j].argmax().item())
                if target == token:
                    count += 1
                else:
                    break
            accepted.append(count + 1)
        return accepted

    # ---------- PP ----------

    def _pp_send(self, hidden: torch.Tensor) -> None:
        group = get_pp_group()
        if group.world_size <= 1:
            return
        group.send(hidden.contiguous(), dst=1)

    def _pp_recv(self, num_tokens: int) -> torch.Tensor:
        group = get_pp_group()
        if group.world_size <= 1 or self.model_cfg is None:
            return torch.empty(0, device=self.device)
        hidden = torch.empty(
            (num_tokens, self.model_cfg.hidden_size),
            dtype=self.config.model.torch_dtype,
            device=self.device,
        )
        group.recv(hidden, src=0)
        return hidden

    # ---------- 状态 ----------

    def set_current_requests(self, requests: list[Request]) -> None:
        self._current_requests = requests

    @property
    def graph_stats(self) -> dict[str, int]:
        return {m.name: c for m, c in self._graph_mode_count.items()}
