"""EngineCore：调度、执行、采样、输出回写的单步循环。"""

from __future__ import annotations

import torch

from leanmoe.config import VllmConfig
from leanmoe.logger import get_logger
from leanmoe.sequence import (
    EngineCoreOutput,
    EngineCoreOutputs,
    Request,
    RequestStatus,
    build_stop_token_ids,
)
from leanmoe.runtime.core.kv_cache import KVCacheManager
from leanmoe.runtime.core.sched.output import SchedulerOutput
from leanmoe.runtime.core.sched.scheduler import AsyncScheduler, Scheduler
from leanmoe.runtime.kv_transfer.connector import KVConnectorBase
from leanmoe.runtime.spec_decode.proposer import SpecDecodeState
from leanmoe.runtime.worker.gpu_worker import Worker

logger = get_logger("engine_core")


class EngineCore:
    """单步循环：调度 → 执行 → 回写。"""
    def __init__(
        self,
        config: VllmConfig,
        worker: Worker,
        kv_manager: KVCacheManager,
        scheduler: Scheduler,
        proposer: SpecDecodeState | None = None,
        kv_connector: KVConnectorBase | None = None,
    ) -> None:
        self.config = config
        self.worker = worker
        self.kv_manager = kv_manager
        self.scheduler = scheduler
        self.proposer = proposer
        self.kv_connector = kv_connector
        self._requests: dict[str, Request] = {}
        # vLLM 式调度：只有 rank0 跑调度器，其余 rank 重放同一计划。
        self._primary = True
        self._sched_group = None
        self._sched_mode = "off"
        self._sched_dirty = True
        try:
            import torch.distributed as dist

            from leanmoe.distributed.parallel_state import get_parallel_state

            import os as _os

            state = get_parallel_state()
            self._sched_mode = _os.environ.get("LEANMOE_SCHED_MODE", "hash")
            if (
                self._sched_mode != "off"
                and state.tp_size > 1
                and dist.is_initialized()
                and not isinstance(scheduler, AsyncScheduler)
            ):
                self._primary = state.tp_rank == 0
                self._sched_group = dist.new_group(backend="gloo")
        except Exception as exc:  # 单卡 / 未初始化分布式 / 无 gloo
            logger.debug("调度广播不可用，退回各 rank 本地调度: %s", exc)
            self._primary, self._sched_group = True, None
            self._sched_mode = "off"
        # 广播用缓冲：primary 模式用完整计划，hash 模式用 8 元指纹
        self._plan_buf = (
            torch.zeros(4096, dtype=torch.int64)
            if self._sched_group is not None
            else None
        )
        self._fp_buf = (
            torch.zeros(8, dtype=torch.int64)
            if self._sched_group is not None
            else None
        )

    # ---------- 请求 ----------

    def add_request(self, request: Request) -> None:
        """登记请求并交给调度器。"""
        self._requests[request.request_id] = request
        # 新增请求是唯一可能让两卡调度分叉的入口，需要一次指纹校验
        self._sched_dirty = True
        self.scheduler.add_request(request)

    def abort_request(self, request_id: str) -> None:
        """中止请求并释放其 KV。"""
        request = self._requests.pop(request_id, None)
        if request is None:
            return
        self._sched_dirty = True
        self.kv_manager.free(request)
        if request in self.scheduler.running:
            self.scheduler.running.remove(request)
        else:
            try:
                self.scheduler.waiting.remove(request)
            except ValueError:
                pass
        request.status = RequestStatus.FINISHED_ABORTED

    # ---------- 单步 ----------

    _PLAN_CAP = 4096

    def _encode_plan(self, out) -> None:
        """把 rank0 的调度结果写进定长 int64 缓冲（用提交序号代替字符串 id）。"""
        sched = self.scheduler
        entries = [(r, False) for r in out.scheduled_cached_reqs] + [
            (r, True) for r in out.scheduled_new_reqs
        ]
        waiting_list = sched.waiting.to_list()
        vals = [
            len(entries),
            len(sched._step_preempted),
            len(sched.running),
            len(waiting_list),
            1 if out.is_prompt else 0,
            out.total_num_scheduled_tokens,
        ]
        for request, is_new in entries:
            vals.append(sched._request_index[request.request_id])
            vals.append(out.num_scheduled_tokens[request.request_id])
            vals.append(1 if is_new else 0)
            vals.append(sched._step_prefix_hits.get(request.request_id, 0))
        for rid in sched._step_preempted:
            vals.append(sched._request_index[rid])
        for request in sched.running:
            vals.append(sched._request_index[request.request_id])
        for request in waiting_list:
            vals.append(sched._request_index[request.request_id])
        if len(vals) > self._PLAN_CAP:
            raise RuntimeError(f"调度计划过大：{len(vals)} > {self._PLAN_CAP}")
        # 整段拷贝，避免逐元素 Python 写入（单次 ~5-10us）
        self._plan_buf[: len(vals)].copy_(torch.tensor(vals, dtype=torch.int64))

    def _decode_plan(self) -> dict:
        head = self._plan_buf[:6].tolist()
        n_sched, n_pre, n_run, n_wait = head[0], head[1], head[2], head[3]
        total_len = 6 + 4 * n_sched + n_pre + n_run + n_wait
        vals = self._plan_buf[:total_len].tolist()
        n_sched, n_pre, n_run, n_wait = vals[0], vals[1], vals[2], vals[3]
        pos = 6
        scheduled = []
        for _ in range(n_sched):
            scheduled.append(
                (vals[pos], vals[pos + 1], bool(vals[pos + 2]), vals[pos + 3])
            )
            pos += 4
        preempted = vals[pos : pos + n_pre]
        pos += n_pre
        running = vals[pos : pos + n_run]
        pos += n_run
        waiting = vals[pos : pos + n_wait]
        return {
            "scheduled": scheduled,
            "preempted": preempted,
            "running": running,
            "waiting": waiting,
            "is_prompt": bool(vals[4]),
            "total": vals[5],
        }

    def _fingerprint(self, out) -> list[int]:
        """8 元指纹：两卡各自调度后比对，确保决策完全一致（开销约 0.1ms）。"""
        sched = self.scheduler
        n_sched = len(out.scheduled_cached_reqs) + len(out.scheduled_new_reqs)
        idx_sum = 0
        tok_sum = 0
        mix = 0
        for request in out.scheduled_cached_reqs + out.scheduled_new_reqs:
            i = sched._request_index[request.request_id]
            n = out.num_scheduled_tokens[request.request_id]
            idx_sum += i
            tok_sum += n
            mix = (mix * 1000003 + i * 97 + n) % (1 << 62)
        return [
            n_sched,
            out.total_num_scheduled_tokens,
            len(sched.running),
            len(sched.waiting),
            idx_sum,
            tok_sum,
            len(sched._step_preempted),
            mix,
        ]

    def step(self) -> EngineCoreOutputs:
        """推进一步：调度、执行、采样、回写。"""
        self._maybe_propose()

        if self._sched_group is None or self._sched_mode == "off":
            out = self.scheduler.schedule()
        else:
            import torch.distributed as dist

            # 两卡状态一致时 has_unfinished 也一致：False 时双方都跳过广播
            active = self.scheduler.has_unfinished()
            if self._sched_mode == "hash":
                out = self.scheduler.schedule()
                if self._sched_dirty and (active or not out.is_empty()):
                    vals = self._fingerprint(out)
                    if self._primary:
                        self._fp_buf.copy_(torch.tensor(vals, dtype=torch.int64))
                    dist.broadcast(self._fp_buf, src=0, group=self._sched_group)
                    if not self._primary:
                        remote = self._fp_buf.tolist()
                        if remote != vals:
                            raise RuntimeError(
                                f"两卡调度不一致：rank0={remote} local={vals}；"
                                "检查请求提交是否对齐"
                            )
                    self._sched_dirty = False
            elif self._primary:
                out = self.scheduler.schedule()
                if active or not out.is_empty():
                    self._encode_plan(out)
                    dist.broadcast(self._plan_buf, src=0, group=self._sched_group)
            elif active:
                dist.broadcast(self._plan_buf, src=0, group=self._sched_group)
                out = self.scheduler.apply_plan(self._decode_plan())
            else:
                out = SchedulerOutput()

        if out.is_empty():
            return EngineCoreOutputs(
                outputs=[], num_preempted=len(out.preempted_req_ids)
            )

        runner_out = self.worker.execute_model(out)

        sampled: dict[str, tuple[int, float]] = {}
        accepted_map: dict[str, list[int]] = {}
        for request_id, token in runner_out.sampled_token_ids.items():
            request = self._requests.get(request_id)
            if request is None:
                continue
            ts = runner_out.sampled_token_times.get(request_id, 0.0)
            accepted = runner_out.accepted_drafts.get(request_id, [])
            if accepted:
                # 投机路径可能一次生成 accepted + 1 个 token，必须按 max_tokens 截断，
                # 否则会在最后一步多输出一个 token。
                remaining = request.sampling_params.max_tokens - request.num_output_tokens
                accepted = accepted[: max(0, remaining - 1)]
                accepted_map[request_id] = accepted
                for draft in accepted:
                    request.append_token(draft, ts)
            sampled[request_id] = (token, ts)

        stop_ids = {
            request.request_id: build_stop_token_ids(
                request.sampling_params, self.worker.runner.model_cfg.eos_token_id
            )
            for request in out.scheduled_new_reqs + out.scheduled_cached_reqs
        }

        finished = self.scheduler.update_from_output(out, sampled, stop_ids)

        # 投机：只要本步用过草稿，就清空并回退被拒绝部分的计算进度
        for request_id in sampled:
            request = self._requests.get(request_id)
            if request is None or not request.spec_token_ids:
                continue
            accepted = accepted_map.get(request_id, [])
            rejected = len(request.spec_token_ids) - len(accepted)
            if rejected > 0:
                request.num_computed_tokens = max(
                    request.num_prompt_tokens,
                    request.num_computed_tokens - rejected,
                )
            request.spec_token_ids.clear()

        outputs: list[EngineCoreOutput] = []
        for request in out.scheduled_new_reqs + out.scheduled_cached_reqs:
            if request.request_id not in sampled:
                continue
            token, ts = sampled[request.request_id]
            extra = accepted_map.get(request.request_id, [])
            outputs.append(
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=extra + [token],
                    new_token_times=[ts] * (len(extra) + 1),
                    finish_reason=self._finish_reason(request),
                    routed_experts=runner_out.routed_experts.get(request.request_id),
                )
            )

        for request in finished:
            self._requests.pop(request.request_id, None)

        return EngineCoreOutputs(
            outputs=outputs, num_preempted=len(out.preempted_req_ids)
        )

    # ---------- 投机 ----------

    def _maybe_propose(self) -> None:
        if self.proposer is None:
            return
        for request in self.scheduler.running:
            if (
                request.num_computed_tokens >= request.num_prompt_tokens
                and not request.spec_token_ids
                and not request.is_finished()
            ):
                request.spec_token_ids = self.proposer.propose(request)

    # ---------- 辅助 ----------

    @staticmethod
    def _finish_reason(request: Request) -> str | None:
        if request.status == RequestStatus.FINISHED_STOPPED:
            return "stop"
        if request.status == RequestStatus.FINISHED_LENGTH_CAPPED:
            return "length"
        if request.status == RequestStatus.FINISHED_ABORTED:
            return "abort"
        return None

    @property
    def has_unfinished(self) -> bool:
        """是否还有未完成请求。"""
        return self.scheduler.has_unfinished()
