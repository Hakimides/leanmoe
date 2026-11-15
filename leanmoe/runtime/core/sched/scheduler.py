"""连续批处理调度器。"""

from __future__ import annotations

from leanmoe.config import SchedulerConfig
from leanmoe.logger import get_logger
from leanmoe.sequence import Request, RequestStatus
from leanmoe.runtime.core.kv_cache import KVCacheManager
from leanmoe.runtime.core.sched.output import SchedulerOutput
from leanmoe.runtime.core.sched.request_queue import create_request_queue

logger = get_logger("scheduler")


class Scheduler:
    """连续批处理调度器：每步先扫 running，再准入 waiting。"""
    def __init__(
        self,
        config: SchedulerConfig,
        kv_cache_manager: KVCacheManager,
    ) -> None:
        self.config = config
        self.kv_cache_manager = kv_cache_manager
        self.waiting = create_request_queue(config.policy)
        self.running: list[Request] = []
        self.finished: list[Request] = []
        # 统计
        self.num_preemptions_total = 0
        # 按 id 索引全部活跃请求（供 rank0 调度结果的重放使用）
        self._requests: dict[str, Request] = {}
        # 提交顺序编号：广播计划里用整数索引代替字符串 id
        self._request_ids: list[str] = []
        self._request_index: dict[str, int] = {}
        # 本步的抢占顺序与 prefix cache 命中数（打包给其它 rank 重放）
        self._step_preempted: list[str] = []
        self._step_prefix_hits: dict[str, int] = {}

    # ---------- 请求 ----------

    def add_request(self, request: Request) -> None:
        """登记请求并放入 waiting 队尾。"""
        request.status = RequestStatus.WAITING
        self._requests[request.request_id] = request
        if request.request_id not in self._request_index:
            self._request_index[request.request_id] = len(self._request_ids)
            self._request_ids.append(request.request_id)
        self.waiting.add(request)

    def abort_request(self, request_id: str) -> None:
        """中止请求：释放 KV，移出等待/运行队列。"""
        for req in list(self.running):
            if req.request_id == request_id:
                self._free_request(req)
                self.running.remove(req)
                req.status = RequestStatus.FINISHED_ABORTED
                self.finished.append(req)
                return
        # 等待队列：两种实现都支持 remove
        pending = self.waiting.peek()
        if pending is not None and pending.request_id == request_id:
            self.waiting.pop()
            pending.status = RequestStatus.FINISHED_ABORTED
            self.finished.append(pending)

    # ---------- 调度 ----------

    def schedule(self) -> SchedulerOutput:
        """生成一步调度计划：running 优先、双预算、必要时抢占。"""
        out = SchedulerOutput()
        self._step_preempted = []
        self._step_prefix_hits = {}
        token_budget = self.config.max_num_scheduled_tokens or self.config.max_num_batched_tokens
        input_budget = self.config.max_num_batched_tokens
        preempted_this_step = False
        import os

        if os.environ.get("LEANMOE_DEBUG_STEP") == "1":
            print(
                f"[sched] waiting={len(self.waiting)} running={len(self.running)} "
                f"budget={token_budget} max_seqs={self.config.max_num_seqs}",
                flush=True,
            )

        running = list(self.running)
        index = 0
        while index < len(running) and token_budget > 0 and input_budget > 0:
            request = running[index]
            num_new = self._num_new_tokens(request, token_budget, input_budget)
            if num_new <= 0:
                index += 1
                continue

            computed_blocks = []
            if request.num_computed_tokens == 0:
                # 被抢占后重算 / 新请求：先查 prefix cache
                computed_blocks, num_hit = self.kv_cache_manager.get_computed_blocks(request)
                if num_hit > 0:
                    request.num_computed_tokens = num_hit
                    self._step_prefix_hits[request.request_id] = num_hit
                    num_new = self._num_new_tokens(request, token_budget, input_budget)
                    if num_new <= 0:
                        index += 1
                        continue

            allocated = self._allocate_with_preempt(
                request, num_new, computed_blocks, running, index, out
            )
            if allocated is None:
                # 当前请求自己被抢占，本步不再继续
                preempted_this_step = True
                break
            if allocated is False:
                # 抢占过别人后仍分不出来，停止本步
                preempted_this_step = True
                break

            self._mark_scheduled(out, request, num_new)
            token_budget -= num_new
            if request.num_computed_tokens < request.num_prompt_tokens:
                input_budget -= num_new
                out.is_prompt = True
            index += 1

        self.running = running

        # 本步发生过抢占就不准入新请求（与 vLLM 一致）
        if not preempted_this_step:
            while (
                len(self.running) < self.config.max_num_seqs
                and token_budget > 0
                and input_budget > 0
                and self.waiting
            ):
                request = self.waiting.peek()
                if request is None:
                    break
                computed_blocks, num_hit = self.kv_cache_manager.get_computed_blocks(request)
                if num_hit > 0:
                    request.num_computed_tokens = num_hit
                    self._step_prefix_hits[request.request_id] = num_hit
                num_new = self._num_new_tokens(request, token_budget, input_budget)
                if num_new <= 0:
                    break
                if self.kv_cache_manager.allocate_slots(
                    request, num_new, computed_blocks, self._lookahead(request)
                ) is None:
                    break
                self.waiting.pop()
                request.status = RequestStatus.RUNNING
                request.num_output_placeholders = 0
                self.running.append(request)
                self._mark_scheduled(out, request, num_new, new=True)
                token_budget -= num_new
                input_budget -= num_new
                if request.num_computed_tokens < request.num_prompt_tokens:
                    out.is_prompt = True

        out.total_num_scheduled_tokens = sum(out.num_scheduled_tokens.values())
        return out

    def apply_plan(self, plan: dict) -> SchedulerOutput:
        """重放 rank0 的调度决定（vLLM 式：只有 rank0 跑调度器）。"""

        def req(i: int) -> Request:
            return self._requests[self._request_ids[i]]

        out = SchedulerOutput()
        for idx in plan["preempted"]:
            request = req(idx)
            self.kv_cache_manager.free(request)
            request.reset_for_recompute()
            request.num_preemptions += 1
            request.status = RequestStatus.PREEMPTED
            self.num_preemptions_total += 1
            out.preempted_req_ids.add(request.request_id)

        for idx, num_new, is_new, num_hit in plan["scheduled"]:
            request = req(idx)
            computed_blocks = []
            if num_hit > 0:
                request.num_computed_tokens = num_hit
                computed_blocks, _ = self.kv_cache_manager.get_computed_blocks(request)
            if is_new:
                request.status = RequestStatus.RUNNING
                request.num_output_placeholders = 0
            if self.kv_cache_manager.allocate_slots(
                request, num_new, computed_blocks, self._lookahead(request)
            ) is None:
                raise RuntimeError(
                    f"调度重放失败：请求 {request.request_id} 无法分配 KV（本地状态与 rank0 不一致）"
                )
            self._mark_scheduled(out, request, num_new, new=is_new)

        self.running = [req(i) for i in plan["running"]]
        self.waiting = create_request_queue(self.config.policy)
        for i in plan["waiting"]:
            self.waiting.add(req(i))

        out.is_prompt = plan["is_prompt"]
        out.total_num_scheduled_tokens = plan["total"]
        return out

    def _num_new_tokens(self, request: Request, token_budget: int, input_budget: int) -> int:
        """本步可为该请求调度多少 token。"""
        # async scheduling 下，decode 请求可以多排一个占位 token
        effective_tokens = request.num_tokens_with_spec + request.num_output_placeholders
        remaining = effective_tokens - request.num_computed_tokens
        if remaining <= 0:
            return 0
        num_new = min(remaining, token_budget)
        # prefill 阶段的额外限制
        if request.num_computed_tokens < request.num_prompt_tokens:
            num_new = min(num_new, input_budget)
            if self.config.long_prefill_token_threshold > 0:
                num_new = min(num_new, self.config.long_prefill_token_threshold)
            if not self.config.enable_chunked_prefill:
                # 不允许切块时，整段放得下才调度
                if num_new < remaining:
                    return 0
        return max(num_new, 0)

    def _allocate_with_preempt(
        self,
        request: Request,
        num_new: int,
        computed_blocks,
        running: list[Request],
        index: int,
        out: SchedulerOutput,
    ):
        """返回 True 表示分配成功；False 表示本步停止；None 表示当前请求被抢占。"""
        lookahead = self._lookahead(request)
        while True:
            if self.kv_cache_manager.allocate_slots(
                request, num_new, computed_blocks, lookahead
            ) is not None:
                return True
            # 只能抢占本步尚未调度的请求（下标 > index）
            if len(running) - 1 <= index:
                self._preempt(request)
                request.status = RequestStatus.PREEMPTED
                out.preempted_req_ids.add(request.request_id)
                return None
            victim = running.pop()
            self._preempt(victim)
            out.preempted_req_ids.add(victim.request_id)
            # 抢占后重新尝试；若还是不够，下一轮会继续抢，直到只剩自己

    def _mark_scheduled(
        self, out: SchedulerOutput, request: Request, num_new: int, new: bool = False
    ) -> None:
        if new:
            out.scheduled_new_reqs.append(request)
        else:
            out.scheduled_cached_reqs.append(request)
        out.num_scheduled_tokens[request.request_id] = num_new

    def _lookahead(self, request: Request) -> int:
        return len(request.spec_token_ids)

    def _preempt(self, request: Request) -> None:
        """recompute 抢占：丢弃 KV，保留 prompt 与已生成 token，回到 waiting 队首。"""
        if request.num_preemptions >= self.config.max_num_preemptions:
            raise RuntimeError(
                f"请求 {request.request_id} 被抢占 {request.num_preemptions} 次，"
                "KV 容量可能严重不足"
            )
        self.kv_cache_manager.free(request)
        request.reset_for_recompute()
        request.num_preemptions += 1
        request.status = RequestStatus.PREEMPTED
        self.num_preemptions_total += 1
        self._step_preempted.append(request.request_id)
        self.waiting.prepend(request)

    def _free_request(self, request: Request) -> None:
        self.kv_cache_manager.free(request)

    # ---------- 输出处理 ----------

    def update_from_output(
        self,
        out: SchedulerOutput,
        sampled_tokens: dict[str, tuple[int, float]],
        stop_token_ids: dict[str, set[int]],
    ) -> list[Request]:
        """把采样结果写回请求。"""
        finished: list[Request] = []
        for request in out.scheduled_cached_reqs + out.scheduled_new_reqs:
            num_new = out.num_scheduled_tokens.get(request.request_id, 0)
            request.num_computed_tokens += num_new

            if request.request_id in sampled_tokens:
                # 异步占位符在真正拿到输出后扣减
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= 1
                token, ts = sampled_tokens[request.request_id]
                request.append_token(token, ts)
                if request.first_token_time is None:
                    request.first_token_time = ts
                stop_ids = stop_token_ids.get(request.request_id, set())
                reason = None
                if request.num_output_tokens >= request.sampling_params.min_tokens:
                    if token in stop_ids:
                        reason = "stop"
                    elif request.num_output_tokens >= request.sampling_params.max_tokens:
                        reason = "length"

                if reason is None and request.num_computed_tokens >= self._max_len(request):
                    reason = "length"

                if reason is not None:
                    # 先登记已写满的块，再释放（释放会清空 block_ids）
                    self.kv_cache_manager.cache_blocks(
                        request, request.num_computed_tokens
                    )
                    request.status = (
                        RequestStatus.FINISHED_STOPPED
                        if reason == "stop"
                        else RequestStatus.FINISHED_LENGTH_CAPPED
                    )
                    request.finished_time = ts
                    self._free_request(request)
                    if request in self.running:
                        self.running.remove(request)
                    finished.append(request)
                    continue

            # 已写满的块登记进 prefix cache
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

        self.finished.extend(finished)
        out.finished_req_ids.update(r.request_id for r in finished)
        return finished

    def _max_len(self, request: Request) -> int:
        # 由调用方在构造时通过 set_max_model_len 注入
        return getattr(self, "max_model_len", 32768)

    def set_max_model_len(self, value: int) -> None:
        self.max_model_len = value

    # ---------- 状态 ----------

    def has_unfinished(self) -> bool:
        """waiting 或 running 非空。"""
        return bool(self.running) or bool(self.waiting)

    @property
    def num_unfinished(self) -> int:
        """未完成请求数。"""
        return len(self.running) + len(self.waiting)


class AsyncScheduler(Scheduler):
    """异步调度器：decode 预加占位符，调度不等上一步采样结果。"""

    def _num_new_tokens(self, request: Request, token_budget: int, input_budget: int) -> int:
        if (
            request.num_computed_tokens >= request.num_prompt_tokens
            and request.num_output_placeholders == 0
            and request.num_tokens_with_spec == request.num_computed_tokens
        ):
            # 纯 decode：预排一个 token 的槽位
            request.num_output_placeholders = 1
        return super()._num_new_tokens(request, token_budget, input_budget)
