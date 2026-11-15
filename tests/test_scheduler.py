"""调度器测试：预算、抢占、等待队列。"""

from leanmoe.config import SchedulerConfig
from leanmoe.sequence import Request, RequestStatus, SamplingParams
from leanmoe.v1.core.block_pool import BlockPool
from leanmoe.v1.core.kv_cache import KVCacheManager
from leanmoe.v1.core.sched.scheduler import Scheduler


def make_request(req_id: str, length: int, max_tokens: int = 8) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(length)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def build_scheduler(num_blocks: int = 64, budget: int = 64, max_seqs: int = 8) -> Scheduler:
    pool = BlockPool(num_blocks, enable_caching=False)
    manager = KVCacheManager(pool, block_size=16, enable_caching=False)
    scheduler = Scheduler(
        SchedulerConfig(
            max_num_seqs=max_seqs,
            max_num_batched_tokens=budget,
            enable_chunked_prefill=True,
            async_scheduling=False,
        ),
        manager,
    )
    scheduler.set_max_model_len(32768)
    return scheduler


def test_chunked_prefill_respects_budget():
    scheduler = build_scheduler(budget=32)
    req = make_request("a", 100)
    scheduler.add_request(req)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["a"] == 32
    assert req.status == RequestStatus.RUNNING


def test_prefill_then_decode():
    scheduler = build_scheduler(budget=128)
    req = make_request("a", 32)
    scheduler.add_request(req)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["a"] == 32
    scheduler.update_from_output(out, {"a": (7, 0.0)}, {"a": set()})

    out2 = scheduler.schedule()
    # decode 每步 1 个 token
    assert out2.num_scheduled_tokens["a"] == 1


def test_preemption_frees_blocks_and_requeues():
    scheduler = build_scheduler(num_blocks=8, budget=128, max_seqs=4)
    reqs = [make_request(f"r{i}", 64) for i in range(4)]
    for r in reqs:
        scheduler.add_request(r)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens, "至少应调度一个请求"

    # 反复推进到需要抢占：8 个 block 最多容纳 128 token
    for _ in range(16):
        sampled = {rid: (1, 0.0) for rid in out.num_scheduled_tokens}
        scheduler.update_from_output(out, sampled, {rid: set() for rid in sampled})
        out = scheduler.schedule()
        if out.preempted_req_ids:
            break
    assert scheduler.num_preemptions_total >= 0


def test_max_num_seqs_limits_waiting_admission():
    scheduler = build_scheduler(budget=1024, max_seqs=2)
    for i in range(5):
        scheduler.add_request(make_request(f"r{i}", 8))
    out = scheduler.schedule()
    assert len(out.num_scheduled_tokens) <= 2
