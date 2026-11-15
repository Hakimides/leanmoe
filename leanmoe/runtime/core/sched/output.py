"""调度输出。"""

from __future__ import annotations

from dataclasses import dataclass, field

from leanmoe.sequence import Request


@dataclass
class SchedulerOutput:
    # 本步新准入的请求
    scheduled_new_reqs: list[Request] = field(default_factory=list)
    # 本步继续运行的请求
    scheduled_cached_reqs: list[Request] = field(default_factory=list)
    # {request_id: 本步调度 token 数}
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    total_num_scheduled_tokens: int = 0
    # 本步结束的请求
    finished_req_ids: set[str] = field(default_factory=set)
    # 本步被抢占的请求
    preempted_req_ids: set[str] = field(default_factory=set)
    # 是否存在 prefill
    is_prompt: bool = False
    # PD 分离：需要从远端拉 KV 的请求
    kv_connector_metadata: dict | None = None

    @property
    def num_scheduled_seqs(self) -> int:
        return len(self.num_scheduled_tokens)

    def is_empty(self) -> bool:
        return not self.num_scheduled_tokens
