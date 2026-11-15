"""等待队列：FCFS 与优先级两种策略。

优先级数值越大越优先；被抢占的请求会按原优先级重新插入。
"""

from __future__ import annotations

import heapq
from collections import deque

from leanmoe.sequence import Request


class FCFSRequestQueue:
    def __init__(self) -> None:
        self._queue: deque[Request] = deque()

    def add(self, request: Request) -> None:
        self._queue.append(request)

    def prepend(self, request: Request) -> None:
        self._queue.appendleft(request)

    def peek(self) -> Request | None:
        return self._queue[0] if self._queue else None

    def pop(self) -> Request | None:
        return self._queue.popleft() if self._queue else None

    def remove(self, request: Request) -> None:
        self._queue.remove(request)

    def __len__(self) -> int:
        return len(self._queue)

    def to_list(self) -> list[Request]:
        return list(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)


class PriorityRequestQueue:
    """小顶堆的变体：priority 越大越优先，故用 -priority。"""

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, int, Request]] = []
        self._counter = 0
        self._removed: set[str] = set()

    def add(self, request: Request) -> None:
        self._counter += 1
        heapq.heappush(
            self._heap,
            (-request.priority, request.arrival_time, self._counter, request),
        )

    def prepend(self, request: Request) -> None:
        # 被抢占的请求按原优先级重新入堆；arrival_time 保持不变，天然排在前面
        self.add(request)

    def peek(self) -> Request | None:
        while self._heap:
            _, _, _, request = self._heap[0]
            if request.request_id in self._removed:
                heapq.heappop(self._heap)
                continue
            return request
        return None

    def pop(self) -> Request | None:
        while self._heap:
            _, _, _, request = heapq.heappop(self._heap)
            if request.request_id in self._removed:
                self._removed.discard(request.request_id)
                continue
            return request
        return None

    def remove(self, request: Request) -> None:
        self._removed.add(request.request_id)

    def __len__(self) -> int:
        return len(self._heap) - len(self._removed)

    def to_list(self) -> list[Request]:
        live = [e for e in self._heap if e[3].request_id not in self._removed]
        return [e[3] for e in sorted(live, key=lambda e: (e[0], e[1], e[2]))]

    def __bool__(self) -> bool:
        return len(self) > 0


def create_request_queue(policy: str):
    if policy == "fcfs":
        return FCFSRequestQueue()
    if policy == "priority":
        return PriorityRequestQueue()
    raise ValueError(f"未知的调度策略: {policy}")
