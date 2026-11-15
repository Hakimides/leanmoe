"""请求状态机、采样参数与输出结构。"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class RequestStatus(enum.IntEnum):
    WAITING = 0
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = 1
    WAITING_FOR_REMOTE_KVS = 2
    RUNNING = 3
    PREEMPTED = 4
    FINISHED_STOPPED = 5
    FINISHED_LENGTH_CAPPED = 6
    FINISHED_ABORTED = 7
    FINISHED_IGNORED = 8

    @property
    def is_finished(self) -> bool:
        return self >= RequestStatus.FINISHED_STOPPED

    @property
    def is_waiting(self) -> bool:
        return self < RequestStatus.RUNNING


@dataclass
class SamplingParams:
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    min_tokens: int = 0
    ignore_eos: bool = False
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    seed: int | None = None
    # 是否返回 prompt 的 logprobs（暂不实现，保留字段）
    logprobs: int | None = None
    # 是否要求该请求输出路由专家（需要引擎级开关）
    collect_routed_experts: bool = False
    # 跳过特殊 token 检测（调试用）
    skip_special_tokens: bool = True

    def __post_init__(self) -> None:
        if self.n != 1:
            raise ValueError("当前实现只支持 n=1")
        if self.temperature < 0:
            raise ValueError("temperature 不能为负")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p 必须在 (0, 1]")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须为正")
        if self.min_tokens < 0 or self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens 必须在 [0, max_tokens]")

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0 or self.top_k == 1


@dataclass(eq=False)
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    # 到达时间（单调时钟秒），用于排队时延统计
    arrival_time: float = 0.0
    priority: int = 0

    output_token_ids: list[int] = field(default_factory=list)
    # 每个输出 token 被采样出来的时刻，与 output_token_ids 一一对应
    output_token_times: list[float] = field(default_factory=list)
    # 投机解码产生的草稿 token（本步验证前）
    spec_token_ids: list[int] = field(default_factory=list)
    # 已写入 KV 的 token 数（prompt + 已确认输出）
    num_computed_tokens: int = 0
    # 已分配的块 id，按逻辑顺序
    block_ids: list[int] = field(default_factory=list)
    # 块哈希链，每 block_size 个 token 一个；重算时用于命中 prefix cache
    block_hashes: list[bytes] = field(default_factory=list)
    # 已登记进 prefix cache 的满块数
    num_cached_blocks: int = 0
    status: RequestStatus = RequestStatus.WAITING
    num_preemptions: int = 0
    # 路由专家记录：[num_tokens, num_layers, top_k]，仅 collect_routed_experts 时非空
    routed_experts: np.ndarray | None = None
    first_token_time: float | None = None
    finished_time: float | None = None
    # PD 分离：远端 KV 的元数据
    kv_transfer_params: dict | None = None
    # 已完成的推理步数（用于 async 占位符对齐）
    num_output_placeholders: int = 0

    # ---------- 派生属性 ----------

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def num_tokens_with_spec(self) -> int:
        return self.num_tokens + len(self.spec_token_ids)

    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def append_token(self, token_id: int, timestamp: float | None = None) -> None:
        self.output_token_ids.append(token_id)
        if timestamp is not None:
            self.output_token_times.append(timestamp)

    # ---------- 状态 ----------

    def set_status(self, status: RequestStatus) -> None:
        self.status = status

    def is_finished(self) -> bool:
        return self.status.is_finished

    # ---------- 调度辅助 ----------

    def reset_for_recompute(self) -> None:
        """抢占后回到 waiting：清空块表与计算进度，保留 prompt 和已生成 token。"""
        self.num_computed_tokens = 0
        self.block_ids.clear()
        self.spec_token_ids.clear()
        self.num_output_placeholders = 0

    def record_routed_experts(self, experts: np.ndarray) -> None:
        """追加本步的路由结果，形状 [num_new_tokens, num_layers, top_k]。"""
        if self.routed_experts is None:
            self.routed_experts = experts
        else:
            self.routed_experts = np.concatenate([self.routed_experts, experts], axis=0)


@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float | None = None
    finish_reason: str | None = None
    # [num_tokens, num_layers, top_k]，仅请求时才有
    routed_experts: np.ndarray | None = None


@dataclass
class RequestOutput:
    request_id: str
    prompt: str | None
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]
    finished: bool
    metrics: dict[str, float] = field(default_factory=dict)
    # 调度器内部使用：本步新增 token 数
    num_new_tokens: int = 0


@dataclass
class EngineCoreOutput:
    """EngineCore 送给前端的增量输出。"""

    request_id: str
    new_token_ids: list[int]
    new_token_times: list[float] = field(default_factory=list)
    finish_reason: str | None = None
    routed_experts: np.ndarray | None = None


@dataclass
class EngineCoreOutputs:
    outputs: list[EngineCoreOutput]
    # 本步是否有请求被抢占（用于前端日志）
    num_preempted: int = 0


def build_stop_token_ids(sampling_params: SamplingParams, eos_token_id: int | None) -> set[int]:
    ids = set(sampling_params.stop_token_ids)
    if eos_token_id is not None and not sampling_params.ignore_eos:
        ids.add(eos_token_id)
    return ids
