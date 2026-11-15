"""投机解码提案器：ngram 与 draft model。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from leanmoe.config import SpeculativeConfig
from leanmoe.logger import get_logger
from leanmoe.sequence import Request

logger = get_logger("spec_decode")


class Proposer:
    def propose(self, request: Request, num_tokens: int) -> list[int]:
        raise NotImplementedError


class NgramProposer(Proposer):
    def __init__(self, max_n: int = 4, min_n: int = 1) -> None:
        self.max_n = max_n
        self.min_n = min_n

    def propose(self, request: Request, num_tokens: int) -> list[int]:
        """从 request 的 token 序列里找最后一个 ngram 的历史匹配。"""
        tokens = request.all_token_ids()
        if len(tokens) < self.min_n + 1:
            return []
        for n in range(self.max_n, self.min_n - 1, -1):
            if len(tokens) < n + 1:
                continue
            suffix = tokens[-n:]
            # 从后往前找上一次出现的位置
            for start in range(len(tokens) - n - 1, -1, -1):
                if tokens[start : start + n] == suffix:
                    draft = tokens[start + n : start + n + num_tokens]
                    if draft:
                        return draft[:num_tokens]
        return []


class DraftModelProposer(Proposer):
    """draft model 提案器。"""

    def __init__(
        self,
        draft_model_path: str,
        num_speculative_tokens: int,
        draft_runner=None,
    ) -> None:
        self.draft_model_path = draft_model_path
        self.num_speculative_tokens = num_speculative_tokens
        self.draft_runner = draft_runner
        self.last_draft_probs: list[torch.Tensor] | None = None

    def propose(self, request: Request, num_tokens: int) -> list[int]:
        if self.draft_runner is None:
            logger.warning("draft model 未加载，投机解码退化为无草稿")
            return []
        tokens = request.all_token_ids()
        drafts, probs = self.draft_runner.generate_draft(
            tokens, min(num_tokens, self.num_speculative_tokens)
        )
        self.last_draft_probs = probs
        return drafts


@dataclass
class SpecDecodeState:
    proposer: Proposer
    num_speculative_tokens: int

    def propose(self, request: Request) -> list[int]:
        return self.proposer.propose(request, self.num_speculative_tokens)


def create_proposer(config: SpeculativeConfig, draft_runner=None) -> SpecDecodeState | None:
    if config.method is None:
        return None
    if config.method == "ngram":
        return SpecDecodeState(
            NgramProposer(config.ngram_prompt_lookup_max, config.ngram_prompt_lookup_min),
            config.num_speculative_tokens,
        )
    if config.method == "draft_model":
        return SpecDecodeState(
            DraftModelProposer(
                config.draft_model or "", config.num_speculative_tokens, draft_runner
            ),
            config.num_speculative_tokens,
        )
    if config.method == "mtp":
        raise NotImplementedError(
            "MTP 需要模型自带多 token 预测头；当前 Qwen3-30B-A3B 无此结构"
        )
    raise ValueError(f"未知的投机解码方法: {config.method}")
