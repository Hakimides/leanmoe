"""采样器：greedy / top-k / top-p，以及投机解码的拒绝采样。"""

from __future__ import annotations

import torch

from leanmoe.sequence import SamplingParams


class Sampler:
    @torch.inference_mode()
    def sample(self, logits: torch.Tensor, params: list[SamplingParams]) -> torch.Tensor:
        """logits: [B, V]；返回 [B] 的 token。"""
        if logits.shape[0] != len(params):
            raise ValueError("logits 行数与采样参数数量不一致")
        if all(p.is_greedy for p in params):
            return logits.argmax(dim=-1)
        tokens = [
            self._sample_one(logits[i], p, i) for i, p in enumerate(params)
        ]
        return torch.stack(tokens)

    def _sample_one(
        self, logits: torch.Tensor, params: SamplingParams, index: int
    ) -> torch.Tensor:
        if params.is_greedy:
            return logits.argmax()

        logits = logits / max(params.temperature, 1e-5)
        if 0 < params.top_k < logits.numel():
            kth = torch.topk(logits, params.top_k).values[-1]
            logits = torch.where(
                logits < kth, torch.full_like(logits, float("-inf")), logits
            )
        probs = torch.softmax(logits, dim=-1)

        generator = None
        if params.seed is not None:
            seed = (params.seed + index * 1_000_003) % (2**63 - 1)
            generator = torch.Generator(device=logits.device).manual_seed(seed)

        if 0 < params.top_p < 1.0:
            sorted_probs, order = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            drop = cumulative - sorted_probs >= params.top_p
            sorted_probs = sorted_probs.masked_fill(drop, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum()
            pick = torch.multinomial(sorted_probs, 1, generator=generator)
            return order[pick].squeeze(0)
        return torch.multinomial(probs, 1, generator=generator).squeeze(0)

    @torch.inference_mode()
    def rejection_sample(
        self,
        target_logits: torch.Tensor,
        draft_token_ids: list[list[int]],
        draft_probs: list[torch.Tensor] | None = None,
        temperature: float = 1.0,
    ) -> list[list[int]]:
        """投机解码拒绝采样（标准 speculative sampling）。"""
        accepted: list[list[int]] = []
        for i, drafts in enumerate(draft_token_ids):
            row = target_logits[i]
            keep: list[int] = []
            for j, token in enumerate(drafts):
                if j >= row.shape[0]:
                    break
                if draft_probs is None:
                    if int(row[j].argmax().item()) == token:
                        keep.append(token)
                    else:
                        break
                else:
                    p = torch.softmax(row[j].float() / max(temperature, 1e-5), dim=-1)
                    q = draft_probs[i][j]
                    ratio = (p[token] / torch.clamp(q[token], min=1e-9)).item()
                    if ratio >= 1.0:
                        keep.append(token)
                    else:
                        if torch.rand(1).item() < ratio:
                            keep.append(token)
                        else:
                            break
            accepted.append(keep)
        return accepted
