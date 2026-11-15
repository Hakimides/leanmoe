"""输出处理：把 Request 的增量输出转成 RequestOutput。"""

from __future__ import annotations

from leanmoe.sequence import CompletionOutput, Request, RequestOutput


class OutputProcessor:
    def __init__(self, tokenizer=None) -> None:
        self.tokenizer = tokenizer
        self._request_outputs: dict[str, list[int]] = {}

    def add_request(self, request: Request) -> None:
        self._request_outputs[request.request_id] = []

    def process(self, request: Request, new_token_ids: list[int]) -> RequestOutput:
        self._request_outputs.setdefault(request.request_id, []).extend(new_token_ids)
        token_ids = list(self._request_outputs[request.request_id])
        text = ""
        if self.tokenizer is not None:
            text = self.tokenizer.decode(
                token_ids, skip_special_tokens=request.sampling_params.skip_special_tokens
            )
        finish_reason = None
        if request.is_finished():
            finish_reason = {
                "FINISHED_STOPPED": "stop",
                "FINISHED_LENGTH_CAPPED": "length",
                "FINISHED_ABORTED": "abort",
            }.get(request.status.name, "stop")
        return RequestOutput(
            request_id=request.request_id,
            prompt=None,
            prompt_token_ids=request.prompt_token_ids,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=text,
                    token_ids=token_ids,
                    finish_reason=finish_reason,
                    routed_experts=request.routed_experts,
                )
            ],
            finished=request.is_finished(),
            metrics=self._metrics(request),
        )

    @staticmethod
    def _metrics(request: Request) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if request.first_token_time is not None:
            metrics["ttft"] = max(0.0, request.first_token_time - request.arrival_time)
        if request.finished_time is not None and request.first_token_time is not None:
            n = max(1, request.num_output_tokens - 1)
            metrics["tpot"] = max(
                0.0, (request.finished_time - request.first_token_time) / n
            )
        return metrics

    def abort(self, request_id: str) -> None:
        self._request_outputs.pop(request_id, None)
