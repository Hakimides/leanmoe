"""FusedMoE 的并行配置推导。"""

from __future__ import annotations

from dataclasses import dataclass

from leanmoe.config import ParallelConfig


@dataclass
class FusedMoEConfig:
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    tp_size: int
    ep_size: int
    use_ep: bool

    @classmethod
    def from_parallel(
        cls,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        parallel: ParallelConfig,
    ) -> "FusedMoEConfig":
        use_ep = parallel.enable_expert_parallel and parallel.expert_parallel_size > 1
        return cls(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_size=parallel.moe_tensor_parallel_size,
            ep_size=parallel.expert_parallel_size if use_ep else 1,
            use_ep=use_ep,
        )

    @property
    def num_local_experts(self) -> int:
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"专家数 {self.num_experts} 不能被 ep_size={self.ep_size} 整除"
            )
        return self.num_experts // self.ep_size

    @property
    def intermediate_per_partition(self) -> int:
        if self.use_ep:
            return self.intermediate_size
        if self.intermediate_size % self.tp_size != 0:
            raise ValueError(
                f"intermediate_size={self.intermediate_size} 不能被 tp_size={self.tp_size} 整除"
            )
        return self.intermediate_size // self.tp_size

    def global_expert_id(self, local_id: int, ep_rank: int) -> int:
        return ep_rank * self.num_local_experts + local_id
