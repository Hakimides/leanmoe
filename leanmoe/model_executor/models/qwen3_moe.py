"""Qwen3-MoE 模型实现。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from leanmoe.config import ParallelConfig, QuantizationConfig
from leanmoe.distributed.parallel_state import get_parallel_state, get_tp_group
from leanmoe.logger import get_logger
from leanmoe.model_executor.layers.activation import SiluAndMul
from leanmoe.model_executor.layers.attention import Qwen3Attention
from leanmoe.model_executor.layers.fused_moe import FusedMoEConfig, create_moe_runner
from leanmoe.model_executor.layers.layernorm import RMSNorm
from leanmoe.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from leanmoe.model_executor.layers.rotary_embedding import RotaryEmbedding
from leanmoe.model_executor.quantization import get_linear_quant_method
from leanmoe.runtime.attention.backend import AttentionMetadata

logger = get_logger("qwen3_moe")

# 加载时允许缺省的键（tie embedding 时 lm_head 可共享）
_IGNORED_MISSING = {"lm_head.weight"}


@dataclass
class Qwen3MoeConfig:
    """Qwen3-MoE 的 HF 配置字段。"""
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    hidden_act: str = "silu"
    eos_token_id: int | None = None
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    decoder_sparse_step: int = 1
    norm_topk_prob: bool = False
    mlp_only_layers: tuple[int, ...] = ()
    rope_scaling: dict | None = None

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    def is_sparse_layer(self, layer_idx: int) -> bool:
        if not self.is_moe or layer_idx in self.mlp_only_layers:
            return False
        return (layer_idx + 1) % self.decoder_sparse_step == 0

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "Qwen3MoeConfig":
        raw = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
        if "text_config" in raw:
            raw = {**raw, **raw["text_config"]}
        heads = raw["num_attention_heads"]
        rope_params = raw.get("rope_parameters") or {}
        rope_theta = raw.get("rope_theta", rope_params.get("rope_theta", 1_000_000.0))
        return cls(
            hidden_size=raw["hidden_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=heads,
            num_key_value_heads=raw.get("num_key_value_heads", heads),
            head_dim=raw.get("head_dim") or raw["hidden_size"] // heads,
            intermediate_size=raw["intermediate_size"],
            vocab_size=raw["vocab_size"],
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=rope_theta,
            max_position_embeddings=raw.get("max_position_embeddings", 32768),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            attention_bias=raw.get("attention_bias", False),
            hidden_act=raw.get("hidden_act", "silu"),
            eos_token_id=raw.get("eos_token_id"),
            num_experts=(raw.get("num_experts") or raw.get("num_local_experts") or 0),
            num_experts_per_tok=raw.get("num_experts_per_tok", 0) or 0,
            moe_intermediate_size=raw.get("moe_intermediate_size", 0) or 0,
            decoder_sparse_step=raw.get("decoder_sparse_step", 1),
            norm_topk_prob=raw.get("norm_topk_prob", False),
            mlp_only_layers=tuple(raw.get("mlp_only_layers", ()) or ()),
            rope_scaling=raw.get("rope_scaling"),
        )


class Qwen3MoeMLP(nn.Module):
    """dense FFN：gate_up 投影 → SwiGLU → down 投影。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
        quant_method=None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size, 2 * intermediate_size, bias=False,
            quant_method=quant_method, params_dtype=params_dtype,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
            quant_method=quant_method, params_dtype=params_dtype,
        )
        self.act = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """dense MLP 前向（非 MoE 层）。"""
        return self.down_proj(self.act(self.gate_up_proj(x)))


class Qwen3MoeSparseBlock(nn.Module):
    """MoE 层：由 FusedMoE 装配。"""

    def __init__(
        self,
        cfg: Qwen3MoeConfig,
        parallel: ParallelConfig,
        quant: QuantizationConfig,
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.is_moe = True
        moe_config = FusedMoEConfig.from_parallel(
            cfg.num_experts,
            cfg.num_experts_per_tok,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            parallel,
        )
        state = get_parallel_state()
        self.runner = create_moe_runner(
            hidden_size=cfg.hidden_size,
            num_experts=cfg.num_experts,
            top_k=cfg.num_experts_per_tok,
            intermediate_size=cfg.moe_intermediate_size,
            norm_topk_prob=cfg.norm_topk_prob,
            config=moe_config,
            params_dtype=params_dtype,
            dp_size=state.dp_size,
            pp_size=state.pp_size,
        )
        self.moe_config = moe_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MoE 前向：路由 → 专家计算 → 合并。"""
        return self.runner(x)


class Qwen3MoeDecoderLayer(nn.Module):
    """一层：norm → attention → norm → MoE。"""
    def __init__(
        self,
        cfg: Qwen3MoeConfig,
        layer_idx: int,
        parallel: ParallelConfig,
        quant: QuantizationConfig,
        params_dtype: torch.dtype,
        rotary: RotaryEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_moe = cfg.is_sparse_layer(layer_idx)
        linear_quant = get_linear_quant_method(quant)
        self.self_attn = Qwen3Attention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            rotary=rotary
            or RotaryEmbedding(
                head_dim=cfg.head_dim,
                max_position_embeddings=cfg.max_position_embeddings,
                base=cfg.rope_theta,
                dtype=params_dtype,
                scaling=cfg.rope_scaling,
            ),
            rms_norm_eps=cfg.rms_norm_eps,
            params_dtype=params_dtype,
            quant_method=linear_quant,
        )
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=params_dtype)
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=params_dtype
        )
        if self.is_moe:
            self.mlp = Qwen3MoeSparseBlock(cfg, parallel, quant, params_dtype)
        else:
            self.mlp = Qwen3MoeMLP(
                cfg.hidden_size, cfg.intermediate_size, params_dtype, quant_method=linear_quant
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor | None,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # vLLM 风格的融合残差：先 norm 再进入子层，残差在外部累加
        """返回本层输出与新的残差。"""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(hidden_states, positions, kv_cache, attn_metadata)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):
    """只构建本 PP stage 的层。"""

    def __init__(
        self,
        cfg: Qwen3MoeConfig,
        parallel: ParallelConfig,
        quant: QuantizationConfig,
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.parallel = parallel
        state = get_parallel_state()
        self.pp_rank = state.pp_rank
        self.pp_size = state.pp_size

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=params_dtype)
        start, end = self._pp_range(cfg.num_hidden_layers, state.pp_rank, state.pp_size)
        self.layer_range = (start, end)
        # 所有层共享一份 RoPE 缓存，避免 48 层各自展开一份 cos/sin
        self.rotary = RotaryEmbedding(
            head_dim=cfg.head_dim,
            max_position_embeddings=cfg.max_position_embeddings,
            base=cfg.rope_theta,
            dtype=params_dtype,
            scaling=cfg.rope_scaling,
        )
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(cfg, i, parallel, quant, params_dtype, self.rotary)
                for i in range(start, end)
            ]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=params_dtype)
        # PP 非首段不需要 embedding；保留但标记为未使用
        self.uses_embedding = state.is_first_pp_stage

    @staticmethod
    def _pp_range(num_layers: int, pp_rank: int, pp_size: int) -> tuple[int, int]:
        if pp_size == 1:
            return 0, num_layers
        per = math.ceil(num_layers / pp_size)
        start = pp_rank * per
        end = min(num_layers, start + per)
        return start, end

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor | None],
        attn_metadata: AttentionMetadata,
        hidden_states: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states is None:
            if input_ids is None:
                raise ValueError("非首段 PP 必须提供 hidden_states")
            hidden_states = self.embed_tokens(input_ids)
        start, end = self.layer_range
        # runner 传的是全模型 KV 视图；PP 时只取本段
        local_caches = kv_caches[start:end] if kv_caches else [None] * len(self.layers)
        for layer, cache in zip(self.layers, local_caches):
            hidden_states, residual = layer(
                hidden_states, positions, cache, attn_metadata, residual
            )
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    def __init__(
        self,
        cfg: Qwen3MoeConfig,
        parallel: ParallelConfig,
        quant: QuantizationConfig,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.quant = quant
        self.parallel = parallel
        self.model = Qwen3MoeModel(cfg, parallel, quant, params_dtype)
        self.lm_head = ColumnParallelLinear(
            cfg.hidden_size, cfg.vocab_size, bias=False,
            quant_method=get_linear_quant_method(quant),
            params_dtype=params_dtype,
        )
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self._routed_experts_enabled = False

    # ---------- 前向 ----------

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor | None],
        attn_metadata: AttentionMetadata,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """embedding → 各层 → final norm。"""
        return self.model(
            input_ids, positions, kv_caches, attn_metadata, hidden_states=hidden_states
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """只对指定行计算 lm_head，返回 logits。"""
        out = self.lm_head(hidden_states)
        tp = get_tp_group().world_size
        # tie embedding 时 lm_head 绑定了完整词表，不需要 all_gather；
        # 只有真正按词表切分的输出才需要。
        if tp > 1 and out.shape[-1] * tp == self.cfg.vocab_size:
            out = get_tp_group().all_gather(out, dim=-1)
        return out.float()

    def compute_greedy_tokens(
        self, hidden_states: torch.Tensor, vocab_chunk: int = 0
    ) -> torch.Tensor:
        """分块 argmax，不物化 [B, V] logits；单卡专用（TP>1 时回退）。"""
        if get_tp_group().world_size > 1:
            return self.compute_logits(hidden_states).argmax(dim=-1)
        weight = self.lm_head.weight
        vocab = weight.shape[0]
        if vocab_chunk <= 0:
            vocab_chunk = vocab
        best_val = None
        best_idx = None
        for start in range(0, vocab, vocab_chunk):
            chunk_w = weight[start : start + vocab_chunk]
            logits = hidden_states @ chunk_w.T
            vals, idx = logits.max(dim=-1)
            idx = idx + start
            if best_val is None:
                best_val, best_idx = vals, idx
            else:
                better = vals > best_val
                best_val = torch.where(better, vals, best_val)
                best_idx = torch.where(better, idx, best_idx)
        return best_idx

    # ---------- 路由观测 ----------

    def set_collect_routed_experts(self, enabled: bool) -> None:
        self._routed_experts_enabled = enabled
        for layer in self.model.layers:
            if getattr(layer, "is_moe", False):
                layer.mlp.runner.router.record_routing = enabled

    def collect_routed_experts(self, num_tokens: int) -> torch.Tensor | None:
        """返回 [num_tokens, num_layers, top_k] 的 logical 路由，未开启时为 None。"""
        if not self._routed_experts_enabled:
            return None
        per_layer = []
        for layer in self.model.layers:
            if not getattr(layer, "is_moe", False):
                continue
            ids = layer.mlp.runner.pop_routed_experts()
            if ids is not None:
                per_layer.append(ids.detach())
        if not per_layer:
            return None
        return torch.stack(per_layer, dim=1)

    # ---------- 权重 ----------

    def load_weights(self, state: dict[str, torch.Tensor], model_path: str) -> list[str]:
        """加载并切分权重，返回缺失的键。"""
        state = normalize_moe_state_dict(state, self.cfg)
        used: set[str] = set()
        state_ps = get_parallel_state()
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group

        def take(name: str) -> torch.Tensor | None:
            if name in state:
                used.add(name)
                return state[name]
            return None

        embed = take("model.embed_tokens.weight")
        if embed is not None and self.model.uses_embedding:
            self.model.embed_tokens.weight.data.copy_(embed)

        for layer_idx in range(*self.model.layer_range):
            local_idx = layer_idx - self.model.layer_range[0]
            layer = self.model.layers[local_idx]
            prefix = f"model.layers.{layer_idx}"

            for norm_name in ("input_layernorm", "post_attention_layernorm"):
                w = take(f"{prefix}.{norm_name}.weight")
                if w is not None:
                    getattr(layer, norm_name).weight.data.copy_(w)

            q = take(f"{prefix}.self_attn.q_proj.weight")
            k = take(f"{prefix}.self_attn.k_proj.weight")
            v = take(f"{prefix}.self_attn.v_proj.weight")
            if q is not None:
                self._copy_qkv(layer.self_attn, q, k, v, tp_rank, tp_size)
            o = take(f"{prefix}.self_attn.o_proj.weight")
            if o is not None:
                chunk = o.shape[1] // tp_size
                layer.self_attn.o_proj.weight.data.copy_(
                    o[:, tp_rank * chunk : (tp_rank + 1) * chunk]
                )
            qn = take(f"{prefix}.self_attn.q_norm.weight")
            kn = take(f"{prefix}.self_attn.k_norm.weight")
            if qn is not None:
                layer.self_attn.q_norm.weight.data.copy_(qn)
                layer.self_attn.k_norm.weight.data.copy_(kn)

            if getattr(layer, "is_moe", False):
                self._load_moe_weights(layer, prefix, state, take, state_ps)
            else:
                gate = take(f"{prefix}.mlp.gate_proj.weight")
                up = take(f"{prefix}.mlp.up_proj.weight")
                down = take(f"{prefix}.mlp.down_proj.weight")
                if gate is not None:
                    inter = gate.shape[0] // tp_size
                    s, e = tp_rank * inter, (tp_rank + 1) * inter
                    layer.mlp.gate_up_proj.weight.data.copy_(
                        torch.cat([gate[s:e], up[s:e]], dim=0)
                    )
                    layer.mlp.down_proj.weight.data.copy_(
                        down[:, s:e]
                    )

        if state_ps.is_last_pp_stage:
            norm = take("model.norm.weight")
            if norm is not None:
                self.model.norm.weight.data.copy_(norm)
            lm = take("lm_head.weight")
            if lm is not None and not self.cfg.tie_word_embeddings:
                vocab_per = lm.shape[0] // tp_size
                self.lm_head.weight.data.copy_(
                    lm[tp_rank * vocab_per : (tp_rank + 1) * vocab_per]
                )

        start, end = self.model.layer_range

        def _is_my_key(key: str) -> bool:
            match = re.search(r"model\.layers\.(\d+)\.", key)
            if match is not None:
                return start <= int(match.group(1)) < end
            if key == "model.embed_tokens.weight":
                return self.model.uses_embedding
            if key in ("model.norm.weight", "lm_head.weight"):
                return state_ps.is_last_pp_stage
            return True

        missing = [
            k for k in state
            if k not in used and k not in _IGNORED_MISSING and _is_my_key(k)
        ]
        return missing

    @staticmethod
    def _copy_qkv(
        attn: Qwen3Attention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        head_dim = attn.head_dim
        q_heads = q.shape[0] // head_dim
        kv_heads = k.shape[0] // head_dim
        q_per = q_heads // tp_size
        kv_per = kv_heads // tp_size
        q_part = q[tp_rank * q_per * head_dim : (tp_rank + 1) * q_per * head_dim]
        k_part = k[tp_rank * kv_per * head_dim : (tp_rank + 1) * kv_per * head_dim]
        v_part = v[tp_rank * kv_per * head_dim : (tp_rank + 1) * kv_per * head_dim]
        attn.qkv_proj.weight.data.copy_(torch.cat([q_part, k_part, v_part], dim=0))

    @staticmethod
    def _load_moe_weights(layer, prefix, state, take, ps) -> None:
        runner = layer.mlp.runner
        router_w = take(f"{prefix}.mlp.router.gate.weight")
        if router_w is not None:
            runner.router.gate.weight.data.copy_(router_w)
        gate_up = take(f"{prefix}.mlp.experts.gate_up_proj")
        down = take(f"{prefix}.mlp.experts.down_proj")
        if gate_up is None or down is None:
            return
        cfg = runner.config
        if cfg.use_ep:
            start = ps.ep_rank * cfg.num_local_experts
            gate_up = gate_up[start : start + cfg.num_local_experts]
            down = down[start : start + cfg.num_local_experts]
        elif cfg.tp_size > 1:
            inter = cfg.intermediate_per_partition
            rank = get_tp_group().rank_in_group
            s, e = rank * inter, (rank + 1) * inter
            # gate_up 的第 2 维是 [gate | up] 拼接，两段各切同一区间
            gate_up = torch.cat(
                [gate_up[..., : cfg.intermediate_size][..., s:e],
                 gate_up[..., cfg.intermediate_size :][..., s:e]],
                dim=-1,
            )
            down = down[:, s:e, :]
        runner.experts.w13.data.copy_(gate_up)
        runner.experts.w2.data.copy_(down)


def normalize_moe_state_dict(
    state: dict[str, torch.Tensor], cfg: Qwen3MoeConfig
) -> dict[str, torch.Tensor]:
    """把 HF 的专家权重整理成堆叠布局，并修正路由键名。"""
    state = dict(state)

    # 1) 已是堆叠格式时先转置到本实现的 [H, 2I] / [I, H]
    for key in list(state):
        weight = state[key]
        if weight.dim() != 3:
            continue
        if key.endswith("mlp.experts.gate_up_proj") and weight.shape[1] != cfg.hidden_size:
            state[key] = weight.transpose(1, 2).contiguous()
        elif key.endswith("mlp.experts.down_proj") and weight.shape[1] != cfg.moe_intermediate_size:
            state[key] = weight.transpose(1, 2).contiguous()

    # 2) 逐专家 → 堆叠
    pattern = re.compile(
        r"(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )
    per_expert: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for key in list(state):
        match = pattern.match(key)
        if match is None:
            continue
        prefix, expert_id, proj = match.group(1), int(match.group(2)), match.group(3)
        per_expert.setdefault(prefix, {}).setdefault(expert_id, {})[proj] = state.pop(key)

    for prefix, experts in per_expert.items():
        ids = sorted(experts)
        gate_up = torch.stack(
            [
                torch.cat(
                    [experts[i]["gate_proj"].T, experts[i]["up_proj"].T], dim=-1
                )
                for i in ids
            ]
        )
        down = torch.stack([experts[i]["down_proj"].T for i in ids])
        state[f"{prefix}.gate_up_proj"] = gate_up
        state[f"{prefix}.down_proj"] = down

    # 3) 路由键名修正（历史坑：漏掉这一步会让 30B 一直用随机路由）
    for key in list(state):
        if key.endswith(".mlp.gate.weight"):
            state[key[: -len(".mlp.gate.weight")] + ".mlp.router.gate.weight"] = state.pop(key)

    return state
