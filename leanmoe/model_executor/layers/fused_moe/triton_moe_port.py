"""本地重写的 grouped MoE（算法与 kernel 结构参考 vLLM fused_moe，Apache-2.0）。

布局约定：B = [E, K, N]（本项目改为 K 连续，与本地 GEMM 共用权重）；
sorted_ids 存扁平 (token, k) 行号，padding 位填 num_valid。
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _align_kernel(
    TOPK, SORTED, EXPERT_IDS, CURSOR, n_pairs,
    BLOCK_M: tl.constexpr, BLOCK: tl.constexpr,
):
    """计数排序：把扁平行号按专家写进 sorted_ids，并记录每个 block 的专家。"""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_pairs
    e = tl.load(TOPK + idx, mask=mask, other=0)
    pos = tl.atomic_add(CURSOR + e, 1, mask=mask)
    tl.store(SORTED + pos, idx, mask=mask)
    # 同一 block 内的多个行写同一个专家号，值相同，非原子写是安全的
    tl.store(EXPERT_IDS + pos // BLOCK_M, e, mask=mask)


@triton.jit
def _moe_gemm_kernel(
    A, B, C, TOPK_W, SORTED, EXPERT_IDS, NUM_POST,
    EM, NUM_VALID,
    N: tl.constexpr, K: tl.constexpr, TOP_K: tl.constexpr,
    MUL_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_M * num_pid_n
    group_id = pid // num_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_in_group) % group_size_m)
    pid_n = (pid % num_in_group) // group_size_m

    if pid_m * BLOCK_M >= tl.load(NUM_POST):
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(SORTED + offs_m).to(tl.int64)      # 扁平行号
    token_mask = tok < NUM_VALID
    expert = tl.load(EXPERT_IDS + pid_m).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    row_a = tok // TOP_K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(
            A + row_a[:, None] * K + offs_k[None, :],
            mask=token_mask[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            # B 为 [E, K, N]（与本地 grouped GEMM 同一布局），N 方向连续
            B + expert * K * N + offs_k[:, None] * N + offs_n[None, :],
            mask=mask_n[None, :] & mask_k[:, None],
            other=0.0,
        )
        acc += tl.dot(a, b)

    if MUL_W:
        w = tl.load(TOPK_W + tok, mask=token_mask, other=0.0).to(tl.float32)
        acc = acc * w[:, None]

    tl.store(
        C + tok[:, None] * N + offs_n[None, :],
        acc.to(C.dtype.element_ty),
        mask=token_mask[:, None] & mask_n[None, :],
    )


def _align(topk_ids: torch.Tensor, num_experts: int, block_m: int):
    """生成 vLLM 布局的 sorted_ids / expert_ids / num_post。"""
    numel = topk_ids.numel()
    flat = topk_ids.reshape(-1).to(torch.int64)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    padded = ((counts + block_m - 1) // block_m) * block_m
    offsets = torch.cumsum(padded, dim=0).to(torch.int32)
    cursor = (offsets - padded).contiguous()

    total_max = numel + num_experts * (block_m - 1)
    sorted_ids = torch.full(
        (total_max,), numel, dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty(
        (triton.cdiv(total_max, block_m),), dtype=torch.int32, device=topk_ids.device
    )
    _align_kernel[(triton.cdiv(numel, 256),)](
        flat.to(torch.int32), sorted_ids, expert_ids, cursor, numel,
        BLOCK_M=block_m, BLOCK=256,
    )
    return sorted_ids, expert_ids, offsets[-1:], total_max


def _moe_gemm(
    a: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_post: torch.Tensor,
    total_max: int,
    num_valid: int,
    top_k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    mul_weight: bool,
) -> None:
    n = w.shape[2]
    grid = (triton.cdiv(total_max, block_m) * triton.cdiv(n, block_n),)
    # 调参开关（默认与 vLLM 默认配置一致）
    group_m = int(os.environ.get("LEANMOE_PORT_GROUP_M", "1"))
    stages = int(os.environ.get("LEANMOE_PORT_STAGES", "3"))
    _moe_gemm_kernel[grid](
        a, w, out,
        topk_weights if topk_weights is not None else a,
        sorted_ids, expert_ids, num_post,
        total_max, num_valid,
        N=n, K=w.shape[1], TOP_K=top_k, MUL_W=mul_weight,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        num_warps=num_warps, num_stages=stages,
    )


@triton.jit
def _moe_sum_kernel(
    INP, OUT,
    TOPK: tl.constexpr, H: tl.constexpr, BLOCK_H: tl.constexpr,
):
    """out[t, h] = sum_k inp[t*TOPK + k, h]，fp32 累加（与 torch.sum 的 bf16 语义一致）。"""
    t = tl.program_id(0)
    hb = tl.program_id(1)
    offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        acc += tl.load(
            INP + (t * TOPK + k) * H + offs, mask=mask, other=0.0
        ).to(tl.float32)
    tl.store(OUT + t * H + offs, acc.to(OUT.dtype.element_ty), mask=mask)


def _moe_sum(inp: torch.Tensor, num_tokens: int, top_k: int, hidden: int) -> torch.Tensor:
    out = torch.empty(
        (num_tokens, hidden), dtype=inp.dtype, device=inp.device
    )
    BLOCK_H = 512
    grid = (num_tokens, triton.cdiv(hidden, BLOCK_H))
    _moe_sum_kernel[grid](
        inp, out,
        TOPK=top_k, H=hidden, BLOCK_H=BLOCK_H, num_warps=4,
    )
    return out


def fused_experts_port(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    silu_and_mul,
) -> torch.Tensor:
    """w1 [E, K, 2N_partition]、w2 [E, I, H]（与本地 grouped GEMM 共用布局）。"""
    num_tokens, k = hidden_states.shape
    top_k = topk_ids.shape[1]
    num_valid = num_tokens * top_k

    if hidden_states.shape[0] > 128:
        block_m, block_n, block_k, warps = 64, 128, 64, 8
    else:
        block_m, block_n, block_k, warps = 16, 64, 128, 4
    # 调参开关（默认留空，即用上面的启发式）
    if os.environ.get("LEANMOE_PORT_TILES"):
        block_m, block_n, block_k, warps = (
            int(x) for x in os.environ["LEANMOE_PORT_TILES"].split(",")
        )

    sorted_ids, expert_ids, num_post, total_max = _align(topk_ids, w1.shape[0], block_m)

    inter = torch.empty(
        (num_valid, w1.shape[2]), dtype=hidden_states.dtype, device=hidden_states.device
    )
    _moe_gemm(
        hidden_states, w1, inter, None, sorted_ids, expert_ids, num_post,
        total_max, num_valid, top_k, block_m, block_n, block_k, warps,
        mul_weight=False,
    )
    act = silu_and_mul(inter)

    out_flat = torch.empty(
        (num_valid, w2.shape[2]), dtype=hidden_states.dtype, device=hidden_states.device
    )
    _moe_gemm(
        act, w2, out_flat, topk_weights.reshape(-1), sorted_ids, expert_ids, num_post,
        total_max, num_valid, 1, block_m, block_n, block_k, warps,
        mul_weight=True,
    )
    # 与 vLLM 的 moe_sum 一致：用融合 kernel 一次读 TOP_K 行并累加，
    # 避免 torch.sum 走通用归约路径（prefill 每层要读 ~4GB）。
    return _moe_sum(out_flat, num_tokens, top_k, w2.shape[2])
