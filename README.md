# leanmoe

从零实现的 Qwen3-30B-A3B（MoE，128 专家 / top-8）推理引擎，2×A100-80GB、TP2。
连续批处理、分页 KV、CUDA Graph、Triton paged attention、MoE grouped GEMM、
CUDA IPC all-reduce 均为本地实现，**零 vLLM 运行时依赖**；vLLM 0.32 只作对照基线。

统一口径下吞吐为 vLLM 的 **96.2%**（Offline 16）、**99.6%**（Offline 256）、
**87.5%–100%**（Server r1–128）。

- [结果](#结果)
- [工程亮点](#工程亮点)
- [架构](#架构)
- [快速开始](#快速开始)
- [范围与限制](#范围与限制)


## 结果

**环境**

| 项 | 值 |
|---|---|
| GPU | 2×A100-SXM4-80GB（NVLink） |
| 并行 | TP2 |
| 软件 | torch 2.13.0+cu130 / CUDA 13.0 / NCCL 2.29.7 |
| 基线 | vLLM 0.32.0 |
| 模型 | Qwen3-30B-A3B-Instruct-2507 |
| 配置 | `max_model_len=2048`、greedy、`ignore_eos`、关闭 prefix cache |

**吞吐与延迟**

| 场景 | 本引擎 | vLLM 0.32 | 比例 |
|---|---:|---:|---:|
| Offline 16×512/128 | 1469.7 tok/s | 1527.3 | **96.2%** |
| Offline 256 | 5176.6 tok/s | 5196.3 | **99.6%** |
| Server r1–128 | 122.9–4666.7 tok/s | 122.9–5335.4 | **87.5%–100%** |
| TPOT p50 | — | — | 慢 8.6%–35.5% |

**测量方法**

- 两侧由同一 harness 驱动：同一请求表、同一 Poisson 到达表、客户端 `perf_counter` 打点；
- TTFT 从计划到达时刻起算，含排队；
- vLLM 使用进程内 `AsyncLLMEngine`，关闭 detokenize 与 log stats；
- 每点重复测量取中位数，校验输出 token 数与完成数。

**口径说明**：vLLM 是多进程架构（EngineCore + IPC），本引擎单进程直调。
延迟只讲方向和量级，跨引擎不比较 TTFT/TPOT 的精确倍数。
Offline 256 按项目配置 `mmbt=131072` 比较；vLLM 自身最优为 65536，此时本引擎为 92.8%。

## 工程亮点

**1. MoE 权重布局统一，释放 28.9 GB**

移植版 kernel 原本需要 `[E,N,K]` 布局，每层缓存一份转置副本（0.56 GiB × 48 层）。
改为与本地 GEMM 共用 `[E,K,N]` 后：激活峰值 27.65 → 0.65 GiB，KV 预算 1.61 → **37.08 GiB**，
batch256 从无法启动到可跑 128k prefill 预算。

**2. 混合批的 TPOT 差距被量化**

混合批退回 eager：r16 时 eager 步占步数 **30.2%**、占总步时 **58.3%**
（单步 48.2 ms vs 纯 decode 16.2 ms）；纯 decode 的 ITL 只慢 **8.7%**。

**3. 非精确批按桶补齐**

图的 key 含请求数，17 条请求命中不了按 16/24 捕获的图。把请求数也 pad 到桶，
dummy 行 `slot=-1` 跳过 KV 写入、`seq_len=1` 防除零：
17 并发 **409.7 → 1005.4 tok/s（+145%）**，图/eager 输出 **1.0 exact**。

**4. prefill 因果早退**

混合批中 `q_len=1` 的 decode 序列不再按最长序列的 grid 白跑 KV 循环：
混合步 **244 → 52 ms**，rate 16 吞吐 **489 → 1705 tok/s（+249%）**。

## 架构

```
entrypoints/llm.py       提交请求、推 step 循环、收集输出
        │
runtime/engine/core.py   一步 = 调度 → 组批/前向 → 采样 → 回写
        ├── runtime/core/sched/    连续批处理、双预算、抢占
        ├── runtime/core/          块池、分页 KV、前缀缓存
        ├── runtime/worker/        组批、CUDA Graph、采样
        └── runtime/attention/     paged decode/prefill kernel 与后端选择
model_executor/          Qwen3-MoE 结构、层、fused MoE
distributed/             TP/DP/PP/EP 组、自研 CUDA IPC all-reduce
```

依赖方向单向：`entrypoints → engine → core/worker → model_executor → attention/distributed`。

## 快速开始

```bash
pip install -e .
pytest tests/
```

单卡最小示例（prompt 为 token id，tokenizer 在引擎外）：

```python
from leanmoe.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from leanmoe.entrypoints import LLM
from leanmoe.sequence import SamplingParams

cfg = VllmConfig(
    model=ModelConfig(model_path="/path/to/Qwen3-30B-A3B", max_model_len=2048),
    cache=CacheConfig(gpu_memory_utilization=0.85, enable_prefix_caching=False),
    scheduler=SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=8192),
)
llm = LLM(cfg)
out = llm.generate([[1, 2, 3]], SamplingParams(temperature=0.0, max_tokens=16))
print(out[0].outputs[0].token_ids)
llm.shutdown()
```

TP2 用 `torchrun --nproc-per-node=2` 启动同一入口。


## 范围与限制

| 已实现 | 未包含 |
|---|---|
| 连续批处理、chunked prefill、recompute 抢占 | HTTP/流式服务 |
| 分页 KV、prefix caching、引用计数块池 | LoRA / EPLB |
| decode CUDA Graph（token×长度双桶） | Qwen3 以外的模型 |
| 自研 Triton paged decode/prefill | |
| MoE 两条路径、TP2、自研 IPC all-reduce | |


