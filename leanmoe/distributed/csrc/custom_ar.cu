// 2 卡 CUDA IPC all-reduce 的最小实现（本项目自研，不依赖 vllm）。
//
// 思路与 vLLM 的 CustomAllreduce 同源：每卡持有一块持久 send buffer，
// 通过 CUDA IPC 打开对端 buffer 指针；同步不用设备端自旋，而用 CUDA IPC event
// （对端写完后 record，本地 stream_wait_event），避免死锁。
//
// 仅支持 2 卡、bf16 的 sum，规模上限由 Python 侧 buffer 决定。

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#define CHECK(expr)                                                          \
  do {                                                                       \
    cudaError_t err = (expr);                                                \
    if (err != cudaSuccess) {                                                \
      throw std::runtime_error(std::string(#expr) + ": " +                  \
                               cudaGetErrorString(err));                     \
    }                                                                        \
  } while (0)

__global__ void sum2_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                            __nv_bfloat16* out, int64_t n,
                            unsigned long long* my_ctr,
                            unsigned long long* peer_ctr) {
  // 每个 block 先登记一次，再等对端同一批登记完成——单调计数器，图回放安全
  if (threadIdx.x == 0) {
    __threadfence();
    unsigned long long mine = atomicAdd(my_ctr, 1ULL) + 1ULL;
    while (atomicAdd(peer_ctr, 0ULL) < mine) {
      __nanosleep(64);
    }
  }
  __syncthreads();

  // 向量化：一次搬 8 个 bf16（16B），与 vLLM 的 packed 访问一致
  int64_t nvec = n >> 3;
  int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  const uint4* a4 = reinterpret_cast<const uint4*>(a);
  const uint4* b4 = reinterpret_cast<const uint4*>(b);
  uint4* o4 = reinterpret_cast<uint4*>(out);
  for (int64_t i = gid; i < nvec; i += stride) {
    uint4 va = a4[i];
    uint4 vb = b4[i];
    uint4 vo;
    const __nv_bfloat16* pa = reinterpret_cast<const __nv_bfloat16*>(&va);
    const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&vb);
    __nv_bfloat16* po = reinterpret_cast<__nv_bfloat16*>(&vo);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      po[k] = __float2bfloat16(__bfloat162float(pa[k]) + __bfloat162float(pb[k]));
    }
    o4[i] = vo;
  }
  // 尾部不足 8 个元素的标量兜底
  for (int64_t i = (nvec << 3) + gid; i < n; i += stride) {
    out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
  }
}

void sum2(uint64_t a, uint64_t b, uint64_t out, int64_t n,
          uint64_t my_ctr, uint64_t peer_ctr) {
  if (n <= 0) return;
  int threads = 256;
  // 块数上限 32（与 vLLM 的 max_blocks 思路一致）：
  // 自旋等待的 block 不会占满全部 SM，对端 kernel 才能被同时调度。
  int64_t nvec = (n >> 3) + ((n & 7) ? 1 : 0);
  int blocks = (int)((nvec + threads - 1) / threads);
  if (blocks > 32) blocks = 32;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  sum2_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(a),
      reinterpret_cast<const __nv_bfloat16*>(b),
      reinterpret_cast<__nv_bfloat16*>(out), n,
      reinterpret_cast<unsigned long long*>(my_ctr),
      reinterpret_cast<unsigned long long*>(peer_ctr));
  CHECK(cudaGetLastError());
}

// ---- IPC 显存 ----

void get_mem_handle(uint64_t ptr, uint64_t out_addr) {
  cudaIpcMemHandle_t handle;
  CHECK(cudaIpcGetMemHandle(&handle, reinterpret_cast<void*>(ptr)));
  memcpy(reinterpret_cast<uint8_t*>(out_addr), &handle, sizeof(handle));
}

uint64_t open_mem_handle(uint64_t in_addr) {
  cudaIpcMemHandle_t handle;
  memcpy(&handle, reinterpret_cast<const uint8_t*>(in_addr), sizeof(handle));
  void* ptr = nullptr;
  CHECK(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<uint64_t>(ptr);
}

// ---- IPC event（跨进程同步）----

uint64_t create_event() {
  cudaEvent_t ev;
  CHECK(cudaEventCreateWithFlags(&ev, cudaEventInterprocess | cudaEventDisableTiming));
  return reinterpret_cast<uint64_t>(ev);
}

void get_event_handle(uint64_t ev, uint64_t out_addr) {
  cudaIpcEventHandle_t handle;
  CHECK(cudaIpcGetEventHandle(&handle, reinterpret_cast<cudaEvent_t>(ev)));
  memcpy(reinterpret_cast<uint8_t*>(out_addr), &handle, sizeof(handle));
}

uint64_t open_event_handle(uint64_t in_addr) {
  cudaIpcEventHandle_t handle;
  memcpy(&handle, reinterpret_cast<const uint8_t*>(in_addr), sizeof(handle));
  cudaEvent_t ev;
  CHECK(cudaIpcOpenEventHandle(&ev, handle));
  return reinterpret_cast<uint64_t>(ev);
}

void record_event(uint64_t ev) {
  CHECK(cudaEventRecord(reinterpret_cast<cudaEvent_t>(ev),
                        at::cuda::getCurrentCUDAStream()));
}

void stream_wait_event(uint64_t ev) {
  CHECK(cudaStreamWaitEvent(at::cuda::getCurrentCUDAStream(),
                            reinterpret_cast<cudaEvent_t>(ev), 0));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sum2", &sum2, "out = a + b (bf16)");
  m.def("get_mem_handle", &get_mem_handle, "导出 IPC 显存句柄");
  m.def("open_mem_handle", &open_mem_handle, "打开对端 IPC 显存句柄");
  m.def("create_event", &create_event, "创建 IPC event");
  m.def("get_event_handle", &get_event_handle, "导出 event 句柄");
  m.def("open_event_handle", &open_event_handle, "打开对端 event 句柄");
  m.def("record_event", &record_event, "在当前流 record event");
  m.def("stream_wait_event", &stream_wait_event, "当前流等待 event");
}
