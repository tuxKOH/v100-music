// SPDX-License-Identifier: GPL-3.0-only
// Modulate computational activity using the GPU's global nanosecond timer.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cstdint>

class StopFlag {
 public:
  StopFlag() {
    C10_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host_), sizeof(int),
                                cudaHostAllocMapped | cudaHostAllocPortable));
    *host_ = 0;
  }
  ~StopFlag() { cudaFreeHost(host_); }
  void stop() { *static_cast<volatile int*>(host_) = 1; }
  volatile int* device_pointer() {
    void* pointer = nullptr;
    C10_CUDA_CHECK(cudaHostGetDevicePointer(&pointer, host_, 0));
    return static_cast<volatile int*>(pointer);
  }
 private:
  int* host_ = nullptr;
};

__device__ __forceinline__ unsigned long long timer_ns() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__global__ void initialize_start(int64_t* stats) {
  // All blocks on this GPU share an epoch; do not derive timing from SM clocks.
  stats[0] = static_cast<int64_t>(timer_ns() + 2000000ULL);
}

template<int Carrier>
__global__ void power_tone_kernel(
    const int64_t* periods, const int64_t* highs, const int64_t* ends,
    int event_count, float4* memory, unsigned int memory_count,
    float* sink, int64_t* stats, int64_t* edges, int edge_capacity,
    volatile int* stop, int work_repeats) {
  const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = gridDim.x * blockDim.x;
  const unsigned long long start = static_cast<unsigned long long>(stats[0]);
  const bool reporter = thread == 0;
  // Only one thread per block polls host memory, at most once per millisecond.
  // Reading a mapped PCIe flag in every thread/iteration would create a large
  // continuous traffic load, defeating the quiet part of the modulation.
  __shared__ volatile int block_stop;
  if (threadIdx.x == 0) block_stop = *stop;
  __syncthreads();
  unsigned long long next_stop_check = 0;
  float acc[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = (thread % 97 + i + 1) * 0.001f;
  unsigned int cursor = thread % memory_count;
  unsigned long long iterations = 0, edge_count = 0;
  int64_t last_cycle = -1;
  int event = 0;
  unsigned long long now;
  while ((now = timer_ns()) < start && !block_stop) {
    if (threadIdx.x == 0 && now >= next_stop_check) {
      block_stop = *stop;
      next_stop_check = now + 1000000ULL;
    }
    __nanosleep(2000);
  }
  while (!block_stop) {
    now = timer_ns();
    if (threadIdx.x == 0 && now >= next_stop_check) {
      block_stop = *stop;
      next_stop_check = now + 1000000ULL;
    }
    if (block_stop) break;
    const unsigned long long elapsed = now - start;
    while (event < event_count && elapsed >= static_cast<unsigned long long>(ends[event])) {
      ++event;
      last_cycle = -1;
    }
    if (event >= event_count) break;
    const int64_t period = periods[event];
    const unsigned long long local = elapsed - (event ? ends[event - 1] : 0);
    bool active = false;
    if (period > 0) {
      const int64_t cycle = static_cast<int64_t>(local / period);
      active = local % period < static_cast<unsigned long long>(highs[event]);
      if (reporter && active && highs[event] < period && cycle != last_cycle) {
        if (edge_count < static_cast<unsigned long long>(edge_capacity)) {
          edges[edge_count * 2] = event;
          edges[edge_count * 2 + 1] = static_cast<int64_t>(elapsed);
        }
        ++edge_count;
        last_cycle = cycle;
      }
    }
    if (!active) {
      // A resident waiting kernel is not a powered-off SM; this reduces activity.
      const unsigned long long wait = period > 0
          ? static_cast<unsigned long long>(period) - local % period
          : static_cast<unsigned long long>(ends[event]) - elapsed;
      // Sleep longer far from an edge. Approach the edge in small steps since
      // nanosleep is approximate, avoiding both a polling load and overshoot.
      const unsigned int sleep = wait > 8000ULL
          ? static_cast<unsigned int>(wait / 4 < 50000ULL ? wait / 4 : 50000ULL)
          : 1000;
      __nanosleep(sleep);
      continue;
    }
    if (Carrier != 1) {
      const unsigned long long phase = local % period;
      const unsigned long long active_until = highs[event] < period
          ? now + static_cast<unsigned long long>(highs[event]) - phase
          : start + static_cast<unsigned long long>(ends[event]);
      for (int repeat = 0; repeat < work_repeats; ++repeat) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
          acc[i] = fmaf(acc[i], 0.999991f, 0.000009f * (i + 1));
        }
        // Recheck time every 32 FMA groups to bound high-phase overshoot.
        if ((repeat & 31) == 31 && timer_ns() >= active_until) break;
      }
    }
    if (Carrier != 0) {
      // Cycle through a buffer larger than L2 instead of repeatedly touching
      // one tiny tensor. No CPU/VRAM copies occur during the tone.
      const float4 value = memory[cursor];
      memory[cursor] = make_float4(
          value.x * 0.99991f + 0.00009f, value.y * 0.99991f + 0.00018f,
          value.z * 0.99991f + 0.00027f, value.w * 0.99991f + 0.00036f);
      cursor += stride;
      if (cursor >= memory_count) cursor -= memory_count;
    }
    ++iterations;
  }
  float total = 0.0f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) total += acc[i];
  sink[thread] = total;
  if (reporter) {
    const auto finished = timer_ns();
    stats[1] = finished > start ? static_cast<int64_t>(finished - start) : 0;
    stats[2] = static_cast<int64_t>(iterations);
    stats[3] = static_cast<int64_t>(edge_count);
    stats[4] = block_stop;
  }
}

// The synthetic weight ring is not an NInfer checkpoint: each warp tile has
// 128 bytes of E2M1 codes, 16 E4M3 block scales, and 16 bytes of alignment pad.
// This exercises the same classes of hardware operations without model files.
constexpr int kWeightTileBytes = 160;

__device__ __forceinline__ float decode_e2m1(unsigned code) {
  const float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float value = magnitudes[code & 7];
  return code & 8 ? -value : value;
}

__device__ __forceinline__ float decode_e4m3(unsigned code) {
  const int exponent = (code >> 3) & 15;
  const unsigned mantissa = code & 7;
  if ((code & 127) == 127) return NAN;
  const float value = exponent ? ldexpf(1.0f + mantissa * 0.125f, exponent - 7)
                              : mantissa * (1.0f / 512.0f);
  return code & 128 ? -value : value;
}

__global__ void decode_weights_kernel(const uint8_t* codes, const uint8_t* scales,
                                      __half* output, int64_t count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const unsigned code = (codes[index / 2] >> ((index & 1) * 4)) & 15;
    output[index] = __float2half_rn(decode_e2m1(code) * decode_e4m3(scales[index / 16]));
  }
}

__global__ void initialize_weights(uint8_t* memory, uint64_t bytes) {
  auto words = reinterpret_cast<uint32_t*>(memory);
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < bytes / 4; index += stride) {
    uint32_t value = static_cast<uint32_t>(index) + 0x9e3779b9u;
    value = (value ^ (value >> 16)) * 0x85ebca6bu;
    value = (value ^ (value >> 13)) * 0xc2b2ae35u;
    value ^= value >> 16;
    // Finite positive E4M3 scales around 1/4 keep accumulated values bounded.
    if ((index * 4) % kWeightTileBytes >= 128) value = (value & 0x07070707u) | 0x28282828u;
    words[index] = value;
  }
}

__device__ __forceinline__ unsigned long long warp_timer(int lane) {
  const auto value = lane == 0 ? timer_ns() : 0ULL;
  return __shfl_sync(0xffffffffu, value, 0);
}

// Burst state: done, absolute busy deadline, event, cycle, report edge, ring step.
// A single-thread controller waits between finite full-grid kernels. Unlike
// the resident engine, the other SMs hold no blocks during the quiet phase.
__global__ void initialize_burst(int64_t* stats, int64_t* state) {
  stats[0] = static_cast<int64_t>(timer_ns() + 2000000ULL);
  for (int index = 1; index < 6; ++index) stats[index] = 0;
  state[0] = state[1] = state[4] = state[5] = 0;
  state[2] = state[3] = -1;
}

__device__ bool burst_wait_until(unsigned long long deadline, int64_t* stats,
                                  int64_t* state, volatile int* stop) {
  unsigned long long next_check = 0;
  while (true) {
    const auto now = timer_ns();
    if (now >= next_check) {
      if (*stop) {
        state[0] = 1;
        stats[1] = now > static_cast<unsigned long long>(stats[0]) ? now - stats[0] : 0;
        stats[4] = 1;
        return false;
      }
      next_check = now + 500000ULL;
    }
    if (now >= deadline) return true;
    const auto remaining = deadline - now;
    __nanosleep(remaining > 8000ULL
        ? static_cast<unsigned int>(remaining / 4 < 50000ULL ? remaining / 4 : 50000ULL) : 1000);
  }
}

__global__ void burst_controller(
    const int64_t* periods, const int64_t* highs, const int64_t* ends,
    int event_count, int64_t* stats, int64_t* state, volatile int* stop) {
  if (state[0]) return;
  const auto start = static_cast<unsigned long long>(stats[0]);
  if (!burst_wait_until(start, stats, state, stop)) return;
  int event = state[2] < 0 ? 0 : static_cast<int>(state[2]);
  while (true) {
    const auto now = timer_ns();
    const auto elapsed = now - start;
    while (event < event_count && elapsed >= static_cast<unsigned long long>(ends[event])) ++event;
    if (event >= event_count) {
      state[0] = 1;
      stats[1] = static_cast<int64_t>(elapsed);
      return;
    }
    const auto event_begin = event ? static_cast<unsigned long long>(ends[event - 1]) : 0ULL;
    const auto event_end = start + static_cast<unsigned long long>(ends[event]);
    if (periods[event] == 0) {
      if (!burst_wait_until(event_end, stats, state, stop)) return;
      continue;
    }
    const auto period = static_cast<unsigned long long>(periods[event]);
    int64_t cycle = static_cast<int64_t>((elapsed - event_begin) / period);
    if (state[2] == event && cycle <= state[3]) cycle = state[3] + 1;
    auto busy_start = start + event_begin + cycle * period;
    auto busy_end = busy_start + static_cast<unsigned long long>(highs[event]);
    if (highs[event] == periods[event]) busy_end = event_end;
    if (now >= busy_end) {
      busy_start += period;
      busy_end += period;
      ++cycle;
    }
    if (busy_start >= event_end) {
      if (!burst_wait_until(event_end, stats, state, stop)) return;
      continue;
    }
    if (!burst_wait_until(busy_start, stats, state, stop)) return;
    // Skip a high phase missed by launch/scheduling latency; never stretch a
    // late pulse into the next period. All cycles remain anchored to the epoch.
    if (timer_ns() >= busy_end) continue;
    state[1] = static_cast<int64_t>(busy_end < event_end ? busy_end : event_end);
    state[2] = event;
    state[3] = cycle;
    state[4] = highs[event] < periods[event];
    state[5] = stats[5];
    return;
  }
}

template<bool Dequantize, bool Burst = false>
__global__ void tensor_tone_kernel(
    const int64_t* periods, const int64_t* highs, const int64_t* ends,
    int event_count, const uint8_t* memory, uint64_t memory_bytes,
    float* sink, int64_t* stats, int64_t* edges, int edge_capacity,
    volatile int* stop, int work_repeats, int tensor_reuse, int64_t* state) {
  namespace wmma = nvcuda::wmma;
  if (Burst && state[0]) return;
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x / 32;
  const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t warp = blockIdx.x * (blockDim.x / 32) + warp_in_block;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * (blockDim.x / 32) * kWeightTileBytes;
  uint64_t cursor = warp * kWeightTileBytes;
  if (Burst) cursor += (static_cast<uint64_t>(state[5]) % (memory_bytes / stride)) * stride;
  const unsigned long long start = static_cast<unsigned long long>(stats[0]);
  const bool reporter = thread == 0;
  __shared__ volatile int block_stop;
  __shared__ __align__(32) __half a_shared[8][256];
  __shared__ __align__(32) __half b_shared[8][256];
  if (threadIdx.x == 0) block_stop = Burst ? 0 : *stop;
  for (int index = lane; index < 256; index += 32) {
    a_shared[warp_in_block][index] = __float2half_rn((index % 13 - 6) * 0.015625f);
    b_shared[warp_in_block][index] = __float2half_rn((index % 11 - 5) * 0.015625f);
  }
  __syncthreads();
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> d0, d1;
  wmma::load_matrix_sync(a, a_shared[warp_in_block], 16);
  wmma::load_matrix_sync(b, b_shared[warp_in_block], 16);
  wmma::fill_fragment(d0, 0.0f);
  wmma::fill_fragment(d1, 0.0f);
  unsigned long long next_stop_check = Burst ? timer_ns() + 1000000ULL : 0;
  unsigned long long iterations = 0, edge_count = Burst && reporter ? stats[3] : 0, tiles = 0;
  int64_t last_cycle = -1;
  int event = 0;
  unsigned long long now;
  // Control flow must be uniform within a warp: WMMA is a collective operation.
  while (!Burst && (now = warp_timer(lane)) < start) {
    __nanosleep(2000);
  }
  while (true) {
    now = warp_timer(lane);
    if (threadIdx.x == 0 && now >= next_stop_check) {
      block_stop = *stop;
      next_stop_check = now + 1000000ULL;
    }
    int cancelled = lane == 0 ? block_stop : 0;
    cancelled = __shfl_sync(0xffffffffu, cancelled, 0);
    if (cancelled) break;
    const unsigned long long elapsed = now - start;
    unsigned long long active_until;
    if (Burst) {
      active_until = static_cast<unsigned long long>(state[1]);
      if (now >= active_until) break;
      event = static_cast<int>(state[2]);
      if (reporter && iterations == 0 && state[4]) {
        if (edge_count < static_cast<unsigned long long>(edge_capacity)) {
          edges[edge_count * 2] = event;
          edges[edge_count * 2 + 1] = static_cast<int64_t>(elapsed);
        }
        ++edge_count;
      }
    } else {
      while (event < event_count && elapsed >= static_cast<unsigned long long>(ends[event])) {
        ++event;
        last_cycle = -1;
      }
      if (event >= event_count) break;
      const int64_t period = periods[event];
      const unsigned long long local = elapsed - (event ? ends[event - 1] : 0);
      const bool active = period > 0 && local % period < static_cast<unsigned long long>(highs[event]);
      if (!active) {
        const unsigned long long wait = period > 0
            ? static_cast<unsigned long long>(period) - local % period
            : static_cast<unsigned long long>(ends[event]) - elapsed;
        __nanosleep(wait > 8000ULL
            ? static_cast<unsigned int>(wait / 4 < 50000ULL ? wait / 4 : 50000ULL) : 1000);
        continue;
      }
      const int64_t cycle = static_cast<int64_t>(local / period);
      if (reporter && highs[event] < period && cycle != last_cycle) {
        if (edge_count < static_cast<unsigned long long>(edge_capacity)) {
          edges[edge_count * 2] = event;
          edges[edge_count * 2 + 1] = static_cast<int64_t>(elapsed);
        }
        ++edge_count;
        last_cycle = cycle;
      }
      const unsigned long long phase = local % period;
      active_until = highs[event] < period
          ? now + static_cast<unsigned long long>(highs[event]) - phase
          : start + static_cast<unsigned long long>(ends[event]);
    }
    if (Dequantize) {
      // Stream several weight tiles before revisiting schedule divisions.
      // Every tile checks its deadline, keeping transitions warp-uniform.
      for (int tile = 0; tile < work_repeats; ++tile) {
        const uint32_t codes = *reinterpret_cast<const uint32_t*>(memory + cursor + lane * 4);
        const float scale = decode_e4m3(memory[cursor + 128 + lane / 2]);
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const float low = decode_e2m1((codes >> (pair * 8)) & 15) * scale;
          const float high = decode_e2m1((codes >> (pair * 8 + 4)) & 15) * scale;
          *reinterpret_cast<__half2*>(&b_shared[warp_in_block][lane * 8 + pair * 2]) =
              __floats2half2_rn(low, high);
        }
        __syncwarp();
        wmma::load_matrix_sync(b, b_shared[warp_in_block], 16);
        cursor += stride;
        if (cursor >= memory_bytes) cursor -= memory_bytes;
        ++tiles;
        for (int repeat = 0; repeat < tensor_reuse; ++repeat) {
          wmma::mma_sync(d0, a, b, d0);
          wmma::mma_sync(d1, a, b, d1);
          if ((repeat & 7) == 7 && warp_timer(lane) >= active_until) break;
        }
        if (warp_timer(lane) >= active_until) break;
      }
    } else {
      for (int repeat = 0; repeat < work_repeats; ++repeat) {
        wmma::mma_sync(d0, a, b, d0);
        wmma::mma_sync(d1, a, b, d1);
        if ((repeat & 7) == 7 && warp_timer(lane) >= active_until) break;
      }
    }
    ++iterations;
  }
  float total = 0.0f;
  #pragma unroll
  for (int index = 0; index < d0.num_elements; ++index) total += d0.x[index] + d1.x[index];
  sink[thread] = total;
  if (reporter) {
    if (!Burst) stats[1] = static_cast<int64_t>(timer_ns() - start);
    stats[2] = (Burst ? stats[2] : 0) + static_cast<int64_t>(iterations);
    stats[3] = static_cast<int64_t>(edge_count);
    if (!Burst) stats[4] = block_stop;
    stats[5] = (Burst ? stats[5] : 0) + static_cast<int64_t>(tiles);
  }
}

int resident_blocks_per_sm(int device, int carrier, bool burst = false) {
  const c10::cuda::CUDAGuard guard(device);
  int blocks = 0;
  if (carrier == 0) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, power_tone_kernel<0>, 256, 0));
  } else if (carrier == 1) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, power_tone_kernel<1>, 256, 0));
  } else if (carrier == 2) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, power_tone_kernel<2>, 256, 0));
  } else if (carrier == 3) {
    if (burst) {
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, tensor_tone_kernel<false, true>, 256, 0));
    } else {
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, tensor_tone_kernel<false>, 256, 0));
    }
  } else {
    TORCH_CHECK(carrier == 4, "invalid carrier");
    if (burst) {
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, tensor_tone_kernel<true, true>, 256, 0));
    } else {
      C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, tensor_tone_kernel<true>, 256, 0));
    }
  }
  return blocks;
}

void launch(torch::Tensor periods, torch::Tensor highs, torch::Tensor ends,
            torch::Tensor memory, torch::Tensor sink, torch::Tensor stats,
            torch::Tensor edges, StopFlag& stop, int blocks, int carrier,
            int work_repeats, int tensor_reuse) {
  TORCH_CHECK(periods.is_cuda() && periods.scalar_type() == torch::kInt64,
              "periods must be CUDA int64");
  const c10::cuda::CUDAGuard guard(periods.device());
  for (const auto& tensor : {highs, ends, memory, sink, stats, edges}) {
    TORCH_CHECK(tensor.device() == periods.device() && tensor.is_contiguous(),
                "all tensors must be contiguous on the same GPU");
  }
  TORCH_CHECK(periods.is_contiguous() && periods.dim() == 1 && periods.numel() > 0,
              "empty or non-contiguous schedule");
  TORCH_CHECK(highs.scalar_type() == torch::kInt64 && ends.scalar_type() == torch::kInt64 &&
              highs.sizes() == periods.sizes() && ends.sizes() == periods.sizes(),
              "invalid timing arrays");
  TORCH_CHECK(blocks > 0 && carrier >= 0 && carrier <= 4, "invalid launch configuration");
  TORCH_CHECK(work_repeats > 0 && work_repeats <= 4096, "invalid compute intensity");
  TORCH_CHECK(tensor_reuse > 0 && tensor_reuse <= 256, "invalid tensor reuse");
  const int64_t bytes = memory.numel() * memory.element_size();
  TORCH_CHECK((carrier >= 3
                  ? memory.scalar_type() == torch::kUInt8 && bytes % (blocks * 8 * kWeightTileBytes) == 0
                  : memory.scalar_type() == torch::kFloat32 && memory.numel() % (blocks * 256 * 4) == 0) &&
              bytes > 0 && sink.scalar_type() == torch::kFloat32 &&
              sink.numel() >= blocks * 256, "invalid work buffers");
  TORCH_CHECK(stats.scalar_type() == torch::kInt64 && stats.numel() >= 6 &&
              edges.scalar_type() == torch::kInt64 && edges.dim() == 2 && edges.size(1) == 2,
              "invalid diagnostics buffers");
  cudaDeviceProp properties;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, periods.get_device()));
  TORCH_CHECK(blocks <= properties.multiProcessorCount *
              resident_blocks_per_sm(periods.get_device(), carrier),
              "persistent blocks must all fit concurrently on the GPU");
  auto stream = at::cuda::getCurrentCUDAStream();
  initialize_start<<<1, 1, 0, stream>>>(stats.data_ptr<int64_t>());
  volatile int* flag = stop.device_pointer();
  #define LAUNCH(kind) power_tone_kernel<kind><<<blocks, 256, 0, stream>>>( \
      periods.data_ptr<int64_t>(), highs.data_ptr<int64_t>(), ends.data_ptr<int64_t>(), \
      periods.numel(), reinterpret_cast<float4*>(memory.data_ptr<float>()), memory.numel() / 4, \
      sink.data_ptr<float>(), stats.data_ptr<int64_t>(), edges.data_ptr<int64_t>(), edges.size(0), flag, work_repeats)
  #define LAUNCH_TENSOR(dequant) tensor_tone_kernel<dequant><<<blocks, 256, 0, stream>>>( \
      periods.data_ptr<int64_t>(), highs.data_ptr<int64_t>(), ends.data_ptr<int64_t>(), \
      periods.numel(), memory.data_ptr<uint8_t>(), bytes, sink.data_ptr<float>(), \
      stats.data_ptr<int64_t>(), edges.data_ptr<int64_t>(), edges.size(0), flag, work_repeats, tensor_reuse, nullptr)
  if (carrier == 0) { LAUNCH(0); }
  else if (carrier == 1) { LAUNCH(1); }
  else if (carrier == 2) { LAUNCH(2); }
  else if (carrier == 3) { LAUNCH_TENSOR(false); }
  else { LAUNCH_TENSOR(true); }
  #undef LAUNCH
  #undef LAUNCH_TENSOR
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void start_burst(torch::Tensor stats, torch::Tensor state) {
  TORCH_CHECK(stats.is_cuda() && stats.is_contiguous() && stats.scalar_type() == torch::kInt64 &&
              stats.numel() >= 6 && state.device() == stats.device() && state.is_contiguous() &&
              state.scalar_type() == torch::kInt64 && state.numel() == 6, "invalid burst control buffers");
  const c10::cuda::CUDAGuard guard(stats.device());
  initialize_burst<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      stats.data_ptr<int64_t>(), state.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void burst_cycle(torch::Tensor periods, torch::Tensor highs, torch::Tensor ends,
                 torch::Tensor memory, torch::Tensor sink, torch::Tensor stats,
                 torch::Tensor edges, torch::Tensor state, StopFlag& stop,
                 int blocks, int carrier, int work_repeats, int tensor_reuse) {
  TORCH_CHECK(carrier == 3 || carrier == 4, "burst engine requires tensor or fp4 carrier");
  const c10::cuda::CUDAGuard guard(periods.device());
  for (const auto& tensor : {periods, highs, ends, memory, sink, stats, edges, state}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == periods.device() && tensor.is_contiguous(),
                "all burst tensors must be contiguous on the same GPU");
  }
  TORCH_CHECK(periods.scalar_type() == torch::kInt64 && periods.dim() == 1 && periods.numel() > 0 &&
              highs.scalar_type() == torch::kInt64 && highs.sizes() == periods.sizes() &&
              ends.scalar_type() == torch::kInt64 && ends.sizes() == periods.sizes() &&
              state.scalar_type() == torch::kInt64 && state.numel() == 6 &&
              stats.scalar_type() == torch::kInt64 && stats.numel() >= 6 &&
              edges.scalar_type() == torch::kInt64 && edges.dim() == 2 && edges.size(1) == 2,
              "invalid burst timing/diagnostics buffers");
  TORCH_CHECK(blocks > 0 && memory.scalar_type() == torch::kUInt8 && memory.numel() > 0 &&
              memory.numel() % (blocks * 8 * kWeightTileBytes) == 0 &&
              sink.scalar_type() == torch::kFloat32 && sink.numel() >= blocks * 256 &&
              work_repeats > 0 && work_repeats <= 4096 && tensor_reuse > 0 && tensor_reuse <= 256,
              "invalid burst work buffers/intensity");
  auto stream = at::cuda::getCurrentCUDAStream();
  volatile int* flag = stop.device_pointer();
  burst_controller<<<1, 1, 0, stream>>>(periods.data_ptr<int64_t>(), highs.data_ptr<int64_t>(),
      ends.data_ptr<int64_t>(), periods.numel(), stats.data_ptr<int64_t>(), state.data_ptr<int64_t>(), flag);
  #define BURST_TENSOR(dequant) tensor_tone_kernel<dequant, true><<<blocks, 256, 0, stream>>>( \
      periods.data_ptr<int64_t>(), highs.data_ptr<int64_t>(), ends.data_ptr<int64_t>(), \
      periods.numel(), memory.data_ptr<uint8_t>(), memory.numel(), sink.data_ptr<float>(), \
      stats.data_ptr<int64_t>(), edges.data_ptr<int64_t>(), edges.size(0), flag, work_repeats, tensor_reuse, \
      state.data_ptr<int64_t>())
  if (carrier == 3) { BURST_TENSOR(false); }
  else { BURST_TENSOR(true); }
  #undef BURST_TENSOR
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fill_weights(torch::Tensor memory) {
  TORCH_CHECK(memory.is_cuda() && memory.is_contiguous() && memory.scalar_type() == torch::kUInt8 &&
              memory.numel() > 0 && memory.numel() % kWeightTileBytes == 0, "invalid weight buffer");
  const c10::cuda::CUDAGuard guard(memory.device());
  initialize_weights<<<1024, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      memory.data_ptr<uint8_t>(), memory.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_link(torch::Tensor destination, torch::Tensor source) {
  TORCH_CHECK(destination.is_cuda() && source.is_cuda() && destination.is_contiguous() &&
              source.is_contiguous() && destination.scalar_type() == torch::kUInt8 &&
              source.scalar_type() == torch::kUInt8 && destination.numel() == source.numel() &&
              destination.device() != source.device(), "invalid link buffers");
  const c10::cuda::CUDAGuard guard(destination.device());
  // Do not force PCIe peer access; the runtime can stage through host memory.
  // This is transport load, not a distributed model or an allreduce operation.
  C10_CUDA_CHECK(cudaMemcpyAsync(destination.data_ptr(), source.data_ptr(),
      source.numel(), cudaMemcpyDeviceToDevice, at::cuda::getCurrentCUDAStream()));
}

void decode_weights(torch::Tensor codes, torch::Tensor scales, torch::Tensor output) {
  TORCH_CHECK(codes.is_cuda() && codes.is_contiguous() && codes.scalar_type() == torch::kUInt8 &&
              scales.device() == codes.device() && scales.is_contiguous() &&
              scales.scalar_type() == torch::kUInt8 && output.device() == codes.device() &&
              output.is_contiguous() && output.scalar_type() == torch::kFloat16 &&
              output.numel() == codes.numel() * 2 && output.numel() > 0 &&
              output.numel() % 16 == 0 && scales.numel() == output.numel() / 16,
              "invalid standalone FP4 decode buffers");
  const c10::cuda::CUDAGuard guard(codes.device());
  decode_weights_kernel<<<(output.numel() + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), output.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<StopFlag>(module, "StopFlag")
      .def(pybind11::init<>()).def("stop", &StopFlag::stop);
  module.def("launch", &launch);
  module.def("resident_blocks_per_sm", &resident_blocks_per_sm,
             pybind11::arg("device"), pybind11::arg("carrier"), pybind11::arg("burst") = false);
  module.def("start_burst", &start_burst);
  module.def("burst_cycle", &burst_cycle);
  module.def("fill_weights", &fill_weights);
  module.def("copy_link", &copy_link);
  module.def("decode_weights", &decode_weights);
}
