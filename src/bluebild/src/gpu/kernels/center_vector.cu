#include "bluebild/config.h"
#include "gpu/kernels/center_vector.hpp"
#include "gpu/util/gpu_runtime.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

#include <cub/cub.cuh>

namespace bluebild {
namespace gpu{

template <typename T>
static __global__ void
sub_from_vector_kernel(int n, const T *__restrict__ value,
                       const T *__restrict__ in, T *__restrict__ out) {
  const T mean = *value / n;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
       i += gridDim.x * blockDim.x) {
    out[i] = in[i] - mean;
  }
}

template <typename T>
auto center_vector_get_worksize(gpu::StreamType stream, int n, const T *in, T *out) -> std::size_t {
  std::size_t size = 0;
  gpu::check_status(::cub::DeviceReduce::Sum<const T *, T *>(nullptr, size, in,
                                                             out, n, stream));
  // Work size includes temporary storage for reduce operation result
  return size + sizeof(T);
}

template <typename T>
auto center_vector(gpu::StreamType stream, int n, const T *in, T *out,
                   std::size_t worksize, void *work) -> void {
  constexpr int blockSize = 256;
  constexpr int maxBlocks = 65535;

  worksize -= sizeof(T);

  // To avoid alignment issues for type T, sum up at beginning of work array and
  // provide remaining memory to reduce function
  T *sumPtr = reinterpret_cast<T *>(work);

  gpu::check_status(::cub::DeviceReduce::Sum<const T *, T *>(
      reinterpret_cast<T *>(work) + 1, worksize, in, sumPtr, n, stream));

  dim3 block(blockSize, 1, 1);
  dim3 grid(std::min<unsigned int>(maxBlocks, (n + block.x - 1) / block.x) / 2 + 1, 1,
            1);
  gpu::launch_kernel(sub_from_vector_kernel<T>, grid, block, 0, stream, n,
                     sumPtr, in, out);
}

template auto center_vector_get_worksize<float>(gpu::StreamType stream, int n,
                                                const float *in, float *out)
    -> std::size_t;

template auto center_vector_get_worksize<double>(gpu::StreamType stream, int n,
                                                 const double *in, double *out)
    -> std::size_t;

template auto center_vector<float>(gpu::StreamType stream, int n,
                                   const float *in, float *out,
                                   std::size_t worksize, void *work) -> void;

template auto center_vector<double>(gpu::StreamType stream, int n,
                                   const double *in, double *out,
                                   std::size_t worksize, void *work) -> void;
}
} // namespace bluebild
