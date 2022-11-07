#pragma once

#include <cstddef>
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild{
namespace gpu{
template <typename T>
auto center_vector_get_worksize(gpu::StreamType stream, int n, const T *in, T *out) -> std::size_t;

template <typename T>
auto center_vector(gpu::StreamType stream, int n, const T *in, T *out,
                   std::size_t worksize, void *work) -> void;
}
} // namespace bluebild
