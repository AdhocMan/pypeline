#pragma once

#include <complex>
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild{
template <typename T>
auto gemmexp_gpu(gpu::StreamType stream, int nEig, int nPixel, int nAntenna,
                 T alpha, const gpu::ComplexType<T> *vUnbeam, int ldv,
                 const T *xyz, int ldxyz, const T *pixelX, const T *pixelY,
                 const T *pixelZ, T *out, int ldout) -> void;
} // namespace bluebild
