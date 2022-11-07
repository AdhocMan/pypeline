#include "bluebild/config.h"
#include "gpu/kernels/gemmexp.hpp"
#include "gpu/util/gpu_runtime.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

#include <cub/cub.cuh>

namespace bluebild {

static __device__ __forceinline__ void calc_sincos(float x, float *sptr,
                                                   float *cptr) {
  sincosf(x, sptr, cptr);
}

static __device__ __forceinline__ void calc_sincos(double x, double *sptr,
                                                   double *cptr) {
  sincos(x, sptr, cptr);
}

namespace {
template <typename T> struct ComplexOp {
  __device__ __forceinline__ ComplexOp() = default;
  __device__ __forceinline__ ComplexOp(T x_, T y_) : x(x_), y(y_) {}
  __device__ __forceinline__ ComplexOp(const gpu::ComplexType<T> &c)
      : x(c.x), y(c.y) {}

  __device__ __forceinline__ ComplexOp<T>
  operator-(const ComplexOp<T> &other) const {
    return ComplexOp{x - other.x, y - other.y};
  }

  __device__ __forceinline__ ComplexOp<T>
  operator+(const ComplexOp<T> &other) const {
    return ComplexOp{x + other.x, y + other.y};
  }

  __device__ __forceinline__ ComplexOp<T>
  operator*(const ComplexOp<T> &other) const {
    return ComplexOp{x * other.x - y * other.y, x * other.y + other.x * y};
  }

  T x, y;
};
} // namespace

template <typename T, int BLOCK_THREADS, cub::BlockReduceAlgorithm ALGORITHM>
static __global__ void
gemmexp_kernel(int nEig, int nPixel, int nAntenna, T alpha,
               const gpu::ComplexType<T> *__restrict__ vUnbeam, int ldv,
               const T *__restrict__ xyz, int ldxyz,
               const T *__restrict__ pixelX, const T *__restrict__ pixelY,
               const T *__restrict__ pixelZ, T *__restrict__ out, int ldout) {
  using BlockReduceType =
      cub::BlockReduce<ComplexOp<T>, BLOCK_THREADS, ALGORITHM>;
  __shared__ typename BlockReduceType::TempStorage tmpStorage;

  for (int idxEig = blockIdx.y; idxEig < nEig; idxEig += gridDim.y) {
    for (int idxPix = blockIdx.x; idxPix < nPixel; idxPix += gridDim.x) {
      const auto pX = pixelX[idxPix];
      const auto pY = pixelY[idxPix];
      const auto pZ = pixelZ[idxPix];

      ComplexOp<T> localSum{0, 0};
      for (int idxAnt = threadIdx.x; idxAnt < nAntenna; idxAnt += blockDim.x) {
        const auto imag = alpha * (pX * xyz[idxAnt] + pY * xyz[idxAnt + ldxyz] +
                                   pZ * xyz[idxAnt + 2 * ldxyz]);
        ComplexOp<T> sc;
        calc_sincos(imag, &(sc.y), &(sc.x));
        localSum = localSum + sc * vUnbeam[idxEig * ldv + idxAnt];
      }

      auto totalSum = BlockReduceType(tmpStorage).Sum(localSum);
      if (threadIdx.x == 0) {
        out[idxEig * ldout + idxPix] =
            totalSum.x * totalSum.x + totalSum.y * totalSum.y;
      }
    }
  }
}

template <typename T>
auto gemmexp_gpu(gpu::StreamType stream, int nEig, int nPixel, int nAntenna,
                 T alpha, const gpu::ComplexType<T> *vUnbeam, int ldv,
                 const T *xyz, int ldxyz, const T *pixelX, const T *pixelY,
                 const T *pixelZ, T *out, int ldout) -> void {
  constexpr int blockSize = 512;
  constexpr int maxBlocks = 65535;

  dim3 block(blockSize, 1, 1);
  dim3 grid(std::min<unsigned int>(maxBlocks, nPixel),
            std::min<unsigned int>(maxBlocks, nEig), 1);

  gpu::launch_kernel(
      gemmexp_kernel<T, blockSize,
                     cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>,
      grid, block, 0, stream, nEig, nPixel, nAntenna, alpha, vUnbeam, ldv, xyz,
      ldxyz, pixelX, pixelY, pixelZ, out, ldout);
}

template auto gemmexp_gpu<float>(
    gpu::StreamType stream, int nEig, int nPixel, int nAntenna, float alpha,
    const gpu::ComplexType<float> *__restrict__ vUnbeam, int ldv,
    const float *__restrict__ xyz, int ldxyz, const float *__restrict__ pixelX,
    const float *__restrict__ pixelY, const float *__restrict__ pixelZ,
    float *__restrict__ out, int ldout) -> void;

template auto gemmexp_gpu<double>(
    gpu::StreamType stream, int nEig, int nPixel, int nAntenna, double alpha,
    const gpu::ComplexType<double> *__restrict__ vUnbeam, int ldv,
    const double *__restrict__ xyz, int ldxyz,
    const double *__restrict__ pixelX, const double *__restrict__ pixelY,
    const double *__restrict__ pixelZ, double *__restrict__ out, int ldout)
    -> void;
} // namespace bluebild
