#pragma once

#include "bluebild/config.h"
#include "context_internal.hpp"
#include "memory/buffer.hpp"
#include "gpu/util/gpu_runtime_api.hpp"


namespace bluebild {

template <typename T> class StandardSynthesisGPU {
public:
  StandardSynthesisGPU(std::shared_ptr<ContextInternal> ctx, int nAntenna,
                       int nBeam, int nIntervals, int nFilter,
                       const BluebildFilter *filterHost, int nPixel,
                       const T *pixelX, const T *pixelY, const T *pixelZ);

  auto collect(int nEig, T wl, const T *intervalsHost, int ldIntervals,
               const gpu::ComplexType<T> *s, int lds,
               const gpu::ComplexType<T> *w, int ldw, const T *xyz, int ldxyz)
      -> void;

  auto get(BluebildFilter f, T *outHostOrDevice, int ld) -> void;

  auto context() -> ContextInternal & { return *ctx_; }

private:
  std::shared_ptr<ContextInternal> ctx_;
  const int nIntervals_, nFilter_, nPixel_, nAntenna_, nBeam_;
  BufferType<BluebildFilter> filterHost_;
  BufferType<T> pixelX_, pixelY_, pixelZ_;
  BufferType<T> img_;
};

} // namespace bluebild
