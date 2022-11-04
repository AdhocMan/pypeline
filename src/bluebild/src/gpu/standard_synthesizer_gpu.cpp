#include <complex>
#include <functional>
#include <memory>
#include <cstring>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "gpu/intensity_field_data_gpu.hpp"
#include "gpu/kernels/add_vector.hpp"
#include "gpu/kernels/apply_filter.hpp"
#include "gpu/kernels/scale_matrix.hpp"
#include "gpu/kernels/gemmexp.hpp"
#include "gpu/sensitivity_field_data_gpu.hpp"
#include "gpu/standard_synthesizer_gpu.hpp"
#include "gpu/util/gpu_blas_api.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "util.hpp"

namespace bluebild {

template <typename T>
StandardSynthesisGPU<T>::StandardSynthesisGPU(
    std::shared_ptr<ContextInternal> ctx, int nAntenna, int nBeam,
    int nIntervals, int nFilter, const BluebildFilter *filterHost, int nPixel,
    const T *pixelX, const T *pixelY, const T *pixelZ)
    : ctx_(std::move(ctx)), nIntervals_(nIntervals), nFilter_(nFilter),
      nPixel_(nPixel), nAntenna_(nAntenna), nBeam_(nBeam) {
  filterHost_ = create_buffer<BluebildFilter>(ctx_->allocators().host(), nFilter_);
  std::memcpy(filterHost_.get(), filterHost, sizeof(BluebildFilter) * nFilter_);
  pixelX_ = create_buffer<T>(ctx_->allocators().gpu(), nPixel_);
  gpu::check_status(gpu::memcpy_async(pixelX_.get(), pixelX, sizeof(T) * nPixel_,
                                      gpu::flag::MemcpyDeviceToDevice,
                                      ctx_->gpu_stream()));
  pixelY_ = create_buffer<T>(ctx_->allocators().gpu(), nPixel_);
  gpu::check_status(gpu::memcpy_async(pixelY_.get(), pixelY, sizeof(T) * nPixel_,
                                      gpu::flag::MemcpyDeviceToDevice,
                                      ctx_->gpu_stream()));
  pixelZ_ = create_buffer<T>(ctx_->allocators().gpu(), nPixel_);
  gpu::check_status(gpu::memcpy_async(pixelZ_.get(), pixelZ, sizeof(T) * nPixel_,
                                      gpu::flag::MemcpyDeviceToDevice,
                                      ctx_->gpu_stream()));

  img_ = create_buffer<T>(ctx_->allocators().gpu(),
                          nPixel_ * nIntervals_ * nFilter_);
  gpu::memset_async(img_.get(), 0, nPixel_ * nIntervals_ * nFilter_ * sizeof(T),
                    ctx_->gpu_stream());
}

template <typename T>
auto StandardSynthesisGPU<T>::collect(int nEig, T wl, const T *intervalsHost,
                                      int ldIntervals,
                                      const gpu::ComplexType<T> *s, int lds,
                                      const gpu::ComplexType<T> *w, int ldw,
                                      const T *xyz, int ldxyz) -> void {

  auto v = create_buffer<gpu::ComplexType<T>>(ctx_->allocators().gpu(),
                                              nBeam_ * nEig);
  auto d = create_buffer<T>(ctx_->allocators().gpu(), nEig);
  auto vUnbeam =
      create_buffer<gpu::ComplexType<T>>(ctx_->allocators().gpu(), nAntenna_ * nEig);
  auto unlayeredStats =
      create_buffer<T>(ctx_->allocators().gpu(), nPixel_ * nEig);
  auto indices = create_buffer<int>(ctx_->allocators().gpu(), nEig);
  auto cluster =
      create_buffer<T>(ctx_->allocators().gpu(),
                       nIntervals_); // dummy input until
                                     // intensity_field_data_host can be updated

  if (s)
    intensity_field_data_gpu(*ctx_, wl, nAntenna_, nBeam_, nEig, s, lds, w,
                              ldw, xyz, ldxyz, d.get(), v.get(), nBeam_,
                              nIntervals_, cluster.get(), indices.get());
  else
    sensitivity_field_data_gpu(*ctx_, wl, nAntenna_, nBeam_, nEig, w, ldw, xyz,
                               ldxyz, d.get(), v.get(), nBeam_);

  auto DBufferHost = create_buffer<T>(ctx_->allocators().pinned(), nEig);
  auto DFilteredBufferHost = create_buffer<T>(ctx_->allocators().host(), nEig);
  gpu::check_status(
      gpu::memcpy_async(DBufferHost.get(), d.get(), nEig * sizeof(T),
                        gpu::flag::MemcpyDeviceToHost, ctx_->gpu_stream()));
  // Make sure D is available on host
  gpu::check_status(gpu::stream_synchronize(ctx_->gpu_stream()));

  gpu::ComplexType<T> one{1, 0};
  gpu::ComplexType<T> zero{0, 0};
  gpu::blas::check_status(gpu::blas::gemm(
      ctx_->gpu_blas_handle(), gpu::blas::operation::None,
      gpu::blas::operation::None, nAntenna_, nEig, nBeam_, &one, w, ldw,
      v.get(), nBeam_, &zero, vUnbeam.get(), nAntenna_));

  T alpha = 2.0 * M_PI / wl;
  gemmexp_gpu<T>(ctx_->gpu_stream(), nEig, nPixel_, nAntenna_, alpha,
                 vUnbeam.get(), nAntenna_, xyz, ldxyz, pixelX_.get(),
                 pixelY_.get(), pixelZ_.get(), unlayeredStats.get(), nPixel_);


  auto filterHost = filterHost_.get();
  for (std::size_t idxFilter = 0; idxFilter < static_cast<std::size_t>(nFilter_); ++idxFilter) {
    apply_filter(filterHost_.get()[idxFilter], nEig, DBufferHost.get(), DFilteredBufferHost.get());

    for (std::size_t idxInt = 0; idxInt < static_cast<std::size_t>(nIntervals_); ++idxInt) {
      std::size_t start, size;
      std::tie(start, size) = find_interval_indices(
          nEig, DBufferHost.get(),
          intervalsHost[idxInt * static_cast<std::size_t>(ldIntervals)],
          intervalsHost[idxInt * static_cast<std::size_t>(ldIntervals) + 1]);

      auto imgCurrent =
          img_.get() + (idxFilter * nIntervals_ + idxInt) * nPixel_;
      for (std::size_t idxEig = start; idxEig < start + size; ++idxEig) {
        const auto scale = DFilteredBufferHost.get()[idxEig];
        auto unlayeredStatsCurrent = unlayeredStats.get() + nPixel_ * idxEig;
        gpu::blas::check_status(
            gpu::blas::axpy(ctx_->gpu_blas_handle(), nPixel_, &scale,
                            unlayeredStatsCurrent, 1, imgCurrent, 1));
      }
    }
  }
}

template <typename T>
auto StandardSynthesisGPU<T>::get(BluebildFilter f, T *outHostOrDevice, int ld) -> void {
  int index = nFilter_;
  const BluebildFilter *filterPtr = filterHost_.get();
  for (int idxFilter = 0; idxFilter < nFilter_; ++idxFilter) {
    if (filterPtr[idxFilter] == f) {
      index = idxFilter;
      break;
    }
  }
  if (index == nFilter_)
    throw InvalidParameterError();

  gpu::check_status(gpu::memcpy_2d_async(
      outHostOrDevice, ld * sizeof(T),
      img_.get() + index * nIntervals_ * nPixel_, nPixel_ * sizeof(T),
      nPixel_ * sizeof(T), nIntervals_, gpu::flag::MemcpyDefault,
      ctx_->gpu_stream()));
}

template class StandardSynthesisGPU<float>;
template class StandardSynthesisGPU<double>;

} // namespace bluebild
