#include "host/standard_synthesis_host.hpp"

#include <complex>
#include <complex.h>
#include <iostream>
#include <algorithm>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "host/blas_api.hpp"
#include "host/intensity_field_data_host.hpp"
#include "host/sensitivity_field_data_host.hpp"
#include "memory/allocator.hpp"
#include "memory/buffer.hpp"
#include "host/gemmexp.hpp"
#include "util.hpp"


namespace bluebild {

template <typename T>
static auto center_vector(std::size_t n, const T *__restrict__ in,
                          T *__restrict__ out) -> void {
  T mean = 0;
  for (std::size_t i = 0; i < n; ++i) {
    mean += in[i];
  }
  mean /= n;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = in[i] - mean;
  }
}

template <typename T>
StandardSynthesisHost<T>::StandardSynthesisHost(
    std::shared_ptr<ContextInternal> ctx, std::size_t nAntenna,
    std::size_t nBeam, std::size_t nIntervals, std::size_t nFilter,
    const BluebildFilter *filter, std::size_t nPixel, const T *pixelX,
    const T *pixelY, const T *pixelZ)
    : ctx_(std::move(ctx)), nIntervals_(nIntervals), nFilter_(nFilter),
      nPixel_(nPixel), nAntenna_(nAntenna), nBeam_(nBeam) {
  filter_ = create_buffer<BluebildFilter>(ctx_->allocators().host(), nFilter_);
  std::memcpy(filter_.get(), filter, sizeof(BluebildFilter) * nFilter_);
  pixelX_ = create_buffer<T>(ctx_->allocators().host(), nPixel_);
  std::memcpy(pixelX_.get(), pixelX, sizeof(T) * nPixel_);
  pixelY_ = create_buffer<T>(ctx_->allocators().host(), nPixel_);
  std::memcpy(pixelY_.get(), pixelY, sizeof(T) * nPixel_);
  pixelZ_ = create_buffer<T>(ctx_->allocators().host(), nPixel_);
  std::memcpy(pixelZ_.get(), pixelZ, sizeof(T) * nPixel_);

  img_ = create_buffer<T>(ctx_->allocators().host(),
                          nPixel_ * nIntervals_ * nFilter_);
  std::memset(img_.get(), 0, nPixel_ * nIntervals_ * nFilter_ * sizeof(T));
}

template <typename T>
auto StandardSynthesisHost<T>::collect(
    std::size_t nEig, T wl, const T *intervals, std::size_t ldIntervals,
    const std::complex<T> *s, std::size_t lds, const std::complex<T> *w,
    std::size_t ldw, const T *xyz, std::size_t ldxyz) -> void {

  auto v =
      create_buffer<std::complex<T>>(ctx_->allocators().host(), nBeam_ * nEig);
  auto vUnbeam =
      create_buffer<std::complex<T>>(ctx_->allocators().host(), nAntenna_ * nEig);
  auto unlayeredStats =
      create_buffer<T>(ctx_->allocators().host(), nPixel_ * nEig);
  auto d = create_buffer<T>(ctx_->allocators().host(), nEig);
  auto dFiltered = create_buffer<T>(ctx_->allocators().host(), nEig);
  auto indices = create_buffer<int>(ctx_->allocators().host(), nEig);
  auto cluster =
      create_buffer<T>(ctx_->allocators().host(),
                       nIntervals_); // dummy input until
                                     // intensity_field_data_host can be updated
  // Center coordinates for much better performance of cos / sin
  auto xyzCentered = create_buffer<T>(ctx_->allocators().host(), 3 * nAntenna_);
  center_vector(nAntenna_, xyz, xyzCentered.get());
  center_vector(nAntenna_, xyz + ldxyz, xyzCentered.get() + nAntenna_);
  center_vector(nAntenna_, xyz + 2 * ldxyz, xyzCentered.get() + 2 * nAntenna_);

  if (s)
    intensity_field_data_host(*ctx_, wl, nAntenna_, nBeam_, nEig, s, lds, w,
                              ldw, xyzCentered.get(), nAntenna_, d.get(), v.get(),
                              nBeam_, nIntervals_, cluster.get(),
                              indices.get());
  else
    sensitivity_field_data_host(*ctx_, wl, nAntenna_, nBeam_, nEig, w, ldw,
                                xyzCentered.get(), nAntenna_, d.get(), v.get(),
                                nBeam_);

  blas::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nAntenna_, nEig, nBeam_,
             {1, 0}, w, ldw, v.get(), nBeam_, {0, 0}, vUnbeam.get(), nAntenna_);


  T alpha = 2.0 * M_PI / wl;
  gemmexp(nEig, nPixel_, nAntenna_, alpha, vUnbeam.get(), nAntenna_,
          xyzCentered.get(), nAntenna_, pixelX_.get(), pixelY_.get(),
          pixelZ_.get(), unlayeredStats.get(), nPixel_);

  // cluster eigenvalues / vectors based on invervals
  for (std::size_t idxFilter = 0; idxFilter < nFilter_; ++idxFilter) {
    apply_filter(filter_.get()[idxFilter], nEig, d.get(), dFiltered.get());
    for (std::size_t idxInt = 0; idxInt < nIntervals_; ++idxInt) {
      std::size_t start, size;
      std::tie(start, size) = find_interval_indices<T>(
          nEig, d.get(), intervals[idxInt * ldIntervals],
          intervals[idxInt * ldIntervals + 1]);

      auto imgCurrent =
          img_.get() + (idxFilter * nIntervals_ + idxInt) * nPixel_;
      for (std::size_t idxEig = start; idxEig < start + size; ++idxEig) {
        const auto scale = dFiltered.get()[idxEig];
        auto unlayeredStatsCurrent = unlayeredStats.get() + nPixel_ * idxEig;
        for (std::size_t idxPix = 0; idxPix < nPixel_; ++idxPix) {
          imgCurrent[idxPix] += scale * unlayeredStatsCurrent[idxPix];
        }
      }
    }
  }
}

template <typename T>
auto StandardSynthesisHost<T>::get(BluebildFilter f, T *out, std::size_t ld)
    -> void {
  std::size_t index = nFilter_;
  const BluebildFilter *filterPtr = filter_.get();
  for (std::size_t i = 0; i < nFilter_; ++i) {
    if (filterPtr[i] == f) {
      index = i;
      break;
    }
  }
  if (index == nFilter_)
    throw InvalidParameterError();

  for (std::size_t i = 0; i < nIntervals_; ++i) {
    std::memcpy(out + i * ld,
                img_.get() + index * nIntervals_ * nPixel_ + i * nPixel_,
                sizeof(T) * nPixel_);
  }
}


template class StandardSynthesisHost<double>;

template class StandardSynthesisHost<float>;


} // namespace bluebild