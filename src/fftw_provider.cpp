#include "skbench/skbench.hpp"

#include <fftw3.h>

#include <chrono>
#include <climits>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace skbench {
namespace {

using Clock = std::chrono::steady_clock;
std::mutex planningMutex;

std::string libraryContaining(const void* symbol) {
    Dl_info information{};
    if (symbol != nullptr && dladdr(symbol, &information) != 0 && information.dli_fname != nullptr) {
        return information.dli_fname;
    }
    return {};
}

double elapsedSeconds(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

} // namespace

class FFTWProvider::Impl {
public:
    Impl(const Workload& workload, std::size_t workers) : workload_(workload), workers_(workers) {
        static_assert(sizeof(Complex) == sizeof(fftw_complex));
        if (workers == 0 || workers > static_cast<std::size_t>(INT_MAX)) {
            throw std::invalid_argument("FFTW workers must lie in [1, INT_MAX].");
        }

        const auto setupStart = Clock::now();
        std::lock_guard<std::mutex> lock(planningMutex);
        if (fftw_init_threads() == 0) throw std::runtime_error("fftw_init_threads failed.");
        fftw_plan_with_nthreads(static_cast<int>(workers_));
        otherSetupSeconds_ = elapsedSeconds(setupStart);

        const auto nxHalf = workload_.nxHalf();
        const auto planes = workload_.planes();
        const auto realPlane = workload_.realPlaneElements();
        const auto realField = realPlane * workload_.nz;

        fftw_iodim64 forwardDimensions[2] = {
            {static_cast<ptrdiff_t>(workload_.ny), static_cast<ptrdiff_t>(workload_.nx), static_cast<ptrdiff_t>(planes * nxHalf)},
            {static_cast<ptrdiff_t>(workload_.nx), 1, static_cast<ptrdiff_t>(planes)}};
        fftw_iodim64 forwardBatches[2] = {
            {static_cast<ptrdiff_t>(workload_.nz), static_cast<ptrdiff_t>(realPlane), 1},
            {static_cast<ptrdiff_t>(workload_.fields), static_cast<ptrdiff_t>(realField), static_cast<ptrdiff_t>(workload_.nz)}};
        fftw_iodim64 inverseDimensions[2] = {
            {forwardDimensions[0].n, forwardDimensions[0].os, forwardDimensions[0].is},
            {forwardDimensions[1].n, forwardDimensions[1].os, forwardDimensions[1].is}};
        fftw_iodim64 inverseBatches[2] = {
            {forwardBatches[0].n, forwardBatches[0].os, forwardBatches[0].is},
            {forwardBatches[1].n, forwardBatches[1].os, forwardBatches[1].is}};

        const auto realBytes = workload_.realElements() * sizeof(double);
        const auto spectrumBytes = workload_.spectrumElements() * sizeof(Complex);
        planningBytes_ = realBytes + spectrumBytes;
        const auto allocationStart = Clock::now();
        auto* realSurrogate = static_cast<double*>(fftw_malloc(realBytes));
        auto* spectrumSurrogate = static_cast<fftw_complex*>(fftw_malloc(spectrumBytes));
        if (realSurrogate == nullptr || spectrumSurrogate == nullptr) {
            fftw_free(realSurrogate);
            fftw_free(spectrumSurrogate);
            throw std::bad_alloc();
        }
        allocationSeconds_ = elapsedSeconds(allocationStart);

        constexpr unsigned flags = FFTW_MEASURE | FFTW_UNALIGNED;
        const auto planningStart = Clock::now();
        forward_ = fftw_plan_guru64_dft_r2c(2, forwardDimensions, 2, forwardBatches, realSurrogate, spectrumSurrogate, flags);
        inverse_ = fftw_plan_guru64_dft_c2r(2, inverseDimensions, 2, inverseBatches, spectrumSurrogate, realSurrogate, flags);
        planningSeconds_ = elapsedSeconds(planningStart);
        fftw_free(realSurrogate);
        fftw_free(spectrumSurrogate);
        if (forward_ == nullptr || inverse_ == nullptr) {
            if (forward_ != nullptr) fftw_destroy_plan(forward_);
            if (inverse_ != nullptr) fftw_destroy_plan(inverse_);
            throw std::runtime_error("FFTW could not create the WVM-compatible guru64 plans.");
        }
    }

    ~Impl() {
        std::lock_guard<std::mutex> lock(planningMutex);
        if (forward_ != nullptr) fftw_destroy_plan(forward_);
        if (inverse_ != nullptr) fftw_destroy_plan(inverse_);
    }

    void forward(const double* input, Complex* output) {
        fftw_execute_dft_r2c(forward_, const_cast<double*>(input), reinterpret_cast<fftw_complex*>(output));
    }

    void inverse(Complex* input, double* output) {
        fftw_execute_dft_c2r(inverse_, reinterpret_cast<fftw_complex*>(input), output);
    }

    Workload workload_;
    std::size_t workers_ = 1;
    fftw_plan forward_ = nullptr;
    fftw_plan inverse_ = nullptr;
    double otherSetupSeconds_ = 0.0;
    double allocationSeconds_ = 0.0;
    double planningSeconds_ = 0.0;
    std::size_t planningBytes_ = 0;
};

FFTWProvider::FFTWProvider(const Workload& workload, std::size_t workers) : impl_(std::make_unique<Impl>(workload, workers)) {}

FFTWProvider::~FFTWProvider() = default;
FFTWProvider::FFTWProvider(FFTWProvider&&) noexcept = default;
FFTWProvider& FFTWProvider::operator=(FFTWProvider&&) noexcept = default;

void FFTWProvider::forward(const double* input, Complex* wvmSpectrum) { impl_->forward(input, wvmSpectrum); }

void FFTWProvider::inverse(Complex* wvmSpectrum, double* output) { impl_->inverse(wvmSpectrum, output); }

double FFTWProvider::otherSetupSeconds() const noexcept { return impl_->otherSetupSeconds_; }

double FFTWProvider::allocationSeconds() const noexcept { return impl_->allocationSeconds_; }

double FFTWProvider::planningSeconds() const noexcept { return impl_->planningSeconds_; }

std::size_t FFTWProvider::planningBytes() const noexcept { return impl_->planningBytes_; }

std::string FFTWProvider::libraryIdentity() const { return libraryContaining(reinterpret_cast<const void*>(&fftw_execute)); }

std::string FFTWProvider::version() const { return fftw_version; }

} // namespace skbench
