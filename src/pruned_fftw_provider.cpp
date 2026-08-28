#include "skbench/skbench.hpp"

#include <fftw3.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstddef>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>

namespace skbench {
namespace {

using Clock = std::chrono::steady_clock;
std::mutex prunedPlanningMutex;

double elapsedSeconds(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

unsigned plannerFlags(FFTWPlanningMode mode) {
    switch (mode) {
        case FFTWPlanningMode::estimate: return FFTW_ESTIMATE | FFTW_UNALIGNED;
        case FFTWPlanningMode::measure: return FFTW_MEASURE | FFTW_UNALIGNED;
        case FFTWPlanningMode::patient: return FFTW_PATIENT | FFTW_UNALIGNED;
        case FFTWPlanningMode::exhaustive: return FFTW_EXHAUSTIVE | FFTW_UNALIGNED;
    }
    throw std::invalid_argument("Unknown FFTW planning mode.");
}

std::string libraryContaining(const void* symbol) {
    Dl_info information{};
    if (symbol != nullptr && dladdr(symbol, &information) != 0 && information.dli_fname != nullptr) {
        return information.dli_fname;
    }
    return {};
}

} // namespace

class FFTWPrunedProvider::Impl {
public:
    Impl(const Workload& workload, const std::vector<RetainedMode>& modes,
         FFTWPlanningMode planningMode, std::size_t internalWorkers)
        : workload_(workload), modes_(modes), planningMode_(planningMode), internalWorkers_(internalWorkers) {
        static_assert(sizeof(Complex) == sizeof(fftw_complex));
        if (modes_.empty()) throw std::invalid_argument("The pruned FFTW provider requires retained modes.");
        if (internalWorkers_ == 0 || internalWorkers_ > static_cast<std::size_t>(INT_MAX)) {
            throw std::invalid_argument("Pruned FFTW internal workers must lie in [1, INT_MAX].");
        }
        for (const auto& mode : modes_) {
            if (mode.storedKx >= workload_.nxHalf() || mode.storedKy >= workload_.ny) {
                throw std::invalid_argument("A retained mode lies outside the FFTW half-spectrum.");
            }
            activeKxCount_ = std::max(activeKxCount_, mode.storedKx + 1);
        }

        const auto setupStart = Clock::now();
        {
            std::lock_guard<std::mutex> lock(prunedPlanningMutex);
            if (fftw_init_threads() == 0) throw std::runtime_error("fftw_init_threads failed.");
        }
        otherSetupSeconds_ = elapsedSeconds(setupStart);

        const auto realBytes = workload_.realElements() * sizeof(double);
        const auto intermediateBytes = workload_.spectrumElements() * sizeof(Complex);
        planningBytes_ = realBytes + intermediateBytes;
        const auto allocationStart = Clock::now();
        realSurrogate_ = static_cast<double*>(fftw_malloc(realBytes));
        intermediate_ = static_cast<Complex*>(fftw_malloc(intermediateBytes));
        if (realSurrogate_ == nullptr || intermediate_ == nullptr) {
            releaseStorage();
            throw std::bad_alloc();
        }
        allocationSeconds_ = elapsedSeconds(allocationStart);

        try {
            std::lock_guard<std::mutex> lock(prunedPlanningMutex);
            fftw_plan_with_nthreads(static_cast<int>(internalWorkers_));
            const auto planningStart = Clock::now();
            createPlans(plannerFlags(planningMode_));
            planningSeconds_ = elapsedSeconds(planningStart);
        } catch (...) {
            destroyPlans();
            releaseStorage();
            throw;
        }
        fftw_free(realSurrogate_);
        realSurrogate_ = nullptr;
    }

    ~Impl() {
        std::lock_guard<std::mutex> lock(prunedPlanningMutex);
        destroyPlans();
        releaseStorage();
    }

    void executeForwardRows(const double* input) {
        fftw_execute_dft_r2c(rowForward_, const_cast<double*>(input),
                             reinterpret_cast<fftw_complex*>(intermediate_));
    }

    void executeForwardColumns() {
        auto* values = reinterpret_cast<fftw_complex*>(intermediate_);
        fftw_execute_dft(columnForward_, values, values);
    }

    void gatherForward(Complex* retainedSpectrum) const {
        for (std::size_t modeIndex = 0; modeIndex < modes_.size(); ++modeIndex) {
            const auto& mode = modes_[modeIndex];
            for (std::size_t field = 0; field < workload_.fields; ++field) {
                for (std::size_t z = 0; z < workload_.nz; ++z) {
                    auto value = intermediate_[planeMajorSpectrumIndex(
                        workload_, mode.storedKx, mode.storedKy, z, field)];
                    if (mode.conjugatesStoredValue) value = conjugate(value);
                    retainedSpectrum[retainedSpectrumIndex(
                        workload_, modeIndex, z, field)] = value;
                }
            }
        }
    }

    void forward(const double* input, Complex* retainedSpectrum) {
        executeForwardRows(input);
        executeForwardColumns();
        gatherForward(retainedSpectrum);
    }

    void embedInverse(const Complex* retainedSpectrum) {
        std::fill_n(intermediate_, workload_.spectrumElements(), Complex{});
        for (std::size_t modeIndex = 0; modeIndex < modes_.size(); ++modeIndex) {
            const auto& mode = modes_[modeIndex];
            for (std::size_t field = 0; field < workload_.fields; ++field) {
                for (std::size_t z = 0; z < workload_.nz; ++z) {
                    const auto compact = retainedSpectrum[retainedSpectrumIndex(
                        workload_, modeIndex, z, field)];
                    const auto stored = mode.conjugatesStoredValue ? conjugate(compact) : compact;
                    intermediate_[planeMajorSpectrumIndex(
                        workload_, mode.storedKx, mode.storedKy, z, field)] = stored;
                    if (mode.storedKx == 0 && mode.storedKy != 0 &&
                        2 * mode.storedKy != workload_.ny) {
                        const auto conjugateKy = (workload_.ny - mode.storedKy) % workload_.ny;
                        intermediate_[planeMajorSpectrumIndex(
                            workload_, 0, conjugateKy, z, field)] = conjugate(stored);
                    }
                }
            }
        }
    }

    void executeInverseColumns() {
        auto* values = reinterpret_cast<fftw_complex*>(intermediate_);
        fftw_execute_dft(columnInverse_, values, values);
    }

    void executeInverseRows(double* output) {
        fftw_execute_dft_c2r(rowInverse_, reinterpret_cast<fftw_complex*>(intermediate_), output);
    }

    void inverse(const Complex* retainedSpectrum, double* output) {
        embedInverse(retainedSpectrum);
        executeInverseColumns();
        executeInverseRows(output);
    }

    Workload workload_;
    std::vector<RetainedMode> modes_;
    FFTWPlanningMode planningMode_ = FFTWPlanningMode::measure;
    std::size_t internalWorkers_ = 1;
    std::size_t activeKxCount_ = 0;
    double* realSurrogate_ = nullptr;
    Complex* intermediate_ = nullptr;
    fftw_plan rowForward_ = nullptr;
    fftw_plan columnForward_ = nullptr;
    fftw_plan columnInverse_ = nullptr;
    fftw_plan rowInverse_ = nullptr;
    double otherSetupSeconds_ = 0.0;
    double allocationSeconds_ = 0.0;
    double planningSeconds_ = 0.0;
    std::size_t planningBytes_ = 0;

private:
    void createPlans(unsigned flags) {
        const auto nx = static_cast<ptrdiff_t>(workload_.nx);
        const auto ny = static_cast<ptrdiff_t>(workload_.ny);
        const auto nxHalf = static_cast<ptrdiff_t>(workload_.nxHalf());
        const auto planes = static_cast<ptrdiff_t>(workload_.planes());
        const auto realPlane = static_cast<ptrdiff_t>(workload_.realPlaneElements());
        const auto spectrumPlane = static_cast<ptrdiff_t>(workload_.halfRows());

        fftw_iodim64 rowForwardDimension[1] = {{nx, 1, 1}};
        fftw_iodim64 rowForwardBatches[2] = {
            {ny, nx, nxHalf},
            {planes, realPlane, spectrumPlane}};
        rowForward_ = fftw_plan_guru64_dft_r2c(
            1, rowForwardDimension, 2, rowForwardBatches, realSurrogate_,
            reinterpret_cast<fftw_complex*>(intermediate_), flags);

        fftw_iodim64 columnDimension[1] = {{ny, nxHalf, nxHalf}};
        fftw_iodim64 columnBatches[2] = {
            {static_cast<ptrdiff_t>(activeKxCount_), 1, 1},
            {planes, spectrumPlane, spectrumPlane}};
        auto* values = reinterpret_cast<fftw_complex*>(intermediate_);
        columnForward_ = fftw_plan_guru64_dft(
            1, columnDimension, 2, columnBatches, values, values, FFTW_FORWARD, flags);
        columnInverse_ = fftw_plan_guru64_dft(
            1, columnDimension, 2, columnBatches, values, values, FFTW_BACKWARD, flags);

        fftw_iodim64 rowInverseDimension[1] = {{nx, 1, 1}};
        fftw_iodim64 rowInverseBatches[2] = {
            {ny, nxHalf, nx},
            {planes, spectrumPlane, realPlane}};
        rowInverse_ = fftw_plan_guru64_dft_c2r(
            1, rowInverseDimension, 2, rowInverseBatches,
            reinterpret_cast<fftw_complex*>(intermediate_), realSurrogate_, flags);

        if (rowForward_ == nullptr || columnForward_ == nullptr ||
            columnInverse_ == nullptr || rowInverse_ == nullptr) {
            destroyPlans();
            throw std::runtime_error("FFTW could not create the partially pruned separable plans.");
        }
    }

    void destroyPlans() {
        if (rowForward_ != nullptr) fftw_destroy_plan(rowForward_);
        if (columnForward_ != nullptr) fftw_destroy_plan(columnForward_);
        if (columnInverse_ != nullptr) fftw_destroy_plan(columnInverse_);
        if (rowInverse_ != nullptr) fftw_destroy_plan(rowInverse_);
        rowForward_ = nullptr;
        columnForward_ = nullptr;
        columnInverse_ = nullptr;
        rowInverse_ = nullptr;
    }

    void releaseStorage() {
        if (realSurrogate_ != nullptr) fftw_free(realSurrogate_);
        if (intermediate_ != nullptr) fftw_free(intermediate_);
        realSurrogate_ = nullptr;
        intermediate_ = nullptr;
    }
};

FFTWPrunedProvider::FFTWPrunedProvider(
    const Workload& workload, const std::vector<RetainedMode>& modes,
    FFTWPlanningMode planningMode, std::size_t internalWorkers)
    : impl_(std::make_unique<Impl>(workload, modes, planningMode, internalWorkers)) {}

FFTWPrunedProvider::~FFTWPrunedProvider() = default;
FFTWPrunedProvider::FFTWPrunedProvider(FFTWPrunedProvider&&) noexcept = default;
FFTWPrunedProvider& FFTWPrunedProvider::operator=(FFTWPrunedProvider&&) noexcept = default;

void FFTWPrunedProvider::executeForwardRows(const double* input) { impl_->executeForwardRows(input); }
void FFTWPrunedProvider::executeForwardColumns() { impl_->executeForwardColumns(); }
void FFTWPrunedProvider::gatherForward(Complex* retainedSpectrum) const { impl_->gatherForward(retainedSpectrum); }
void FFTWPrunedProvider::forward(const double* input, Complex* retainedSpectrum) {
    impl_->forward(input, retainedSpectrum);
}
void FFTWPrunedProvider::embedInverse(const Complex* retainedSpectrum) { impl_->embedInverse(retainedSpectrum); }
void FFTWPrunedProvider::executeInverseColumns() { impl_->executeInverseColumns(); }
void FFTWPrunedProvider::executeInverseRows(double* output) { impl_->executeInverseRows(output); }
void FFTWPrunedProvider::inverse(const Complex* retainedSpectrum, double* output) {
    impl_->inverse(retainedSpectrum, output);
}

std::size_t FFTWPrunedProvider::activeKxCount() const noexcept { return impl_->activeKxCount_; }
std::size_t FFTWPrunedProvider::fullKxCount() const noexcept { return impl_->workload_.nxHalf(); }
std::size_t FFTWPrunedProvider::rowTransformsPerDirection() const noexcept {
    return impl_->workload_.ny * impl_->workload_.planes();
}
std::size_t FFTWPrunedProvider::columnTransformsPerDirection() const noexcept {
    return impl_->activeKxCount_ * impl_->workload_.planes();
}
std::size_t FFTWPrunedProvider::omittedColumnTransformsPerDirection() const noexcept {
    return (impl_->workload_.nxHalf() - impl_->activeKxCount_) * impl_->workload_.planes();
}
std::size_t FFTWPrunedProvider::scratchBytes() const noexcept {
    return impl_->workload_.spectrumElements() * sizeof(Complex);
}
std::size_t FFTWPrunedProvider::planningBytes() const noexcept { return impl_->planningBytes_; }
std::size_t FFTWPrunedProvider::minimumAlignmentBytes() const noexcept { return 1; }
std::size_t FFTWPrunedProvider::internalWorkers() const noexcept { return impl_->internalWorkers_; }
double FFTWPrunedProvider::otherSetupSeconds() const noexcept { return impl_->otherSetupSeconds_; }
double FFTWPrunedProvider::allocationSeconds() const noexcept { return impl_->allocationSeconds_; }
double FFTWPrunedProvider::planningSeconds() const noexcept { return impl_->planningSeconds_; }
FFTWPlanningMode FFTWPrunedProvider::planningMode() const noexcept { return impl_->planningMode_; }
bool FFTWPrunedProvider::completeHalfSpectrumOutputMaterialized() const noexcept { return false; }
bool FFTWPrunedProvider::inPlaceRetainedOperatorSupported() const noexcept { return false; }
std::string FFTWPrunedProvider::inPlaceRetainedOperatorCapability() const {
    return "unsupported in the initial candidate: the logical retained output is disjoint from the real input, while the selected complex column transforms execute in-place inside reusable full-sized plane-major row-spectrum scratch";
}
std::string FFTWPrunedProvider::libraryIdentity() const {
    return libraryContaining(reinterpret_cast<const void*>(&fftw_execute));
}
std::string FFTWPrunedProvider::version() const { return fftw_version; }

} // namespace skbench
