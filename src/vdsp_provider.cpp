#include "skbench/skbench.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if SKBENCH_HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace skbench {
namespace {

using Clock = std::chrono::steady_clock;

double elapsedSeconds(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string libraryContaining(const void* symbol) {
    Dl_info information{};
    if (symbol != nullptr && dladdr(symbol, &information) != 0 && information.dli_fname != nullptr) {
        return information.dli_fname;
    }
    return {};
}

class PlaneExecutor {
public:
    using Task = void (*)(void*, std::size_t, std::size_t, std::size_t);

    PlaneExecutor(std::size_t planes, std::size_t requestedWorkers)
        : planes_(planes), workers_(std::max<std::size_t>(1, std::min(planes, requestedWorkers))) {
        threads_.reserve(workers_ - 1);
        for (std::size_t worker = 1; worker < workers_; ++worker) {
            threads_.emplace_back([this, worker] { workerLoop(worker); });
        }
    }

    ~PlaneExecutor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_.notify_all();
        for (auto& thread : threads_) thread.join();
    }

    PlaneExecutor(const PlaneExecutor&) = delete;
    PlaneExecutor& operator=(const PlaneExecutor&) = delete;

    std::size_t workerCount() const noexcept { return workers_; }

    void run(void* context, Task task) {
        if (workers_ == 1) {
            task(context, 0, 0, planes_);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            context_ = context;
            task_ = task;
            remaining_ = workers_ - 1;
            ++generation_;
        }
        start_.notify_all();
        task(context, 0, 0, planes_ / workers_);
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return remaining_ == 0; });
        context_ = nullptr;
        task_ = nullptr;
    }

private:
    void workerLoop(std::size_t worker) {
        std::size_t observedGeneration = 0;
        for (;;) {
            void* context = nullptr;
            Task task = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_.wait(lock, [this, observedGeneration] { return stopping_ || generation_ != observedGeneration; });
                if (stopping_) return;
                observedGeneration = generation_;
                context = context_;
                task = task_;
            }
            task(context, worker, planes_ * worker / workers_, planes_ * (worker + 1) / workers_);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--remaining_ == 0) done_.notify_one();
            }
        }
    }

    std::size_t planes_ = 0;
    std::size_t workers_ = 1;
    std::size_t generation_ = 0;
    std::size_t remaining_ = 0;
    bool stopping_ = false;
    void* context_ = nullptr;
    Task task_ = nullptr;
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable done_;
    std::vector<std::thread> threads_;
};

bool powerOfTwo(std::size_t value) {
    return value >= 2 && (value & (value - 1)) == 0;
}

#if SKBENCH_HAVE_ACCELERATE
vDSP_Length log2Length(std::size_t value) {
    vDSP_Length result = 0;
    while ((std::size_t{1} << result) < value) ++result;
    return result;
}
#endif

} // namespace

class VDSPProvider::Impl {
public:
    Impl(const Workload& workload, std::size_t workers) : workload_(workload) {
        if (workers == 0) throw std::invalid_argument("vDSP workers must be positive.");
#if SKBENCH_HAVE_ACCELERATE
        if (!powerOfTwo(workload.nx) || !powerOfTwo(workload.ny)) {
            capability_ = "unsupported: vDSP radix-2 requires power-of-two Nx and Ny";
            return;
        }
        const auto setupStart = Clock::now();
        executor_ = std::make_unique<PlaneExecutor>(workload.planes(), workers);
        otherSetupSeconds_ = elapsedSeconds(setupStart);
        const auto nativeCount = workload.planes() * (workload.nx / 2) * workload.ny;
        const auto allocationStart = Clock::now();
        real_.resize(nativeCount);
        imag_.resize(nativeCount);
        setups_.resize(executor_->workerCount(), nullptr);
        allocationSeconds_ = elapsedSeconds(allocationStart);
        const auto planningStart = Clock::now();
        const auto setupLog2 = log2Length(std::max(workload.nx, workload.ny));
        for (auto& setup : setups_) {
            setup = vDSP_create_fftsetupD(setupLog2, kFFTRadix2);
            if (setup == nullptr) {
                for (auto created : setups_) if (created != nullptr) vDSP_destroy_fftsetupD(created);
                std::fill(setups_.begin(), setups_.end(), nullptr);
                throw std::runtime_error("vDSP setup creation failed.");
            }
        }
        supported_ = true;
        capability_ = "supported: double split-complex radix-2 2-D";
        planningSeconds_ = elapsedSeconds(planningStart);
#else
        (void)workers;
        capability_ = "unsupported: Accelerate is available only on Apple platforms";
#endif
    }

    ~Impl() {
#if SKBENCH_HAVE_ACCELERATE
        for (auto setup : setups_) {
            if (setup != nullptr) vDSP_destroy_fftsetupD(setup);
        }
#endif
    }

    void requireSupported() const {
        if (!supported_) throw std::runtime_error(capability_);
    }

    std::size_t nativePlaneElements() const { return (workload_.nx / 2) * workload_.ny; }

    void packForward(const double* input) {
        requireSupported();
        const auto half = workload_.nx / 2;
        const auto nativePlane = nativePlaneElements();
        for (std::size_t plane = 0; plane < workload_.planes(); ++plane) {
            const auto sourceOffset = plane * workload_.realPlaneElements();
            const auto destinationOffset = plane * nativePlane;
            for (std::size_t y = 0; y < workload_.ny; ++y) {
                for (std::size_t x = 0; x < half; ++x) {
                    const auto packed = destinationOffset + y * half + x;
                    real_[packed] = input[sourceOffset + y * workload_.nx + 2 * x];
                    imag_[packed] = input[sourceOffset + y * workload_.nx + 2 * x + 1];
                }
            }
        }
    }

    void executeForward() {
        requireSupported();
#if SKBENCH_HAVE_ACCELERATE
        executor_->run(this, &Impl::forwardTask);
#endif
    }

    void unpackForward(Complex* output) const {
        requireSupported();
        const auto half = workload_.nx / 2;
        const auto nativePlane = nativePlaneElements();
        const auto yNyquist = workload_.ny / 2;
        for (std::size_t plane = 0; plane < workload_.planes(); ++plane) {
            const auto z = plane % workload_.nz;
            const auto field = plane / workload_.nz;
            const auto offset = plane * nativePlane;
            for (std::size_t y = 0; y < workload_.ny; ++y) {
                for (std::size_t x = 1; x < half; ++x) {
                    const auto packed = offset + y * half + x;
                    output[wvmSpectrumIndex(workload_, x, y, z, field)] = {0.5 * real_[packed], 0.5 * imag_[packed]};
                }
            }
            output[wvmSpectrumIndex(workload_, 0, 0, z, field)] = {0.5 * real_[offset], 0.0};
            output[wvmSpectrumIndex(workload_, half, 0, z, field)] = {0.5 * imag_[offset], 0.0};
            output[wvmSpectrumIndex(workload_, 0, yNyquist, z, field)] = {0.5 * real_[offset + half], 0.0};
            output[wvmSpectrumIndex(workload_, half, yNyquist, z, field)] = {0.5 * imag_[offset + half], 0.0};
            for (std::size_t y = 1; y < yNyquist; ++y) {
                const auto first = offset + (2 * y) * half;
                const auto second = offset + (2 * y + 1) * half;
                const Complex zero{0.5 * real_[first], 0.5 * real_[second]};
                const Complex nyquist{0.5 * imag_[first], 0.5 * imag_[second]};
                output[wvmSpectrumIndex(workload_, 0, y, z, field)] = zero;
                output[wvmSpectrumIndex(workload_, 0, workload_.ny - y, z, field)] = conjugate(zero);
                output[wvmSpectrumIndex(workload_, half, y, z, field)] = nyquist;
                output[wvmSpectrumIndex(workload_, half, workload_.ny - y, z, field)] = conjugate(nyquist);
            }
        }
    }

    void packInverse(const Complex* input) {
        requireSupported();
        std::fill(real_.begin(), real_.end(), 0.0);
        std::fill(imag_.begin(), imag_.end(), 0.0);
        const auto half = workload_.nx / 2;
        const auto nativePlane = nativePlaneElements();
        const auto yNyquist = workload_.ny / 2;
        for (std::size_t plane = 0; plane < workload_.planes(); ++plane) {
            const auto z = plane % workload_.nz;
            const auto field = plane / workload_.nz;
            const auto offset = plane * nativePlane;
            for (std::size_t y = 0; y < workload_.ny; ++y) {
                for (std::size_t x = 1; x < half; ++x) {
                    const auto value = input[wvmSpectrumIndex(workload_, x, y, z, field)];
                    const auto packed = offset + y * half + x;
                    real_[packed] = value.real;
                    imag_[packed] = value.imag;
                }
            }
            real_[offset] = input[wvmSpectrumIndex(workload_, 0, 0, z, field)].real;
            imag_[offset] = input[wvmSpectrumIndex(workload_, half, 0, z, field)].real;
            real_[offset + half] = input[wvmSpectrumIndex(workload_, 0, yNyquist, z, field)].real;
            imag_[offset + half] = input[wvmSpectrumIndex(workload_, half, yNyquist, z, field)].real;
            for (std::size_t y = 1; y < yNyquist; ++y) {
                const auto zero = input[wvmSpectrumIndex(workload_, 0, y, z, field)];
                const auto nyquist = input[wvmSpectrumIndex(workload_, half, y, z, field)];
                const auto first = offset + (2 * y) * half;
                const auto second = offset + (2 * y + 1) * half;
                real_[first] = zero.real;
                real_[second] = zero.imag;
                imag_[first] = nyquist.real;
                imag_[second] = nyquist.imag;
            }
        }
    }

    void executeInverse() {
        requireSupported();
#if SKBENCH_HAVE_ACCELERATE
        executor_->run(this, &Impl::inverseTask);
#endif
    }

    void unpackInverse(double* output) const {
        requireSupported();
        const auto half = workload_.nx / 2;
        const auto nativePlane = nativePlaneElements();
        for (std::size_t plane = 0; plane < workload_.planes(); ++plane) {
            const auto sourceOffset = plane * nativePlane;
            const auto destinationOffset = plane * workload_.realPlaneElements();
            for (std::size_t y = 0; y < workload_.ny; ++y) {
                for (std::size_t x = 0; x < half; ++x) {
                    const auto packed = sourceOffset + y * half + x;
                    output[destinationOffset + y * workload_.nx + 2 * x] = real_[packed];
                    output[destinationOffset + y * workload_.nx + 2 * x + 1] = imag_[packed];
                }
            }
        }
    }

#if SKBENCH_HAVE_ACCELERATE
    static void forwardTask(void* context, std::size_t worker, std::size_t begin, std::size_t end) {
        auto& self = *static_cast<Impl*>(context);
        const auto nativePlane = self.nativePlaneElements();
        const auto half = self.workload_.nx / 2;
        for (std::size_t plane = begin; plane < end; ++plane) {
            DSPDoubleSplitComplex split{self.real_.data() + plane * nativePlane, self.imag_.data() + plane * nativePlane};
            vDSP_fft2d_zripD(self.setups_[worker], &split, 1, static_cast<vDSP_Stride>(half),
                             log2Length(self.workload_.nx), log2Length(self.workload_.ny), FFT_FORWARD);
        }
    }

    static void inverseTask(void* context, std::size_t worker, std::size_t begin, std::size_t end) {
        auto& self = *static_cast<Impl*>(context);
        const auto nativePlane = self.nativePlaneElements();
        const auto half = self.workload_.nx / 2;
        for (std::size_t plane = begin; plane < end; ++plane) {
            DSPDoubleSplitComplex split{self.real_.data() + plane * nativePlane, self.imag_.data() + plane * nativePlane};
            vDSP_fft2d_zripD(self.setups_[worker], &split, 1, static_cast<vDSP_Stride>(half),
                             log2Length(self.workload_.nx), log2Length(self.workload_.ny), FFT_INVERSE);
        }
    }
#endif

    Workload workload_;
    bool supported_ = false;
    std::string capability_;
    double otherSetupSeconds_ = 0.0;
    double allocationSeconds_ = 0.0;
    double planningSeconds_ = 0.0;
    std::unique_ptr<PlaneExecutor> executor_;
    std::vector<double> real_;
    std::vector<double> imag_;
#if SKBENCH_HAVE_ACCELERATE
    std::vector<FFTSetupD> setups_;
#endif
};

VDSPProvider::VDSPProvider(const Workload& workload, std::size_t workers) : impl_(std::make_unique<Impl>(workload, workers)) {}

VDSPProvider::~VDSPProvider() = default;
VDSPProvider::VDSPProvider(VDSPProvider&&) noexcept = default;
VDSPProvider& VDSPProvider::operator=(VDSPProvider&&) noexcept = default;

bool VDSPProvider::supported() const noexcept { return impl_->supported_; }
std::string VDSPProvider::capability() const { return impl_->capability_; }
void VDSPProvider::packForwardInput(const double* input) { impl_->packForward(input); }
void VDSPProvider::executeForwardNative() { impl_->executeForward(); }
void VDSPProvider::unpackForwardOutput(Complex* output) const { impl_->unpackForward(output); }
void VDSPProvider::packInverseInput(const Complex* input) { impl_->packInverse(input); }
void VDSPProvider::executeInverseNative() { impl_->executeInverse(); }
void VDSPProvider::unpackInverseOutput(double* output) const { impl_->unpackInverse(output); }

void VDSPProvider::forwardAdapter(const double* input, Complex* output) {
    packForwardInput(input);
    executeForwardNative();
    unpackForwardOutput(output);
}

void VDSPProvider::inverseAdapter(const Complex* input, double* output) {
    packInverseInput(input);
    executeInverseNative();
    unpackInverseOutput(output);
}

double VDSPProvider::otherSetupSeconds() const noexcept { return impl_->otherSetupSeconds_; }

double VDSPProvider::allocationSeconds() const noexcept { return impl_->allocationSeconds_; }

double VDSPProvider::planningSeconds() const noexcept { return impl_->planningSeconds_; }

std::size_t VDSPProvider::nativeBufferBytes() const noexcept {
    return (impl_->real_.size() + impl_->imag_.size()) * sizeof(double);
}

std::size_t VDSPProvider::explicitPersistentBytes() const noexcept {
    return nativeBufferBytes();
}

std::string VDSPProvider::libraryIdentity() const {
#if SKBENCH_HAVE_ACCELERATE
    return libraryContaining(reinterpret_cast<const void*>(&vDSP_fft2d_zripD));
#else
    return {};
#endif
}

} // namespace skbench
