#include "skbench/skbench.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if SKBENCH_HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif
#if defined(__APPLE__) && SKBENCH_HAVE_ACCELERATE
#include <dispatch/dispatch.h>
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

class BatchExecutor {
public:
    using Task = void (*)(void*, std::size_t, std::size_t, std::size_t);

    virtual ~BatchExecutor() = default;
    virtual std::size_t workerCount() const noexcept = 0;
    virtual void run(void* context, Task task) = 0;
};

class PersistentPlaneExecutor final : public BatchExecutor {
public:

    PersistentPlaneExecutor(std::size_t planes, std::size_t requestedWorkers)
        : planes_(planes), workers_(std::max<std::size_t>(1, std::min(planes, requestedWorkers))) {
        threads_.reserve(workers_ - 1);
        for (std::size_t worker = 1; worker < workers_; ++worker) {
            threads_.emplace_back([this, worker] { workerLoop(worker); });
        }
    }

    ~PersistentPlaneExecutor() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_.notify_all();
        for (auto& thread : threads_) thread.join();
    }

    PersistentPlaneExecutor(const PersistentPlaneExecutor&) = delete;
    PersistentPlaneExecutor& operator=(const PersistentPlaneExecutor&) = delete;

    std::size_t workerCount() const noexcept override { return workers_; }

    void run(void* context, Task task) override {
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

#if defined(__APPLE__) && SKBENCH_HAVE_ACCELERATE
class GcdPlaneExecutor final : public BatchExecutor {
public:
    GcdPlaneExecutor(std::size_t planes, std::size_t requestedWorkers)
        : planes_(planes), workers_(std::max<std::size_t>(1, std::min(planes, requestedWorkers))),
          queue_(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0)) {}

    std::size_t workerCount() const noexcept override { return workers_; }

    void run(void* context, Task task) override {
        RunContext run{context, task, planes_, workers_};
        dispatch_apply_f(workers_, queue_, &run, &GcdPlaneExecutor::invoke);
    }

private:
    struct RunContext {
        void* taskContext;
        Task task;
        std::size_t planes;
        std::size_t workers;
    };

    static void invoke(void* context, std::size_t worker) {
        const auto& run = *static_cast<const RunContext*>(context);
        run.task(run.taskContext, worker, run.planes * worker / run.workers,
                 run.planes * (worker + 1) / run.workers);
    }

    std::size_t planes_ = 0;
    std::size_t workers_ = 1;
    dispatch_queue_t queue_ = nullptr;
};
#endif

bool powerOfTwo(std::size_t value) {
    return value >= 2 && (value & (value - 1)) == 0;
}

class AlignedBuffer {
public:
    static constexpr std::size_t alignment = 64;

    AlignedBuffer() = default;
    explicit AlignedBuffer(std::size_t count) { resize(count); }
    ~AlignedBuffer() { std::free(data_); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            std::free(data_);
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    void resize(std::size_t count) {
        if (count == size_) return;
        std::free(data_);
        data_ = nullptr;
        size_ = 0;
        if (count == 0) return;
        void* storage = nullptr;
        if (posix_memalign(&storage, alignment, count * sizeof(double)) != 0) throw std::bad_alloc();
        data_ = static_cast<double*>(storage);
        size_ = count;
        std::fill(data_, data_ + size_, 0.0);
    }

    double* data() noexcept { return data_; }
    const double* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    double* begin() noexcept { return data_; }
    double* end() noexcept { return data_ + size_; }

private:
    double* data_ = nullptr;
    std::size_t size_ = 0;
};

#if SKBENCH_HAVE_ACCELERATE
vDSP_Length log2Length(std::size_t value) {
    vDSP_Length result = 0;
    while ((std::size_t{1} << result) < value) ++result;
    return result;
}
#endif

} // namespace

std::string_view vdspTransformStrategyName(VDSPTransformStrategy strategy) noexcept {
    switch (strategy) {
        case VDSPTransformStrategy::inPlace: return "in-place";
        case VDSPTransformStrategy::inPlaceExplicitScratch: return "in-place-explicit-scratch";
        case VDSPTransformStrategy::outOfPlace: return "out-of-place";
        case VDSPTransformStrategy::outOfPlaceExplicitScratch: return "out-of-place-explicit-scratch";
    }
    return "unknown";
}

VDSPTransformStrategy vdspTransformStrategyNamed(std::string_view name) {
    if (name == "in-place") return VDSPTransformStrategy::inPlace;
    if (name == "in-place-explicit-scratch") return VDSPTransformStrategy::inPlaceExplicitScratch;
    if (name == "out-of-place") return VDSPTransformStrategy::outOfPlace;
    if (name == "out-of-place-explicit-scratch") return VDSPTransformStrategy::outOfPlaceExplicitScratch;
    throw std::invalid_argument("Unknown vDSP strategy: " + std::string(name));
}

std::string_view vdspBatchStrategyName(VDSPBatchStrategy strategy) noexcept {
    switch (strategy) {
        case VDSPBatchStrategy::directPersistent: return "direct-persistent";
        case VDSPBatchStrategy::directGcd: return "direct-gcd";
        case VDSPBatchStrategy::separablePersistent: return "separable-persistent";
        case VDSPBatchStrategy::separableGcd: return "separable-gcd";
    }
    return "unknown";
}

VDSPBatchStrategy vdspBatchStrategyNamed(std::string_view name) {
    if (name == "direct-persistent") return VDSPBatchStrategy::directPersistent;
    if (name == "direct-gcd") return VDSPBatchStrategy::directGcd;
    if (name == "separable-persistent") return VDSPBatchStrategy::separablePersistent;
    if (name == "separable-gcd") return VDSPBatchStrategy::separableGcd;
    throw std::invalid_argument("Unknown vDSP batch strategy: " + std::string(name));
}

class VDSPProvider::Impl {
public:
    Impl(const Workload& workload, std::size_t workers, VDSPTransformStrategy strategy,
         VDSPBatchStrategy batchStrategy)
        : workload_(workload), strategy_(strategy), batchStrategy_(batchStrategy) {
        if (workers == 0) throw std::invalid_argument("vDSP workers must be positive.");
#if SKBENCH_HAVE_ACCELERATE
        if (!powerOfTwo(workload.nx) || !powerOfTwo(workload.ny)) {
            capability_ = "unsupported: vDSP radix-2 requires power-of-two Nx and Ny";
            return;
        }
        if (separable() && strategy_ != VDSPTransformStrategy::inPlace) {
            capability_ = "unsupported: separable packed-real prototype currently supports only in-place placement";
            return;
        }
        const auto setupStart = Clock::now();
        if (usesGcd()) {
#if defined(__APPLE__)
            executor_ = std::make_unique<GcdPlaneExecutor>(workload.planes(), workers);
#else
            capability_ = "unsupported: Grand Central Dispatch is available only on Apple platforms";
            return;
#endif
        } else {
            executor_ = std::make_unique<PersistentPlaneExecutor>(workload.planes(), workers);
        }
        otherSetupSeconds_ = elapsedSeconds(setupStart);
        const auto nativeCount = workload.planes() * (workload.nx / 2) * workload.ny;
        const auto allocationStart = Clock::now();
        inputReal_.resize(nativeCount);
        inputImag_.resize(nativeCount);
        if (outOfPlace()) {
            outputReal_.resize(nativeCount);
            outputImag_.resize(nativeCount);
        }
        if (usesExplicitScratch() || separable()) {
            scratchElementsPerWorker_ = separable() ? workload.ny : std::max(workload.ny, workload.nx / 2);
            scratchReal_.resize(executor_->workerCount() * scratchElementsPerWorker_);
            scratchImag_.resize(executor_->workerCount() * scratchElementsPerWorker_);
        }
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
        capability_ = "supported: double split-complex radix-2; " +
            std::string(vdspTransformStrategyName(strategy_)) + "; " +
            std::string(vdspBatchStrategyName(batchStrategy_));
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

    void requireSeparable() const {
        requireSupported();
        if (!separable()) throw std::logic_error("vDSP row/column phase requested for a direct 2-D strategy");
    }

    std::size_t nativePlaneElements() const { return (workload_.nx / 2) * workload_.ny; }

    bool separable() const noexcept {
        return batchStrategy_ == VDSPBatchStrategy::separablePersistent ||
               batchStrategy_ == VDSPBatchStrategy::separableGcd;
    }

    bool usesGcd() const noexcept {
        return batchStrategy_ == VDSPBatchStrategy::directGcd ||
               batchStrategy_ == VDSPBatchStrategy::separableGcd;
    }

    bool outOfPlace() const noexcept {
        return strategy_ == VDSPTransformStrategy::outOfPlace ||
               strategy_ == VDSPTransformStrategy::outOfPlaceExplicitScratch;
    }

    bool usesExplicitScratch() const noexcept {
        return strategy_ == VDSPTransformStrategy::inPlaceExplicitScratch ||
               strategy_ == VDSPTransformStrategy::outOfPlaceExplicitScratch;
    }

    const double* resultReal() const noexcept { return outOfPlace() ? outputReal_.data() : inputReal_.data(); }
    const double* resultImag() const noexcept { return outOfPlace() ? outputImag_.data() : inputImag_.data(); }

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
                    inputReal_.data()[packed] = input[sourceOffset + y * workload_.nx + 2 * x];
                    inputImag_.data()[packed] = input[sourceOffset + y * workload_.nx + 2 * x + 1];
                }
            }
        }
    }

    void executeForward() {
        requireSupported();
#if SKBENCH_HAVE_ACCELERATE
        if (separable()) {
            executeForwardRows();
            executeForwardColumns();
        } else {
            executor_->run(this, &Impl::forwardDirectTask);
        }
#endif
    }

    void executeForwardRows() {
        requireSeparable();
#if SKBENCH_HAVE_ACCELERATE
        executor_->run(this, &Impl::forwardRowsTask);
#endif
    }

    void executeForwardColumns() {
        requireSeparable();
#if SKBENCH_HAVE_ACCELERATE
        executor_->run(this, &Impl::forwardColumnsTask);
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
                    output[wvmSpectrumIndex(workload_, x, y, z, field)] = {0.5 * resultReal()[packed], 0.5 * resultImag()[packed]};
                }
            }
            output[wvmSpectrumIndex(workload_, 0, 0, z, field)] = {0.5 * resultReal()[offset], 0.0};
            output[wvmSpectrumIndex(workload_, half, 0, z, field)] = {0.5 * resultImag()[offset], 0.0};
            output[wvmSpectrumIndex(workload_, 0, yNyquist, z, field)] = {0.5 * resultReal()[offset + half], 0.0};
            output[wvmSpectrumIndex(workload_, half, yNyquist, z, field)] = {0.5 * resultImag()[offset + half], 0.0};
            for (std::size_t y = 1; y < yNyquist; ++y) {
                const auto first = offset + (2 * y) * half;
                const auto second = offset + (2 * y + 1) * half;
                const Complex zero{0.5 * resultReal()[first], 0.5 * resultReal()[second]};
                const Complex nyquist{0.5 * resultImag()[first], 0.5 * resultImag()[second]};
                output[wvmSpectrumIndex(workload_, 0, y, z, field)] = zero;
                output[wvmSpectrumIndex(workload_, 0, workload_.ny - y, z, field)] = conjugate(zero);
                output[wvmSpectrumIndex(workload_, half, y, z, field)] = nyquist;
                output[wvmSpectrumIndex(workload_, half, workload_.ny - y, z, field)] = conjugate(nyquist);
            }
        }
    }

    void packInverse(const Complex* input) {
        requireSupported();
        std::fill(inputReal_.begin(), inputReal_.end(), 0.0);
        std::fill(inputImag_.begin(), inputImag_.end(), 0.0);
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
                    inputReal_.data()[packed] = value.real;
                    inputImag_.data()[packed] = value.imag;
                }
            }
            inputReal_.data()[offset] = input[wvmSpectrumIndex(workload_, 0, 0, z, field)].real;
            inputImag_.data()[offset] = input[wvmSpectrumIndex(workload_, half, 0, z, field)].real;
            inputReal_.data()[offset + half] = input[wvmSpectrumIndex(workload_, 0, yNyquist, z, field)].real;
            inputImag_.data()[offset + half] = input[wvmSpectrumIndex(workload_, half, yNyquist, z, field)].real;
            for (std::size_t y = 1; y < yNyquist; ++y) {
                const auto zero = input[wvmSpectrumIndex(workload_, 0, y, z, field)];
                const auto nyquist = input[wvmSpectrumIndex(workload_, half, y, z, field)];
                const auto first = offset + (2 * y) * half;
                const auto second = offset + (2 * y + 1) * half;
                inputReal_.data()[first] = zero.real;
                inputReal_.data()[second] = zero.imag;
                inputImag_.data()[first] = nyquist.real;
                inputImag_.data()[second] = nyquist.imag;
            }
        }
    }

    void executeInverse() {
        requireSupported();
#if SKBENCH_HAVE_ACCELERATE
        if (separable()) {
            executeInverseColumns();
            executeInverseRows();
        } else {
            executor_->run(this, &Impl::inverseDirectTask);
        }
#endif
    }

    void executeInverseColumns() {
        requireSeparable();
#if SKBENCH_HAVE_ACCELERATE
        executor_->run(this, &Impl::inverseColumnsTask);
#endif
    }

    void executeInverseRows() {
        requireSeparable();
#if SKBENCH_HAVE_ACCELERATE
        executor_->run(this, &Impl::inverseRowsTask);
#endif
    }

    void executeSchedulerNoop() {
        requireSupported();
#if SKBENCH_HAVE_ACCELERATE
        executor_->run(nullptr, &Impl::noopTask);
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
                    output[destinationOffset + y * workload_.nx + 2 * x] = resultReal()[packed];
                    output[destinationOffset + y * workload_.nx + 2 * x + 1] = resultImag()[packed];
                }
            }
        }
    }

#if SKBENCH_HAVE_ACCELERATE
    DSPDoubleSplitComplex scratch(std::size_t worker) {
        const auto offset = worker * scratchElementsPerWorker_;
        return {scratchReal_.data() + offset, scratchImag_.data() + offset};
    }

    void executePlane(std::size_t worker, std::size_t plane, FFTDirection direction) {
        const auto nativePlane = nativePlaneElements();
        const auto offset = plane * nativePlane;
        const auto half = workload_.nx / 2;
        DSPDoubleSplitComplex input{inputReal_.data() + offset, inputImag_.data() + offset};
        DSPDoubleSplitComplex output{
            outOfPlace() ? outputReal_.data() + offset : inputReal_.data() + offset,
            outOfPlace() ? outputImag_.data() + offset : inputImag_.data() + offset};
        const auto log2Nx = log2Length(workload_.nx);
        const auto log2Ny = log2Length(workload_.ny);
        switch (strategy_) {
            case VDSPTransformStrategy::inPlace:
                vDSP_fft2d_zripD(setups_[worker], &input, 1, static_cast<vDSP_Stride>(half),
                                 log2Nx, log2Ny, direction);
                break;
            case VDSPTransformStrategy::inPlaceExplicitScratch: {
                auto work = scratch(worker);
                vDSP_fft2d_zriptD(setups_[worker], &input, 1, static_cast<vDSP_Stride>(half), &work,
                                  log2Nx, log2Ny, direction);
                break;
            }
            case VDSPTransformStrategy::outOfPlace:
                vDSP_fft2d_zropD(setups_[worker], &input, 1, static_cast<vDSP_Stride>(half),
                                 &output, 1, static_cast<vDSP_Stride>(half), log2Nx, log2Ny, direction);
                break;
            case VDSPTransformStrategy::outOfPlaceExplicitScratch: {
                auto work = scratch(worker);
                vDSP_fft2d_zroptD(setups_[worker], &input, 1, static_cast<vDSP_Stride>(half),
                                  &output, 1, static_cast<vDSP_Stride>(half), &work, log2Nx, log2Ny, direction);
                break;
            }
        }
    }

    void executeSeparableRows(std::size_t worker, std::size_t plane, FFTDirection direction) {
        const auto half = workload_.nx / 2;
        const auto offset = plane * nativePlaneElements();
        const auto log2Nx = log2Length(workload_.nx);
        for (std::size_t y = 0; y < workload_.ny; ++y) {
            DSPDoubleSplitComplex row{inputReal_.data() + offset + y * half,
                                      inputImag_.data() + offset + y * half};
            vDSP_fft_zripD(setups_[worker], &row, 1, log2Nx, direction);
        }
    }

    void executeInteriorColumns(std::size_t worker, std::size_t plane, FFTDirection direction) {
        const auto half = workload_.nx / 2;
        const auto offset = plane * nativePlaneElements();
        const auto log2Ny = log2Length(workload_.ny);
        for (std::size_t x = 1; x < half; ++x) {
            DSPDoubleSplitComplex column{inputReal_.data() + offset + x,
                                         inputImag_.data() + offset + x};
            vDSP_fft_zipD(setups_[worker], &column, static_cast<vDSP_Stride>(half), log2Ny, direction);
        }
    }

    void executeForwardBoundaryColumn(std::size_t worker, std::size_t plane, bool xNyquist) {
        const auto half = workload_.nx / 2;
        const auto offset = plane * nativePlaneElements();
        auto work = scratch(worker);
        auto* target = xNyquist ? inputImag_.data() : inputReal_.data();
        for (std::size_t y = 0; y < workload_.ny; ++y) {
            work.realp[y] = target[offset + y * half];
            work.imagp[y] = 0.0;
        }
        vDSP_fft_zipD(setups_[worker], &work, 1, log2Length(workload_.ny), FFT_FORWARD);

        const auto yNyquist = workload_.ny / 2;
        target[offset] = work.realp[0];
        target[offset + half] = work.realp[yNyquist];
        for (std::size_t y = 1; y < yNyquist; ++y) {
            target[offset + (2 * y) * half] = work.realp[y];
            target[offset + (2 * y + 1) * half] = work.imagp[y];
        }
    }

    void executeInverseBoundaryColumn(std::size_t worker, std::size_t plane, bool xNyquist) {
        const auto half = workload_.nx / 2;
        const auto offset = plane * nativePlaneElements();
        const auto yNyquist = workload_.ny / 2;
        auto work = scratch(worker);
        auto* source = xNyquist ? inputImag_.data() : inputReal_.data();
        work.realp[0] = source[offset];
        work.imagp[0] = 0.0;
        work.realp[yNyquist] = source[offset + half];
        work.imagp[yNyquist] = 0.0;
        for (std::size_t y = 1; y < yNyquist; ++y) {
            const auto real = source[offset + (2 * y) * half];
            const auto imag = source[offset + (2 * y + 1) * half];
            work.realp[y] = real;
            work.imagp[y] = imag;
            work.realp[workload_.ny - y] = real;
            work.imagp[workload_.ny - y] = -imag;
        }
        vDSP_fft_zipD(setups_[worker], &work, 1, log2Length(workload_.ny), FFT_INVERSE);
        for (std::size_t y = 0; y < workload_.ny; ++y) {
            source[offset + y * half] = work.realp[y];
        }
    }

    static void forwardDirectTask(void* context, std::size_t worker, std::size_t begin, std::size_t end) {
        auto& self = *static_cast<Impl*>(context);
        for (std::size_t plane = begin; plane < end; ++plane) {
            self.executePlane(worker, plane, FFT_FORWARD);
        }
    }

    static void inverseDirectTask(void* context, std::size_t worker, std::size_t begin, std::size_t end) {
        auto& self = *static_cast<Impl*>(context);
        for (std::size_t plane = begin; plane < end; ++plane) {
            self.executePlane(worker, plane, FFT_INVERSE);
        }
    }

    static void forwardRowsTask(void* context, std::size_t worker, std::size_t begin, std::size_t end) {
        auto& self = *static_cast<Impl*>(context);
        for (std::size_t plane = begin; plane < end; ++plane) {
            self.executeSeparableRows(worker, plane, FFT_FORWARD);
        }
    }

    static void forwardColumnsTask(void* context, std::size_t worker, std::size_t begin, std::size_t end) {
        auto& self = *static_cast<Impl*>(context);
        for (std::size_t plane = begin; plane < end; ++plane) {
            self.executeInteriorColumns(worker, plane, FFT_FORWARD);
            self.executeForwardBoundaryColumn(worker, plane, false);
            self.executeForwardBoundaryColumn(worker, plane, true);
        }
    }

    static void inverseColumnsTask(void* context, std::size_t worker, std::size_t begin, std::size_t end) {
        auto& self = *static_cast<Impl*>(context);
        for (std::size_t plane = begin; plane < end; ++plane) {
            self.executeInteriorColumns(worker, plane, FFT_INVERSE);
            self.executeInverseBoundaryColumn(worker, plane, false);
            self.executeInverseBoundaryColumn(worker, plane, true);
        }
    }

    static void inverseRowsTask(void* context, std::size_t worker, std::size_t begin, std::size_t end) {
        auto& self = *static_cast<Impl*>(context);
        for (std::size_t plane = begin; plane < end; ++plane) {
            self.executeSeparableRows(worker, plane, FFT_INVERSE);
        }
    }

    static void noopTask(void*, std::size_t, std::size_t, std::size_t) {}
#endif

    Workload workload_;
    VDSPTransformStrategy strategy_ = VDSPTransformStrategy::inPlace;
    VDSPBatchStrategy batchStrategy_ = VDSPBatchStrategy::directPersistent;
    bool supported_ = false;
    std::string capability_;
    double otherSetupSeconds_ = 0.0;
    double allocationSeconds_ = 0.0;
    double planningSeconds_ = 0.0;
    std::size_t scratchElementsPerWorker_ = 0;
    std::unique_ptr<BatchExecutor> executor_;
    AlignedBuffer inputReal_;
    AlignedBuffer inputImag_;
    AlignedBuffer outputReal_;
    AlignedBuffer outputImag_;
    AlignedBuffer scratchReal_;
    AlignedBuffer scratchImag_;
#if SKBENCH_HAVE_ACCELERATE
    std::vector<FFTSetupD> setups_;
#endif
};

VDSPProvider::VDSPProvider(const Workload& workload, std::size_t workers, VDSPTransformStrategy strategy,
                           VDSPBatchStrategy batchStrategy)
    : impl_(std::make_unique<Impl>(workload, workers, strategy, batchStrategy)) {}

VDSPProvider::~VDSPProvider() = default;
VDSPProvider::VDSPProvider(VDSPProvider&&) noexcept = default;
VDSPProvider& VDSPProvider::operator=(VDSPProvider&&) noexcept = default;

bool VDSPProvider::supported() const noexcept { return impl_->supported_; }
std::string VDSPProvider::capability() const { return impl_->capability_; }
void VDSPProvider::packForwardInput(const double* input) { impl_->packForward(input); }
void VDSPProvider::executeForwardNative() { impl_->executeForward(); }
void VDSPProvider::executeForwardRowsNative() { impl_->executeForwardRows(); }
void VDSPProvider::executeForwardColumnsNative() { impl_->executeForwardColumns(); }
void VDSPProvider::unpackForwardOutput(Complex* output) const { impl_->unpackForward(output); }
void VDSPProvider::packInverseInput(const Complex* input) { impl_->packInverse(input); }
void VDSPProvider::executeInverseNative() { impl_->executeInverse(); }
void VDSPProvider::executeInverseColumnsNative() { impl_->executeInverseColumns(); }
void VDSPProvider::executeInverseRowsNative() { impl_->executeInverseRows(); }
void VDSPProvider::executeSchedulerNoop() { impl_->executeSchedulerNoop(); }
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

VDSPTransformStrategy VDSPProvider::strategy() const noexcept { return impl_->strategy_; }

VDSPBatchStrategy VDSPProvider::batchStrategy() const noexcept { return impl_->batchStrategy_; }

bool VDSPProvider::separable() const noexcept { return impl_->separable(); }

std::size_t VDSPProvider::workers() const noexcept {
    return impl_->executor_ == nullptr ? 0 : impl_->executor_->workerCount();
}

std::size_t VDSPProvider::nativeOperandBytes() const noexcept {
    return (impl_->inputReal_.size() + impl_->inputImag_.size()) * sizeof(double);
}

std::size_t VDSPProvider::nativeBufferBytes() const noexcept {
    return (impl_->inputReal_.size() + impl_->inputImag_.size() +
            impl_->outputReal_.size() + impl_->outputImag_.size()) * sizeof(double);
}

std::size_t VDSPProvider::explicitPersistentBytes() const noexcept {
    return nativeBufferBytes();
}

std::size_t VDSPProvider::scratchBytes() const noexcept {
    return (impl_->scratchReal_.size() + impl_->scratchImag_.size()) * sizeof(double);
}

std::size_t VDSPProvider::minimumAlignmentBytes() const noexcept { return AlignedBuffer::alignment; }

std::string VDSPProvider::libraryIdentity() const {
#if SKBENCH_HAVE_ACCELERATE
    return libraryContaining(reinterpret_cast<const void*>(&vDSP_fft2d_zripD));
#else
    return {};
#endif
}

} // namespace skbench
