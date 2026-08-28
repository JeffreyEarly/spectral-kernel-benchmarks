#include "skbench/skbench.hpp"

#include <fftw3.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstddef>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

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

class PersistentIndexExecutor {
public:
    using Task = void (*)(void*, std::size_t);

    explicit PersistentIndexExecutor(std::size_t workers) : workers_(workers) {
        threads_.reserve(workers_ - 1);
        for (std::size_t worker = 1; worker < workers_; ++worker) {
            threads_.emplace_back([this, worker] { workerLoop(worker); });
        }
    }

    ~PersistentIndexExecutor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        ready_.notify_all();
        for (auto& thread : threads_) thread.join();
    }

    PersistentIndexExecutor(const PersistentIndexExecutor&) = delete;
    PersistentIndexExecutor& operator=(const PersistentIndexExecutor&) = delete;

    void run(Task task, void* context) {
        if (workers_ == 1) {
            task(context, 0);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = task;
            context_ = context;
            remaining_ = workers_ - 1;
            ++generation_;
        }
        ready_.notify_all();
        task(context, 0);
        std::unique_lock<std::mutex> lock(mutex_);
        complete_.wait(lock, [this] { return remaining_ == 0; });
    }

private:
    void workerLoop(std::size_t worker) {
        std::size_t observedGeneration = 0;
        while (true) {
            Task task = nullptr;
            void* context = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this, observedGeneration] {
                    return stopping_ || generation_ != observedGeneration;
                });
                if (stopping_) return;
                observedGeneration = generation_;
                task = task_;
                context = context_;
            }
            task(context, worker);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --remaining_;
                if (remaining_ == 0) complete_.notify_one();
            }
        }
    }

    std::size_t workers_ = 1;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable complete_;
    Task task_ = nullptr;
    void* context_ = nullptr;
    std::size_t generation_ = 0;
    std::size_t remaining_ = 0;
    bool stopping_ = false;
};

} // namespace

class FFTWPrunedProvider::Impl {
public:
    struct PlanSet {
        fftw_plan rowForward = nullptr;
        fftw_plan columnForward = nullptr;
        fftw_plan columnInverse = nullptr;
        fftw_plan rowInverse = nullptr;
        std::size_t beginPlane = 0;
        std::size_t planeCount = 0;
    };

    Impl(const Workload& workload, const std::vector<RetainedMode>& modes,
         FFTWPlanningMode planningMode, std::size_t internalWorkers,
         std::size_t outerWorkers)
        : workload_(workload), modes_(modes), planningMode_(planningMode),
          internalWorkers_(internalWorkers), outerWorkers_(outerWorkers) {
        static_assert(sizeof(Complex) == sizeof(fftw_complex));
        if (modes_.empty()) throw std::invalid_argument("The pruned FFTW provider requires retained modes.");
        if (internalWorkers_ == 0 || internalWorkers_ > static_cast<std::size_t>(INT_MAX)) {
            throw std::invalid_argument("Pruned FFTW internal workers must lie in [1, INT_MAX].");
        }
        if (outerWorkers_ == 0 || outerWorkers_ > workload_.planes()) {
            throw std::invalid_argument("Pruned FFTW outer workers must lie in [1, plane count].");
        }
        if (internalWorkers_ > std::numeric_limits<std::size_t>::max() / outerWorkers_) {
            throw std::invalid_argument("Pruned FFTW logical worker count overflows size_t.");
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

        const auto executorStart = Clock::now();
        executor_ = std::make_unique<PersistentIndexExecutor>(outerWorkers_);
        otherSetupSeconds_ += elapsedSeconds(executorStart);
    }

    ~Impl() {
        executor_.reset();
        std::lock_guard<std::mutex> lock(prunedPlanningMutex);
        destroyPlans();
        releaseStorage();
    }

    void executeForwardRows(const double* input) {
        ExecuteContext context{this, input, nullptr, nullptr, nullptr};
        executor_->run(&forwardRowsShard, &context);
    }

    void executeForwardColumns() {
        ExecuteContext context{this, nullptr, nullptr, nullptr, nullptr};
        executor_->run(&forwardColumnsShard, &context);
    }

    void gatherForward(Complex* retainedSpectrum) {
        ExecuteContext context{this, nullptr, nullptr, nullptr, retainedSpectrum};
        executor_->run(&gatherForwardShard, &context);
    }

    void gatherForwardSplit(double* retainedReal, double* retainedImag,
                            double scale) {
        SplitRetainedContext context{
            this, retainedReal, retainedImag, nullptr, nullptr, scale};
        executor_->run(&gatherForwardSplitShard, &context);
    }

    void forward(const double* input, Complex* retainedSpectrum) {
        executeForwardRows(input);
        executeForwardColumns();
        gatherForward(retainedSpectrum);
    }

    void forwardSplit(const double* input, double* retainedReal,
                      double* retainedImag, double scale) {
        executeForwardRows(input);
        executeForwardColumns();
        gatherForwardSplit(retainedReal, retainedImag, scale);
    }

    void embedInverse(const Complex* retainedSpectrum) {
        ExecuteContext context{this, nullptr, nullptr, retainedSpectrum, nullptr};
        executor_->run(&embedInverseShard, &context);
    }

    void embedInverseSplit(const double* retainedReal,
                           const double* retainedImag) {
        SplitRetainedContext context{
            this, nullptr, nullptr, retainedReal, retainedImag, 1.0};
        executor_->run(&embedInverseSplitShard, &context);
    }

    void executeInverseColumns() {
        ExecuteContext context{this, nullptr, nullptr, nullptr, nullptr};
        executor_->run(&inverseColumnsShard, &context);
    }

    void executeInverseRows(double* output) {
        ExecuteContext context{this, nullptr, output, nullptr, nullptr};
        executor_->run(&inverseRowsShard, &context);
    }

    void inverse(const Complex* retainedSpectrum, double* output) {
        embedInverse(retainedSpectrum);
        executeInverseColumns();
        executeInverseRows(output);
    }

    void inverseSplit(const double* retainedReal, const double* retainedImag,
                      double* output) {
        embedInverseSplit(retainedReal, retainedImag);
        executeInverseColumns();
        executeInverseRows(output);
    }

    void executeSchedulerNoop() {
        executor_->run(&noopShard, nullptr);
    }

    Workload workload_;
    std::vector<RetainedMode> modes_;
    FFTWPlanningMode planningMode_ = FFTWPlanningMode::measure;
    std::size_t internalWorkers_ = 1;
    std::size_t outerWorkers_ = 1;
    std::size_t activeKxCount_ = 0;
    double* realSurrogate_ = nullptr;
    Complex* intermediate_ = nullptr;
    std::vector<PlanSet> plans_;
    std::unique_ptr<PersistentIndexExecutor> executor_;
    double otherSetupSeconds_ = 0.0;
    double allocationSeconds_ = 0.0;
    double planningSeconds_ = 0.0;
    std::size_t planningBytes_ = 0;

private:
    struct ExecuteContext {
        Impl* provider;
        const double* realInput;
        double* realOutput;
        const Complex* retainedInput;
        Complex* retainedOutput;
    };

    struct SplitRetainedContext {
        Impl* provider;
        double* retainedRealOutput;
        double* retainedImagOutput;
        const double* retainedRealInput;
        const double* retainedImagInput;
        double scale;
    };

    void createPlans(unsigned flags) {
        const auto nx = static_cast<ptrdiff_t>(workload_.nx);
        const auto ny = static_cast<ptrdiff_t>(workload_.ny);
        const auto nxHalf = static_cast<ptrdiff_t>(workload_.nxHalf());
        const auto realPlane = static_cast<ptrdiff_t>(workload_.realPlaneElements());
        const auto spectrumPlane = static_cast<ptrdiff_t>(workload_.halfRows());

        plans_.reserve(outerWorkers_);
        for (std::size_t shardIndex = 0; shardIndex < outerWorkers_; ++shardIndex) {
            PlanSet shard;
            shard.beginPlane = workload_.planes() * shardIndex / outerWorkers_;
            const auto endPlane = workload_.planes() * (shardIndex + 1) / outerWorkers_;
            shard.planeCount = endPlane - shard.beginPlane;
            const auto shardPlanes = static_cast<ptrdiff_t>(shard.planeCount);
            auto* real = realSurrogate_ + shard.beginPlane * workload_.realPlaneElements();
            auto* scratch = reinterpret_cast<fftw_complex*>(
                intermediate_ + shard.beginPlane * workload_.halfRows());

            fftw_iodim64 rowForwardDimension[1] = {{nx, 1, 1}};
            fftw_iodim64 rowForwardBatches[2] = {
                {ny, nx, nxHalf},
                {shardPlanes, realPlane, spectrumPlane}};
            shard.rowForward = fftw_plan_guru64_dft_r2c(
                1, rowForwardDimension, 2, rowForwardBatches, real, scratch, flags);

            fftw_iodim64 columnDimension[1] = {{ny, nxHalf, nxHalf}};
            fftw_iodim64 columnBatches[2] = {
                {static_cast<ptrdiff_t>(activeKxCount_), 1, 1},
                {shardPlanes, spectrumPlane, spectrumPlane}};
            shard.columnForward = fftw_plan_guru64_dft(
                1, columnDimension, 2, columnBatches, scratch, scratch, FFTW_FORWARD, flags);
            shard.columnInverse = fftw_plan_guru64_dft(
                1, columnDimension, 2, columnBatches, scratch, scratch, FFTW_BACKWARD, flags);

            fftw_iodim64 rowInverseDimension[1] = {{nx, 1, 1}};
            fftw_iodim64 rowInverseBatches[2] = {
                {ny, nxHalf, nx},
                {shardPlanes, spectrumPlane, realPlane}};
            shard.rowInverse = fftw_plan_guru64_dft_c2r(
                1, rowInverseDimension, 2, rowInverseBatches, scratch, real, flags);

            if (shard.rowForward == nullptr || shard.columnForward == nullptr ||
                shard.columnInverse == nullptr || shard.rowInverse == nullptr) {
                destroyPlan(shard);
                destroyPlans();
                throw std::runtime_error(
                    "FFTW could not create the outer-sharded partially pruned plans.");
            }
            plans_.push_back(shard);
        }
    }

    static void destroyPlan(PlanSet& plan) {
        if (plan.rowForward != nullptr) fftw_destroy_plan(plan.rowForward);
        if (plan.columnForward != nullptr) fftw_destroy_plan(plan.columnForward);
        if (plan.columnInverse != nullptr) fftw_destroy_plan(plan.columnInverse);
        if (plan.rowInverse != nullptr) fftw_destroy_plan(plan.rowInverse);
        plan.rowForward = nullptr;
        plan.columnForward = nullptr;
        plan.columnInverse = nullptr;
        plan.rowInverse = nullptr;
    }

    void destroyPlans() {
        for (auto& plan : plans_) destroyPlan(plan);
        plans_.clear();
    }

    static void forwardRowsShard(void* rawContext, std::size_t shardIndex) {
        auto& context = *static_cast<ExecuteContext*>(rawContext);
        auto& provider = *context.provider;
        const auto& shard = provider.plans_[shardIndex];
        auto* input = const_cast<double*>(context.realInput) +
            shard.beginPlane * provider.workload_.realPlaneElements();
        auto* scratch = reinterpret_cast<fftw_complex*>(
            provider.intermediate_ + shard.beginPlane * provider.workload_.halfRows());
        fftw_execute_dft_r2c(shard.rowForward, input, scratch);
    }

    static void forwardColumnsShard(void* rawContext, std::size_t shardIndex) {
        auto& provider = *static_cast<ExecuteContext*>(rawContext)->provider;
        const auto& shard = provider.plans_[shardIndex];
        auto* scratch = reinterpret_cast<fftw_complex*>(
            provider.intermediate_ + shard.beginPlane * provider.workload_.halfRows());
        fftw_execute_dft(shard.columnForward, scratch, scratch);
    }

    static void gatherForwardShard(void* rawContext, std::size_t shardIndex) {
        auto& context = *static_cast<ExecuteContext*>(rawContext);
        const auto& provider = *context.provider;
        const auto& shard = provider.plans_[shardIndex];
        const auto planes = provider.workload_.planes();
        const auto spectrumPlane = provider.workload_.halfRows();
        for (std::size_t modeIndex = 0; modeIndex < provider.modes_.size(); ++modeIndex) {
            const auto& mode = provider.modes_[modeIndex];
            const auto frequency = mode.storedKx + provider.workload_.nxHalf() * mode.storedKy;
            for (std::size_t plane = shard.beginPlane;
                 plane < shard.beginPlane + shard.planeCount; ++plane) {
                auto value = provider.intermediate_[plane * spectrumPlane + frequency];
                if (mode.conjugatesStoredValue) value = conjugate(value);
                context.retainedOutput[plane + planes * modeIndex] = value;
            }
        }
    }

    static void gatherForwardSplitShard(void* rawContext,
                                        std::size_t shardIndex) {
        auto& context = *static_cast<SplitRetainedContext*>(rawContext);
        const auto& provider = *context.provider;
        const auto& shard = provider.plans_[shardIndex];
        const auto planes = provider.workload_.planes();
        const auto spectrumPlane = provider.workload_.halfRows();
        for (std::size_t modeIndex = 0; modeIndex < provider.modes_.size(); ++modeIndex) {
            const auto& mode = provider.modes_[modeIndex];
            const auto frequency = mode.storedKx +
                provider.workload_.nxHalf() * mode.storedKy;
            for (std::size_t plane = shard.beginPlane;
                 plane < shard.beginPlane + shard.planeCount; ++plane) {
                auto value = provider.intermediate_[
                    plane * spectrumPlane + frequency];
                if (mode.conjugatesStoredValue) value = conjugate(value);
                const auto retained = plane + planes * modeIndex;
                context.retainedRealOutput[retained] = context.scale * value.real;
                context.retainedImagOutput[retained] = context.scale * value.imag;
            }
        }
    }

    static void embedInverseShard(void* rawContext, std::size_t shardIndex) {
        auto& context = *static_cast<ExecuteContext*>(rawContext);
        auto& provider = *context.provider;
        const auto& shard = provider.plans_[shardIndex];
        const auto planes = provider.workload_.planes();
        const auto nxHalf = provider.workload_.nxHalf();
        const auto spectrumPlane = provider.workload_.halfRows();
        for (std::size_t plane = shard.beginPlane;
             plane < shard.beginPlane + shard.planeCount; ++plane) {
            auto* scratch = provider.intermediate_ + plane * spectrumPlane;
            std::fill_n(scratch, spectrumPlane, Complex{});
            for (std::size_t modeIndex = 0; modeIndex < provider.modes_.size(); ++modeIndex) {
                const auto& mode = provider.modes_[modeIndex];
                const auto compact = context.retainedInput[plane + planes * modeIndex];
                const auto stored = mode.conjugatesStoredValue ? conjugate(compact) : compact;
                scratch[mode.storedKx + nxHalf * mode.storedKy] = stored;
                if (mode.storedKx == 0 && mode.storedKy != 0 &&
                    2 * mode.storedKy != provider.workload_.ny) {
                    const auto conjugateKy =
                        (provider.workload_.ny - mode.storedKy) % provider.workload_.ny;
                    scratch[nxHalf * conjugateKy] = conjugate(stored);
                }
            }
        }
    }

    static void embedInverseSplitShard(void* rawContext,
                                       std::size_t shardIndex) {
        auto& context = *static_cast<SplitRetainedContext*>(rawContext);
        auto& provider = *context.provider;
        const auto& shard = provider.plans_[shardIndex];
        const auto planes = provider.workload_.planes();
        const auto nxHalf = provider.workload_.nxHalf();
        const auto spectrumPlane = provider.workload_.halfRows();
        for (std::size_t plane = shard.beginPlane;
             plane < shard.beginPlane + shard.planeCount; ++plane) {
            auto* scratch = provider.intermediate_ + plane * spectrumPlane;
            std::fill_n(scratch, spectrumPlane, Complex{});
            for (std::size_t modeIndex = 0; modeIndex < provider.modes_.size(); ++modeIndex) {
                const auto& mode = provider.modes_[modeIndex];
                const auto retained = plane + planes * modeIndex;
                Complex stored{
                    context.retainedRealInput[retained],
                    context.retainedImagInput[retained]};
                if (mode.conjugatesStoredValue) stored = conjugate(stored);
                scratch[mode.storedKx + nxHalf * mode.storedKy] = stored;
                if (mode.storedKx == 0 && mode.storedKy != 0 &&
                    2 * mode.storedKy != provider.workload_.ny) {
                    const auto conjugateKy =
                        (provider.workload_.ny - mode.storedKy) %
                        provider.workload_.ny;
                    scratch[nxHalf * conjugateKy] = conjugate(stored);
                }
            }
        }
    }

    static void inverseColumnsShard(void* rawContext, std::size_t shardIndex) {
        auto& provider = *static_cast<ExecuteContext*>(rawContext)->provider;
        const auto& shard = provider.plans_[shardIndex];
        auto* scratch = reinterpret_cast<fftw_complex*>(
            provider.intermediate_ + shard.beginPlane * provider.workload_.halfRows());
        fftw_execute_dft(shard.columnInverse, scratch, scratch);
    }

    static void inverseRowsShard(void* rawContext, std::size_t shardIndex) {
        auto& context = *static_cast<ExecuteContext*>(rawContext);
        auto& provider = *context.provider;
        const auto& shard = provider.plans_[shardIndex];
        auto* scratch = reinterpret_cast<fftw_complex*>(
            provider.intermediate_ + shard.beginPlane * provider.workload_.halfRows());
        auto* output = context.realOutput +
            shard.beginPlane * provider.workload_.realPlaneElements();
        fftw_execute_dft_c2r(shard.rowInverse, scratch, output);
    }

    static void noopShard(void*, std::size_t) {}

    void releaseStorage() {
        if (realSurrogate_ != nullptr) fftw_free(realSurrogate_);
        if (intermediate_ != nullptr) fftw_free(intermediate_);
        realSurrogate_ = nullptr;
        intermediate_ = nullptr;
    }
};

FFTWPrunedProvider::FFTWPrunedProvider(
    const Workload& workload, const std::vector<RetainedMode>& modes,
    FFTWPlanningMode planningMode, std::size_t internalWorkers,
    std::size_t outerWorkers)
    : impl_(std::make_unique<Impl>(
          workload, modes, planningMode, internalWorkers, outerWorkers)) {}

FFTWPrunedProvider::~FFTWPrunedProvider() = default;
FFTWPrunedProvider::FFTWPrunedProvider(FFTWPrunedProvider&&) noexcept = default;
FFTWPrunedProvider& FFTWPrunedProvider::operator=(FFTWPrunedProvider&&) noexcept = default;

void FFTWPrunedProvider::executeForwardRows(const double* input) { impl_->executeForwardRows(input); }
void FFTWPrunedProvider::executeForwardColumns() { impl_->executeForwardColumns(); }
void FFTWPrunedProvider::gatherForward(Complex* retainedSpectrum) { impl_->gatherForward(retainedSpectrum); }
void FFTWPrunedProvider::gatherForwardSplit(
    double* retainedReal, double* retainedImag, double scale) {
    impl_->gatherForwardSplit(retainedReal, retainedImag, scale);
}
void FFTWPrunedProvider::forward(const double* input, Complex* retainedSpectrum) {
    impl_->forward(input, retainedSpectrum);
}
void FFTWPrunedProvider::forwardSplit(
    const double* input, double* retainedReal, double* retainedImag,
    double scale) {
    impl_->forwardSplit(input, retainedReal, retainedImag, scale);
}
void FFTWPrunedProvider::embedInverse(const Complex* retainedSpectrum) { impl_->embedInverse(retainedSpectrum); }
void FFTWPrunedProvider::embedInverseSplit(
    const double* retainedReal, const double* retainedImag) {
    impl_->embedInverseSplit(retainedReal, retainedImag);
}
void FFTWPrunedProvider::executeInverseColumns() { impl_->executeInverseColumns(); }
void FFTWPrunedProvider::executeInverseRows(double* output) { impl_->executeInverseRows(output); }
void FFTWPrunedProvider::inverse(const Complex* retainedSpectrum, double* output) {
    impl_->inverse(retainedSpectrum, output);
}
void FFTWPrunedProvider::inverseSplit(
    const double* retainedReal, const double* retainedImag, double* output) {
    impl_->inverseSplit(retainedReal, retainedImag, output);
}
void FFTWPrunedProvider::executeSchedulerNoop() { impl_->executeSchedulerNoop(); }

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
std::size_t FFTWPrunedProvider::outerWorkers() const noexcept { return impl_->outerWorkers_; }
std::size_t FFTWPrunedProvider::totalLogicalWorkers() const noexcept {
    return impl_->internalWorkers_ * impl_->outerWorkers_;
}
std::size_t FFTWPrunedProvider::maximumShardScratchBytes() const noexcept {
    std::size_t maximumPlanes = 0;
    for (const auto& plan : impl_->plans_) maximumPlanes = std::max(maximumPlanes, plan.planeCount);
    return maximumPlanes * impl_->workload_.halfRows() * sizeof(Complex);
}
double FFTWPrunedProvider::otherSetupSeconds() const noexcept { return impl_->otherSetupSeconds_; }
double FFTWPrunedProvider::allocationSeconds() const noexcept { return impl_->allocationSeconds_; }
double FFTWPrunedProvider::planningSeconds() const noexcept { return impl_->planningSeconds_; }
FFTWPlanningMode FFTWPrunedProvider::planningMode() const noexcept { return impl_->planningMode_; }
bool FFTWPrunedProvider::completeHalfSpectrumOutputMaterialized() const noexcept { return false; }
bool FFTWPrunedProvider::inPlaceRetainedOperatorSupported() const noexcept { return false; }
std::string FFTWPrunedProvider::inPlaceRetainedOperatorCapability() const {
    return "unsupported: the logical retained output is disjoint from the real input, while each outer shard executes selected complex column transforms in-place inside its disjoint reusable slice of full-sized plane-major row-spectrum scratch";
}
std::string FFTWPrunedProvider::libraryIdentity() const {
    return libraryContaining(reinterpret_cast<const void*>(&fftw_execute));
}
std::string FFTWPrunedProvider::version() const { return fftw_version; }

} // namespace skbench
