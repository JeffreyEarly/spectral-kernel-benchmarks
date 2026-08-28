#include "skbench/skbench.hpp"

#include <fftw3.h>

#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstring>
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

unsigned plannerFlags(const FFTWStrategy& strategy) {
    unsigned flags = 0;
    switch (strategy.planningMode) {
        case FFTWPlanningMode::estimate: flags = FFTW_ESTIMATE; break;
        case FFTWPlanningMode::measure: flags = FFTW_MEASURE; break;
        case FFTWPlanningMode::patient: flags = FFTW_PATIENT; break;
        case FFTWPlanningMode::exhaustive: flags = FFTW_EXHAUSTIVE; break;
    }
    if (strategy.alignment == FFTWAlignmentStrategy::unaligned) flags |= FFTW_UNALIGNED;
    return flags;
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

std::string_view fftwPlanningModeName(FFTWPlanningMode mode) noexcept {
    switch (mode) {
        case FFTWPlanningMode::estimate: return "estimate";
        case FFTWPlanningMode::measure: return "measure";
        case FFTWPlanningMode::patient: return "patient";
        case FFTWPlanningMode::exhaustive: return "exhaustive";
    }
    return "unknown";
}

FFTWPlanningMode fftwPlanningModeNamed(std::string_view name) {
    if (name == "estimate") return FFTWPlanningMode::estimate;
    if (name == "measure") return FFTWPlanningMode::measure;
    if (name == "patient") return FFTWPlanningMode::patient;
    if (name == "exhaustive") return FFTWPlanningMode::exhaustive;
    throw std::invalid_argument("Unknown FFTW planning mode: " + std::string(name));
}

std::string_view fftwAlignmentStrategyName(FFTWAlignmentStrategy strategy) noexcept {
    switch (strategy) {
        case FFTWAlignmentStrategy::aligned: return "aligned";
        case FFTWAlignmentStrategy::unaligned: return "unaligned";
    }
    return "unknown";
}

FFTWAlignmentStrategy fftwAlignmentStrategyNamed(std::string_view name) {
    if (name == "aligned") return FFTWAlignmentStrategy::aligned;
    if (name == "unaligned") return FFTWAlignmentStrategy::unaligned;
    throw std::invalid_argument("Unknown FFTW alignment strategy: " + std::string(name));
}

std::string_view fftwWisdomStrategyName(FFTWWisdomStrategy strategy) noexcept {
    switch (strategy) {
        case FFTWWisdomStrategy::cold: return "cold";
        case FFTWWisdomStrategy::generatedImport: return "generated-import";
    }
    return "unknown";
}

FFTWWisdomStrategy fftwWisdomStrategyNamed(std::string_view name) {
    if (name == "cold") return FFTWWisdomStrategy::cold;
    if (name == "generated-import") return FFTWWisdomStrategy::generatedImport;
    throw std::invalid_argument("Unknown FFTW wisdom strategy: " + std::string(name));
}

std::string_view fftwDataLayoutName(FFTWDataLayout layout) noexcept {
    switch (layout) {
        case FFTWDataLayout::interleaved: return "interleaved";
        case FFTWDataLayout::split: return "split";
    }
    return "unknown";
}

FFTWDataLayout fftwDataLayoutNamed(std::string_view name) {
    if (name == "interleaved") return FFTWDataLayout::interleaved;
    if (name == "split") return FFTWDataLayout::split;
    throw std::invalid_argument("Unknown FFTW data layout: " + std::string(name));
}

class FFTWProvider::Impl {
public:
    struct PlanPair {
        fftw_plan forward = nullptr;
        fftw_plan inverse = nullptr;
        std::size_t beginPlane = 0;
        std::size_t planeCount = 0;
        int realAlignmentClass = 0;
        int spectrumAlignmentClass = 0;
        int spectrumRealAlignmentClass = 0;
        int spectrumImagAlignmentClass = 0;
    };

    Impl(const Workload& workload, FFTWStrategy strategy) : workload_(workload), strategy_(strategy) {
        static_assert(sizeof(Complex) == sizeof(fftw_complex));
        validateStrategy();

        const auto setupStart = Clock::now();
        {
            std::lock_guard<std::mutex> lock(planningMutex);
            if (fftw_init_threads() == 0) throw std::runtime_error("fftw_init_threads failed.");
        }
        otherSetupSeconds_ = elapsedSeconds(setupStart);

        const auto realBytes = workload_.realElements() * sizeof(double);
        const auto spectrumComponentBytes = workload_.spectrumElements() * sizeof(double);
        planningBytes_ = realBytes + 2 * spectrumComponentBytes;
        const auto allocationStart = Clock::now();
        realSurrogate_ = static_cast<double*>(fftw_malloc(realBytes));
        if (strategy_.layout == FFTWDataLayout::interleaved) {
            spectrumSurrogate_ = static_cast<fftw_complex*>(fftw_malloc(2 * spectrumComponentBytes));
        } else {
            spectrumRealSurrogate_ = static_cast<double*>(fftw_malloc(spectrumComponentBytes));
            spectrumImagSurrogate_ = static_cast<double*>(fftw_malloc(spectrumComponentBytes));
        }
        const bool spectrumAllocationFailed = strategy_.layout == FFTWDataLayout::interleaved
            ? spectrumSurrogate_ == nullptr
            : spectrumRealSurrogate_ == nullptr || spectrumImagSurrogate_ == nullptr;
        if (realSurrogate_ == nullptr || spectrumAllocationFailed) {
            releaseSurrogates();
            throw std::bad_alloc();
        }
        allocationSeconds_ = elapsedSeconds(allocationStart);

        try {
            std::lock_guard<std::mutex> lock(planningMutex);
            fftw_forget_wisdom();
            const auto flags = plannerFlags(strategy_);
            if (strategy_.wisdom == FFTWWisdomStrategy::generatedImport) {
                const auto generationStart = Clock::now();
                auto generatedPlans = createPlans(flags, true);
                char* exported = fftw_export_wisdom_to_string();
                if (exported == nullptr) {
                    destroyPlans(generatedPlans);
                    throw std::runtime_error("FFTW could not export generated wisdom.");
                }
                wisdom_.assign(exported);
                fftw_free(exported);
                wisdomBytes_ = wisdom_.size();
                destroyPlans(generatedPlans);
                wisdomGenerationSeconds_ = elapsedSeconds(generationStart);

                fftw_forget_wisdom();
                const auto importStart = Clock::now();
                if (fftw_import_wisdom_from_string(wisdom_.c_str()) == 0) {
                    throw std::runtime_error("FFTW could not import the generated wisdom.");
                }
                wisdomImportSeconds_ = elapsedSeconds(importStart);
                const auto planningStart = Clock::now();
                plans_ = createPlans(flags | FFTW_WISDOM_ONLY, false);
                planningSeconds_ = elapsedSeconds(planningStart);
            } else {
                const auto planningStart = Clock::now();
                plans_ = createPlans(flags, true);
                planningSeconds_ = elapsedSeconds(planningStart);
            }
            fftw_set_timelimit(FFTW_NO_TIMELIMIT);
            fftw_forget_wisdom();
            std::string{}.swap(wisdom_);
        } catch (...) {
            fftw_set_timelimit(FFTW_NO_TIMELIMIT);
            fftw_forget_wisdom();
            releaseSurrogates();
            throw;
        }
        releaseSurrogates();

        const auto executorStart = Clock::now();
        executor_ = std::make_unique<PersistentIndexExecutor>(strategy_.outerWorkers);
        otherSetupSeconds_ += elapsedSeconds(executorStart);
    }

    ~Impl() {
        executor_.reset();
        std::lock_guard<std::mutex> lock(planningMutex);
        destroyPlans(plans_);
        releaseSurrogates();
    }

    void forward(const double* input, Complex* output) {
        requireLayout(FFTWDataLayout::interleaved);
        validateForwardAlignment(input, output);
        ExecuteContext context{this, input, output, nullptr, nullptr, nullptr, nullptr, true};
        executor_->run(&executeShard, &context);
    }

    void inverse(Complex* input, double* output) {
        requireLayout(FFTWDataLayout::interleaved);
        validateInverseAlignment(input, output);
        ExecuteContext context{this, nullptr, nullptr, input, output, nullptr, nullptr, false};
        executor_->run(&executeShard, &context);
    }

    void forwardSplit(const double* input, double* outputReal, double* outputImag) {
        requireLayout(FFTWDataLayout::split);
        validateForwardSplitAlignment(input, outputReal, outputImag);
        ExecuteContext context{this, input, nullptr, nullptr, nullptr, outputReal, outputImag, true};
        executor_->run(&executeShard, &context);
    }

    void inverseSplit(double* inputReal, double* inputImag, double* output) {
        requireLayout(FFTWDataLayout::split);
        validateInverseSplitAlignment(inputReal, inputImag, output);
        ExecuteContext context{this, nullptr, nullptr, nullptr, output, inputReal, inputImag, false};
        executor_->run(&executeShard, &context);
    }

    void schedulerNoop() { executor_->run(&noopShard, nullptr); }

    Workload workload_;
    FFTWStrategy strategy_;
    std::vector<PlanPair> plans_;
    std::unique_ptr<PersistentIndexExecutor> executor_;
    double* realSurrogate_ = nullptr;
    fftw_complex* spectrumSurrogate_ = nullptr;
    double* spectrumRealSurrogate_ = nullptr;
    double* spectrumImagSurrogate_ = nullptr;
    std::string wisdom_;
    double otherSetupSeconds_ = 0.0;
    double allocationSeconds_ = 0.0;
    double planningSeconds_ = 0.0;
    double wisdomGenerationSeconds_ = 0.0;
    double wisdomImportSeconds_ = 0.0;
    std::size_t wisdomBytes_ = 0;
    std::size_t planningBytes_ = 0;
    bool planningBudgetExhausted_ = false;

private:
    struct ExecuteContext {
        Impl* provider;
        const double* forwardInput;
        Complex* forwardOutput;
        Complex* inverseInput;
        double* inverseOutput;
        double* splitReal;
        double* splitImag;
        bool forward;
    };

    void requireLayout(FFTWDataLayout expected) const {
        if (strategy_.layout != expected) {
            throw std::logic_error("FFTW execution entry point does not match the planned data layout.");
        }
    }

    void validateStrategy() {
        if (strategy_.internalWorkers == 0 || strategy_.internalWorkers > static_cast<std::size_t>(INT_MAX)) {
            throw std::invalid_argument("FFTW internal workers must lie in [1, INT_MAX].");
        }
        if (strategy_.outerWorkers == 0 || strategy_.outerWorkers > workload_.planes()) {
            throw std::invalid_argument("FFTW outer workers must lie in [1, plane count].");
        }
        if (!std::isfinite(strategy_.planningTimeLimitSeconds) || strategy_.planningTimeLimitSeconds < 0.0) {
            throw std::invalid_argument("FFTW planning time limit must be finite and nonnegative.");
        }
        if (strategy_.internalWorkers > std::numeric_limits<std::size_t>::max() / strategy_.outerWorkers) {
            throw std::invalid_argument("FFTW logical worker count overflows size_t.");
        }
    }

    std::vector<PlanPair> createPlans(unsigned flags, bool observePlanningBudget) {
        std::vector<PlanPair> plans;
        plans.reserve(strategy_.outerWorkers);
        fftw_plan_with_nthreads(static_cast<int>(strategy_.internalWorkers));
        for (std::size_t shard = 0; shard < strategy_.outerWorkers; ++shard) {
            const auto begin = workload_.planes() * shard / strategy_.outerWorkers;
            const auto end = workload_.planes() * (shard + 1) / strategy_.outerWorkers;
            PlanPair plan;
            plan.beginPlane = begin;
            plan.planeCount = end - begin;
            auto* real = realSurrogate_ + begin * workload_.realPlaneElements();
            plan.realAlignmentClass = fftw_alignment_of(real);
            auto* spectrum = strategy_.layout == FFTWDataLayout::interleaved ? spectrumSurrogate_ + begin : nullptr;
            auto* spectrumReal = strategy_.layout == FFTWDataLayout::split ? spectrumRealSurrogate_ + begin : nullptr;
            auto* spectrumImag = strategy_.layout == FFTWDataLayout::split ? spectrumImagSurrogate_ + begin : nullptr;
            if (strategy_.layout == FFTWDataLayout::interleaved) {
                plan.spectrumAlignmentClass = fftw_alignment_of(reinterpret_cast<double*>(spectrum));
            } else {
                plan.spectrumRealAlignmentClass = fftw_alignment_of(spectrumReal);
                plan.spectrumImagAlignmentClass = fftw_alignment_of(spectrumImag);
            }

            const auto limit = observePlanningBudget && strategy_.planningTimeLimitSeconds > 0.0
                ? strategy_.planningTimeLimitSeconds
                : FFTW_NO_TIMELIMIT;
            fftw_set_timelimit(limit);
            const auto forwardStart = Clock::now();
            plan.forward = makeForwardPlan(plan, real, spectrum, spectrumReal, spectrumImag, flags);
            observeLimit(forwardStart, limit);
            const auto inverseStart = Clock::now();
            plan.inverse = makeInversePlan(plan, spectrum, spectrumReal, spectrumImag, real, flags);
            observeLimit(inverseStart, limit);
            if (plan.forward == nullptr || plan.inverse == nullptr) {
                if (plan.forward != nullptr) fftw_destroy_plan(plan.forward);
                if (plan.inverse != nullptr) fftw_destroy_plan(plan.inverse);
                destroyPlans(plans);
                throw std::runtime_error("FFTW could not create a WVM-compatible guru64 strategy plan.");
            }
            plans.push_back(plan);
        }
        return plans;
    }

    fftw_plan makeForwardPlan(const PlanPair& shard, double* real, fftw_complex* spectrum,
                              double* spectrumReal, double* spectrumImag, unsigned flags) const {
        const auto planes = workload_.planes();
        const auto nxHalf = workload_.nxHalf();
        fftw_iodim64 dimensions[2] = {
            {static_cast<ptrdiff_t>(workload_.ny), static_cast<ptrdiff_t>(workload_.nx), static_cast<ptrdiff_t>(planes * nxHalf)},
            {static_cast<ptrdiff_t>(workload_.nx), 1, static_cast<ptrdiff_t>(planes)}};
        if (strategy_.outerWorkers == 1) {
            const auto realPlane = workload_.realPlaneElements();
            fftw_iodim64 batches[2] = {
                {static_cast<ptrdiff_t>(workload_.nz), static_cast<ptrdiff_t>(realPlane), 1},
                {static_cast<ptrdiff_t>(workload_.fields), static_cast<ptrdiff_t>(realPlane * workload_.nz), static_cast<ptrdiff_t>(workload_.nz)}};
            return strategy_.layout == FFTWDataLayout::interleaved
                ? fftw_plan_guru64_dft_r2c(2, dimensions, 2, batches, real, spectrum, flags)
                : fftw_plan_guru64_split_dft_r2c(2, dimensions, 2, batches, real, spectrumReal, spectrumImag, flags);
        }
        fftw_iodim64 batch = {static_cast<ptrdiff_t>(shard.planeCount),
                              static_cast<ptrdiff_t>(workload_.realPlaneElements()), 1};
        return strategy_.layout == FFTWDataLayout::interleaved
            ? fftw_plan_guru64_dft_r2c(2, dimensions, 1, &batch, real, spectrum, flags)
            : fftw_plan_guru64_split_dft_r2c(2, dimensions, 1, &batch, real, spectrumReal, spectrumImag, flags);
    }

    fftw_plan makeInversePlan(const PlanPair& shard, fftw_complex* spectrum,
                              double* spectrumReal, double* spectrumImag, double* real, unsigned flags) const {
        const auto planes = workload_.planes();
        const auto nxHalf = workload_.nxHalf();
        fftw_iodim64 dimensions[2] = {
            {static_cast<ptrdiff_t>(workload_.ny), static_cast<ptrdiff_t>(planes * nxHalf), static_cast<ptrdiff_t>(workload_.nx)},
            {static_cast<ptrdiff_t>(workload_.nx), static_cast<ptrdiff_t>(planes), 1}};
        if (strategy_.outerWorkers == 1) {
            const auto realPlane = workload_.realPlaneElements();
            fftw_iodim64 batches[2] = {
                {static_cast<ptrdiff_t>(workload_.nz), 1, static_cast<ptrdiff_t>(realPlane)},
                {static_cast<ptrdiff_t>(workload_.fields), static_cast<ptrdiff_t>(workload_.nz), static_cast<ptrdiff_t>(realPlane * workload_.nz)}};
            return strategy_.layout == FFTWDataLayout::interleaved
                ? fftw_plan_guru64_dft_c2r(2, dimensions, 2, batches, spectrum, real, flags)
                : fftw_plan_guru64_split_dft_c2r(2, dimensions, 2, batches, spectrumReal, spectrumImag, real, flags);
        }
        fftw_iodim64 batch = {static_cast<ptrdiff_t>(shard.planeCount), 1,
                              static_cast<ptrdiff_t>(workload_.realPlaneElements())};
        return strategy_.layout == FFTWDataLayout::interleaved
            ? fftw_plan_guru64_dft_c2r(2, dimensions, 1, &batch, spectrum, real, flags)
            : fftw_plan_guru64_split_dft_c2r(2, dimensions, 1, &batch, spectrumReal, spectrumImag, real, flags);
    }

    void observeLimit(const Clock::time_point& start, double limit) {
        if (limit > 0.0 && elapsedSeconds(start) >= 0.95 * limit) planningBudgetExhausted_ = true;
    }

    static void executeShard(void* rawContext, std::size_t shardIndex) {
        auto& context = *static_cast<ExecuteContext*>(rawContext);
        auto& provider = *context.provider;
        const auto& shard = provider.plans_[shardIndex];
        if (context.forward) {
            auto* input = const_cast<double*>(context.forwardInput) +
                shard.beginPlane * provider.workload_.realPlaneElements();
            if (provider.strategy_.layout == FFTWDataLayout::interleaved) {
                auto* output = reinterpret_cast<fftw_complex*>(context.forwardOutput) + shard.beginPlane;
                fftw_execute_dft_r2c(shard.forward, input, output);
            } else {
                fftw_execute_split_dft_r2c(shard.forward,
                                           input,
                                           context.splitReal + shard.beginPlane,
                                           context.splitImag + shard.beginPlane);
            }
        } else {
            auto* output = context.inverseOutput +
                shard.beginPlane * provider.workload_.realPlaneElements();
            if (provider.strategy_.layout == FFTWDataLayout::interleaved) {
                auto* input = reinterpret_cast<fftw_complex*>(context.inverseInput) + shard.beginPlane;
                fftw_execute_dft_c2r(shard.inverse, input, output);
            } else {
                fftw_execute_split_dft_c2r(shard.inverse,
                                           context.splitReal + shard.beginPlane,
                                           context.splitImag + shard.beginPlane,
                                           output);
            }
        }
    }

    static void noopShard(void*, std::size_t) {}

    void validateForwardAlignment(const double* input, Complex* output) const {
        if (strategy_.alignment == FFTWAlignmentStrategy::unaligned) return;
        for (const auto& shard : plans_) {
            auto* real = const_cast<double*>(input) + shard.beginPlane * workload_.realPlaneElements();
            auto* spectrum = reinterpret_cast<double*>(output + shard.beginPlane);
            if (fftw_alignment_of(real) != shard.realAlignmentClass ||
                fftw_alignment_of(spectrum) != shard.spectrumAlignmentClass) {
                throw std::invalid_argument("Aligned FFTW execution buffers do not match the planning alignment classes.");
            }
        }
    }

    void validateInverseAlignment(Complex* input, double* output) const {
        if (strategy_.alignment == FFTWAlignmentStrategy::unaligned) return;
        for (const auto& shard : plans_) {
            auto* spectrum = reinterpret_cast<double*>(input + shard.beginPlane);
            auto* real = output + shard.beginPlane * workload_.realPlaneElements();
            if (fftw_alignment_of(spectrum) != shard.spectrumAlignmentClass ||
                fftw_alignment_of(real) != shard.realAlignmentClass) {
                throw std::invalid_argument("Aligned FFTW execution buffers do not match the planning alignment classes.");
            }
        }
    }

    void validateForwardSplitAlignment(const double* input, double* outputReal, double* outputImag) const {
        if (strategy_.alignment == FFTWAlignmentStrategy::unaligned) return;
        for (const auto& shard : plans_) {
            auto* real = const_cast<double*>(input) + shard.beginPlane * workload_.realPlaneElements();
            auto* spectrumReal = outputReal + shard.beginPlane;
            auto* spectrumImag = outputImag + shard.beginPlane;
            if (fftw_alignment_of(real) != shard.realAlignmentClass ||
                fftw_alignment_of(spectrumReal) != shard.spectrumRealAlignmentClass ||
                fftw_alignment_of(spectrumImag) != shard.spectrumImagAlignmentClass) {
                throw std::invalid_argument("Aligned split FFTW execution buffers do not match the planning alignment classes.");
            }
        }
    }

    void validateInverseSplitAlignment(double* inputReal, double* inputImag, double* output) const {
        if (strategy_.alignment == FFTWAlignmentStrategy::unaligned) return;
        for (const auto& shard : plans_) {
            auto* spectrumReal = inputReal + shard.beginPlane;
            auto* spectrumImag = inputImag + shard.beginPlane;
            auto* real = output + shard.beginPlane * workload_.realPlaneElements();
            if (fftw_alignment_of(spectrumReal) != shard.spectrumRealAlignmentClass ||
                fftw_alignment_of(spectrumImag) != shard.spectrumImagAlignmentClass ||
                fftw_alignment_of(real) != shard.realAlignmentClass) {
                throw std::invalid_argument("Aligned split FFTW execution buffers do not match the planning alignment classes.");
            }
        }
    }

    static void destroyPlans(std::vector<PlanPair>& plans) {
        for (auto& plan : plans) {
            if (plan.forward != nullptr) fftw_destroy_plan(plan.forward);
            if (plan.inverse != nullptr) fftw_destroy_plan(plan.inverse);
            plan.forward = nullptr;
            plan.inverse = nullptr;
        }
        plans.clear();
    }

    void releaseSurrogates() {
        if (realSurrogate_ != nullptr) fftw_free(realSurrogate_);
        if (spectrumSurrogate_ != nullptr) fftw_free(spectrumSurrogate_);
        if (spectrumRealSurrogate_ != nullptr) fftw_free(spectrumRealSurrogate_);
        if (spectrumImagSurrogate_ != nullptr) fftw_free(spectrumImagSurrogate_);
        realSurrogate_ = nullptr;
        spectrumSurrogate_ = nullptr;
        spectrumRealSurrogate_ = nullptr;
        spectrumImagSurrogate_ = nullptr;
    }
};

FFTWProvider::FFTWProvider(const Workload& workload, std::size_t workers)
    : FFTWProvider(workload, FFTWStrategy{FFTWPlanningMode::measure, FFTWAlignmentStrategy::unaligned,
                                         FFTWWisdomStrategy::cold, workers, 1, 0.0}) {}

FFTWProvider::FFTWProvider(const Workload& workload, FFTWStrategy strategy)
    : impl_(std::make_unique<Impl>(workload, strategy)) {}

FFTWProvider::~FFTWProvider() = default;
FFTWProvider::FFTWProvider(FFTWProvider&&) noexcept = default;
FFTWProvider& FFTWProvider::operator=(FFTWProvider&&) noexcept = default;

void FFTWProvider::forward(const double* input, Complex* wvmSpectrum) { impl_->forward(input, wvmSpectrum); }
void FFTWProvider::inverse(Complex* wvmSpectrum, double* output) { impl_->inverse(wvmSpectrum, output); }
void FFTWProvider::forwardSplit(const double* input, double* wvmSpectrumReal, double* wvmSpectrumImag) {
    impl_->forwardSplit(input, wvmSpectrumReal, wvmSpectrumImag);
}
void FFTWProvider::inverseSplit(double* wvmSpectrumReal, double* wvmSpectrumImag, double* output) {
    impl_->inverseSplit(wvmSpectrumReal, wvmSpectrumImag, output);
}
void FFTWProvider::executeSchedulerNoop() { impl_->schedulerNoop(); }
bool FFTWProvider::splitInPlaceWvmOrderSupported() const noexcept { return false; }
std::string FFTWProvider::splitInPlaceWvmOrderCapability() const {
    return "unsupported: one padded x-fastest real grid cannot alias two disjoint WVM-frequency-major split arrays under the exact guru64 stride contract; both forward and inverse planners return null";
}
double FFTWProvider::otherSetupSeconds() const noexcept { return impl_->otherSetupSeconds_; }
double FFTWProvider::allocationSeconds() const noexcept { return impl_->allocationSeconds_; }
double FFTWProvider::planningSeconds() const noexcept { return impl_->planningSeconds_; }
double FFTWProvider::wisdomGenerationSeconds() const noexcept { return impl_->wisdomGenerationSeconds_; }
double FFTWProvider::wisdomImportSeconds() const noexcept { return impl_->wisdomImportSeconds_; }
double FFTWProvider::planningTimeLimitSeconds() const noexcept { return impl_->strategy_.planningTimeLimitSeconds; }
bool FFTWProvider::planningBudgetExhausted() const noexcept { return impl_->planningBudgetExhausted_; }
std::size_t FFTWProvider::wisdomBytes() const noexcept { return impl_->wisdomBytes_; }
std::size_t FFTWProvider::planningBytes() const noexcept { return impl_->planningBytes_; }
std::size_t FFTWProvider::internalWorkers() const noexcept { return impl_->strategy_.internalWorkers; }
std::size_t FFTWProvider::outerWorkers() const noexcept { return impl_->strategy_.outerWorkers; }
std::size_t FFTWProvider::totalLogicalWorkers() const noexcept {
    return impl_->strategy_.internalWorkers * impl_->strategy_.outerWorkers;
}
std::size_t FFTWProvider::minimumAlignmentBytes() const noexcept {
    return impl_->strategy_.alignment == FFTWAlignmentStrategy::aligned ? 64 : 1;
}
FFTWStrategy FFTWProvider::strategy() const noexcept { return impl_->strategy_; }
std::string FFTWProvider::libraryIdentity() const { return libraryContaining(reinterpret_cast<const void*>(&fftw_execute)); }
std::string FFTWProvider::version() const { return fftw_version; }

} // namespace skbench
