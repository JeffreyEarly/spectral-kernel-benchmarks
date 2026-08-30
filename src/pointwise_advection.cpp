#include "pointwise_advection.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace skbench {
namespace {

using Clock = std::chrono::steady_clock;

class PersistentIndexExecutor {
public:
    using Task = void (*)(void*, std::size_t);

    explicit PersistentIndexExecutor(std::size_t workers) : workers_(workers) {
        if (workers_ == 0) {
            throw std::invalid_argument(
                "Persistent pointwise worker count must be positive.");
        }
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

    std::size_t workers() const noexcept { return workers_; }

    std::size_t persistentBytes() const noexcept {
        return threads_.capacity() * sizeof(std::thread);
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

void applySerial(std::size_t volume, double scale,
                 const double* shared, const double* derivative,
                 double* target) {
    const auto* u = shared;
    const auto* v = shared + volume;
    const auto* w = shared + 2 * volume;
    const auto* qx = derivative;
    const auto* qy = derivative + volume;
    const auto* qz = derivative + 2 * volume;
    for (std::size_t point = 0; point < volume; ++point) {
        target[point] = -scale *
            (u[point] * qx[point] + v[point] * qy[point] +
             w[point] * qz[point]);
    }
}

#if defined(__clang__) || defined(__GNUC__)
#define SKBENCH_RESTRICT __restrict__
#else
#define SKBENCH_RESTRICT
#endif

void applyVectorRange(std::size_t volume, std::size_t begin, std::size_t end,
                      double scale,
                      const double* SKBENCH_RESTRICT shared,
                      const double* SKBENCH_RESTRICT derivative,
                      double* SKBENCH_RESTRICT target) {
    const auto* SKBENCH_RESTRICT u = shared;
    const auto* SKBENCH_RESTRICT v = shared + volume;
    const auto* SKBENCH_RESTRICT w = shared + 2 * volume;
    const auto* SKBENCH_RESTRICT qx = derivative;
    const auto* SKBENCH_RESTRICT qy = derivative + volume;
    const auto* SKBENCH_RESTRICT qz = derivative + 2 * volume;
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#endif
    for (std::size_t point = begin; point < end; ++point) {
        target[point] = -scale *
            (u[point] * qx[point] + v[point] * qy[point] +
             w[point] * qz[point]);
    }
}

#undef SKBENCH_RESTRICT

} // namespace

class PointwiseAdvectionExecutor::Impl {
public:
    Impl(PointwiseAdvectionPolicy policy, std::size_t workers,
         std::size_t volumeElements, double scale)
        : policy_(policy), workers_(workers), volume_(volumeElements), scale_(scale) {
        if (volume_ == 0) {
            throw std::invalid_argument(
                "Pointwise advection volume must be positive.");
        }
        if (workers_ == 0) {
            throw std::invalid_argument(
                "Pointwise advection worker count must be positive.");
        }
        if (policy_ != PointwiseAdvectionPolicy::spatialStatic && workers_ != 1) {
            throw std::invalid_argument(
                "Serial pointwise policies require exactly one worker.");
        }
        const auto setupStart = Clock::now();
        if (policy_ == PointwiseAdvectionPolicy::spatialStatic) {
            executor_ = std::make_unique<PersistentIndexExecutor>(workers_);
        }
        setupSeconds_ = std::chrono::duration<double>(Clock::now() - setupStart).count();
    }

    void execute(const double* shared, const double* derivative,
                 double* target) {
        if (shared == nullptr || derivative == nullptr || target == nullptr) {
            throw std::invalid_argument(
                "Pointwise advection buffers must be non-null.");
        }
        if (policy_ == PointwiseAdvectionPolicy::serial) {
            applySerial(volume_, scale_, shared, derivative, target);
            return;
        }
        if (policy_ == PointwiseAdvectionPolicy::vectorSerial) {
            applyVectorRange(volume_, 0, volume_, scale_, shared, derivative,
                             target);
            return;
        }
        ExecuteContext context{this, shared, derivative, target};
        executor_->run(&executeShard, &context);
    }

    void executeSchedulerNoop() {
        if (executor_) executor_->run(&noopShard, nullptr);
    }

    PointwiseAdvectionPolicy policy() const noexcept { return policy_; }
    std::size_t workers() const noexcept { return workers_; }
    std::size_t persistentBytes() const noexcept {
        return executor_ ? executor_->persistentBytes() : 0;
    }
    double setupSeconds() const noexcept { return setupSeconds_; }

private:
    struct ExecuteContext {
        Impl* executor;
        const double* shared;
        const double* derivative;
        double* target;
    };

    static void executeShard(void* rawContext, std::size_t worker) {
        auto& context = *static_cast<ExecuteContext*>(rawContext);
        const auto begin = context.executor->volume_ * worker /
            context.executor->workers_;
        const auto end = context.executor->volume_ * (worker + 1) /
            context.executor->workers_;
        applyVectorRange(context.executor->volume_, begin, end,
                         context.executor->scale_, context.shared,
                         context.derivative, context.target);
    }

    static void noopShard(void*, std::size_t) {}

    PointwiseAdvectionPolicy policy_ = PointwiseAdvectionPolicy::serial;
    std::size_t workers_ = 1;
    std::size_t volume_ = 0;
    double scale_ = 0.0;
    std::unique_ptr<PersistentIndexExecutor> executor_;
    double setupSeconds_ = 0.0;
};

PointwiseAdvectionPolicy pointwiseAdvectionPolicyNamed(std::string_view name) {
    if (name == "serial") return PointwiseAdvectionPolicy::serial;
    if (name == "vector-serial") return PointwiseAdvectionPolicy::vectorSerial;
    if (name == "spatial-static") return PointwiseAdvectionPolicy::spatialStatic;
    throw std::invalid_argument(
        "Unknown pointwise policy '" + std::string(name) +
        "'. Expected serial, vector-serial, or spatial-static.");
}

const char* pointwiseAdvectionPolicyName(PointwiseAdvectionPolicy policy) {
    switch (policy) {
        case PointwiseAdvectionPolicy::serial: return "serial";
        case PointwiseAdvectionPolicy::vectorSerial: return "vector-serial";
        case PointwiseAdvectionPolicy::spatialStatic: return "spatial-static";
    }
    throw std::invalid_argument("Unknown pointwise advection policy.");
}

PointwiseAdvectionExecutor::PointwiseAdvectionExecutor(
    PointwiseAdvectionPolicy policy, std::size_t workers,
    std::size_t volumeElements, double scale)
    : impl_(std::make_unique<Impl>(policy, workers, volumeElements, scale)) {}

PointwiseAdvectionExecutor::~PointwiseAdvectionExecutor() = default;
PointwiseAdvectionExecutor::PointwiseAdvectionExecutor(
    PointwiseAdvectionExecutor&&) noexcept = default;
PointwiseAdvectionExecutor& PointwiseAdvectionExecutor::operator=(
    PointwiseAdvectionExecutor&&) noexcept = default;

void PointwiseAdvectionExecutor::execute(
    const double* shared, const double* derivative, double* target) {
    impl_->execute(shared, derivative, target);
}

void PointwiseAdvectionExecutor::executeSchedulerNoop() {
    impl_->executeSchedulerNoop();
}

PointwiseAdvectionPolicy PointwiseAdvectionExecutor::policy() const noexcept {
    return impl_->policy();
}

std::size_t PointwiseAdvectionExecutor::workers() const noexcept {
    return impl_->workers();
}

std::size_t PointwiseAdvectionExecutor::persistentBytes() const noexcept {
    return impl_->persistentBytes();
}

double PointwiseAdvectionExecutor::setupSeconds() const noexcept {
    return impl_->setupSeconds();
}

} // namespace skbench
