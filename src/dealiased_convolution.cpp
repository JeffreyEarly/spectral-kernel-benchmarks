#include "skbench/skbench.hpp"

#include <fftw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>

#if SKBENCH_HAVE_FFTWPP
#include "Complex.h"
#include "convolve.h"
#endif

#ifndef SKBENCH_FFTWPP_COMMIT
#define SKBENCH_FFTWPP_COMMIT "unavailable"
#endif

#if SKBENCH_HAVE_FFTWPP
namespace {
void issue17MultiplyOffset(std::size_t offset, Complex** values, std::size_t count) {
    double* realValues[12]{};
    for (std::size_t array = 0; array < 4; ++array)
        realValues[array] = reinterpret_cast<double*>(values[array]);
    for (std::size_t index = 0; index < count; ++index) {
        double input[4] = {realValues[0][index], realValues[1][index],
                           realValues[2][index], realValues[3][index]};
        for (std::size_t product = 0; product < 4; ++product) {
            const auto logicalProduct = offset + product;
            const auto first = logicalProduct / 4;
            const auto second = logicalProduct % 4;
            realValues[product][index] = input[first] * input[second];
        }
    }
}
void issue17Multiply0(Complex** values, std::size_t count, fftwpp::Indices*, std::size_t) {
    issue17MultiplyOffset(0, values, count);
}
void issue17Multiply4(Complex** values, std::size_t count, fftwpp::Indices*, std::size_t) {
    issue17MultiplyOffset(4, values, count);
}
void issue17Multiply8(Complex** values, std::size_t count, fftwpp::Indices*, std::size_t) {
    issue17MultiplyOffset(8, values, count);
}

void issue17AdvectiveStreamed(Complex** values, std::size_t count,
                              fftwpp::Indices*, std::size_t) {
    double* realValues[6]{};
    for (std::size_t array = 0; array < 6; ++array)
        realValues[array] = reinterpret_cast<double*>(values[array]);
    for (std::size_t index = 0; index < count; ++index) {
        const double u = realValues[0][index];
        const double v = realValues[1][index];
        const double w = realValues[2][index];
        realValues[0][index] = -(u * realValues[3][index] +
                                 v * realValues[4][index] +
                                 w * realValues[5][index]);
    }
}

void issue17AdvectiveAllTargets(Complex** values, std::size_t count,
                                fftwpp::Indices*, std::size_t) {
    double* realValues[15]{};
    for (std::size_t array = 0; array < 15; ++array)
        realValues[array] = reinterpret_cast<double*>(values[array]);
    for (std::size_t index = 0; index < count; ++index) {
        const double u = realValues[0][index];
        const double v = realValues[1][index];
        const double w = realValues[2][index];
        for (std::size_t target = 0; target < 4; ++target) {
            const auto derivative = 3 + 3 * target;
            realValues[target][index] =
                -(u * realValues[derivative][index] +
                  v * realValues[derivative + 1][index] +
                  w * realValues[derivative + 2][index]);
        }
    }
}
}
#endif

namespace skbench {
namespace {

using Clock = std::chrono::steady_clock;

#if SKBENCH_HAVE_FFTWPP
std::uint64_t processHighWaterBytes() noexcept {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) return 0;
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
#endif
}
constexpr double tolerance = 1.0e-12;
struct StoredMode {
    int k = 0;
    int l = 0;
    std::size_t rectangleIndex = 0;
    std::size_t fullIndex = 0;
};

std::vector<StoredMode> storedDisk(std::size_t n) {
    const auto radius = static_cast<int>(n / 3);
    const auto length = static_cast<std::size_t>(2 * radius + 1);
    const auto half = static_cast<std::size_t>(radius + 1);
    std::vector<StoredMode> modes;
    for (int l = 0; l <= radius; ++l) {
        for (int k = -radius; k <= radius; ++k) {
            if (k * k + l * l > radius * radius) continue;
            const auto centeredK = static_cast<std::size_t>(k + radius);
            const auto fullK = k < 0 ? n - static_cast<std::size_t>(-k)
                                     : static_cast<std::size_t>(k);
            modes.push_back({k, l, centeredK * half + static_cast<std::size_t>(l),
                             fullK * (n / 2 + 1) + static_cast<std::size_t>(l)});
        }
    }
    (void)length;
    return modes;
}

std::vector<Complex> compactFixture(const std::vector<StoredMode>& modes,
                                    std::size_t fields, std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::vector<Complex> values(fields * modes.size());
    for (std::size_t field = 0; field < fields; ++field) {
        for (std::size_t mode = 0; mode < modes.size(); ++mode) {
            const auto& key = modes[mode];
            if (key.l == 0 && key.k < 0) continue;
            Complex value{distribution(generator), distribution(generator)};
            if (key.l == 0 && key.k == 0) value.imag = 0.0;
            values[field * modes.size() + mode] = value;
            if (key.l == 0 && key.k > 0) {
                const auto match = std::find_if(modes.begin(), modes.end(), [&](const StoredMode& other) {
                    return other.k == -key.k && other.l == 0;
                });
                values[field * modes.size() + static_cast<std::size_t>(match - modes.begin())] =
                    {value.real, -value.imag};
            }
        }
    }
    return values;
}

template <class Prepare, class Action>
std::vector<double> timed(std::size_t warmups, std::size_t samples,
                          Prepare prepare, Action action) {
    for (std::size_t i = 0; i < warmups; ++i) { prepare(); action(); }
    std::vector<double> result;
    result.reserve(samples);
    for (std::size_t i = 0; i < samples; ++i) {
        prepare();
        const auto start = Clock::now();
        action();
        result.push_back(std::chrono::duration<double>(Clock::now() - start).count());
    }
    return result;
}

TimingSeries timing(std::string scope, std::string stage, StageState state,
                    std::uint64_t moved, std::vector<double> samples = {}) {
    return {std::move(scope), std::move(stage), "forward", state, moved, std::move(samples)};
}

class PersistentTaskExecutor {
public:
    using Task = void (*)(void*, std::size_t);

    explicit PersistentTaskExecutor(std::size_t workers) : workers_(workers) {
        threads_.reserve(workers_ - 1);
        for (std::size_t worker = 1; worker < workers_; ++worker)
            threads_.emplace_back([this, worker] { workerLoop(worker); });
    }
    ~PersistentTaskExecutor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        ready_.notify_all();
        for (auto& thread : threads_) thread.join();
    }
    PersistentTaskExecutor(const PersistentTaskExecutor&) = delete;
    PersistentTaskExecutor& operator=(const PersistentTaskExecutor&) = delete;

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
    std::uint64_t persistentBytes() const {
        return threads_.capacity() * sizeof(std::thread) +
               sizeof(mutex_) + sizeof(ready_) + sizeof(complete_);
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

class ExplicitConvolution {
public:
    ExplicitConvolution(std::size_t n, std::size_t products,
                        const std::vector<StoredMode>& modes)
        : n_(n), products_(products), count_(std::max<std::size_t>(4, products)),
          halfCount_(n * (n / 2 + 1)), realCount_(n * n),
          modes_(modes), output_(products * modes.size()) {
        for (std::size_t i = 0; i < count_; ++i) {
            spectra_.push_back(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * halfCount_)));
            physical_.push_back(static_cast<double*>(fftw_malloc(sizeof(double) * realCount_)));
            if (!spectra_.back() || !physical_.back()) throw std::bad_alloc();
        }
        const auto flags = FFTW_MEASURE | FFTW_UNALIGNED;
        inverse_ = fftw_plan_dft_c2r_2d(static_cast<int>(n_), static_cast<int>(n_),
                                        spectra_[0], physical_[0], flags);
        forward_ = fftw_plan_dft_r2c_2d(static_cast<int>(n_), static_cast<int>(n_),
                                        physical_[0], spectra_[0], flags);
        if (!inverse_ || !forward_) throw std::runtime_error("Unable to create explicit FFTW plans.");
    }
    ~ExplicitConvolution() {
        if (inverse_) fftw_destroy_plan(inverse_);
        if (forward_) fftw_destroy_plan(forward_);
        for (auto* value : spectra_) fftw_free(value);
        for (auto* value : physical_) fftw_free(value);
    }
    void load(const std::vector<Complex>& compact) {
        for (std::size_t field = 0; field < 4; ++field) {
            std::memset(spectra_[field], 0, sizeof(fftw_complex) * halfCount_);
            for (std::size_t mode = 0; mode < modes_.size(); ++mode) {
                spectra_[field][modes_[mode].fullIndex][0] = compact[field * modes_.size() + mode].real;
                spectra_[field][modes_[mode].fullIndex][1] = compact[field * modes_.size() + mode].imag;
            }
        }
    }
    void inverse() {
        for (std::size_t field = 0; field < 4; ++field)
            fftw_execute_dft_c2r(inverse_, spectra_[field], physical_[field]);
    }
    void multiply() {
        for (std::size_t point = 0; point < realCount_; ++point) {
            double input[4] = {physical_[0][point], physical_[1][point],
                               physical_[2][point], physical_[3][point]};
            for (std::size_t product = 0; product < products_; ++product)
                physical_[product][point] = input[product / 4] * input[product % 4];
        }
    }
    void forward() {
        for (std::size_t product = 0; product < products_; ++product)
            fftw_execute_dft_r2c(forward_, physical_[product], spectra_[product]);
    }
    void gather() {
        const double scale = 1.0 / static_cast<double>(realCount_);
        for (std::size_t product = 0; product < products_; ++product)
            for (std::size_t mode = 0; mode < modes_.size(); ++mode)
                output_[product * modes_.size() + mode] = {
                    scale * spectra_[product][modes_[mode].fullIndex][0],
                    scale * spectra_[product][modes_[mode].fullIndex][1]};
    }
    void execute(const std::vector<Complex>& compact) { load(compact); inverse(); multiply(); forward(); gather(); }
    const std::vector<Complex>& output() const { return output_; }
    std::uint64_t residentBytes() const {
        return count_ * (halfCount_ * sizeof(fftw_complex) + realCount_ * sizeof(double)) +
               output_.size() * sizeof(Complex);
    }
private:
    std::size_t n_, products_, count_, halfCount_, realCount_;
    const std::vector<StoredMode>& modes_;
    std::vector<fftw_complex*> spectra_;
    std::vector<double*> physical_;
    std::vector<Complex> output_;
    fftw_plan inverse_ = nullptr;
    fftw_plan forward_ = nullptr;
};

class ExplicitAdvectiveConvolution {
public:
    ExplicitAdvectiveConvolution(std::size_t n,
                                 const std::vector<StoredMode>& modes)
        : n_(n), halfCount_(n * (n / 2 + 1)), realCount_(n * n),
          modes_(modes), output_(4 * modes.size()) {
        spectraStorage_ = static_cast<fftw_complex*>(
            fftw_malloc(sizeof(fftw_complex) * 3 * halfCount_));
        physicalStorage_ = static_cast<double*>(
            fftw_malloc(sizeof(double) * 6 * realCount_));
        if (!spectraStorage_ || !physicalStorage_) {
            if (spectraStorage_) fftw_free(spectraStorage_);
            if (physicalStorage_) fftw_free(physicalStorage_);
            throw std::bad_alloc();
        }
        for (std::size_t i = 0; i < 3; ++i)
            spectra_.push_back(spectraStorage_ + i * halfCount_);
        for (std::size_t i = 0; i < 6; ++i)
            physical_.push_back(physicalStorage_ + i * realCount_);
        const auto flags = FFTW_MEASURE | FFTW_UNALIGNED;
        inverse_ = fftw_plan_dft_c2r_2d(
            static_cast<int>(n_), static_cast<int>(n_), spectra_[0],
            physical_[0], flags);
        forward_ = fftw_plan_dft_r2c_2d(static_cast<int>(n_), static_cast<int>(n_),
                                        physical_[0], spectra_[0], flags);
        if (!inverse_ || !forward_) {
            if (inverse_) fftw_destroy_plan(inverse_);
            if (forward_) fftw_destroy_plan(forward_);
            inverse_ = nullptr;
            forward_ = nullptr;
            fftw_free(spectraStorage_);
            fftw_free(physicalStorage_);
            spectraStorage_ = nullptr;
            physicalStorage_ = nullptr;
            throw std::runtime_error("Unable to create streamed explicit FFTW plans.");
        }
    }
    ~ExplicitAdvectiveConvolution() {
        if (inverse_) fftw_destroy_plan(inverse_);
        if (forward_) fftw_destroy_plan(forward_);
        if (spectraStorage_) fftw_free(spectraStorage_);
        if (physicalStorage_) fftw_free(physicalStorage_);
    }
    void loadField(std::size_t slot, const std::vector<Complex>& compact,
                   std::size_t field) {
        std::memset(spectra_[slot], 0, sizeof(fftw_complex) * halfCount_);
        for (std::size_t mode = 0; mode < modes_.size(); ++mode) {
            const auto value = compact[field * modes_.size() + mode];
            spectra_[slot][modes_[mode].fullIndex][0] = value.real;
            spectra_[slot][modes_[mode].fullIndex][1] = value.imag;
        }
    }
    void loadAdvectors(const std::vector<Complex>& compact) {
        for (std::size_t field = 0; field < 3; ++field)
            loadField(field, compact, field);
    }
    void inverseAdvectors() {
        for (std::size_t field = 0; field < 3; ++field)
            fftw_execute_dft_c2r(inverse_, spectra_[field], physical_[field]);
    }
    void loadDerivatives(const std::vector<Complex>& compact, std::size_t target) {
        for (std::size_t derivative = 0; derivative < 3; ++derivative)
            loadField(derivative, compact, 3 + 3 * target + derivative);
    }
    void inverseDerivatives() {
        for (std::size_t derivative = 0; derivative < 3; ++derivative)
            fftw_execute_dft_c2r(
                inverse_, spectra_[derivative], physical_[3 + derivative]);
    }
    void multiplyTarget() {
        const auto* u = physical_[0];
        const auto* v = physical_[1];
        const auto* w = physical_[2];
        auto* dx = physical_[3];
        const auto* dy = physical_[4];
        const auto* dz = physical_[5];
        for (std::size_t point = 0; point < realCount_; ++point)
            dx[point] = -(u[point] * dx[point] +
                          v[point] * dy[point] +
                          w[point] * dz[point]);
    }
    void forwardTarget() {
        fftw_execute_dft_r2c(forward_, physical_[3], spectra_[0]);
    }
    void gatherTarget(std::size_t target) {
        const double scale = 1.0 / static_cast<double>(realCount_);
        for (std::size_t mode = 0; mode < modes_.size(); ++mode)
            output_[target * modes_.size() + mode] = {
                scale * spectra_[0][modes_[mode].fullIndex][0],
                scale * spectra_[0][modes_[mode].fullIndex][1]};
    }
    void execute(const std::vector<Complex>& compact) {
        loadAdvectors(compact);
        inverseAdvectors();
        for (std::size_t target = 0; target < 4; ++target) {
            loadDerivatives(compact, target);
            inverseDerivatives();
            multiplyTarget();
            forwardTarget();
            gatherTarget(target);
        }
    }
    const std::vector<Complex>& output() const { return output_; }
    std::uint64_t residentBytes() const {
        return 3 * halfCount_ * sizeof(fftw_complex) +
               6 * realCount_ * sizeof(double) +
               output_.size() * sizeof(Complex);
    }
private:
    std::size_t n_, halfCount_, realCount_;
    const std::vector<StoredMode>& modes_;
    std::vector<fftw_complex*> spectra_;
    std::vector<double*> physical_;
    fftw_complex* spectraStorage_ = nullptr;
    double* physicalStorage_ = nullptr;
    std::vector<Complex> output_;
    fftw_plan inverse_ = nullptr;
    fftw_plan forward_ = nullptr;
};

class ExplicitParallelAdvectiveConvolution {
public:
    ExplicitParallelAdvectiveConvolution(std::size_t n,
                                         const std::vector<StoredMode>& modes)
        : n_(n), halfCount_(n * (n / 2 + 1)), realCount_(n * n),
          modes_(modes), output_(4 * modes.size()), executor_(4) {
        sharedSpectra_ = static_cast<fftw_complex*>(
            fftw_malloc(sizeof(fftw_complex) * 3 * halfCount_));
        advectors_ = static_cast<double*>(
            fftw_malloc(sizeof(double) * 3 * realCount_));
        for (auto& target : targets_) {
            target.spectra = static_cast<fftw_complex*>(
                fftw_malloc(sizeof(fftw_complex) * 3 * halfCount_));
            target.physical = static_cast<double*>(
                fftw_malloc(sizeof(double) * 3 * realCount_));
        }
        if (!sharedSpectra_ || !advectors_ ||
            std::any_of(targets_.begin(), targets_.end(), [](const auto& target) {
                return target.spectra == nullptr || target.physical == nullptr;
            })) {
            release();
            throw std::bad_alloc();
        }
        const auto flags = FFTW_MEASURE | FFTW_UNALIGNED;
        inverse_ = fftw_plan_dft_c2r_2d(
            static_cast<int>(n_), static_cast<int>(n_), sharedSpectra_,
            advectors_, flags);
        forward_ = fftw_plan_dft_r2c_2d(
            static_cast<int>(n_), static_cast<int>(n_), targets_[0].physical,
            targets_[0].spectra, flags);
        if (!inverse_ || !forward_) {
            if (inverse_) fftw_destroy_plan(inverse_);
            if (forward_) fftw_destroy_plan(forward_);
            inverse_ = nullptr;
            forward_ = nullptr;
            release();
            throw std::runtime_error("Unable to create parallel explicit FFTW plans.");
        }
    }
    ~ExplicitParallelAdvectiveConvolution() {
        if (inverse_) fftw_destroy_plan(inverse_);
        if (forward_) fftw_destroy_plan(forward_);
        release();
    }
    void execute(const std::vector<Complex>& compact) {
        executeAdvectors(compact);
        executeTargets(compact);
    }
    void executeAdvectors(const std::vector<Complex>& compact) {
        for (std::size_t field = 0; field < 3; ++field) {
            auto* spectrum = sharedSpectra_ + field * halfCount_;
            loadField(spectrum, compact, field);
            fftw_execute_dft_c2r(
                inverse_, spectrum, advectors_ + field * realCount_);
        }
    }
    void executeTargets(const std::vector<Complex>& compact) {
        Context context{this, &compact};
        executor_.run(&executeTargetTask, &context);
    }
    const std::vector<Complex>& output() const { return output_; }
    std::uint64_t residentBytes() const {
        return 15 * halfCount_ * sizeof(fftw_complex) +
               15 * realCount_ * sizeof(double) +
               output_.size() * sizeof(Complex) + executor_.persistentBytes();
    }
private:
    struct TargetBuffers {
        fftw_complex* spectra = nullptr;
        double* physical = nullptr;
    };
    struct Context {
        ExplicitParallelAdvectiveConvolution* self;
        const std::vector<Complex>* compact;
    };
    static void executeTargetTask(void* raw, std::size_t target) {
        auto& context = *static_cast<Context*>(raw);
        context.self->executeTarget(*context.compact, target);
    }
    void loadField(fftw_complex* spectrum,
                   const std::vector<Complex>& compact,
                   std::size_t field) {
        std::memset(spectrum, 0, sizeof(fftw_complex) * halfCount_);
        for (std::size_t mode = 0; mode < modes_.size(); ++mode) {
            const auto value = compact[field * modes_.size() + mode];
            spectrum[modes_[mode].fullIndex][0] = value.real;
            spectrum[modes_[mode].fullIndex][1] = value.imag;
        }
    }
    void executeTarget(const std::vector<Complex>& compact,
                       std::size_t targetIndex) {
        auto& target = targets_[targetIndex];
        for (std::size_t derivative = 0; derivative < 3; ++derivative) {
            auto* spectrum = target.spectra + derivative * halfCount_;
            loadField(spectrum, compact, 3 + 3 * targetIndex + derivative);
            fftw_execute_dft_c2r(
                inverse_, spectrum, target.physical + derivative * realCount_);
        }
        const auto* u = advectors_;
        const auto* v = advectors_ + realCount_;
        const auto* w = advectors_ + 2 * realCount_;
        auto* dx = target.physical;
        const auto* dy = target.physical + realCount_;
        const auto* dz = target.physical + 2 * realCount_;
        for (std::size_t point = 0; point < realCount_; ++point)
            dx[point] = -(u[point] * dx[point] +
                          v[point] * dy[point] +
                          w[point] * dz[point]);
        fftw_execute_dft_r2c(forward_, dx, target.spectra);
        const double scale = 1.0 / static_cast<double>(realCount_);
        auto* destination = output_.data() + targetIndex * modes_.size();
        for (std::size_t mode = 0; mode < modes_.size(); ++mode)
            destination[mode] = {
                scale * target.spectra[modes_[mode].fullIndex][0],
                scale * target.spectra[modes_[mode].fullIndex][1]};
    }
    void release() {
        if (sharedSpectra_) fftw_free(sharedSpectra_);
        if (advectors_) fftw_free(advectors_);
        sharedSpectra_ = nullptr;
        advectors_ = nullptr;
        for (auto& target : targets_) {
            if (target.spectra) fftw_free(target.spectra);
            if (target.physical) fftw_free(target.physical);
            target = {};
        }
    }

    std::size_t n_, halfCount_, realCount_;
    const std::vector<StoredMode>& modes_;
    fftw_complex* sharedSpectra_ = nullptr;
    double* advectors_ = nullptr;
    std::array<TargetBuffers, 4> targets_{};
    std::vector<Complex> output_;
    PersistentTaskExecutor executor_;
    fftw_plan inverse_ = nullptr;
    fftw_plan forward_ = nullptr;
};

#if SKBENCH_HAVE_FFTWPP
class ImplicitConvolution {
    struct Kernel;
public:
    ImplicitConvolution(std::size_t n, std::size_t products,
                        const std::vector<StoredMode>& modes)
        : n_(n), products_(products), modes_(modes), radius_(n / 3),
          length_(2 * radius_ + 1), half_(radius_ + 1),
          count_(4), output_(products * modes.size()) {
        fftwpp::fftw::maxthreads = 1;
        fftwpp::fftw::effort = FFTW_MEASURE | FFTW_UNALIGNED;
        fftwpp::multiplier* multipliers[] = {issue17Multiply0, issue17Multiply4, issue17Multiply8};
        for (std::size_t group = 0; group < (products_ + 3) / 4; ++group)
            kernels_.push_back(std::make_unique<Kernel>(n_, length_, half_, multipliers[group]));
    }
    void load(Kernel& kernel, const std::vector<Complex>& compact) {
        for (std::size_t field = 0; field < count_; ++field) {
            std::fill(kernel.arrays[field],
                      kernel.arrays[field] + length_ * half_,
                      ::Complex(0.0, 0.0));
            for (std::size_t mode = 0; mode < modes_.size(); ++mode) {
                const auto value = compact[field * modes_.size() + mode];
                kernel.arrays[field][modes_[mode].rectangleIndex] = ::Complex(value.real, value.imag);
            }
        }
    }
    void gatherGroup(const Kernel& kernel, std::size_t firstProduct) {
        const auto stop = std::min(products_, firstProduct + 4);
        for (std::size_t product = firstProduct; product < stop; ++product)
            for (std::size_t mode = 0; mode < modes_.size(); ++mode) {
                const auto value = kernel.arrays[product - firstProduct][modes_[mode].rectangleIndex];
                output_[product * modes_.size() + mode] = {value.real(), value.imag()};
            }
    }
    void executeNativeSequence(const std::vector<Complex>& compact, bool gatherOutput) {
        for (std::size_t group = 0; group < kernels_.size(); ++group) {
            auto& kernel = *kernels_[group];
            load(kernel, compact);
            kernel.convolution->convolve(kernel.arrays);
            if (gatherOutput) gatherGroup(kernel, group * 4);
        }
    }
    void execute(const std::vector<Complex>& compact) { executeNativeSequence(compact, true); }
    const std::vector<Complex>& output() const { return output_; }
    std::uint64_t residentBytes() const {
        std::uint64_t bytes = output_.size() * sizeof(Complex);
        for (const auto& kernel : kernels_) {
            const auto caller = static_cast<std::uint64_t>(
                                    kernel->arrays[count_ - 1] - kernel->arrays[0]) +
                                length_ * half_;
            const auto internal = count_ * kernel->fftx->outputSize() + kernel->fftx->workSizeW() +
                                  4 * kernel->fftx->workSizeV();
            bytes += (caller + internal) * sizeof(::Complex);
        }
        return bytes;
    }
    bool nativeInPlace() const {
        return std::all_of(kernels_.begin(), kernels_.end(), [](const auto& kernel) {
            return kernel->fftx->inplace && kernel->ffty->inplace;
        });
    }
    std::string optimizerParameters() const {
        std::string result;
        for (std::size_t group = 0; group < kernels_.size(); ++group) {
            auto describe = [](auto& fft) {
                return "m=" + std::to_string(fft.m) +
                       ",p=" + std::to_string(fft.p) +
                       ",q=" + std::to_string(fft.q) +
                       ",n=" + std::to_string(fft.n) +
                       ",D=" + std::to_string(fft.D) +
                       ",loops=" + std::to_string(fft.nloops()) +
                       ",loop2=" + std::string(fft.loop2() ? "true" : "false");
            };
            if (!result.empty()) result += "; ";
            result += "group" + std::to_string(group) + " x{" +
                      describe(*kernels_[group]->fftx) + "} y{" +
                      describe(*kernels_[group]->ffty) + "}";
        }
        return result;
    }
private:
    struct Kernel {
        Kernel(std::size_t n, std::size_t length, std::size_t half,
               fftwpp::multiplier* multiplier) {
            arrays = utils::ComplexAlign(4, length * half);
            if (!arrays) throw std::bad_alloc();
            appx = std::make_unique<fftwpp::Application>(4, 4, fftwpp::multNone, 1);
            fftx = std::make_unique<fftwpp::fftPadCentered>(length, n, *appx, half);
            appy = std::make_unique<fftwpp::Application>(4, 4, multiplier, *appx);
            ffty = std::make_unique<fftwpp::fftPadHermitian>(length, n, *appy);
            convolution = std::make_unique<fftwpp::Convolution2>(fftx.get(), ffty.get());
        }
        ~Kernel() {
            convolution.reset();
            ffty.reset();
            fftx.reset();
            appy.reset();
            appx.reset();
            if (arrays) { utils::deleteAlign(arrays[0]); delete[] arrays; }
        }
        ::Complex** arrays = nullptr;
        std::unique_ptr<fftwpp::Application> appx, appy;
        std::unique_ptr<fftwpp::fftPadCentered> fftx;
        std::unique_ptr<fftwpp::fftPadHermitian> ffty;
        std::unique_ptr<fftwpp::Convolution2> convolution;
    };
    std::size_t n_, products_;
    const std::vector<StoredMode>& modes_;
    std::size_t radius_, length_, half_, count_;
    std::vector<Complex> output_;
    std::vector<std::unique_ptr<Kernel>> kernels_;
};

class ImplicitAdvectiveConvolution {
    struct Kernel;
public:
    enum class Topology { streamedTarget, allTargets };

    ImplicitAdvectiveConvolution(std::size_t n,
                                 const std::vector<StoredMode>& modes,
                                 Topology topology,
                                 std::size_t centeredM = 0,
                                 bool storeOutput = true)
        : n_(n), modes_(modes), topology_(topology), radius_(n / 3),
          length_(2 * radius_ + 1), half_(radius_ + 1),
          inputs_(topology == Topology::streamedTarget ? 6 : 15),
          outputs_(topology == Topology::streamedTarget ? 1 : 4),
          forcedCenteredM_(topology == Topology::allTargets
                               ? (centeredM != 0 ? centeredM : n)
                               : 0),
          output_((storeOutput ? 4 : 0) * modes.size()) {
        fftwpp::fftw::maxthreads = 1;
        fftwpp::fftw::effort = FFTW_MEASURE | FFTW_UNALIGNED;
        kernel_ = std::make_unique<Kernel>(
            n_, length_, half_, inputs_, outputs_,
            topology == Topology::streamedTarget
                ? issue17AdvectiveStreamed
                : issue17AdvectiveAllTargets,
            forcedCenteredM_);
    }
    void loadArray(std::size_t array, const std::vector<Complex>& compact,
                   std::size_t field) {
        std::fill(kernel_->arrays[array],
                  kernel_->arrays[array] + length_ * half_,
                  ::Complex(0.0, 0.0));
        for (std::size_t mode = 0; mode < modes_.size(); ++mode) {
            const auto value = compact[field * modes_.size() + mode];
            kernel_->arrays[array][modes_[mode].rectangleIndex] =
                ::Complex(value.real, value.imag);
        }
    }
    void loadStreamedTarget(const std::vector<Complex>& compact,
                            std::size_t target) {
        for (std::size_t field = 0; field < 3; ++field)
            loadArray(field, compact, field);
        for (std::size_t derivative = 0; derivative < 3; ++derivative)
            loadArray(3 + derivative, compact, 3 + 3 * target + derivative);
    }
    void loadAllTargets(const std::vector<Complex>& compact) {
        for (std::size_t field = 0; field < 15; ++field)
            loadArray(field, compact, field);
    }
    void gatherArrayTo(std::size_t array, Complex* destination) {
        for (std::size_t mode = 0; mode < modes_.size(); ++mode) {
            const auto value = kernel_->arrays[array][modes_[mode].rectangleIndex];
            destination[mode] = {value.real(), value.imag()};
        }
    }
    void gatherArray(std::size_t array, std::size_t target) {
        gatherArrayTo(array, output_.data() + target * modes_.size());
    }
    void executeOneStreamedTarget(const std::vector<Complex>& compact,
                                  std::size_t target,
                                  Complex* destination) {
        if (topology_ != Topology::streamedTarget)
            throw std::logic_error("executeOneStreamedTarget requires the streamed topology.");
        loadStreamedTarget(compact, target);
        kernel_->convolution->convolve(kernel_->arrays);
        if (destination != nullptr) gatherArrayTo(0, destination);
    }
    void executeNativeSequence(const std::vector<Complex>& compact,
                               bool gatherOutput) {
        if (topology_ == Topology::allTargets) {
            loadAllTargets(compact);
            kernel_->convolution->convolve(kernel_->arrays);
            if (gatherOutput)
                for (std::size_t target = 0; target < 4; ++target)
                    gatherArray(target, target);
            return;
        }
        for (std::size_t target = 0; target < 4; ++target) {
            loadStreamedTarget(compact, target);
            kernel_->convolution->convolve(kernel_->arrays);
            if (gatherOutput) gatherArray(0, target);
        }
    }
    void execute(const std::vector<Complex>& compact) {
        executeNativeSequence(compact, true);
    }
    const std::vector<Complex>& output() const { return output_; }
    std::uint64_t residentBytes() const {
        const auto caller = static_cast<std::uint64_t>(
                                kernel_->arrays[inputs_ - 1] - kernel_->arrays[0]) +
                            length_ * half_;
        const auto internal = inputs_ * kernel_->fftx->outputSize() +
                              kernel_->fftx->workSizeW() +
                              outputs_ * kernel_->fftx->workSizeV();
        return (caller + internal) * sizeof(::Complex) +
               output_.size() * sizeof(Complex);
    }
    bool nativeInPlace() const {
        return kernel_->fftx->inplace && kernel_->ffty->inplace;
    }
    std::string optimizerParameters() const {
        const auto describe = [](auto& fft) {
            return "m=" + std::to_string(fft.m) +
                   ",p=" + std::to_string(fft.p) +
                   ",q=" + std::to_string(fft.q) +
                   ",n=" + std::to_string(fft.n) +
                   ",D=" + std::to_string(fft.D) +
                   ",loops=" + std::to_string(fft.nloops()) +
                   ",loop2=" + std::string(fft.loop2() ? "true" : "false");
        };
        return "x{" + describe(*kernel_->fftx) + "}; y{" +
               describe(*kernel_->ffty) + "}";
    }
    Topology topology() const { return topology_; }
    std::size_t forcedCenteredM() const { return forcedCenteredM_; }
private:
    struct Kernel {
        Kernel(std::size_t n, std::size_t length, std::size_t half,
               std::size_t inputs, std::size_t outputs,
               fftwpp::multiplier* multiplier,
               std::size_t forcedCenteredM) {
            arrays = utils::ComplexAlign(std::max(inputs, outputs), length * half);
            if (!arrays) throw std::bad_alloc();
            appx = std::make_unique<fftwpp::Application>(inputs, outputs, fftwpp::multNone, 1);
            if (forcedCenteredM == 0) {
                fftx = std::make_unique<fftwpp::fftPadCentered>(length, n, *appx, half);
            } else {
                fftx = std::make_unique<fftwpp::fftPadCentered>(
                    length, n, *appx, half, half, forcedCenteredM, 1, false);
            }
            appy = std::make_unique<fftwpp::Application>(inputs, outputs, multiplier, *appx);
            ffty = std::make_unique<fftwpp::fftPadHermitian>(length, n, *appy);
            convolution = std::make_unique<fftwpp::Convolution2>(fftx.get(), ffty.get());
        }
        ~Kernel() {
            convolution.reset();
            ffty.reset();
            fftx.reset();
            appy.reset();
            appx.reset();
            if (arrays) { utils::deleteAlign(arrays[0]); delete[] arrays; }
        }
        ::Complex** arrays = nullptr;
        std::unique_ptr<fftwpp::Application> appx, appy;
        std::unique_ptr<fftwpp::fftPadCentered> fftx;
        std::unique_ptr<fftwpp::fftPadHermitian> ffty;
        std::unique_ptr<fftwpp::Convolution2> convolution;
    };
    std::size_t n_;
    const std::vector<StoredMode>& modes_;
    Topology topology_;
    std::size_t radius_, length_, half_, inputs_, outputs_, forcedCenteredM_;
    std::vector<Complex> output_;
    std::unique_ptr<Kernel> kernel_;
};

class ImplicitParallelAdvectiveConvolution {
public:
    ImplicitParallelAdvectiveConvolution(
        std::size_t n, const std::vector<StoredMode>& modes)
        : modes_(modes), output_(4 * modes.size()), executor_(4) {
        for (std::size_t target = 0; target < 4; ++target)
            workers_.push_back(std::make_unique<ImplicitAdvectiveConvolution>(
                n, modes, ImplicitAdvectiveConvolution::Topology::streamedTarget,
                0, false));
    }
    void execute(const std::vector<Complex>& compact) {
        executeNativeSequence(compact, true);
    }
    void executeNativeSequence(const std::vector<Complex>& compact,
                               bool gatherOutput) {
        Context context{this, &compact, gatherOutput};
        executor_.run(&executeTargetTask, &context);
    }
    const std::vector<Complex>& output() const { return output_; }
    std::uint64_t residentBytes() const {
        std::uint64_t bytes = output_.size() * sizeof(Complex) +
                              executor_.persistentBytes();
        for (const auto& worker : workers_) bytes += worker->residentBytes();
        return bytes;
    }
    bool nativeInPlace() const {
        return std::all_of(workers_.begin(), workers_.end(), [](const auto& worker) {
            return worker->nativeInPlace();
        });
    }
    std::string optimizerParameters() const {
        return workers_.front()->optimizerParameters();
    }
private:
    struct Context {
        ImplicitParallelAdvectiveConvolution* self;
        const std::vector<Complex>* compact;
        bool gatherOutput;
    };
    static void executeTargetTask(void* raw, std::size_t target) {
        auto& context = *static_cast<Context*>(raw);
        context.self->workers_[target]->executeOneStreamedTarget(
            *context.compact, target,
            context.gatherOutput
                ? context.self->output_.data() + target * context.self->modes_.size()
                : nullptr);
    }

    const std::vector<StoredMode>& modes_;
    std::vector<std::unique_ptr<ImplicitAdvectiveConvolution>> workers_;
    std::vector<Complex> output_;
    PersistentTaskExecutor executor_;
};
#endif

CorrectnessMetric correctness(const std::vector<Complex>& actual,
                              const std::vector<Complex>& expected) {
    const auto maximum = maximumRelativeError(actual.data(), expected.data(), actual.size());
    const auto l2 = relativeL2Error(actual.data(), expected.data(), actual.size());
    return {"mode-keyed retained convolution versus explicit FFTW oracle", maximum,
            tolerance, maximum <= tolerance && l2 <= tolerance, l2};
}

CorrectnessMetric correctnessGroup(const std::vector<Complex>& actual,
                                   const std::vector<Complex>& expected,
                                   std::size_t modes, std::size_t firstProduct) {
    const auto count = std::min<std::size_t>(4, actual.size() / modes - firstProduct) * modes;
    const auto* actualStart = actual.data() + firstProduct * modes;
    const auto* expectedStart = expected.data() + firstProduct * modes;
    const auto maximum = maximumRelativeError(actualStart, expectedStart, count);
    const auto l2 = relativeL2Error(actualStart, expectedStart, count);
    return {"product group " + std::to_string(firstProduct / 4 + 1) + " versus explicit oracle",
            maximum, tolerance, maximum <= tolerance && l2 <= tolerance, l2};
}

CorrectnessMetric correctnessTarget(const std::vector<Complex>& actual,
                                    const std::vector<Complex>& expected,
                                    std::size_t modes, std::size_t target) {
    const auto* actualStart = actual.data() + target * modes;
    const auto* expectedStart = expected.data() + target * modes;
    const auto maximum = maximumRelativeError(actualStart, expectedStart, modes);
    const auto l2 = relativeL2Error(actualStart, expectedStart, modes);
    return {"advective target " + std::to_string(target) + " versus explicit oracle",
            maximum, tolerance, maximum <= tolerance && l2 <= tolerance, l2};
}

struct ComposedModeSet {
    std::vector<RetainedMode> logical;
    std::vector<StoredMode> stored;
};

ComposedModeSet composedModeSet(std::size_t n) {
    auto stored = storedDisk(n);
    std::stable_sort(stored.begin(), stored.end(), [](const auto& first,
                                                       const auto& second) {
        const auto firstSquared = first.k * first.k + first.l * first.l;
        const auto secondSquared = second.k * second.k + second.l * second.l;
        if (firstSquared != secondSquared) return firstSquared < secondSquared;
        if (first.k != second.k) return first.k < second.k;
        return first.l < second.l;
    });

    std::vector<RetainedMode> logical;
    logical.reserve(stored.size());
    for (const auto& mode : stored) {
        const auto storedKx = mode.k < 0
            ? n - static_cast<std::size_t>(-mode.k)
            : static_cast<std::size_t>(mode.k);
        logical.push_back({
            mode.k, mode.l, storedKx, static_cast<std::size_t>(mode.l),
            false, std::hypot(static_cast<double>(mode.k),
                              static_cast<double>(mode.l))});
    }
    return {std::move(logical), std::move(stored)};
}

std::vector<Complex> composedModalFixture(
    const Workload& workload, const std::vector<StoredMode>& modes,
    std::uint64_t seed) {
    const auto radius = static_cast<int>(workload.nx / 3);
    const auto missing = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> zeroLineByK(
        static_cast<std::size_t>(2 * radius + 1), missing);
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
        if (modes[mode].l == 0) {
            zeroLineByK[static_cast<std::size_t>(modes[mode].k + radius)] = mode;
        }
    }

    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::vector<Complex> values(
        modes.size() * workload.retainedVerticalModes() * workload.fields);
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
        const auto& key = modes[mode];
        if (key.l == 0 && key.k < 0) continue;
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t j = 0; j < workload.retainedVerticalModes(); ++j) {
                Complex value{distribution(generator), distribution(generator)};
                if (key.k == 0 && key.l == 0) value.imag = 0.0;
                values[modalSpectrumIndex(workload, mode, j, field)] = value;
                if (key.l == 0 && key.k > 0) {
                    const auto conjugateMode = zeroLineByK[
                        static_cast<std::size_t>(-key.k + radius)];
                    if (conjugateMode == missing) {
                        throw std::logic_error(
                            "Composed modal fixture lacks a Hermitian boundary partner.");
                    }
                    values[modalSpectrumIndex(
                        workload, conjugateMode, j, field)] = conjugate(value);
                }
            }
        }
    }
    return values;
}

void packInterleavedLevel(const Workload& workload, std::size_t modeCount,
                          std::size_t z, const Complex* physical,
                          std::vector<Complex>& level) {
    for (std::size_t field = 0; field < workload.fields; ++field) {
        for (std::size_t mode = 0; mode < modeCount; ++mode) {
            level[field * modeCount + mode] = physical[
                retainedSpectrumIndex(workload, mode, z, field)];
        }
    }
}

void packSplitLevel(const Workload& workload, std::size_t modeCount,
                    std::size_t z, const VerticalGemmProvider& provider,
                    std::vector<Complex>& level) {
    const auto* real = provider.splitPhysicalOutputRealData();
    const auto* imaginary = provider.splitPhysicalOutputImaginaryData();
    for (std::size_t field = 0; field < workload.fields; ++field) {
        for (std::size_t mode = 0; mode < modeCount; ++mode) {
            const auto index = retainedSpectrumIndex(workload, mode, z, field);
            level[field * modeCount + mode] = {real[index], imaginary[index]};
        }
    }
}

void scatterInterleavedLevel(const Workload& workload, std::size_t modeCount,
                             std::size_t z, const std::vector<Complex>& level,
                             Complex* physical) {
    for (std::size_t field = 0; field < workload.fields; ++field) {
        for (std::size_t mode = 0; mode < modeCount; ++mode) {
            physical[retainedSpectrumIndex(workload, mode, z, field)] =
                level[field * modeCount + mode];
        }
    }
}

void scatterSplitLevel(const Workload& workload, std::size_t modeCount,
                       std::size_t z, const std::vector<Complex>& level,
                       VerticalGemmProvider& provider) {
    auto* real = provider.splitPhysicalInputRealData();
    auto* imaginary = provider.splitPhysicalInputImaginaryData();
    for (std::size_t field = 0; field < workload.fields; ++field) {
        for (std::size_t mode = 0; mode < modeCount; ++mode) {
            const auto index = retainedSpectrumIndex(workload, mode, z, field);
            const auto value = level[field * modeCount + mode];
            real[index] = value.real;
            imaginary[index] = value.imag;
        }
    }
}

template <class HorizontalPath>
void executeComposedSplit(
    const Workload& inputWorkload, const Workload& outputWorkload,
    std::size_t modeCount, VerticalGemmProvider& reconstruction,
    HorizontalPath& horizontal, VerticalGemmProvider& projection,
    std::vector<Complex>& levelInput) {
    reconstruction.executeInverse();
    for (std::size_t z = 0; z < inputWorkload.nz; ++z) {
        packSplitLevel(
            inputWorkload, modeCount, z, reconstruction, levelInput);
        horizontal.execute(levelInput);
        scatterSplitLevel(
            outputWorkload, modeCount, z, horizontal.output(), projection);
    }
    projection.executeForward();
}

CorrectnessMetric splitMetric(
    std::string name, const double* actualReal, const double* actualImaginary,
    const Complex* expected, std::size_t count) {
    double maximumDifference = 0.0;
    double maximumReference = 0.0;
    long double squaredDifference = 0.0;
    long double squaredReference = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const Complex difference{
            actualReal[index] - expected[index].real,
            actualImaginary[index] - expected[index].imag};
        maximumDifference = std::max(maximumDifference, magnitude(difference));
        maximumReference = std::max(maximumReference, magnitude(expected[index]));
        squaredDifference +=
            static_cast<long double>(difference.real) * difference.real +
            static_cast<long double>(difference.imag) * difference.imag;
        squaredReference +=
            static_cast<long double>(expected[index].real) * expected[index].real +
            static_cast<long double>(expected[index].imag) * expected[index].imag;
    }
    const double maximum = maximumDifference / std::max(maximumReference, 1.0);
    const double l2 = squaredReference == 0.0
        ? std::sqrt(static_cast<double>(squaredDifference))
        : std::sqrt(static_cast<double>(squaredDifference / squaredReference));
    return {std::move(name), maximum, tolerance,
            maximum <= tolerance && l2 <= tolerance, l2};
}

CorrectnessMetric splitTargetMetric(
    std::size_t target, const Workload& workload, std::size_t modeCount,
    const double* actualReal, const double* actualImaginary,
    const std::vector<Complex>& expected) {
    double maximumDifference = 0.0;
    double maximumReference = 0.0;
    long double squaredDifference = 0.0;
    long double squaredReference = 0.0;
    for (std::size_t mode = 0; mode < modeCount; ++mode) {
        for (std::size_t j = 0; j < workload.retainedVerticalModes(); ++j) {
            const auto index = modalSpectrumIndex(workload, mode, j, target);
            const Complex difference{
                actualReal[index] - expected[index].real,
                actualImaginary[index] - expected[index].imag};
            maximumDifference = std::max(maximumDifference, magnitude(difference));
            maximumReference = std::max(maximumReference, magnitude(expected[index]));
            squaredDifference +=
                static_cast<long double>(difference.real) * difference.real +
                static_cast<long double>(difference.imag) * difference.imag;
            squaredReference +=
                static_cast<long double>(expected[index].real) * expected[index].real +
                static_cast<long double>(expected[index].imag) * expected[index].imag;
        }
    }
    const double maximum = maximumDifference / std::max(maximumReference, 1.0);
    const double l2 = squaredReference == 0.0
        ? std::sqrt(static_cast<double>(squaredDifference))
        : std::sqrt(static_cast<double>(squaredDifference / squaredReference));
    return {"projected advective target " + std::to_string(target) +
                " versus composed oracle",
            maximum, tolerance, maximum <= tolerance && l2 <= tolerance, l2};
}

CorrectnessMetric splitProbeMetric(
    const std::vector<std::size_t>& indices,
    const std::vector<Complex>& expected,
    const VerticalGemmProvider& reconstruction) {
    const auto* real = reconstruction.splitPhysicalOutputRealData();
    const auto* imaginary = reconstruction.splitPhysicalOutputImaginaryData();
    std::vector<Complex> actual(indices.size());
    for (std::size_t probe = 0; probe < indices.size(); ++probe) {
        actual[probe] = {real[indices[probe]], imaginary[indices[probe]]};
    }
    const auto maximum = maximumRelativeError(
        actual.data(), expected.data(), actual.size());
    const auto l2 = relativeL2Error(actual.data(), expected.data(), actual.size());
    return {"inverse vertical reconstruction probes versus independent complex GEMM",
            maximum, tolerance, maximum <= tolerance && l2 <= tolerance, l2};
}

CorrectnessMetric splitHermitianBoundaryMetric(
    const Workload& workload, const std::vector<StoredMode>& modes,
    const double* real, const double* imaginary) {
    const auto radius = static_cast<int>(workload.nx / 3);
    const auto missing = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> zeroLineByK(
        static_cast<std::size_t>(2 * radius + 1), missing);
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
        if (modes[mode].l == 0)
            zeroLineByK[static_cast<std::size_t>(modes[mode].k + radius)] = mode;
    }
    double maximumDifference = 0.0;
    double maximumReference = 0.0;
    long double squaredDifference = 0.0;
    long double squaredReference = 0.0;
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
        if (modes[mode].l != 0 || modes[mode].k < 0) continue;
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t j = 0; j < workload.retainedVerticalModes(); ++j) {
                const auto positive = modalSpectrumIndex(
                    workload, mode, j, field);
                Complex difference;
                Complex reference{real[positive], imaginary[positive]};
                if (modes[mode].k == 0) {
                    difference = {0.0, imaginary[positive]};
                } else {
                    const auto negativeMode = zeroLineByK[
                        static_cast<std::size_t>(-modes[mode].k + radius)];
                    if (negativeMode == missing)
                        throw std::logic_error(
                            "Composed output lacks a Hermitian boundary partner.");
                    const auto negative = modalSpectrumIndex(
                        workload, negativeMode, j, field);
                    difference = {
                        real[negative] - reference.real,
                        imaginary[negative] + reference.imag};
                }
                maximumDifference = std::max(
                    maximumDifference, magnitude(difference));
                maximumReference = std::max(
                    maximumReference, magnitude(reference));
                squaredDifference +=
                    static_cast<long double>(difference.real) * difference.real +
                    static_cast<long double>(difference.imag) * difference.imag;
                squaredReference +=
                    static_cast<long double>(reference.real) * reference.real +
                    static_cast<long double>(reference.imag) * reference.imag;
            }
        }
    }
    const double maximum = maximumDifference / std::max(maximumReference, 1.0);
    const double l2 = squaredReference == 0.0
        ? std::sqrt(static_cast<double>(squaredDifference))
        : std::sqrt(static_cast<double>(squaredDifference / squaredReference));
    return {"DC and stored Hermitian boundary constraints",
            maximum, tolerance, maximum <= tolerance && l2 <= tolerance, l2};
}

ExecutionContract convolutionContract(std::size_t n, std::size_t products,
                                      bool nativeInPlace, bool implicit) {
    DirectionExecutionContract forward;
    forward.nativePlacement = nativeInPlace ? "in-place" : "out-of-place";
    forward.adapterPlacement = "out-of-place";
    forward.destroysNativeInput = true;
    forward.adapterPreservesCallerInput = true;
    forward.requiresPreservationCopyForRepeatedExecution = true;
    forward.preservationIncludedInAdapterTiming = true;
    forward.nativeInputRepresentationId = implicit ? "centered-rectangular-hermitian" : "full-fftw-half-spectrum";
    forward.nativeOutputRepresentationId = forward.nativeInputRepresentationId;
    forward.adapterInputRepresentationId = "mode-keyed-radial-hermitian-four-field";
    forward.adapterOutputRepresentationId = "mode-keyed-radial-hermitian-products";
    forward.physicalExtents = "Nx=Ny=" + std::to_string(n) + "; fields=4; products=" + std::to_string(products);
    forward.aliasing = "native execution overwrites provider buffers; the timed complete adapter copies from and preserves the caller-owned compact input";
    DirectionExecutionContract inverse;
    inverse.nativePlacement = "unsupported";
    inverse.adapterPlacement = "unsupported";
    inverse.nativeInputRepresentationId = "not-applicable";
    inverse.nativeOutputRepresentationId = "not-applicable";
    return {std::move(forward), std::move(inverse)};
}

ExecutionContract advectiveConvolutionContract(std::size_t n,
                                               bool nativeInPlace,
                                               bool implicit,
                                               std::string topology) {
    auto contract = convolutionContract(n, 4, nativeInPlace, implicit);
    contract.forward.physicalExtents =
        "Nx=Ny=" + std::to_string(n) +
        "; advectors=3; derivative-fields=12; flux-targets=4; topology=" +
        std::move(topology);
    contract.forward.adapterInputRepresentationId =
        "mode-keyed-radial-hermitian-wvm-advection-inputs-15";
    contract.forward.adapterOutputRepresentationId =
        "mode-keyed-radial-hermitian-wvm-flux-targets-4";
    return contract;
}

BenchmarkReport runWvmAdvectiveFinalistBenchmark(const RunOptions& options) {
    const auto selected = profileNamed(options.profile);
    if (selected.workload.nx != selected.workload.ny)
        throw std::invalid_argument(
            "dealiased-convolution currently requires a square horizontal grid.");
    if (options.convolutionCandidate != "explicit-parallel" &&
        options.convolutionCandidate != "fftwpp-parallel")
        throw std::invalid_argument(
            "convolution-candidate must be all, explicit-parallel, or fftwpp-parallel.");

    const auto n = selected.workload.nx;
    const auto warmups = options.warmups ? options.warmups : 1;
    const auto samples = options.samples ? options.samples : 5;
    const auto modes = storedDisk(n);
    const auto input = compactFixture(modes, 15, options.seed);
    std::vector<Complex> expected;
    {
        ExplicitAdvectiveConvolution oracle(n, modes);
        oracle.execute(input);
        expected = oracle.output();
    }
    fftw_forget_wisdom();

    BenchmarkReport report;
    report.environment = environmentRecord();
    auto id = report.environment.timestampUtc;
    id.erase(std::remove_if(id.begin(), id.end(),
                            [](char c) { return c == '-' || c == ':'; }),
             id.end());
    report.runId = id + "-issue17-ref-n" + std::to_string(n) + "-" +
                   options.convolutionCandidate;
    report.profile = options.profile;
    report.seed = options.seed;
    report.warmups = warmups;
    report.samples = samples;
    report.workload = {n, n, 1, 4, 1.0, 1.0, true};
    report.retainedHorizontalModeCount = modes.size();
    report.retainedModeOrderHash =
        "centered-k-then-nonnegative-l-radial-disk-v1";
    report.fullRealBytes = n * n * 15 * sizeof(double);
    report.fullSpectrumBytes =
        n * (n / 2 + 1) * 15 * sizeof(Complex);
    report.retainedSpectrumBytes = modes.size() * 15 * sizeof(Complex);
    const auto harnessBytes =
        (input.size() + expected.size()) * sizeof(Complex);

    if (options.convolutionCandidate == "explicit-parallel") {
        const auto setupStart = Clock::now();
        ExplicitParallelAdvectiveConvolution path(n, modes);
        const auto setupSeconds =
            std::chrono::duration<double>(Clock::now() - setupStart).count();
        path.execute(input);
        const auto metric = correctness(path.output(), expected);

        ProviderRecord provider;
        provider.id = "fftw-explicit-parallel-target-wvm-advection";
        provider.version = "3.3.11";
        provider.libraryIdentity = "FFTW 3.3.11 pthread build";
        provider.algorithmId =
            "explicit-full-grid-shared-advectors-parallel-target-wvm-advection-15to4";
        provider.nativeRepresentationId = "full-fftw-half-spectrum";
        provider.modeOrderId = report.retainedModeOrderHash;
        provider.schedulingId =
            "persistent-outer-static-4-targets-shared-advectors";
        provider.workers = 4;
        provider.internalWorkers = 1;
        provider.outerWorkers = 4;
        provider.planningConfiguration = "FFTW_MEASURE|FFTW_UNALIGNED|cold";
        provider.execution = advectiveConvolutionContract(
            n, false, false,
            "advectors-once-then-four-persistent-parallel-targets");
        provider.explicitPersistentBytes = path.residentBytes();
        provider.algorithmResidentBytes = path.residentBytes();
        provider.benchmarkHarnessBytes = harnessBytes;
        provider.estimatedProcessPeakBytes =
            provider.algorithmResidentBytes + provider.benchmarkHarnessBytes;
        provider.opaqueProviderMemory = true;
        provider.planningSeconds = setupSeconds;
        provider.correctness = {metric};
        for (std::size_t target = 0; target < 4; ++target)
            provider.correctness.push_back(correctnessTarget(
                path.output(), expected, modes.size(), target));
        provider.timings.push_back(timing(
            "adapter-component", "shared advector embedding and inverse FFTs",
            StageState::executed,
            3 * (n * (n / 2 + 1) * sizeof(Complex) +
                 n * n * sizeof(double)),
            timed(warmups, samples, [] {},
                  [&] { path.executeAdvectors(input); })));
        provider.timings.push_back(timing(
            "adapter-component", "parallel target transforms and reduction",
            StageState::executed, path.residentBytes(),
            timed(warmups, samples,
                  [&] { path.executeAdvectors(input); },
                  [&] { path.executeTargets(input); })));
        provider.timings.push_back(timing(
            "uninstrumented-total", "WVM-like four-target horizontal advection",
            StageState::executed, path.residentBytes(),
            timed(warmups, samples, [] {}, [&] { path.execute(input); })));
        provider.ledger = {
            {"setup/planning", StageState::setupOnly,
             "two reusable cold FFTW_MEASURE plans and a persistent four-worker pool"},
            {"shared advector stage", StageState::executed,
             "three compact embeddings and inverse FFTs execute once"},
            {"target stage", StageState::executed,
             "four derivative transform-reduce-forward-retain tasks execute concurrently"},
            {"scheduler", StageState::executed,
             "persistent outer-static four-target dispatch is included"},
            {"oracle", StageState::setupOnly,
             "independent serial explicit FFTW oracle is destroyed before candidate setup"},
            {"complete nonlinear WVM flux", StageState::unsupported,
             "vertical transforms, phase, and coefficient projection remain excluded"}};
        provider.observedProcessHighWaterBytes = processHighWaterBytes();
        report.status = metric.passed ? "passed" : "failed";
        report.providers = {std::move(provider)};
        return report;
    }

    const auto setupStart = Clock::now();
    ImplicitParallelAdvectiveConvolution path(n, modes);
    const auto setupSeconds =
        std::chrono::duration<double>(Clock::now() - setupStart).count();
    path.execute(input);
    const auto metric = correctness(path.output(), expected);

    ProviderRecord provider;
    provider.id = "fftwpp-parallel-target-wvm-advection";
    provider.version = "3.04";
    provider.libraryIdentity =
        "FFTW++ public master pinned at " SKBENCH_FFTWPP_COMMIT;
    provider.algorithmId =
        "hybrid-centered-hermitian-parallel-target-wvm-advection-6to1x4";
    provider.nativeRepresentationId = "centered-rectangular-hermitian";
    provider.modeOrderId = report.retainedModeOrderHash;
    provider.schedulingId =
        "persistent-outer-static-4-target-applications";
    provider.sourceIdentity =
        "https://github.com/dealias/fftwpp/commit/" SKBENCH_FFTWPP_COMMIT;
    provider.configureFlags =
        "FFTWPP_SINGLE_THREAD=1; " + path.optimizerParameters();
    provider.workers = 4;
    provider.internalWorkers = 1;
    provider.outerWorkers = 4;
    provider.planningConfiguration = "FFTW_MEASURE|FFTW_UNALIGNED|cold";
    provider.execution = advectiveConvolutionContract(
        n, path.nativeInPlace(), true,
        "four-independent-6-input-1-output-applications");
    provider.explicitPersistentBytes = path.residentBytes();
    provider.algorithmResidentBytes = path.residentBytes();
    provider.benchmarkHarnessBytes = harnessBytes;
    provider.estimatedProcessPeakBytes =
        provider.algorithmResidentBytes + provider.benchmarkHarnessBytes;
    provider.opaqueProviderMemory = true;
    provider.planningSeconds = setupSeconds;
    provider.correctness = {metric};
    for (std::size_t target = 0; target < 4; ++target)
        provider.correctness.push_back(correctnessTarget(
            path.output(), expected, modes.size(), target));
    provider.timings.push_back(timing(
        "fused-primitive", "implicit hybrid transform-reduce-transform",
        StageState::fused, path.residentBytes(),
        timed(warmups, samples, [] {},
              [&] { path.executeNativeSequence(input, false); })));
    provider.timings.push_back(timing(
        "uninstrumented-total", "WVM-like four-target horizontal advection",
        StageState::executed, path.residentBytes(),
        timed(warmups, samples, [] {}, [&] { path.execute(input); })));
    provider.ledger = {
        {"setup/planning", StageState::setupOnly,
         "four persistent FFTW++ applications with cold reusable FFTW plans"},
        {"input embedding", StageState::executed,
         "each target receives three shared advectors and three target derivatives"},
        {"inverse FFTs", StageState::fused,
         "four concurrent six-inverse/one-forward implicit-hybrid applications"},
        {"advective products", StageState::fused,
         "four concurrent negative three-term dot products"},
        {"forward FFTs", StageState::fused,
         "inseparable FFTW++ implicit-hybrid stage"},
        {"radial retention", StageState::executed,
         "each worker writes one disjoint compact target"},
        {"scheduler", StageState::executed,
         "persistent outer-static four-target dispatch is included"},
        {"oracle", StageState::setupOnly,
         "independent serial explicit FFTW oracle is destroyed before candidate setup"},
        {"complete nonlinear WVM flux", StageState::unsupported,
         "vertical transforms, phase, and coefficient projection remain excluded"}};
    provider.observedProcessHighWaterBytes = processHighWaterBytes();
    report.status = metric.passed ? "passed" : "failed";
    report.providers = {std::move(provider)};
    return report;
}

BenchmarkReport runWvmAdvectiveConvolutionBenchmark(const RunOptions& options) {
    const auto selected = profileNamed(options.profile);
    if (selected.workload.nx != selected.workload.ny)
        throw std::invalid_argument(
            "dealiased-convolution currently requires a square horizontal grid.");
    const auto n = selected.workload.nx;
    const auto warmups = options.warmups ? options.warmups : 1;
    const auto samples = options.samples ? options.samples : 5;
    const auto modes = storedDisk(n);
    constexpr std::size_t inputFields = 15;
    const auto input = compactFixture(modes, inputFields, options.seed);

    const auto explicitSetupStart = Clock::now();
    ExplicitAdvectiveConvolution explicitPath(n, modes);
    const auto explicitSetup =
        std::chrono::duration<double>(Clock::now() - explicitSetupStart).count();
    const auto explicitParallelSetupStart = Clock::now();
    ExplicitParallelAdvectiveConvolution explicitParallelPath(n, modes);
    const auto explicitParallelSetup =
        std::chrono::duration<double>(Clock::now() - explicitParallelSetupStart).count();
    const auto streamedSetupStart = Clock::now();
    ImplicitAdvectiveConvolution streamedPath(
        n, modes, ImplicitAdvectiveConvolution::Topology::streamedTarget);
    const auto streamedSetup =
        std::chrono::duration<double>(Clock::now() - streamedSetupStart).count();
    const auto allTargetSetupStart = Clock::now();
    ImplicitAdvectiveConvolution allTargetPath(
        n, modes, ImplicitAdvectiveConvolution::Topology::allTargets,
        options.convolutionCenteredM);
    const auto allTargetSetup =
        std::chrono::duration<double>(Clock::now() - allTargetSetupStart).count();
    const auto implicitParallelSetupStart = Clock::now();
    ImplicitParallelAdvectiveConvolution implicitParallelPath(n, modes);
    const auto implicitParallelSetup =
        std::chrono::duration<double>(Clock::now() - implicitParallelSetupStart).count();

    explicitPath.execute(input);
    explicitParallelPath.execute(input);
    streamedPath.execute(input);
    allTargetPath.execute(input);
    implicitParallelPath.execute(input);
    const auto explicitParallelCorrectness =
        correctness(explicitParallelPath.output(), explicitPath.output());
    const auto streamedCorrectness =
        correctness(streamedPath.output(), explicitPath.output());
    const auto allTargetCorrectness =
        correctness(allTargetPath.output(), explicitPath.output());
    const auto implicitParallelCorrectness =
        correctness(implicitParallelPath.output(), explicitPath.output());

    BenchmarkReport report;
    report.status = explicitParallelCorrectness.passed &&
                            streamedCorrectness.passed &&
                            allTargetCorrectness.passed &&
                            implicitParallelCorrectness.passed
                        ? "passed"
                        : "failed";
    report.environment = environmentRecord();
    auto id = report.environment.timestampUtc;
    id.erase(std::remove_if(id.begin(), id.end(),
                            [](char c) { return c == '-' || c == ':'; }),
             id.end());
    report.runId = id + "-issue17-n" + std::to_string(n) + "-wvmadv-m" +
                   std::to_string(allTargetPath.forcedCenteredM());
    report.profile = options.profile;
    report.seed = options.seed;
    report.warmups = warmups;
    report.samples = samples;
    report.workload = {n, n, 1, 4, 1.0, 1.0, true};
    report.retainedHorizontalModeCount = modes.size();
    report.retainedModeOrderHash =
        "centered-k-then-nonnegative-l-radial-disk-v1";
    report.fullRealBytes = n * n * inputFields * sizeof(double);
    report.fullSpectrumBytes =
        n * (n / 2 + 1) * inputFields * sizeof(Complex);
    report.retainedSpectrumBytes =
        modes.size() * inputFields * sizeof(Complex);

    ProviderRecord baseline;
    baseline.id = "fftw-explicit-streamed-wvm-advection";
    baseline.version = "3.3.11";
    baseline.libraryIdentity = "FFTW 3.3.11 pthread build";
    baseline.algorithmId =
        "explicit-full-grid-streamed-target-wvm-advection-15to4";
    baseline.nativeRepresentationId = "full-fftw-half-spectrum";
    baseline.modeOrderId = report.retainedModeOrderHash;
    baseline.schedulingId = "single-thread-streamed-target";
    baseline.planningConfiguration = "FFTW_MEASURE|FFTW_UNALIGNED";
    baseline.execution = advectiveConvolutionContract(
        n, false, false, "advectors-once-then-four-streamed-targets");
    baseline.explicitPersistentBytes = explicitPath.residentBytes();
    baseline.algorithmResidentBytes = explicitPath.residentBytes();
    baseline.opaqueProviderMemory = true;
    baseline.planningSeconds = explicitSetup;
    baseline.correctness = {
        {"independent explicit FFTW oracle baseline", 0.0, tolerance, true, 0.0}};
    baseline.timings.push_back(timing(
        "primitive-component", "advecting inverse FFT batch (3 fields)",
        StageState::executed,
        3 * (n * (n / 2 + 1) * sizeof(Complex) + n * n * sizeof(double)),
        timed(warmups, samples,
              [&] { explicitPath.loadAdvectors(input); },
              [&] { explicitPath.inverseAdvectors(); })));
    baseline.timings.push_back(timing(
        "primitive-component", "one target derivative inverse FFT batch (3 fields)",
        StageState::executed,
        3 * (n * (n / 2 + 1) * sizeof(Complex) + n * n * sizeof(double)),
        timed(warmups, samples,
              [&] { explicitPath.loadDerivatives(input, 0); },
              [&] { explicitPath.inverseDerivatives(); })));
    baseline.timings.push_back(timing(
        "primitive-component", "one target advective reduction",
        StageState::executed, 7 * n * n * sizeof(double),
        timed(warmups, samples,
              [&] {
                  explicitPath.loadAdvectors(input);
                  explicitPath.inverseAdvectors();
                  explicitPath.loadDerivatives(input, 0);
                  explicitPath.inverseDerivatives();
              },
              [&] { explicitPath.multiplyTarget(); })));
    baseline.timings.push_back(timing(
        "primitive-component", "one target forward FFT",
        StageState::executed,
        n * n * sizeof(double) + n * (n / 2 + 1) * sizeof(Complex),
        timed(warmups, samples,
              [&] {
                  explicitPath.loadAdvectors(input);
                  explicitPath.inverseAdvectors();
                  explicitPath.loadDerivatives(input, 0);
                  explicitPath.inverseDerivatives();
                  explicitPath.multiplyTarget();
              },
              [&] { explicitPath.forwardTarget(); })));
    baseline.timings.push_back(timing(
        "uninstrumented-total", "WVM-like four-target horizontal advection",
        StageState::executed, explicitPath.residentBytes(),
        timed(warmups, samples, [] {}, [&] { explicitPath.execute(input); })));
    baseline.ledger = {
        {"setup/planning", StageState::setupOnly,
         "two reusable FFTW_MEASURE plans"},
        {"advecting inverse FFTs", StageState::executed,
         "three fields transformed once"},
        {"target derivative inverse FFTs", StageState::executed,
         "three derivative fields for each of four streamed targets"},
        {"advective products", StageState::executed,
         "four fused negative three-term dot products; 12 multiplies and 8 adds per point"},
        {"forward FFTs", StageState::executed,
         "four scalar flux targets"},
        {"radial retention", StageState::executed,
         "included in the authoritative total"},
        {"vertical reconstruction/projection", StageState::unsupported,
         "derivative inputs are ready horizontal spectra"},
        {"complete nonlinear WVM flux", StageState::unsupported,
         "phase, vertical transforms, and coefficient projection remain outside issue #17"}};

    ProviderRecord explicitParallel;
    explicitParallel.id = "fftw-explicit-parallel-target-wvm-advection";
    explicitParallel.version = "3.3.11";
    explicitParallel.libraryIdentity = "FFTW 3.3.11 pthread build";
    explicitParallel.algorithmId =
        "explicit-full-grid-shared-advectors-parallel-target-wvm-advection-15to4";
    explicitParallel.nativeRepresentationId = "full-fftw-half-spectrum";
    explicitParallel.modeOrderId = report.retainedModeOrderHash;
    explicitParallel.schedulingId =
        "persistent-outer-static-4-targets-shared-advectors";
    explicitParallel.workers = 4;
    explicitParallel.internalWorkers = 1;
    explicitParallel.outerWorkers = 4;
    explicitParallel.planningConfiguration = "FFTW_MEASURE|FFTW_UNALIGNED";
    explicitParallel.execution = advectiveConvolutionContract(
        n, false, false,
        "advectors-once-then-four-persistent-parallel-targets");
    explicitParallel.explicitPersistentBytes = explicitParallelPath.residentBytes();
    explicitParallel.algorithmResidentBytes = explicitParallelPath.residentBytes();
    explicitParallel.opaqueProviderMemory = true;
    explicitParallel.planningSeconds = explicitParallelSetup;
    explicitParallel.correctness = {explicitParallelCorrectness};
    for (std::size_t target = 0; target < 4; ++target)
        explicitParallel.correctness.push_back(correctnessTarget(
            explicitParallelPath.output(), explicitPath.output(),
            modes.size(), target));
    explicitParallel.timings.push_back(timing(
        "uninstrumented-total", "WVM-like four-target horizontal advection",
        StageState::executed, explicitParallelPath.residentBytes(),
        timed(warmups, samples, [] {},
              [&] { explicitParallelPath.execute(input); })));
    explicitParallel.ledger = {
        {"setup/planning", StageState::setupOnly,
         "two reusable FFTW_MEASURE plans and a persistent four-worker pool"},
        {"advecting inverse FFTs", StageState::executed,
         "three fields transformed once before target dispatch"},
        {"target derivative inverse FFTs", StageState::executed,
         "four independent three-field batches execute concurrently"},
        {"advective products", StageState::executed,
         "four concurrent negative three-term dot products"},
        {"forward FFTs", StageState::executed,
         "four scalar targets execute concurrently"},
        {"scheduler", StageState::executed,
         "persistent outer-static four-target dispatch is included in the total"},
        {"radial retention", StageState::executed,
         "each worker writes a disjoint compact target"},
        {"complete nonlinear WVM flux", StageState::unsupported,
         "phase, vertical transforms, and coefficient projection remain outside issue #17"}};

    auto implicitRecord = [&](auto& path,
                              const CorrectnessMetric& metric,
                              double setupSeconds,
                              std::string idValue,
                              std::string algorithm,
                              std::string scheduling,
                              std::string topology,
                              std::string transformDescription) {
        ProviderRecord candidate;
        candidate.id = std::move(idValue);
        candidate.version = "3.04";
        candidate.libraryIdentity =
            "FFTW++ public master pinned at " SKBENCH_FFTWPP_COMMIT;
        candidate.algorithmId = std::move(algorithm);
        candidate.nativeRepresentationId = "centered-rectangular-hermitian";
        candidate.modeOrderId = report.retainedModeOrderHash;
        candidate.schedulingId = std::move(scheduling);
        candidate.sourceIdentity =
            "https://github.com/dealias/fftwpp/commit/" SKBENCH_FFTWPP_COMMIT;
        candidate.configureFlags =
            "FFTWPP_SINGLE_THREAD=1; " + path.optimizerParameters();
        candidate.planningConfiguration = "FFTW_MEASURE|FFTW_UNALIGNED";
        candidate.execution = advectiveConvolutionContract(
            n, path.nativeInPlace(), true, std::move(topology));
        candidate.explicitPersistentBytes = path.residentBytes();
        candidate.algorithmResidentBytes = path.residentBytes();
        candidate.opaqueProviderMemory = false;
        candidate.planningSeconds = setupSeconds;
        candidate.correctness = {metric};
        for (std::size_t target = 0; target < 4; ++target)
            candidate.correctness.push_back(correctnessTarget(
                path.output(), explicitPath.output(), modes.size(), target));
        candidate.timings.push_back(timing(
            "fused-primitive", "implicit hybrid transform-reduce-transform",
            StageState::fused, path.residentBytes(),
            timed(warmups, samples, [] {},
                  [&] { path.executeNativeSequence(input, false); })));
        candidate.timings.push_back(timing(
            "uninstrumented-total", "WVM-like four-target horizontal advection",
            StageState::executed, path.residentBytes(),
            timed(warmups, samples, [] {}, [&] { path.execute(input); })));
        candidate.ledger = {
            {"setup/planning", StageState::setupOnly,
             "FFTW++ optimizer and reusable FFTW plans"},
            {"input embedding", StageState::executed,
             "ready compact retained spectra copied into centered rectangular storage"},
            {"inverse FFTs", StageState::fused, transformDescription},
            {"advective products", StageState::fused,
             "four negative three-term dot products; 12 multiplies and 8 adds per point"},
            {"forward FFTs", StageState::fused, transformDescription},
            {"radial retention", StageState::executed,
             "included in the authoritative total"},
            {"vertical reconstruction/projection", StageState::unsupported,
             "derivative inputs are ready horizontal spectra"},
            {"complete nonlinear WVM flux", StageState::unsupported,
             "phase, vertical transforms, and coefficient projection remain outside issue #17"}};
        return candidate;
    };

    auto streamed = implicitRecord(
        streamedPath, streamedCorrectness, streamedSetup,
        "fftwpp-streamed-target-wvm-advection",
        "hybrid-centered-hermitian-streamed-target-wvm-advection-6to1x4",
        "single-thread-four-sequential-target-applications",
        "one-reused-6-input-1-output-application-four-times",
        "six inverse and one forward transform per target; advectors are recomputed");
    auto allTarget = implicitRecord(
        allTargetPath, allTargetCorrectness, allTargetSetup,
        "fftwpp-all-target-wvm-advection",
        "hybrid-centered-hermitian-all-target-wvm-advection-15to4-mx" +
            std::to_string(allTargetPath.forcedCenteredM()),
        "single-thread-one-all-target-application",
        "one-15-input-4-output-application",
        "15 inverse transforms and four forward transforms in one application");
    auto implicitParallel = implicitRecord(
        implicitParallelPath, implicitParallelCorrectness,
        implicitParallelSetup,
        "fftwpp-parallel-target-wvm-advection",
        "hybrid-centered-hermitian-parallel-target-wvm-advection-6to1x4",
        "persistent-outer-static-4-target-applications",
        "four-independent-6-input-1-output-applications",
        "four six-inverse/one-forward target applications execute concurrently");
    implicitParallel.workers = 4;
    implicitParallel.internalWorkers = 1;
    implicitParallel.outerWorkers = 4;
    implicitParallel.ledger.push_back(
        {"scheduler", StageState::executed,
         "persistent outer-static four-target dispatch is included in the total"});

    report.providers = {
        std::move(baseline), std::move(explicitParallel),
        std::move(streamed), std::move(allTarget),
        std::move(implicitParallel)};
    return report;
}

BenchmarkReport runVerticallyBatchedAdvectionBenchmarkImpl(
    const RunOptions& options) {
    if (options.convolutionCandidate != "explicit-parallel" &&
        options.convolutionCandidate != "fftwpp-parallel") {
        throw std::invalid_argument(
            "vertically-batched-advection requires --convolution-candidate "
            "explicit-parallel or fftwpp-parallel.");
    }
    if (options.verticalGemmFamily != "k2-grouped") {
        throw std::invalid_argument(
            "vertically-batched-advection requires --vertical-gemm-family "
            "k2-grouped.");
    }
    const auto selected = profileNamed(options.profile);
    if (selected.workload.nx != selected.workload.ny) {
        throw std::invalid_argument(
            "vertically-batched-advection requires a square horizontal grid.");
    }
    if (options.workers != 0) {
        throw std::invalid_argument(
            "vertically-batched-advection uses the fixed four-target horizontal "
            "schedule and independent vertical worker controls; omit --workers.");
    }
    const VerticalGemmStrategy verticalStrategy{
        verticalGemmScheduleNamed(options.verticalGemmSchedule),
        options.verticalGemmOuterWorkers};
    if (verticalStrategy.schedule == VerticalGemmSchedule::serial &&
        verticalStrategy.outerWorkers != 1) {
        throw std::invalid_argument(
            "A serial vertical schedule requires one outer worker.");
    }

    const auto n = selected.workload.nx;
    const auto nz = selected.workload.nz;
    const auto warmups = options.warmups == 0 ? 1 : options.warmups;
    const auto samples = options.samples == 0 ? 3 : options.samples;
    if (samples == 0) {
        throw std::invalid_argument(
            "vertically-batched-advection requires at least one sample.");
    }

    Workload outputWorkload = selected.workload;
    outputWorkload.fields = 4;
    Workload inputWorkload = outputWorkload;
    inputWorkload.fields = 15;
    const auto modeSet = composedModeSet(n);
    const auto modeCount = modeSet.stored.size();
    const auto nj = outputWorkload.retainedVerticalModes();
    const auto fixtureStart = Clock::now();
    auto vertical = squaredWavenumberVerticalFixture(
        outputWorkload, modeSet.logical);
    const auto fixtureSeconds =
        std::chrono::duration<double>(Clock::now() - fixtureStart).count();
    const auto inputModal = composedModalFixture(
        inputWorkload, modeSet.stored, options.seed);

    BenchmarkReport report;
    report.environment = environmentRecord();
    auto id = report.environment.timestampUtc;
    id.erase(std::remove_if(
                 id.begin(), id.end(),
                 [](char character) {
                     return character == '-' || character == ':';
                 }),
             id.end());
    report.runId = id + "-issue18-n" + std::to_string(n) + "-nz" +
        std::to_string(nz) + "-" + options.convolutionCandidate;
    report.profile = options.profile;
    report.seed = options.seed;
    report.warmups = warmups;
    report.samples = samples;
    report.workload = outputWorkload;
    report.retainedHorizontalModeCount = modeCount;
    report.retainedModeOrderHash = modeOrderHash(modeSet.logical);
    report.fullRealBytes = static_cast<std::uint64_t>(
        n * n * inputWorkload.fields) * sizeof(double);
    report.fullSpectrumBytes = static_cast<std::uint64_t>(
        n * (n / 2 + 1) * inputWorkload.fields) * sizeof(Complex);
    const auto inputPhysicalCount = modeCount * nz * inputWorkload.fields;
    const auto outputPhysicalCount = modeCount * nz * outputWorkload.fields;
    const auto inputModalCount = modeCount * nj * inputWorkload.fields;
    const auto outputModalCount = modeCount * nj * outputWorkload.fields;
    report.retainedSpectrumBytes = static_cast<std::uint64_t>(
        inputPhysicalCount + outputPhysicalCount) * sizeof(Complex);
    report.modalSpectrumBytes = static_cast<std::uint64_t>(
        inputModalCount + outputModalCount) * sizeof(Complex);
    report.verticalMatrixFamilySourceBytes = static_cast<std::uint64_t>(
        vertical.forward.size() + vertical.inverse.size()) * sizeof(double);
    report.verticalMatrixFamilyId = vertical.id;
    report.verticalGroupCount = vertical.groups.size();
    report.verticalGroupOrderHash = verticalModeGroupHash(vertical.groups);
    std::vector<double> groupModes;
    std::vector<double> groupColumns;
    groupModes.reserve(vertical.groups.size());
    groupColumns.reserve(vertical.groups.size());
    for (const auto& group : vertical.groups) {
        groupModes.push_back(static_cast<double>(group.modeCount));
        groupColumns.push_back(static_cast<double>(
            group.modeCount * inputWorkload.fields));
    }
    report.minimumVerticalGroupModes = static_cast<std::size_t>(
        *std::min_element(groupModes.begin(), groupModes.end()));
    report.medianVerticalGroupModes = median(groupModes);
    report.maximumVerticalGroupModes = static_cast<std::size_t>(
        *std::max_element(groupModes.begin(), groupModes.end()));
    report.minimumVerticalGroupColumns = static_cast<std::size_t>(
        *std::min_element(groupColumns.begin(), groupColumns.end()));
    report.medianVerticalGroupColumns = median(groupColumns);
    report.maximumVerticalGroupColumns = static_cast<std::size_t>(
        *std::max_element(groupColumns.begin(), groupColumns.end()));

    std::vector<Complex> oraclePhysicalOutput(outputPhysicalCount);
    std::vector<Complex> expectedModal(outputModalCount);
    std::vector<std::size_t> reconstructionProbeIndices;
    std::vector<Complex> reconstructionProbeExpected;
    {
        VerticalGemmProvider reconstructionOracle(
            inputWorkload, vertical, VerticalGemmLayout::complexInterleaved,
            {VerticalGemmSchedule::serial, 1},
            VerticalGemmBufferPolicy::inverseOnly);
        VerticalGemmProvider projectionOracle(
            outputWorkload, vertical, VerticalGemmLayout::complexInterleaved,
            {VerticalGemmSchedule::serial, 1},
            VerticalGemmBufferPolicy::forwardOnly);
        if (!reconstructionOracle.supported()) {
            throw std::runtime_error(reconstructionOracle.capability());
        }
        if (!projectionOracle.supported()) {
            throw std::runtime_error(projectionOracle.capability());
        }
        reconstructionOracle.loadModalInput(inputModal.data());
        reconstructionOracle.executeInverse();
        std::vector<Complex> oraclePhysicalInput(inputPhysicalCount);
        reconstructionOracle.copyInverseOutput(oraclePhysicalInput.data());

        const std::array<std::size_t, 3> probeModes{
            0, modeCount / 2, modeCount - 1};
        const std::array<std::size_t, 3> probeLevels{
            0, nz / 2, nz - 1};
        const std::array<std::size_t, 2> probeFields{
            0, inputWorkload.fields - 1};
        for (const auto mode : probeModes) {
            for (const auto field : probeFields) {
                for (const auto z : probeLevels) {
                    const auto index = retainedSpectrumIndex(
                        inputWorkload, mode, z, field);
                    if (std::find(
                            reconstructionProbeIndices.begin(),
                            reconstructionProbeIndices.end(), index) ==
                        reconstructionProbeIndices.end()) {
                        reconstructionProbeIndices.push_back(index);
                        reconstructionProbeExpected.push_back(
                            oraclePhysicalInput[index]);
                    }
                }
            }
        }

        std::vector<Complex> levelInput(
            modeCount * inputWorkload.fields);
        ExplicitAdvectiveConvolution horizontalOracle(n, modeSet.stored);
        for (std::size_t z = 0; z < nz; ++z) {
            packInterleavedLevel(
                inputWorkload, modeCount, z, oraclePhysicalInput.data(),
                levelInput);
            horizontalOracle.execute(levelInput);
            scatterInterleavedLevel(
                outputWorkload, modeCount, z, horizontalOracle.output(),
                oraclePhysicalOutput.data());
        }
        projectionOracle.loadPhysicalInput(oraclePhysicalOutput.data());
        projectionOracle.executeForward();
        projectionOracle.copyForwardOutput(expectedModal.data());
    }
    fftw_forget_wisdom();

    auto benchmarkPath = [&](auto& path, double horizontalSetupSeconds,
                             bool implicit, bool nativeInPlace,
                             const std::string& optimizerParameters) {
        VerticalGemmProvider reconstruction(
            inputWorkload, vertical, VerticalGemmLayout::split,
            verticalStrategy, VerticalGemmBufferPolicy::inverseOnly);
        VerticalGemmProvider projection(
            outputWorkload, vertical, VerticalGemmLayout::split,
            verticalStrategy, VerticalGemmBufferPolicy::forwardOnly);
        if (!reconstruction.supported()) {
            throw std::runtime_error(reconstruction.capability());
        }
        if (!projection.supported()) {
            throw std::runtime_error(projection.capability());
        }
        reconstruction.loadModalInput(inputModal.data());
        std::vector<Complex> levelInput(
            modeCount * inputWorkload.fields);
        executeComposedSplit(
            inputWorkload, outputWorkload, modeCount, reconstruction,
            path, projection, levelInput);

        const auto reconstructionMetric = splitProbeMetric(
            reconstructionProbeIndices, reconstructionProbeExpected,
            reconstruction);
        const auto physicalMetric = splitMetric(
            "vertically batched horizontal outputs versus explicit oracle",
            projection.splitPhysicalInputRealData(),
            projection.splitPhysicalInputImaginaryData(),
            oraclePhysicalOutput.data(), oraclePhysicalOutput.size());
        const auto modalMetric = splitMetric(
            "complete vertically batched modal output versus composed oracle",
            projection.splitModalOutputRealData(),
            projection.splitModalOutputImaginaryData(),
            expectedModal.data(), expectedModal.size());
        const auto preservationMetric = splitMetric(
            "caller modal input preserved across repeated execution",
            reconstruction.splitModalInputRealData(),
            reconstruction.splitModalInputImaginaryData(),
            inputModal.data(), inputModal.size());
        const auto hermitianMetric = splitHermitianBoundaryMetric(
            outputWorkload, modeSet.stored,
            projection.splitModalOutputRealData(),
            projection.splitModalOutputImaginaryData());

        ProviderRecord provider;
        provider.id = implicit
            ? "composed-fftwpp-parallel-target-wvm-advection"
            : "composed-fftw-explicit-parallel-target-wvm-advection";
        provider.version = implicit
            ? "FFTW++ 3.04 + FFTW 3.3.11 + Apple Accelerate"
            : "FFTW 3.3.11 + Apple Accelerate";
        provider.libraryIdentity = implicit
            ? "FFTW++ pinned at " SKBENCH_FFTWPP_COMMIT
                ", FFTW 3.3.11, and Apple Accelerate"
            : "FFTW 3.3.11 pthread build and Apple Accelerate";
        provider.algorithmId = implicit
            ? "split-k2-inverse+level-streamed-hybrid-6to1x4+split-k2-forward-v1"
            : "split-k2-inverse+level-streamed-explicit-shared-advectors+split-k2-forward-v1";
        provider.nativeRepresentationId = implicit
            ? "directional-k2-compact-split+centered-rectangular-hermitian"
            : "directional-k2-compact-split+full-fftw-half-spectrum";
        provider.modeOrderId =
            "k2-grouped-floor-n-over-3-radial-hermitian-with-stored-zero-line";
        provider.schedulingId =
            "vertical-" + std::string(verticalGemmScheduleName(
                reconstruction.strategy().schedule)) + "-" +
            std::to_string(reconstruction.outerWorkers()) +
            ";horizontal-persistent-outer-static-4-targets;vertical-levels-streamed";
        provider.sourceIdentity = implicit
            ? "https://github.com/dealias/fftwpp/commit/" SKBENCH_FFTWPP_COMMIT
            : "https://fftw.org/pub/fftw/fftw-3.3.11.tar.gz";
        provider.configureFlags = implicit
            ? "FFTWPP_SINGLE_THREAD=1; " + optimizerParameters
            : "FFTW --host=aarch64-apple-darwin --enable-neon --enable-threads";
        provider.compilerFlags = report.environment.compilerFlags;
        provider.planningConfiguration =
            "FFTW_MEASURE|FFTW_UNALIGNED|cold; Float64; horizontal radius=floor(N/3); "
            "Nj=floor(2*(Nz-1)/3); directional split K2-grouped vertical GEMM";
        provider.workers = std::max<std::size_t>(
            4, reconstruction.outerWorkers());
        provider.internalWorkers = 1;
        provider.outerWorkers = reconstruction.outerWorkers();
        provider.gemmCallsPerExecution =
            reconstruction.gemmCallsPerExecution() +
            projection.gemmCallsPerExecution();
        provider.execution = advectiveConvolutionContract(
            n, implicit && nativeInPlace, implicit,
            "directional-vertical-reconstruction-then-level-streaming-then-directional-projection");
        provider.execution.forward.nativeInputRepresentationId =
            "radial-k2-grouped-split-modal-15-fields";
        provider.execution.forward.nativeOutputRepresentationId =
            "radial-k2-grouped-split-modal-4-fields";
        provider.execution.forward.adapterInputRepresentationId =
            "mode-keyed-retained-truncated-modal-15-fields";
        provider.execution.forward.adapterOutputRepresentationId =
            "mode-keyed-retained-truncated-modal-4-fields";
        provider.execution.forward.physicalExtents =
            "Nx=Ny=" + std::to_string(n) + "; Nz=" +
            std::to_string(nz) + "; Nj=" + std::to_string(nj) +
            "; inputs=15; outputs=4; horizontal radius=floor(N/3)";
        provider.execution.forward.stridesElements =
            "modal/physical split: j-or-z fastest, then field, then K2-grouped mode; "
            "one reusable field-major interleaved level adapter";
        provider.execution.forward.paddingElements =
            implicit ? 4 * (2 * (n / 3) + 1) * (n / 3 + 1) :
                15 * n * (n / 2 + 1);
        provider.execution.forward.minimumAlignmentBytes = 64;
        provider.execution.forward.aliasing =
            "caller modal input, directional vertical buffers, reusable level adapter, "
            "horizontal provider storage, and projected modal output do not overlap";

        const auto verticalPersistent =
            reconstruction.persistentBytes() + projection.persistentBytes();
        const auto horizontalPersistent = path.residentBytes();
        const auto levelScratch = static_cast<std::uint64_t>(
            levelInput.size()) * sizeof(Complex);
        provider.explicitPersistentBytes =
            verticalPersistent + horizontalPersistent;
        provider.scratchBytes = levelScratch;
        provider.algorithmResidentBytes =
            provider.explicitPersistentBytes + provider.scratchBytes;
        provider.benchmarkHarnessBytes =
            static_cast<std::uint64_t>(inputModal.size() +
                                       oraclePhysicalOutput.size() +
                                       expectedModal.size()) * sizeof(Complex) +
            report.verticalMatrixFamilySourceBytes +
            static_cast<std::uint64_t>(
                reconstructionProbeExpected.size()) * sizeof(Complex);
        provider.estimatedProcessPeakBytes =
            provider.algorithmResidentBytes + provider.benchmarkHarnessBytes;
        report.spectralPipelineEstimatedExplicitPeakBytes =
            provider.estimatedProcessPeakBytes;
        provider.opaqueProviderMemory = true;
        provider.otherSetupSeconds = fixtureSeconds +
            reconstruction.matrixPreparationSeconds() +
            projection.matrixPreparationSeconds() +
            reconstruction.schedulerSetupSeconds() +
            projection.schedulerSetupSeconds();
        provider.allocationSeconds =
            reconstruction.allocationSeconds() + projection.allocationSeconds();
        provider.planningSeconds = horizontalSetupSeconds;
        provider.correctness = {
            reconstructionMetric, physicalMetric, modalMetric,
            preservationMetric, hermitianMetric};
        for (std::size_t target = 0; target < outputWorkload.fields; ++target) {
            provider.correctness.push_back(splitTargetMetric(
                target, outputWorkload, modeCount,
                projection.splitModalOutputRealData(),
                projection.splitModalOutputImaginaryData(), expectedModal));
        }

        const auto inputVerticalBytes = static_cast<std::uint64_t>(2) *
            (reconstruction.matrixBytesPerDirection() +
             inputModalCount * sizeof(double) +
             inputPhysicalCount * sizeof(double));
        const auto outputVerticalBytes = static_cast<std::uint64_t>(2) *
            (projection.matrixBytesPerDirection() +
             outputPhysicalCount * sizeof(double) +
             outputModalCount * sizeof(double));
        const auto movementBytes = static_cast<std::uint64_t>(2) *
            (inputPhysicalCount + outputPhysicalCount) * sizeof(Complex);
        const auto oneLevelBytes = static_cast<std::uint64_t>(
            (inputWorkload.fields + outputWorkload.fields) * modeCount) *
            sizeof(Complex);
        packSplitLevel(
            inputWorkload, modeCount, nz / 2, reconstruction, levelInput);
        const auto horizontalState = implicit
            ? StageState::fused : StageState::executed;
        provider.timings = {
            {"setup-shared-component", "K2-grouped vertical fixture generation",
             "shared", StageState::setupOnly,
             report.verticalMatrixFamilySourceBytes, {fixtureSeconds}},
            {"setup-component", "directional vertical matrix preparation",
             "shared", StageState::setupOnly,
             reconstruction.matrixBytesPerDirection() +
                 projection.matrixBytesPerDirection(),
             {reconstruction.matrixPreparationSeconds() +
              projection.matrixPreparationSeconds()}},
            {"setup-component", "horizontal planning and persistent scheduler setup",
             "shared", StageState::setupOnly, horizontalPersistent,
             {horizontalSetupSeconds}},
            {"primitive", "raw inverse vertical GEMM (15 fields)", "inverse",
             StageState::executed, inputVerticalBytes,
             timed(warmups, samples, [] {},
                   [&] { reconstruction.executeInverse(); })},
            {implicit ? "fused-primitive" : "operator-component",
             "one physical-level four-target horizontal advection", "horizontal",
             horizontalState, oneLevelBytes,
             timed(warmups, samples, [] {},
                   [&] { path.execute(levelInput); })},
            {"adapter-component",
             "all-level split/field-major packing and projected-output scatter",
             "horizontal", StageState::executed, movementBytes,
             timed(warmups, samples, [] {}, [&] {
                 for (std::size_t z = 0; z < nz; ++z) {
                     packSplitLevel(
                         inputWorkload, modeCount, z, reconstruction,
                         levelInput);
                     scatterSplitLevel(
                         outputWorkload, modeCount, z, path.output(),
                         projection);
                 }
             })},
            {"component",
             "vertically batched horizontal advection including level movement",
             "horizontal", StageState::executed,
             movementBytes + static_cast<std::uint64_t>(nz) * oneLevelBytes,
             timed(warmups, samples, [] {}, [&] {
                 for (std::size_t z = 0; z < nz; ++z) {
                     packSplitLevel(
                         inputWorkload, modeCount, z, reconstruction,
                         levelInput);
                     path.execute(levelInput);
                     scatterSplitLevel(
                         outputWorkload, modeCount, z, path.output(),
                         projection);
                 }
             })},
            {"primitive", "raw forward vertical GEMM (4 fields)", "forward",
             StageState::executed, outputVerticalBytes,
             timed(warmups, samples, [] {},
                   [&] { projection.executeForward(); })},
            {"uninstrumented-total",
             "vertically batched WVM-derived advection pipeline", "forward",
             StageState::executed,
             inputVerticalBytes + outputVerticalBytes + movementBytes +
                 static_cast<std::uint64_t>(nz) * oneLevelBytes,
             timed(warmups, samples, [] {}, [&] {
                 executeComposedSplit(
                     inputWorkload, outputWorkload, modeCount,
                     reconstruction, path, projection, levelInput);
             })}};
        provider.ledger = {
            {"setup/planning", StageState::setupOnly,
             "K2-grouped fixture, directional split matrices, horizontal plans, and persistent schedulers"},
            {"vertical reconstruction", StageState::executed,
             "inverse-only split GEMM reconstructs 15 inputs over physical levels"},
            {"horizontal input ordering", StageState::executed,
             "one reusable level is packed from mode-major split to field-major interleaved compact spectra"},
            {"horizontal convolution", horizontalState,
             implicit
                 ? "four persistent FFTW++ 6-to-1 applications execute per physical level"
                 : "shared explicit advectors and four persistent target tasks execute per physical level"},
            {"horizontal retention", StageState::fused,
             "each level writes the fixed floor(N/3) radial compact disk"},
            {"vertical projection", StageState::executed,
             "forward-only split GEMM retains Nj=floor(2*(Nz-1)/3) modes"},
            {"steady-state allocation", StageState::elided,
             "all buffers and schedulers are persistent; allocator interposer is a focused test"},
            {"authoritative total", StageState::executed,
             "ready 15-input modal coefficients to four ready projected modal outputs"},
            {"complete nonlinear WVM flux", StageState::unsupported,
             "phase evolution, coefficient accumulation, remaining flux bookkeeping, and time stepping are excluded"}};
        provider.observedProcessHighWaterBytes = processHighWaterBytes();
        return provider;
    };

    if (options.convolutionCandidate == "explicit-parallel") {
        const auto setupStart = Clock::now();
        ExplicitParallelAdvectiveConvolution path(n, modeSet.stored);
        const auto setupSeconds =
            std::chrono::duration<double>(Clock::now() - setupStart).count();
        report.providers.push_back(benchmarkPath(
            path, setupSeconds, false, false, ""));
    } else {
        const auto setupStart = Clock::now();
        ImplicitParallelAdvectiveConvolution path(n, modeSet.stored);
        const auto setupSeconds =
            std::chrono::duration<double>(Clock::now() - setupStart).count();
        report.providers.push_back(benchmarkPath(
            path, setupSeconds, true, path.nativeInPlace(),
            path.optimizerParameters()));
    }
    report.status = std::all_of(
        report.providers.front().correctness.begin(),
        report.providers.front().correctness.end(),
        [](const CorrectnessMetric& metric) { return metric.passed; })
        ? "passed" : "failed";
    return report;
}
#endif

} // namespace

BenchmarkReport runDealiasedConvolutionBenchmark(const RunOptions& options) {
#if !SKBENCH_HAVE_FFTWPP
    (void)options;
    throw std::runtime_error("dealiased-convolution requires configuring with SKBENCH_ENABLE_FFTWPP=ON");
#else
    if (options.convolutionMap == "wvm-advection" &&
        options.convolutionCandidate != "all")
        return runWvmAdvectiveFinalistBenchmark(options);
    if (options.convolutionMap == "wvm-advection")
        return runWvmAdvectiveConvolutionBenchmark(options);
    if (options.convolutionCandidate != "all")
        throw std::invalid_argument(
            "convolution-candidate is only supported by the wvm-advection map.");
    if (options.convolutionMap != "independent-products")
        throw std::invalid_argument(
            "convolution-map must be independent-products or wvm-advection.");
    if (options.convolutionProducts != 4 && options.convolutionProducts != 12)
        throw std::invalid_argument("convolution-products must be 4 or 12.");
    const auto selected = profileNamed(options.profile);
    if (selected.workload.nx != selected.workload.ny)
        throw std::invalid_argument("dealiased-convolution currently requires a square horizontal grid.");
    const auto n = selected.workload.nx;
    const auto products = options.convolutionProducts;
    const auto warmups = options.warmups ? options.warmups : 1;
    const auto samples = options.samples ? options.samples : 5;
    const auto modes = storedDisk(n);
    const auto input = compactFixture(modes, 4, options.seed);

    const auto setupStart = Clock::now();
    ExplicitConvolution explicitPath(n, products, modes);
    const auto explicitSetup = std::chrono::duration<double>(Clock::now() - setupStart).count();
    const auto implicitSetupStart = Clock::now();
    ImplicitConvolution implicitPath(n, products, modes);
    const auto implicitSetup = std::chrono::duration<double>(Clock::now() - implicitSetupStart).count();

    explicitPath.execute(input);
    implicitPath.execute(input);
    const auto implicitCorrectness = correctness(implicitPath.output(), explicitPath.output());

    BenchmarkReport report;
    report.status = implicitCorrectness.passed ? "passed" : "failed";
    report.environment = environmentRecord();
    auto id = report.environment.timestampUtc;
    id.erase(std::remove_if(id.begin(), id.end(), [](char c) { return c == '-' || c == ':'; }), id.end());
    report.runId = id + "-issue17-n" + std::to_string(n) + "-p" + std::to_string(products);
    report.profile = options.profile;
    report.seed = options.seed;
    report.warmups = warmups;
    report.samples = samples;
    report.workload = {n, n, 1, 4, 1.0, 1.0, true};
    report.retainedHorizontalModeCount = modes.size();
    report.retainedModeOrderHash = "centered-k-then-nonnegative-l-radial-disk-v1";
    report.fullRealBytes = n * n * 4 * sizeof(double);
    report.fullSpectrumBytes = n * (n / 2 + 1) * 4 * sizeof(Complex);
    report.retainedSpectrumBytes = modes.size() * 4 * sizeof(Complex);

    ProviderRecord baseline;
    baseline.id = "fftw-explicit-dealiased-convolution";
    baseline.version = "3.3.11";
    baseline.libraryIdentity = "FFTW 3.3.11 pthread build";
    baseline.algorithmId = "explicit-full-grid-four-field-quadratic-p" + std::to_string(products);
    baseline.nativeRepresentationId = "full-fftw-half-spectrum";
    baseline.modeOrderId = report.retainedModeOrderHash;
    baseline.schedulingId = "single-thread";
    baseline.planningConfiguration = "FFTW_MEASURE|FFTW_UNALIGNED";
    baseline.execution = convolutionContract(n, products, false, false);
    baseline.explicitPersistentBytes = explicitPath.residentBytes();
    baseline.algorithmResidentBytes = explicitPath.residentBytes();
    baseline.opaqueProviderMemory = true;
    baseline.planningSeconds = explicitSetup;
    baseline.correctness = {{"independent oracle baseline", 0.0, tolerance, true, 0.0}};
    baseline.timings.push_back(timing("primitive-component", "inverse FFT batch", StageState::executed, report.fullSpectrumBytes + report.fullRealBytes,
        timed(warmups, samples, [&] { explicitPath.load(input); }, [&] { explicitPath.inverse(); })));
    baseline.timings.push_back(timing("primitive-component", "pointwise quadratic products", StageState::executed, (4 + products) * n * n * sizeof(double),
        timed(warmups, samples, [&] { explicitPath.load(input); explicitPath.inverse(); }, [&] { explicitPath.multiply(); })));
    baseline.timings.push_back(timing("primitive-component", "forward FFT batch", StageState::executed, products * n * n * sizeof(double) + products * n * (n / 2 + 1) * sizeof(Complex),
        timed(warmups, samples, [&] { explicitPath.load(input); explicitPath.inverse(); explicitPath.multiply(); }, [&] { explicitPath.forward(); })));
    baseline.timings.push_back(timing("uninstrumented-total", "dealiased four-field convolution", StageState::executed, explicitPath.residentBytes(),
        timed(warmups, samples, [] {}, [&] { explicitPath.execute(input); })));
    baseline.ledger = {
        {"setup/planning", StageState::setupOnly, "two reusable FFTW_MEASURE plans"},
        {"compact-to-full embedding", StageState::executed, "included in the authoritative total"},
        {"inverse FFTs", StageState::executed, "four full-grid c2r transforms"},
        {"quadratic products", StageState::executed, std::to_string(products) + " deterministic pointwise products"},
        {"forward FFTs", StageState::executed, std::to_string(products) + " full-grid r2c transforms"},
        {"radial retention", StageState::executed, "included in the authoritative total"},
        {"nonlinear WVM flux", StageState::unsupported, "explicitly outside issue #17"}};

    ProviderRecord candidate;
    candidate.id = "fftwpp-hybrid-hermitian-convolution";
    candidate.version = "3.04";
    candidate.libraryIdentity = "FFTW++ public master pinned at " SKBENCH_FFTWPP_COMMIT;
    candidate.algorithmId = "hybrid-centered-hermitian-four-field-quadratic-p" + std::to_string(products);
    candidate.nativeRepresentationId = "centered-rectangular-hermitian";
    candidate.modeOrderId = report.retainedModeOrderHash;
    candidate.schedulingId = "single-thread";
    candidate.sourceIdentity = "https://github.com/dealias/fftwpp/commit/" SKBENCH_FFTWPP_COMMIT;
    candidate.configureFlags =
        "FFTWPP_SINGLE_THREAD=1; " + implicitPath.optimizerParameters();
    candidate.planningConfiguration = "FFTW_MEASURE|FFTW_UNALIGNED";
    candidate.execution = convolutionContract(n, products, implicitPath.nativeInPlace(), true);
    candidate.explicitPersistentBytes = implicitPath.residentBytes();
    candidate.algorithmResidentBytes = implicitPath.residentBytes();
    candidate.opaqueProviderMemory = false;
    candidate.planningSeconds = implicitSetup;
    candidate.correctness = {implicitCorrectness};
    for (std::size_t firstProduct = 0; firstProduct < products; firstProduct += 4)
        candidate.correctness.push_back(correctnessGroup(
            implicitPath.output(), explicitPath.output(), modes.size(), firstProduct));
    candidate.timings.push_back(timing("fused-primitive", "implicit hybrid transform-multiply-transform", StageState::fused, implicitPath.residentBytes(),
        timed(warmups, samples, [] {}, [&] { implicitPath.executeNativeSequence(input, false); })));
    candidate.timings.push_back(timing("uninstrumented-total", "dealiased four-field convolution", StageState::executed, implicitPath.residentBytes(),
        timed(warmups, samples, [] {}, [&] { implicitPath.execute(input); })));
    candidate.ledger = {
        {"setup/planning", StageState::setupOnly, "FFTW++ optimizer and reusable FFTW plans"},
        {"compact-to-centered embedding", StageState::executed, "included in the authoritative total"},
        {"inverse FFTs", StageState::fused, "inseparable FFTW++ implicit/hybrid stage"},
        {"quadratic products", StageState::fused, std::to_string(products) + " outputs from " + std::to_string((products + 3) / 4) + " four-output custom-multiplier call(s); FFTW++ B>A is excluded after a reproducible optimizer crash"},
        {"forward FFTs", StageState::fused, "inseparable FFTW++ implicit/hybrid stage"},
        {"radial retention", StageState::executed, "included in the authoritative total"},
        {"nonlinear WVM flux", StageState::unsupported, "explicitly outside issue #17"}};

    report.providers = {std::move(baseline), std::move(candidate)};
    return report;
#endif
}

BenchmarkReport runVerticallyBatchedAdvectionBenchmark(
    const RunOptions& options) {
#if !SKBENCH_HAVE_FFTWPP
    (void)options;
    throw std::runtime_error(
        "vertically-batched-advection requires configuring with "
        "SKBENCH_ENABLE_FFTWPP=ON");
#else
    return runVerticallyBatchedAdvectionBenchmarkImpl(options);
#endif
}

std::uint64_t probeDealiasedConvolutionSteadyStateAllocationsForTesting(
    std::size_t n, std::size_t products,
    void (*beginTracking)(), std::uint64_t (*endTracking)()) {
#if !SKBENCH_HAVE_FFTWPP
    (void)n;
    (void)products;
    (void)beginTracking;
    (void)endTracking;
    throw std::runtime_error("FFTW++ allocation probe is unavailable in this build.");
#else
    if (products != 4 && products != 12)
        throw std::invalid_argument("allocation probe products must be 4 or 12.");
    const auto modes = storedDisk(n);
    const auto input = compactFixture(modes, 4, 129);
    ExplicitConvolution explicitPath(n, products, modes);
    ImplicitConvolution implicitPath(n, products, modes);
    explicitPath.execute(input);
    implicitPath.execute(input);
    beginTracking();
    explicitPath.execute(input);
    implicitPath.execute(input);
    return endTracking();
#endif
}

std::uint64_t probeWvmAdvectiveConvolutionSteadyStateAllocationsForTesting(
    std::size_t n, void (*beginTracking)(), std::uint64_t (*endTracking)()) {
#if !SKBENCH_HAVE_FFTWPP
    (void)n;
    (void)beginTracking;
    (void)endTracking;
    throw std::runtime_error("FFTW++ allocation probe is unavailable in this build.");
#else
    const auto modes = storedDisk(n);
    const auto input = compactFixture(modes, 15, 129);
    ExplicitAdvectiveConvolution explicitPath(n, modes);
    ExplicitParallelAdvectiveConvolution explicitParallelPath(n, modes);
    ImplicitAdvectiveConvolution streamedPath(
        n, modes, ImplicitAdvectiveConvolution::Topology::streamedTarget);
    ImplicitAdvectiveConvolution allTargetPath(
        n, modes, ImplicitAdvectiveConvolution::Topology::allTargets);
    ImplicitParallelAdvectiveConvolution implicitParallelPath(n, modes);
    explicitPath.execute(input);
    explicitParallelPath.execute(input);
    streamedPath.execute(input);
    allTargetPath.execute(input);
    implicitParallelPath.execute(input);
    beginTracking();
    explicitPath.execute(input);
    explicitParallelPath.execute(input);
    streamedPath.execute(input);
    allTargetPath.execute(input);
    implicitParallelPath.execute(input);
    return endTracking();
#endif
}

std::uint64_t probeVerticallyBatchedAdvectionSteadyStateAllocationsForTesting(
    std::size_t n, std::size_t nz,
    void (*beginTracking)(), std::uint64_t (*endTracking)()) {
#if !SKBENCH_HAVE_FFTWPP
    (void)n;
    (void)nz;
    (void)beginTracking;
    (void)endTracking;
    throw std::runtime_error(
        "Vertically batched allocation probe is unavailable in this build.");
#else
    Workload outputWorkload{n, n, nz, 4, 1.0, 1.0, true};
    Workload inputWorkload = outputWorkload;
    inputWorkload.fields = 15;
    const auto modeSet = composedModeSet(n);
    const auto vertical = squaredWavenumberVerticalFixture(
        outputWorkload, modeSet.logical);
    const auto inputModal = composedModalFixture(
        inputWorkload, modeSet.stored, 129);
    const VerticalGemmStrategy strategy{
        VerticalGemmSchedule::outerDynamic, 2};

    VerticalGemmProvider explicitReconstruction(
        inputWorkload, vertical, VerticalGemmLayout::split, strategy,
        VerticalGemmBufferPolicy::inverseOnly);
    VerticalGemmProvider explicitProjection(
        outputWorkload, vertical, VerticalGemmLayout::split, strategy,
        VerticalGemmBufferPolicy::forwardOnly);
    VerticalGemmProvider implicitReconstruction(
        inputWorkload, vertical, VerticalGemmLayout::split, strategy,
        VerticalGemmBufferPolicy::inverseOnly);
    VerticalGemmProvider implicitProjection(
        outputWorkload, vertical, VerticalGemmLayout::split, strategy,
        VerticalGemmBufferPolicy::forwardOnly);
    explicitReconstruction.loadModalInput(inputModal.data());
    implicitReconstruction.loadModalInput(inputModal.data());

    ExplicitParallelAdvectiveConvolution explicitPath(n, modeSet.stored);
    ImplicitParallelAdvectiveConvolution implicitPath(n, modeSet.stored);
    std::vector<Complex> explicitLevel(
        modeSet.stored.size() * inputWorkload.fields);
    std::vector<Complex> implicitLevel(explicitLevel.size());
    for (std::size_t repetition = 0; repetition < 3; ++repetition) {
        executeComposedSplit(
            inputWorkload, outputWorkload, modeSet.stored.size(),
            explicitReconstruction, explicitPath, explicitProjection,
            explicitLevel);
        executeComposedSplit(
            inputWorkload, outputWorkload, modeSet.stored.size(),
            implicitReconstruction, implicitPath, implicitProjection,
            implicitLevel);
    }
    beginTracking();
    executeComposedSplit(
        inputWorkload, outputWorkload, modeSet.stored.size(),
        explicitReconstruction, explicitPath, explicitProjection,
        explicitLevel);
    executeComposedSplit(
        inputWorkload, outputWorkload, modeSet.stored.size(),
        implicitReconstruction, implicitPath, implicitProjection,
        implicitLevel);
    return endTracking();
#endif
}

} // namespace skbench
