#include "skbench/skbench.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <complex>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if SKBENCH_HAVE_ACCELERATE
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

namespace skbench {
namespace {

using Clock = std::chrono::steady_clock;
#if SKBENCH_HAVE_ACCELERATE
using BlasComplex = __LAPACK_double_complex;
#else
using BlasComplex = std::complex<double>;
#endif
static_assert(sizeof(BlasComplex) == 2 * sizeof(double));
static_assert(sizeof(Complex) == sizeof(BlasComplex));
static_assert(alignof(Complex) >= alignof(BlasComplex));

double elapsedSeconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::size_t checkedProduct(std::size_t left, std::size_t right, const char* label) {
    if (left != 0 && right > static_cast<std::size_t>(-1) / left) {
        throw std::overflow_error(std::string(label) + " size overflows size_t.");
    }
    return left * right;
}

int checkedBlasDimension(std::size_t value, const char* label) {
    if (value > static_cast<std::size_t>(INT_MAX)) {
        throw std::invalid_argument(std::string(label) + " exceeds the Accelerate BLAS integer range.");
    }
    return static_cast<int>(value);
}

class PersistentGroupExecutor {
public:
    using Task = void (*)(void*, std::size_t);

    PersistentGroupExecutor(std::vector<int> groupWeights, std::size_t requestedWorkers,
                            VerticalGemmSchedule schedule)
        : groupCount_(groupWeights.size()),
          workers_(std::max<std::size_t>(1, std::min(groupCount_, requestedWorkers))),
          schedule_(schedule) {
        if (groupCount_ == 0) throw std::invalid_argument("Grouped executor requires at least one group.");
        if (schedule_ == VerticalGemmSchedule::serial) {
            workers_ = 1;
        } else if (schedule_ == VerticalGemmSchedule::outerStatic) {
            staticBounds_.resize(workers_ + 1);
            std::vector<std::size_t> prefix(groupCount_ + 1, 0);
            for (std::size_t group = 0; group < groupCount_; ++group) {
                prefix[group + 1] = prefix[group] + static_cast<std::size_t>(groupWeights[group]);
            }
            staticBounds_.front() = 0;
            staticBounds_.back() = groupCount_;
            for (std::size_t worker = 1; worker < workers_; ++worker) {
                const auto minimum = staticBounds_[worker - 1] + 1;
                const auto maximum = groupCount_ - (workers_ - worker);
                const auto target = prefix.back() * worker / workers_;
                const auto match = std::lower_bound(prefix.begin() + static_cast<std::ptrdiff_t>(minimum),
                                                    prefix.begin() + static_cast<std::ptrdiff_t>(maximum + 1), target);
                staticBounds_[worker] = static_cast<std::size_t>(std::distance(prefix.begin(), match));
            }
        }
        threads_.reserve(workers_ - 1);
        for (std::size_t worker = 1; worker < workers_; ++worker) {
            threads_.emplace_back([this, worker] { workerLoop(worker); });
        }
    }

    ~PersistentGroupExecutor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_.notify_all();
        for (auto& thread : threads_) thread.join();
    }

    PersistentGroupExecutor(const PersistentGroupExecutor&) = delete;
    PersistentGroupExecutor& operator=(const PersistentGroupExecutor&) = delete;

    std::size_t workerCount() const noexcept { return workers_; }

    std::size_t explicitBytes() const noexcept {
        return threads_.capacity() * sizeof(std::thread) +
            staticBounds_.capacity() * sizeof(std::size_t);
    }

    void run(void* context, Task task) {
        if (workers_ == 1) {
            runWorker(0, context, task);
            return;
        }
        nextGroup_.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            context_ = context;
            task_ = task;
            remaining_ = workers_ - 1;
            ++generation_;
        }
        start_.notify_all();
        runWorker(0, context, task);
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return remaining_ == 0; });
        context_ = nullptr;
        task_ = nullptr;
    }

private:
    void runWorker(std::size_t worker, void* context, Task task) {
        if (schedule_ == VerticalGemmSchedule::outerDynamic) {
            for (;;) {
                const auto group = nextGroup_.fetch_add(1, std::memory_order_relaxed);
                if (group >= groupCount_) return;
                task(context, group);
            }
        }
        const auto begin = schedule_ == VerticalGemmSchedule::outerStatic ? staticBounds_[worker] : 0;
        const auto end = schedule_ == VerticalGemmSchedule::outerStatic ? staticBounds_[worker + 1] : groupCount_;
        for (std::size_t group = begin; group < end; ++group) task(context, group);
    }

    void workerLoop(std::size_t worker) {
        std::size_t observedGeneration = 0;
        for (;;) {
            void* context = nullptr;
            Task task = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_.wait(lock, [this, observedGeneration] {
                    return stopping_ || generation_ != observedGeneration;
                });
                if (stopping_) return;
                observedGeneration = generation_;
                context = context_;
                task = task_;
            }
            runWorker(worker, context, task);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--remaining_ == 0) done_.notify_one();
            }
        }
    }

    std::size_t groupCount_ = 0;
    std::size_t workers_ = 1;
    VerticalGemmSchedule schedule_ = VerticalGemmSchedule::serial;
    std::vector<std::thread> threads_;
    std::vector<std::size_t> staticBounds_;
    std::atomic<std::size_t> nextGroup_{0};
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable done_;
    void* context_ = nullptr;
    Task task_ = nullptr;
    std::size_t generation_ = 0;
    std::size_t remaining_ = 0;
    bool stopping_ = false;
};

void noopGroup(void*, std::size_t) {}

template <typename Value>
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    ~AlignedBuffer() { std::free(data_); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    void allocate(std::size_t count) {
        if (data_ != nullptr) throw std::logic_error("Aligned buffer is already allocated.");
        if (count == 0) return;
        void* storage = nullptr;
        if (posix_memalign(&storage, 64, checkedProduct(count, sizeof(Value), "aligned buffer")) != 0 || storage == nullptr) {
            throw std::bad_alloc();
        }
        data_ = static_cast<Value*>(storage);
        count_ = count;
    }

    Value* data() noexcept { return data_; }
    const Value* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(Value); }

private:
    Value* data_ = nullptr;
    std::size_t count_ = 0;
};

} // namespace

std::string_view verticalGemmLayoutName(VerticalGemmLayout layout) noexcept {
    switch (layout) {
        case VerticalGemmLayout::complexInterleaved: return "complex-interleaved";
        case VerticalGemmLayout::split: return "split";
    }
    return "unknown";
}

std::string_view verticalGemmScheduleName(VerticalGemmSchedule schedule) noexcept {
    switch (schedule) {
        case VerticalGemmSchedule::serial: return "serial";
        case VerticalGemmSchedule::outerStatic: return "outer-static";
        case VerticalGemmSchedule::outerDynamic: return "outer-dynamic";
    }
    return "unknown";
}

VerticalGemmSchedule verticalGemmScheduleNamed(std::string_view name) {
    if (name == "serial") return VerticalGemmSchedule::serial;
    if (name == "outer-static") return VerticalGemmSchedule::outerStatic;
    if (name == "outer-dynamic") return VerticalGemmSchedule::outerDynamic;
    throw std::invalid_argument("Unknown vertical GEMM schedule: " + std::string(name));
}

struct VerticalGemmProvider::Impl {
    Workload workload;
    std::size_t horizontalModeCount = 0;
    std::size_t columnCount = 0;
    std::size_t physicalCount = 0;
    std::size_t modalCount = 0;
    std::size_t matrixElementsPerGroup = 0;
    VerticalGemmLayout layout = VerticalGemmLayout::complexInterleaved;
    VerticalGemmBufferPolicy bufferPolicy = VerticalGemmBufferPolicy::bidirectional;
    VerticalGemmStrategy strategy;
    bool available = SKBENCH_HAVE_ACCELERATE != 0;
    std::string capabilityText;
    double allocationTime = 0.0;
    double preparationTime = 0.0;
    double schedulerSetupTime = 0.0;
    std::vector<VerticalModeGroup> groups;
    std::vector<int> groupColumns;
    std::vector<std::size_t> columnOffsets;
    std::unique_ptr<PersistentGroupExecutor> executor;

    AlignedBuffer<BlasComplex> complexForwardMatrix;
    AlignedBuffer<BlasComplex> complexInverseMatrix;
    AlignedBuffer<BlasComplex> complexPhysicalInput;
    AlignedBuffer<BlasComplex> complexModalInput;
    AlignedBuffer<BlasComplex> complexModalOutput;
    AlignedBuffer<BlasComplex> complexPhysicalOutput;

    AlignedBuffer<double> realForwardMatrix;
    AlignedBuffer<double> realInverseMatrix;
    AlignedBuffer<double> physicalInputReal;
    AlignedBuffer<double> physicalInputImaginary;
    AlignedBuffer<double> modalInputReal;
    AlignedBuffer<double> modalInputImaginary;
    AlignedBuffer<double> modalOutputReal;
    AlignedBuffer<double> modalOutputImaginary;
    AlignedBuffer<double> physicalOutputReal;
    AlignedBuffer<double> physicalOutputImaginary;

    int nz = 0;
    int nj = 0;

    Impl(const Workload& inputWorkload, const GroupedVerticalOperators& operators,
         VerticalGemmLayout inputLayout, VerticalGemmStrategy inputStrategy,
         VerticalGemmBufferPolicy inputBufferPolicy)
        : workload(inputWorkload), layout(inputLayout),
          bufferPolicy(inputBufferPolicy), strategy(inputStrategy),
          groups(operators.groups) {
        if (operators.nz != workload.nz || operators.nj != workload.retainedVerticalModes()) {
            throw std::invalid_argument("Vertical GEMM operator dimensions do not match the workload.");
        }
        if (groups.empty() || workload.fields == 0) {
            throw std::invalid_argument("Vertical GEMM requires at least one horizontal mode and field.");
        }
        if (strategy.outerWorkers == 0) {
            throw std::invalid_argument("Vertical GEMM outer workers must be positive.");
        }
        if (strategy.schedule == VerticalGemmSchedule::serial && strategy.outerWorkers != 1) {
            throw std::invalid_argument("The serial vertical GEMM schedule requires exactly one outer worker.");
        }
        std::size_t expectedFirstMode = 0;
        groupColumns.reserve(groups.size());
        columnOffsets.reserve(groups.size());
        for (const auto& group : groups) {
            if (group.modeCount == 0 || group.firstMode != expectedFirstMode) {
                throw std::invalid_argument("Vertical GEMM groups must be nonempty and contiguous.");
            }
            columnOffsets.push_back(checkedProduct(group.firstMode, workload.fields, "group column offset"));
            groupColumns.push_back(checkedBlasDimension(
                checkedProduct(group.modeCount, workload.fields, "group column count"),
                "group K"));
            expectedFirstMode += group.modeCount;
        }
        horizontalModeCount = expectedFirstMode;
        columnCount = checkedProduct(horizontalModeCount, workload.fields, "vertical GEMM column");
        physicalCount = checkedProduct(workload.nz, columnCount, "vertical GEMM physical operand");
        modalCount = checkedProduct(operators.nj, columnCount, "vertical GEMM modal operand");
        matrixElementsPerGroup = checkedProduct(operators.nj, operators.nz, "vertical GEMM matrix");
        const auto familyElements = checkedProduct(groups.size(), matrixElementsPerGroup, "vertical GEMM matrix family");
        if (operators.forward.size() != familyElements || operators.inverse.size() != familyElements) {
            throw std::invalid_argument("Vertical GEMM matrix-family storage does not match its groups and dimensions.");
        }
        nz = checkedBlasDimension(workload.nz, "Nz");
        nj = checkedBlasDimension(operators.nj, "Nj");
        checkedBlasDimension(columnCount, "K");

        if (!available) {
            capabilityText = "unsupported: Accelerate BLAS is available only on Apple platforms";
            return;
        }

        const auto allocationStart = Clock::now();
        const bool allocateForward =
            bufferPolicy != VerticalGemmBufferPolicy::inverseOnly;
        const bool allocateInverse =
            bufferPolicy != VerticalGemmBufferPolicy::forwardOnly;
        if (layout == VerticalGemmLayout::complexInterleaved) {
            if (allocateForward) {
                complexForwardMatrix.allocate(familyElements);
                complexPhysicalInput.allocate(physicalCount);
                complexModalOutput.allocate(modalCount);
            }
            if (allocateInverse) {
                complexInverseMatrix.allocate(familyElements);
                complexModalInput.allocate(modalCount);
                complexPhysicalOutput.allocate(physicalCount);
            }
        } else {
            if (allocateForward) {
                realForwardMatrix.allocate(familyElements);
                physicalInputReal.allocate(physicalCount);
                physicalInputImaginary.allocate(physicalCount);
                modalOutputReal.allocate(modalCount);
                modalOutputImaginary.allocate(modalCount);
            }
            if (allocateInverse) {
                realInverseMatrix.allocate(familyElements);
                modalInputReal.allocate(modalCount);
                modalInputImaginary.allocate(modalCount);
                physicalOutputReal.allocate(physicalCount);
                physicalOutputImaginary.allocate(physicalCount);
            }
        }
        allocationTime = elapsedSeconds(allocationStart);

        const auto preparationStart = Clock::now();
        for (std::size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
            const auto offset = groupIndex * matrixElementsPerGroup;
            for (std::size_t z = 0; z < operators.nz; ++z) {
                for (std::size_t j = 0; j < operators.nj; ++j) {
                    const double forwardValue = operators.forward[offset + j * operators.nz + z];
                    const double inverseValue = operators.inverse[offset + z * operators.nj + j];
                    const auto forwardIndex = offset + j + operators.nj * z;
                    const auto inverseIndex = offset + z + operators.nz * j;
                    if (layout == VerticalGemmLayout::complexInterleaved) {
                        if (allocateForward)
                            complexForwardMatrix.data()[forwardIndex] = {forwardValue, 0.0};
                        if (allocateInverse)
                            complexInverseMatrix.data()[inverseIndex] = {inverseValue, 0.0};
                    } else {
                        if (allocateForward)
                            realForwardMatrix.data()[forwardIndex] = forwardValue;
                        if (allocateInverse)
                            realInverseMatrix.data()[inverseIndex] = inverseValue;
                    }
                }
            }
        }
        preparationTime = elapsedSeconds(preparationStart);
        const auto schedulerStart = Clock::now();
        executor = std::make_unique<PersistentGroupExecutor>(groupColumns, strategy.outerWorkers, strategy.schedule);
        strategy.outerWorkers = executor->workerCount();
        schedulerSetupTime = elapsedSeconds(schedulerStart);
        capabilityText = "supported: " + std::string(verticalGemmScheduleName(strategy.schedule)) +
            "; outer workers=" + std::to_string(strategy.outerWorkers);
    }

    void requireAvailable() const {
        if (!available) throw std::runtime_error(capabilityText);
    }

    void requireForward() const {
        requireAvailable();
        if (bufferPolicy == VerticalGemmBufferPolicy::inverseOnly) {
            throw std::logic_error(
                "Forward GEMM component requested from an inverse-only provider.");
        }
    }

    void requireInverse() const {
        requireAvailable();
        if (bufferPolicy == VerticalGemmBufferPolicy::forwardOnly) {
            throw std::logic_error(
                "Inverse GEMM component requested from a forward-only provider.");
        }
    }

    void requireSplit() const {
        requireAvailable();
        if (layout != VerticalGemmLayout::split) {
            throw std::logic_error("Split GEMM component requested from the complex GEMM provider.");
        }
    }

    void requireInterleaved() const {
        requireAvailable();
        if (layout != VerticalGemmLayout::complexInterleaved) {
            throw std::logic_error("Interleaved GEMM component requested from the split GEMM provider.");
        }
    }

    void validateRetainedModes(const std::vector<RetainedMode>& modes) const {
        if (modes.size() != horizontalModeCount) {
            throw std::invalid_argument("Retained-mode count does not match the vertical GEMM columns.");
        }
    }

    void packPhysicalInputFromWvm(const std::vector<RetainedMode>& modes, const Complex* wvmSpectrum) {
        requireForward();
        validateRetainedModes(modes);
        for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
            const auto& mode = modes[modeIndex];
            for (std::size_t field = 0; field < workload.fields; ++field) {
                for (std::size_t z = 0; z < workload.nz; ++z) {
                    auto value = wvmSpectrum[wvmSpectrumIndex(
                        workload, mode.storedKx, mode.storedKy, z, field)];
                    if (mode.conjugatesStoredValue) value = conjugate(value);
                    const auto retainedIndex = retainedSpectrumIndex(workload, modeIndex, z, field);
                    if (layout == VerticalGemmLayout::complexInterleaved) {
                        complexPhysicalInput.data()[retainedIndex] = {value.real, value.imag};
                    } else {
                        physicalInputReal.data()[retainedIndex] = value.real;
                        physicalInputImaginary.data()[retainedIndex] = value.imag;
                    }
                }
            }
        }
    }

    void embedPhysicalOutputToWvm(const std::vector<RetainedMode>& modes, Complex* wvmSpectrum) const {
        requireInverse();
        validateRetainedModes(modes);
        std::fill_n(wvmSpectrum, workload.spectrumElements(), Complex{});
        for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
            const auto& mode = modes[modeIndex];
            for (std::size_t field = 0; field < workload.fields; ++field) {
                for (std::size_t z = 0; z < workload.nz; ++z) {
                    const auto retainedIndex = retainedSpectrumIndex(workload, modeIndex, z, field);
                    const Complex compact = layout == VerticalGemmLayout::complexInterleaved
                        ? Complex{complexPhysicalOutput.data()[retainedIndex].real(),
                                  complexPhysicalOutput.data()[retainedIndex].imag()}
                        : Complex{physicalOutputReal.data()[retainedIndex],
                                  physicalOutputImaginary.data()[retainedIndex]};
                    const auto stored = mode.conjugatesStoredValue ? conjugate(compact) : compact;
                    wvmSpectrum[wvmSpectrumIndex(
                        workload, mode.storedKx, mode.storedKy, z, field)] = stored;
                    if (mode.storedKx == 0 && mode.storedKy != 0 &&
                        2 * mode.storedKy != workload.ny) {
                        const auto conjugateKy = (workload.ny - mode.storedKy) % workload.ny;
                        wvmSpectrum[wvmSpectrumIndex(workload, 0, conjugateKy, z, field)] =
                            conjugate(stored);
                    }
                }
            }
        }
    }

    std::size_t persistentBytes() const noexcept {
        return complexForwardMatrix.bytes() + complexInverseMatrix.bytes() +
            complexPhysicalInput.bytes() + complexModalInput.bytes() + complexModalOutput.bytes() +
            complexPhysicalOutput.bytes() + realForwardMatrix.bytes() + realInverseMatrix.bytes() +
            physicalInputReal.bytes() + physicalInputImaginary.bytes() + modalInputReal.bytes() +
            modalInputImaginary.bytes() + modalOutputReal.bytes() + modalOutputImaginary.bytes() +
            physicalOutputReal.bytes() + physicalOutputImaginary.bytes() +
            groups.size() * sizeof(VerticalModeGroup) + groupColumns.size() * sizeof(int) +
            columnOffsets.size() * sizeof(std::size_t) +
            (executor == nullptr ? 0 : executor->explicitBytes());
    }

#if SKBENCH_HAVE_ACCELERATE
    void forwardComplexGroup(std::size_t groupIndex) {
        const BlasComplex alpha{1.0, 0.0};
        const BlasComplex beta{0.0, 0.0};
        const auto matrixOffset = groupIndex * matrixElementsPerGroup;
        const auto columnOffset = columnOffsets[groupIndex];
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nj, groupColumns[groupIndex], nz,
                    &alpha, complexForwardMatrix.data() + matrixOffset, nj,
                    complexPhysicalInput.data() + workload.nz * columnOffset, nz,
                    &beta, complexModalOutput.data() + workload.retainedVerticalModes() * columnOffset, nj);
    }

    void inverseComplexGroup(std::size_t groupIndex) {
        const BlasComplex alpha{1.0, 0.0};
        const BlasComplex beta{0.0, 0.0};
        const auto matrixOffset = groupIndex * matrixElementsPerGroup;
        const auto columnOffset = columnOffsets[groupIndex];
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nz, groupColumns[groupIndex], nj,
                    &alpha, complexInverseMatrix.data() + matrixOffset, nz,
                    complexModalInput.data() + workload.retainedVerticalModes() * columnOffset, nj,
                    &beta, complexPhysicalOutput.data() + workload.nz * columnOffset, nz);
    }

    void forwardRealGroup(std::size_t groupIndex) {
        const auto matrixOffset = groupIndex * matrixElementsPerGroup;
        const auto columnOffset = columnOffsets[groupIndex];
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nj, groupColumns[groupIndex], nz, 1.0,
                    realForwardMatrix.data() + matrixOffset, nj,
                    physicalInputReal.data() + workload.nz * columnOffset, nz, 0.0,
                    modalOutputReal.data() + workload.retainedVerticalModes() * columnOffset, nj);
    }

    void forwardImaginaryGroup(std::size_t groupIndex) {
        const auto matrixOffset = groupIndex * matrixElementsPerGroup;
        const auto columnOffset = columnOffsets[groupIndex];
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nj, groupColumns[groupIndex], nz, 1.0,
                    realForwardMatrix.data() + matrixOffset, nj,
                    physicalInputImaginary.data() + workload.nz * columnOffset, nz, 0.0,
                    modalOutputImaginary.data() + workload.retainedVerticalModes() * columnOffset, nj);
    }

    void inverseRealGroup(std::size_t groupIndex) {
        const auto matrixOffset = groupIndex * matrixElementsPerGroup;
        const auto columnOffset = columnOffsets[groupIndex];
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nz, groupColumns[groupIndex], nj, 1.0,
                    realInverseMatrix.data() + matrixOffset, nz,
                    modalInputReal.data() + workload.retainedVerticalModes() * columnOffset, nj, 0.0,
                    physicalOutputReal.data() + workload.nz * columnOffset, nz);
    }

    void inverseImaginaryGroup(std::size_t groupIndex) {
        const auto matrixOffset = groupIndex * matrixElementsPerGroup;
        const auto columnOffset = columnOffsets[groupIndex];
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nz, groupColumns[groupIndex], nj, 1.0,
                    realInverseMatrix.data() + matrixOffset, nz,
                    modalInputImaginary.data() + workload.retainedVerticalModes() * columnOffset, nj, 0.0,
                    physicalOutputImaginary.data() + workload.nz * columnOffset, nz);
    }

    static void forwardComplexTask(void* context, std::size_t group) {
        static_cast<Impl*>(context)->forwardComplexGroup(group);
    }

    static void inverseComplexTask(void* context, std::size_t group) {
        static_cast<Impl*>(context)->inverseComplexGroup(group);
    }

    static void forwardRealTask(void* context, std::size_t group) {
        static_cast<Impl*>(context)->forwardRealGroup(group);
    }

    static void forwardImaginaryTask(void* context, std::size_t group) {
        static_cast<Impl*>(context)->forwardImaginaryGroup(group);
    }

    static void inverseRealTask(void* context, std::size_t group) {
        static_cast<Impl*>(context)->inverseRealGroup(group);
    }

    static void inverseImaginaryTask(void* context, std::size_t group) {
        static_cast<Impl*>(context)->inverseImaginaryGroup(group);
    }

    static void forwardSplitTask(void* context, std::size_t group) {
        auto* self = static_cast<Impl*>(context);
        self->forwardRealGroup(group);
        self->forwardImaginaryGroup(group);
    }

    static void inverseSplitTask(void* context, std::size_t group) {
        auto* self = static_cast<Impl*>(context);
        self->inverseRealGroup(group);
        self->inverseImaginaryGroup(group);
    }
#endif
};

VerticalGemmProvider::VerticalGemmProvider(const Workload& workload, std::size_t horizontalModeCount,
                                           const VerticalOperators& operators, VerticalGemmLayout layout)
    : VerticalGemmProvider(workload, commonVerticalFixture(horizontalModeCount, operators), layout) {}

VerticalGemmProvider::VerticalGemmProvider(const Workload& workload,
                                           const GroupedVerticalOperators& operators,
                                           VerticalGemmLayout layout)
    : VerticalGemmProvider(workload, operators, layout, VerticalGemmStrategy{}) {}

VerticalGemmProvider::VerticalGemmProvider(const Workload& workload,
                                           const GroupedVerticalOperators& operators,
                                           VerticalGemmLayout layout, VerticalGemmStrategy strategy)
    : VerticalGemmProvider(
          workload, operators, layout, strategy,
          VerticalGemmBufferPolicy::bidirectional) {}

VerticalGemmProvider::VerticalGemmProvider(
    const Workload& workload, const GroupedVerticalOperators& operators,
    VerticalGemmLayout layout, VerticalGemmStrategy strategy,
    VerticalGemmBufferPolicy bufferPolicy)
    : impl_(std::make_unique<Impl>(
          workload, operators, layout, strategy, bufferPolicy)) {}

VerticalGemmProvider::~VerticalGemmProvider() = default;
VerticalGemmProvider::VerticalGemmProvider(VerticalGemmProvider&&) noexcept = default;
VerticalGemmProvider& VerticalGemmProvider::operator=(VerticalGemmProvider&&) noexcept = default;

bool VerticalGemmProvider::supported() const noexcept { return impl_->available; }
std::string VerticalGemmProvider::capability() const { return impl_->capabilityText; }
VerticalGemmLayout VerticalGemmProvider::layout() const noexcept { return impl_->layout; }
std::size_t VerticalGemmProvider::columns() const noexcept { return impl_->columnCount; }
std::size_t VerticalGemmProvider::physicalElements() const noexcept { return impl_->physicalCount; }
std::size_t VerticalGemmProvider::modalElements() const noexcept { return impl_->modalCount; }
std::size_t VerticalGemmProvider::groupCount() const noexcept { return impl_->groups.size(); }
std::size_t VerticalGemmProvider::gemmCallsPerExecution() const noexcept {
    return impl_->groups.size() * (impl_->layout == VerticalGemmLayout::split ? 2 : 1);
}
VerticalGemmStrategy VerticalGemmProvider::strategy() const noexcept { return impl_->strategy; }
std::size_t VerticalGemmProvider::outerWorkers() const noexcept { return impl_->strategy.outerWorkers; }
std::size_t VerticalGemmProvider::persistentBytes() const noexcept { return impl_->persistentBytes(); }
std::size_t VerticalGemmProvider::schedulerPersistentBytes() const noexcept {
    return impl_->executor == nullptr ? 0 : impl_->executor->explicitBytes();
}
std::size_t VerticalGemmProvider::matrixBytesPerDirection() const noexcept {
    const auto scalarBytes = impl_->layout == VerticalGemmLayout::complexInterleaved ? sizeof(BlasComplex) : sizeof(double);
    return impl_->groups.size() * impl_->matrixElementsPerGroup * scalarBytes;
}
std::size_t VerticalGemmProvider::minimumAlignmentBytes() const noexcept { return 64; }
double VerticalGemmProvider::allocationSeconds() const noexcept { return impl_->allocationTime; }
double VerticalGemmProvider::matrixPreparationSeconds() const noexcept { return impl_->preparationTime; }
double VerticalGemmProvider::schedulerSetupSeconds() const noexcept { return impl_->schedulerSetupTime; }
bool VerticalGemmProvider::hasOpaqueSchedulerMemory() const noexcept {
    return impl_->strategy.outerWorkers > 1;
}
std::string VerticalGemmProvider::libraryIdentity() const {
#if SKBENCH_HAVE_ACCELERATE
    return "/System/Library/Frameworks/Accelerate.framework";
#else
    return "unavailable";
#endif
}

void VerticalGemmProvider::loadPhysicalInput(const Complex* input) {
    impl_->requireForward();
    if (impl_->layout == VerticalGemmLayout::complexInterleaved) {
        for (std::size_t index = 0; index < impl_->physicalCount; ++index) {
            impl_->complexPhysicalInput.data()[index] = {input[index].real, input[index].imag};
        }
        return;
    }
    for (std::size_t index = 0; index < impl_->physicalCount; ++index) {
        impl_->physicalInputReal.data()[index] = input[index].real;
        impl_->physicalInputImaginary.data()[index] = input[index].imag;
    }
}

void VerticalGemmProvider::packPhysicalInputFromWvm(
    const std::vector<RetainedMode>& modes, const Complex* wvmSpectrum) {
    impl_->packPhysicalInputFromWvm(modes, wvmSpectrum);
}

void VerticalGemmProvider::loadModalInput(const Complex* input) {
    impl_->requireInverse();
    if (impl_->layout == VerticalGemmLayout::complexInterleaved) {
        for (std::size_t index = 0; index < impl_->modalCount; ++index) {
            impl_->complexModalInput.data()[index] = {input[index].real, input[index].imag};
        }
        return;
    }
    for (std::size_t index = 0; index < impl_->modalCount; ++index) {
        impl_->modalInputReal.data()[index] = input[index].real;
        impl_->modalInputImaginary.data()[index] = input[index].imag;
    }
}

void VerticalGemmProvider::executeForward() {
    impl_->requireForward();
#if SKBENCH_HAVE_ACCELERATE
    if (impl_->layout == VerticalGemmLayout::complexInterleaved) {
        if (impl_->strategy.schedule == VerticalGemmSchedule::serial) {
            for (std::size_t group = 0; group < impl_->groups.size(); ++group) impl_->forwardComplexGroup(group);
        } else {
            impl_->executor->run(impl_.get(), &Impl::forwardComplexTask);
        }
    } else {
        if (impl_->strategy.schedule == VerticalGemmSchedule::serial) {
            executeForwardReal();
            executeForwardImaginary();
        } else {
            impl_->executor->run(impl_.get(), &Impl::forwardSplitTask);
        }
    }
#endif
}

void VerticalGemmProvider::executeInverse() {
    impl_->requireInverse();
#if SKBENCH_HAVE_ACCELERATE
    if (impl_->layout == VerticalGemmLayout::complexInterleaved) {
        if (impl_->strategy.schedule == VerticalGemmSchedule::serial) {
            for (std::size_t group = 0; group < impl_->groups.size(); ++group) impl_->inverseComplexGroup(group);
        } else {
            impl_->executor->run(impl_.get(), &Impl::inverseComplexTask);
        }
    } else {
        if (impl_->strategy.schedule == VerticalGemmSchedule::serial) {
            executeInverseReal();
            executeInverseImaginary();
        } else {
            impl_->executor->run(impl_.get(), &Impl::inverseSplitTask);
        }
    }
#endif
}

void VerticalGemmProvider::executeForwardReal() {
    impl_->requireForward();
    impl_->requireSplit();
#if SKBENCH_HAVE_ACCELERATE
    if (impl_->strategy.schedule == VerticalGemmSchedule::serial) {
        for (std::size_t group = 0; group < impl_->groups.size(); ++group) impl_->forwardRealGroup(group);
    } else {
        impl_->executor->run(impl_.get(), &Impl::forwardRealTask);
    }
#endif
}

void VerticalGemmProvider::executeForwardImaginary() {
    impl_->requireForward();
    impl_->requireSplit();
#if SKBENCH_HAVE_ACCELERATE
    if (impl_->strategy.schedule == VerticalGemmSchedule::serial) {
        for (std::size_t group = 0; group < impl_->groups.size(); ++group) impl_->forwardImaginaryGroup(group);
    } else {
        impl_->executor->run(impl_.get(), &Impl::forwardImaginaryTask);
    }
#endif
}

void VerticalGemmProvider::executeInverseReal() {
    impl_->requireInverse();
    impl_->requireSplit();
#if SKBENCH_HAVE_ACCELERATE
    if (impl_->strategy.schedule == VerticalGemmSchedule::serial) {
        for (std::size_t group = 0; group < impl_->groups.size(); ++group) impl_->inverseRealGroup(group);
    } else {
        impl_->executor->run(impl_.get(), &Impl::inverseRealTask);
    }
#endif
}

void VerticalGemmProvider::executeInverseImaginary() {
    impl_->requireInverse();
    impl_->requireSplit();
#if SKBENCH_HAVE_ACCELERATE
    if (impl_->strategy.schedule == VerticalGemmSchedule::serial) {
        for (std::size_t group = 0; group < impl_->groups.size(); ++group) impl_->inverseImaginaryGroup(group);
    } else {
        impl_->executor->run(impl_.get(), &Impl::inverseImaginaryTask);
    }
#endif
}

void VerticalGemmProvider::executeSchedulerNoop() {
    impl_->requireAvailable();
    if (impl_->strategy.schedule != VerticalGemmSchedule::serial) {
        impl_->executor->run(nullptr, &noopGroup);
    }
}

Complex* VerticalGemmProvider::interleavedPhysicalInputData() {
    impl_->requireForward();
    impl_->requireInterleaved();
    static_assert(sizeof(Complex) == sizeof(BlasComplex));
    return reinterpret_cast<Complex*>(impl_->complexPhysicalInput.data());
}

Complex* VerticalGemmProvider::interleavedModalInputData() {
    impl_->requireInverse();
    impl_->requireInterleaved();
    static_assert(sizeof(Complex) == sizeof(BlasComplex));
    return reinterpret_cast<Complex*>(impl_->complexModalInput.data());
}

const Complex* VerticalGemmProvider::interleavedModalOutputData() const {
    impl_->requireForward();
    impl_->requireInterleaved();
    static_assert(sizeof(Complex) == sizeof(BlasComplex));
    return reinterpret_cast<const Complex*>(impl_->complexModalOutput.data());
}

const Complex* VerticalGemmProvider::interleavedPhysicalOutputData() const {
    impl_->requireInverse();
    impl_->requireInterleaved();
    static_assert(sizeof(Complex) == sizeof(BlasComplex));
    return reinterpret_cast<const Complex*>(impl_->complexPhysicalOutput.data());
}

double* VerticalGemmProvider::splitPhysicalInputRealData() {
    impl_->requireForward();
    impl_->requireSplit();
    return impl_->physicalInputReal.data();
}

double* VerticalGemmProvider::splitPhysicalInputImaginaryData() {
    impl_->requireForward();
    impl_->requireSplit();
    return impl_->physicalInputImaginary.data();
}

double* VerticalGemmProvider::splitModalInputRealData() {
    impl_->requireInverse();
    impl_->requireSplit();
    return impl_->modalInputReal.data();
}

double* VerticalGemmProvider::splitModalInputImaginaryData() {
    impl_->requireInverse();
    impl_->requireSplit();
    return impl_->modalInputImaginary.data();
}

const double* VerticalGemmProvider::splitModalOutputRealData() const {
    impl_->requireForward();
    impl_->requireSplit();
    return impl_->modalOutputReal.data();
}

const double* VerticalGemmProvider::splitModalOutputImaginaryData() const {
    impl_->requireForward();
    impl_->requireSplit();
    return impl_->modalOutputImaginary.data();
}

const double* VerticalGemmProvider::splitPhysicalOutputRealData() const {
    impl_->requireInverse();
    impl_->requireSplit();
    return impl_->physicalOutputReal.data();
}

const double* VerticalGemmProvider::splitPhysicalOutputImaginaryData() const {
    impl_->requireInverse();
    impl_->requireSplit();
    return impl_->physicalOutputImaginary.data();
}

void VerticalGemmProvider::copyForwardOutput(Complex* output) const {
    impl_->requireForward();
    if (impl_->layout == VerticalGemmLayout::complexInterleaved) {
        for (std::size_t index = 0; index < impl_->modalCount; ++index) {
            output[index] = {impl_->complexModalOutput.data()[index].real(),
                             impl_->complexModalOutput.data()[index].imag()};
        }
        return;
    }
    for (std::size_t index = 0; index < impl_->modalCount; ++index) {
        output[index] = {impl_->modalOutputReal.data()[index], impl_->modalOutputImaginary.data()[index]};
    }
}

void VerticalGemmProvider::copyInverseOutput(Complex* output) const {
    impl_->requireInverse();
    if (impl_->layout == VerticalGemmLayout::complexInterleaved) {
        for (std::size_t index = 0; index < impl_->physicalCount; ++index) {
            output[index] = {impl_->complexPhysicalOutput.data()[index].real(),
                             impl_->complexPhysicalOutput.data()[index].imag()};
        }
        return;
    }
    for (std::size_t index = 0; index < impl_->physicalCount; ++index) {
        output[index] = {impl_->physicalOutputReal.data()[index], impl_->physicalOutputImaginary.data()[index]};
    }
}

void VerticalGemmProvider::embedPhysicalOutputToWvm(
    const std::vector<RetainedMode>& modes, Complex* wvmSpectrum) const {
    impl_->embedPhysicalOutputToWvm(modes, wvmSpectrum);
}

struct WvmDirectVerticalGemmProvider::Impl {
    struct ExecutionContext {
        Impl* self = nullptr;
        const Complex* input = nullptr;
        Complex* output = nullptr;
    };

    Workload workload;
    std::size_t modalSpectrumCount = 0;
    std::size_t matrixElementsPerGroup = 0;
    VerticalGemmStrategy strategy;
    bool available = SKBENCH_HAVE_ACCELERATE != 0;
    std::string capabilityText;
    double allocationTime = 0.0;
    double preparationTime = 0.0;
    double schedulerSetupTime = 0.0;
    std::vector<RetainedMode> modes;
    std::vector<std::size_t> modeGroupIndices;
    std::vector<std::size_t> spectrumOffsets;
    std::vector<std::size_t> modalOffsets;
    std::unique_ptr<PersistentGroupExecutor> executor;
    AlignedBuffer<BlasComplex> complexForwardMatrix;
    AlignedBuffer<BlasComplex> complexInverseMatrix;
    int nz = 0;
    int nj = 0;
    int fields = 0;

    Impl(const Workload& inputWorkload, const std::vector<RetainedMode>& inputModes,
         const GroupedVerticalOperators& operators, VerticalGemmStrategy inputStrategy)
        : workload(inputWorkload), strategy(inputStrategy), modes(inputModes) {
        if (operators.nz != workload.nz || operators.nj != workload.retainedVerticalModes()) {
            throw std::invalid_argument("Direct WVM vertical GEMM operator dimensions do not match the workload.");
        }
        if (modes.empty() || operators.groups.empty() || workload.fields == 0) {
            throw std::invalid_argument("Direct WVM vertical GEMM requires modes, groups, and fields.");
        }
        if (strategy.outerWorkers == 0) {
            throw std::invalid_argument("Direct WVM vertical GEMM outer workers must be positive.");
        }
        if (strategy.schedule == VerticalGemmSchedule::serial && strategy.outerWorkers != 1) {
            throw std::invalid_argument(
                "The serial direct WVM vertical GEMM schedule requires exactly one outer worker.");
        }

        matrixElementsPerGroup = checkedProduct(operators.nz, operators.nj, "direct WVM matrix");
        const auto familyElements = checkedProduct(
            operators.groups.size(), matrixElementsPerGroup, "direct WVM matrix family");
        if (operators.forward.size() != familyElements || operators.inverse.size() != familyElements) {
            throw std::invalid_argument(
                "Direct WVM vertical GEMM matrix-family storage does not match its groups and dimensions.");
        }
        std::size_t expectedFirstMode = 0;
        modeGroupIndices.resize(modes.size());
        for (std::size_t groupIndex = 0; groupIndex < operators.groups.size(); ++groupIndex) {
            const auto& group = operators.groups[groupIndex];
            if (group.modeCount == 0 || group.firstMode != expectedFirstMode ||
                group.firstMode + group.modeCount > modes.size()) {
                throw std::invalid_argument(
                    "Direct WVM vertical GEMM groups must be nonempty, contiguous, and cover the modes.");
            }
            std::fill_n(modeGroupIndices.begin() + static_cast<std::ptrdiff_t>(group.firstMode),
                        group.modeCount, groupIndex);
            expectedFirstMode += group.modeCount;
        }
        if (expectedFirstMode != modes.size()) {
            throw std::invalid_argument("Direct WVM vertical GEMM groups do not cover every retained mode.");
        }

        spectrumOffsets.reserve(modes.size());
        modalOffsets.reserve(modes.size());
        std::vector<std::size_t> storedFrequencyIndices;
        storedFrequencyIndices.reserve(modes.size());
        for (const auto& mode : modes) {
            const auto frequencyIndex = mode.storedKx + workload.nxHalf() * mode.storedKy;
            storedFrequencyIndices.push_back(frequencyIndex);
            spectrumOffsets.push_back(workload.planes() * frequencyIndex);
            modalOffsets.push_back(
                workload.retainedVerticalModes() * workload.fields * frequencyIndex);
        }
        std::sort(storedFrequencyIndices.begin(), storedFrequencyIndices.end());
        if (std::adjacent_find(storedFrequencyIndices.begin(), storedFrequencyIndices.end()) !=
            storedFrequencyIndices.end()) {
            throw std::invalid_argument(
                "Direct WVM vertical GEMM requires a unique stored frequency for every retained mode.");
        }

        modalSpectrumCount = checkedProduct(
            workload.halfRows(), checkedProduct(operators.nj, workload.fields, "direct WVM modal plane"),
            "direct WVM modal spectrum");
        nz = checkedBlasDimension(workload.nz, "Nz");
        nj = checkedBlasDimension(operators.nj, "Nj");
        fields = checkedBlasDimension(workload.fields, "fields");

        if (!available) {
            capabilityText = "unsupported: Accelerate BLAS is available only on Apple platforms";
            return;
        }

        const auto allocationStart = Clock::now();
        complexForwardMatrix.allocate(familyElements);
        complexInverseMatrix.allocate(familyElements);
        allocationTime = elapsedSeconds(allocationStart);

        const auto preparationStart = Clock::now();
        for (std::size_t groupIndex = 0; groupIndex < operators.groups.size(); ++groupIndex) {
            const auto offset = groupIndex * matrixElementsPerGroup;
            for (std::size_t z = 0; z < operators.nz; ++z) {
                for (std::size_t j = 0; j < operators.nj; ++j) {
                    const auto forwardValue = operators.forward[offset + j * operators.nz + z];
                    const auto inverseValue = operators.inverse[offset + z * operators.nj + j];
                    complexForwardMatrix.data()[offset + j + operators.nj * z] =
                        {forwardValue, 0.0};
                    complexInverseMatrix.data()[offset + z + operators.nz * j] =
                        {inverseValue, 0.0};
                }
            }
        }
        preparationTime = elapsedSeconds(preparationStart);

        const auto schedulerStart = Clock::now();
        std::vector<int> modeWeights(modes.size(), fields);
        executor = std::make_unique<PersistentGroupExecutor>(
            std::move(modeWeights), strategy.outerWorkers, strategy.schedule);
        strategy.outerWorkers = executor->workerCount();
        schedulerSetupTime = elapsedSeconds(schedulerStart);
        capabilityText = "supported: frequency-major direct per-mode zgemm; " +
            std::string(verticalGemmScheduleName(strategy.schedule)) + "; outer workers=" +
            std::to_string(strategy.outerWorkers);
    }

    void requireAvailable() const {
        if (!available) throw std::runtime_error(capabilityText);
    }

    std::size_t persistentBytes() const noexcept {
        return complexForwardMatrix.bytes() + complexInverseMatrix.bytes() +
            modes.size() * sizeof(RetainedMode) +
            modeGroupIndices.size() * sizeof(std::size_t) +
            spectrumOffsets.size() * sizeof(std::size_t) +
            modalOffsets.size() * sizeof(std::size_t) +
            (executor == nullptr ? 0 : executor->explicitBytes());
    }

    void copyModalBoundary(std::size_t modeIndex, Complex* output) const {
        const auto& mode = modes[modeIndex];
        if (mode.storedKx != 0 || mode.storedKy == 0 || 2 * mode.storedKy == workload.ny) return;
        const auto conjugateKy = (workload.ny - mode.storedKy) % workload.ny;
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t j = 0; j < workload.retainedVerticalModes(); ++j) {
                output[wvmModalSpectrumIndex(workload, 0, conjugateKy, j, field)] =
                    conjugate(output[wvmModalSpectrumIndex(
                        workload, mode.storedKx, mode.storedKy, j, field)]);
            }
        }
    }

    void copySpectrumBoundary(std::size_t modeIndex, Complex* output) const {
        const auto& mode = modes[modeIndex];
        if (mode.storedKx != 0 || mode.storedKy == 0 || 2 * mode.storedKy == workload.ny) return;
        const auto conjugateKy = (workload.ny - mode.storedKy) % workload.ny;
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t z = 0; z < workload.nz; ++z) {
                output[wvmSpectrumIndex(workload, 0, conjugateKy, z, field)] =
                    conjugate(output[wvmSpectrumIndex(
                        workload, mode.storedKx, mode.storedKy, z, field)]);
            }
        }
    }

#if SKBENCH_HAVE_ACCELERATE
    void forwardMode(std::size_t modeIndex, const Complex* input, Complex* output) {
        const BlasComplex alpha{1.0, 0.0};
        const BlasComplex beta{0.0, 0.0};
        const auto matrixOffset = modeGroupIndices[modeIndex] * matrixElementsPerGroup;
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nj, fields, nz,
                    &alpha, complexForwardMatrix.data() + matrixOffset, nj,
                    reinterpret_cast<const BlasComplex*>(input + spectrumOffsets[modeIndex]), nz,
                    &beta, reinterpret_cast<BlasComplex*>(output + modalOffsets[modeIndex]), nj);
        copyModalBoundary(modeIndex, output);
    }

    void inverseMode(std::size_t modeIndex, const Complex* input, Complex* output) {
        const BlasComplex alpha{1.0, 0.0};
        const BlasComplex beta{0.0, 0.0};
        const auto matrixOffset = modeGroupIndices[modeIndex] * matrixElementsPerGroup;
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nz, fields, nj,
                    &alpha, complexInverseMatrix.data() + matrixOffset, nz,
                    reinterpret_cast<const BlasComplex*>(input + modalOffsets[modeIndex]), nj,
                    &beta, reinterpret_cast<BlasComplex*>(output + spectrumOffsets[modeIndex]), nz);
        copySpectrumBoundary(modeIndex, output);
    }

    static void forwardTask(void* opaque, std::size_t modeIndex) {
        auto* context = static_cast<ExecutionContext*>(opaque);
        context->self->forwardMode(modeIndex, context->input, context->output);
    }

    static void inverseTask(void* opaque, std::size_t modeIndex) {
        auto* context = static_cast<ExecutionContext*>(opaque);
        context->self->inverseMode(modeIndex, context->input, context->output);
    }
#endif
};

WvmDirectVerticalGemmProvider::WvmDirectVerticalGemmProvider(
    const Workload& workload, const std::vector<RetainedMode>& modes,
    const GroupedVerticalOperators& operators, VerticalGemmStrategy strategy)
    : impl_(std::make_unique<Impl>(workload, modes, operators, strategy)) {}

WvmDirectVerticalGemmProvider::~WvmDirectVerticalGemmProvider() = default;
WvmDirectVerticalGemmProvider::WvmDirectVerticalGemmProvider(
    WvmDirectVerticalGemmProvider&&) noexcept = default;
WvmDirectVerticalGemmProvider& WvmDirectVerticalGemmProvider::operator=(
    WvmDirectVerticalGemmProvider&&) noexcept = default;

bool WvmDirectVerticalGemmProvider::supported() const noexcept { return impl_->available; }
std::string WvmDirectVerticalGemmProvider::capability() const { return impl_->capabilityText; }
std::size_t WvmDirectVerticalGemmProvider::modalSpectrumElements() const noexcept {
    return impl_->modalSpectrumCount;
}
std::size_t WvmDirectVerticalGemmProvider::gemmCallsPerExecution() const noexcept {
    return impl_->modes.size();
}
std::size_t WvmDirectVerticalGemmProvider::outerWorkers() const noexcept {
    return impl_->strategy.outerWorkers;
}
VerticalGemmStrategy WvmDirectVerticalGemmProvider::strategy() const noexcept {
    return impl_->strategy;
}
std::size_t WvmDirectVerticalGemmProvider::persistentBytes() const noexcept {
    return impl_->persistentBytes();
}
std::size_t WvmDirectVerticalGemmProvider::schedulerPersistentBytes() const noexcept {
    return impl_->executor == nullptr ? 0 : impl_->executor->explicitBytes();
}
std::size_t WvmDirectVerticalGemmProvider::matrixBytesPerDirection() const noexcept {
    return impl_->complexForwardMatrix.bytes();
}
double WvmDirectVerticalGemmProvider::allocationSeconds() const noexcept {
    return impl_->allocationTime;
}
double WvmDirectVerticalGemmProvider::matrixPreparationSeconds() const noexcept {
    return impl_->preparationTime;
}
double WvmDirectVerticalGemmProvider::schedulerSetupSeconds() const noexcept {
    return impl_->schedulerSetupTime;
}
bool WvmDirectVerticalGemmProvider::hasOpaqueSchedulerMemory() const noexcept {
    return impl_->strategy.outerWorkers > 1;
}
std::string WvmDirectVerticalGemmProvider::libraryIdentity() const {
#if SKBENCH_HAVE_ACCELERATE
    return "/System/Library/Frameworks/Accelerate.framework";
#else
    return "unavailable";
#endif
}

void WvmDirectVerticalGemmProvider::initializeModalOutput(Complex* fullModalSpectrum) const {
    impl_->requireAvailable();
    std::fill_n(fullModalSpectrum, impl_->modalSpectrumCount, Complex{});
}

void WvmDirectVerticalGemmProvider::initializeSpectrumOutput(Complex* fullSpectrum) const {
    impl_->requireAvailable();
    std::fill_n(fullSpectrum, impl_->workload.spectrumElements(), Complex{});
}

void WvmDirectVerticalGemmProvider::executeForward(
    const Complex* fullSpectrum, Complex* fullModalSpectrum) {
    impl_->requireAvailable();
#if SKBENCH_HAVE_ACCELERATE
    Impl::ExecutionContext context{impl_.get(), fullSpectrum, fullModalSpectrum};
    if (impl_->strategy.schedule == VerticalGemmSchedule::serial) {
        for (std::size_t mode = 0; mode < impl_->modes.size(); ++mode) {
            impl_->forwardMode(mode, fullSpectrum, fullModalSpectrum);
        }
    } else {
        impl_->executor->run(&context, &Impl::forwardTask);
    }
#endif
}

void WvmDirectVerticalGemmProvider::executeInverse(
    const Complex* fullModalSpectrum, Complex* fullSpectrum) {
    impl_->requireAvailable();
#if SKBENCH_HAVE_ACCELERATE
    Impl::ExecutionContext context{impl_.get(), fullModalSpectrum, fullSpectrum};
    if (impl_->strategy.schedule == VerticalGemmSchedule::serial) {
        for (std::size_t mode = 0; mode < impl_->modes.size(); ++mode) {
            impl_->inverseMode(mode, fullModalSpectrum, fullSpectrum);
        }
    } else {
        impl_->executor->run(&context, &Impl::inverseTask);
    }
#endif
}

void WvmDirectVerticalGemmProvider::executeSchedulerNoop() {
    impl_->requireAvailable();
    if (impl_->strategy.schedule != VerticalGemmSchedule::serial) {
        impl_->executor->run(nullptr, &noopGroup);
    }
}

struct PlaneMajorDirectVerticalGemmProvider::Impl {
    struct ExecutionContext {
        Impl* self = nullptr;
        const Complex* input = nullptr;
        Complex* output = nullptr;
    };

    Workload workload;
    std::size_t modalSpectrumCount = 0;
    std::size_t matrixElementsPerGroup = 0;
    std::size_t spectrumPlane = 0;
    VerticalGemmStrategy strategy;
    bool available = SKBENCH_HAVE_ACCELERATE != 0;
    std::string capabilityText;
    double allocationTime = 0.0;
    double preparationTime = 0.0;
    double schedulerSetupTime = 0.0;
    std::vector<RetainedMode> modes;
    std::vector<std::size_t> modeGroupIndices;
    std::vector<std::size_t> frequencyOffsets;
    std::unique_ptr<PersistentGroupExecutor> executor;
    AlignedBuffer<BlasComplex> complexForwardMatrix;
    AlignedBuffer<BlasComplex> complexInverseMatrix;
    int nz = 0;
    int nj = 0;

    Impl(const Workload& inputWorkload, const std::vector<RetainedMode>& inputModes,
         const GroupedVerticalOperators& operators, VerticalGemmStrategy inputStrategy)
        : workload(inputWorkload), strategy(inputStrategy), modes(inputModes) {
        if (operators.nz != workload.nz || operators.nj != workload.retainedVerticalModes()) {
            throw std::invalid_argument(
                "Plane-major direct vertical GEMV operator dimensions do not match the workload.");
        }
        if (modes.empty() || operators.groups.empty() || workload.fields == 0) {
            throw std::invalid_argument(
                "Plane-major direct vertical GEMV requires modes, groups, and fields.");
        }
        if (strategy.outerWorkers == 0) {
            throw std::invalid_argument("Plane-major direct vertical GEMV workers must be positive.");
        }
        if (strategy.schedule == VerticalGemmSchedule::serial && strategy.outerWorkers != 1) {
            throw std::invalid_argument(
                "The serial plane-major direct vertical GEMV schedule requires one outer worker.");
        }

        matrixElementsPerGroup = checkedProduct(operators.nz, operators.nj, "plane-major matrix");
        const auto familyElements = checkedProduct(
            operators.groups.size(), matrixElementsPerGroup, "plane-major matrix family");
        if (operators.forward.size() != familyElements || operators.inverse.size() != familyElements) {
            throw std::invalid_argument(
                "Plane-major vertical GEMV matrix-family storage does not match its groups and dimensions.");
        }

        std::size_t expectedFirstMode = 0;
        modeGroupIndices.resize(modes.size());
        for (std::size_t groupIndex = 0; groupIndex < operators.groups.size(); ++groupIndex) {
            const auto& group = operators.groups[groupIndex];
            if (group.modeCount == 0 || group.firstMode != expectedFirstMode ||
                group.firstMode + group.modeCount > modes.size()) {
                throw std::invalid_argument(
                    "Plane-major vertical GEMV groups must be contiguous and cover the modes.");
            }
            std::fill_n(modeGroupIndices.begin() + static_cast<std::ptrdiff_t>(group.firstMode),
                        group.modeCount, groupIndex);
            expectedFirstMode += group.modeCount;
        }
        if (expectedFirstMode != modes.size()) {
            throw std::invalid_argument("Plane-major vertical GEMV groups do not cover every mode.");
        }

        spectrumPlane = workload.halfRows();
        frequencyOffsets.reserve(modes.size());
        std::vector<std::size_t> storedFrequencies;
        storedFrequencies.reserve(modes.size());
        for (const auto& mode : modes) {
            const auto frequency = mode.storedKx + workload.nxHalf() * mode.storedKy;
            frequencyOffsets.push_back(frequency);
            storedFrequencies.push_back(frequency);
        }
        std::sort(storedFrequencies.begin(), storedFrequencies.end());
        if (std::adjacent_find(storedFrequencies.begin(), storedFrequencies.end()) !=
            storedFrequencies.end()) {
            throw std::invalid_argument(
                "Plane-major direct vertical GEMV requires unique stored frequencies.");
        }

        modalSpectrumCount = checkedProduct(
            spectrumPlane,
            checkedProduct(operators.nj, workload.fields, "plane-major modal planes"),
            "plane-major modal spectrum");
        nz = checkedBlasDimension(workload.nz, "Nz");
        nj = checkedBlasDimension(operators.nj, "Nj");
        checkedBlasDimension(spectrumPlane, "plane-major stride");

        if (!available) {
            capabilityText = "unsupported: Accelerate BLAS is available only on Apple platforms";
            return;
        }

        const auto allocationStart = Clock::now();
        complexForwardMatrix.allocate(familyElements);
        complexInverseMatrix.allocate(familyElements);
        allocationTime = elapsedSeconds(allocationStart);

        const auto preparationStart = Clock::now();
        for (std::size_t groupIndex = 0; groupIndex < operators.groups.size(); ++groupIndex) {
            const auto offset = groupIndex * matrixElementsPerGroup;
            for (std::size_t z = 0; z < operators.nz; ++z) {
                for (std::size_t j = 0; j < operators.nj; ++j) {
                    complexForwardMatrix.data()[offset + j + operators.nj * z] =
                        {operators.forward[offset + j * operators.nz + z], 0.0};
                    complexInverseMatrix.data()[offset + z + operators.nz * j] =
                        {operators.inverse[offset + z * operators.nj + j], 0.0};
                }
            }
        }
        preparationTime = elapsedSeconds(preparationStart);

        const auto schedulerStart = Clock::now();
        std::vector<int> modeWeights(modes.size(), checkedBlasDimension(workload.fields, "fields"));
        executor = std::make_unique<PersistentGroupExecutor>(
            std::move(modeWeights), strategy.outerWorkers, strategy.schedule);
        strategy.outerWorkers = executor->workerCount();
        schedulerSetupTime = elapsedSeconds(schedulerStart);
        capabilityText = "supported: plane-major retained view with strided per-field zgemv; " +
            std::string(verticalGemmScheduleName(strategy.schedule)) + "; outer workers=" +
            std::to_string(strategy.outerWorkers);
    }

    void requireAvailable() const {
        if (!available) throw std::runtime_error(capabilityText);
    }

    std::size_t persistentBytes() const noexcept {
        return complexForwardMatrix.bytes() + complexInverseMatrix.bytes() +
            modes.size() * sizeof(RetainedMode) +
            modeGroupIndices.size() * sizeof(std::size_t) +
            frequencyOffsets.size() * sizeof(std::size_t) +
            (executor == nullptr ? 0 : executor->explicitBytes());
    }

    std::size_t modalIndex(std::size_t frequency, std::size_t j, std::size_t field) const {
        return frequency + spectrumPlane * (j + workload.retainedVerticalModes() * field);
    }

    void repairModalBoundaries(Complex* output) const {
        for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
            const auto& mode = modes[modeIndex];
            if (mode.storedKx != 0 || mode.storedKy == 0 || 2 * mode.storedKy == workload.ny) {
                continue;
            }
            const auto source = frequencyOffsets[modeIndex];
            const auto target = workload.nxHalf() * ((workload.ny - mode.storedKy) % workload.ny);
            for (std::size_t field = 0; field < workload.fields; ++field) {
                for (std::size_t j = 0; j < workload.retainedVerticalModes(); ++j) {
                    output[modalIndex(target, j, field)] = conjugate(output[modalIndex(source, j, field)]);
                }
            }
        }
    }

    void repairSpectrumBoundaries(Complex* output) const {
        for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
            const auto& mode = modes[modeIndex];
            if (mode.storedKx != 0 || mode.storedKy == 0 || 2 * mode.storedKy == workload.ny) {
                continue;
            }
            const auto source = frequencyOffsets[modeIndex];
            const auto target = workload.nxHalf() * ((workload.ny - mode.storedKy) % workload.ny);
            for (std::size_t field = 0; field < workload.fields; ++field) {
                for (std::size_t z = 0; z < workload.nz; ++z) {
                    const auto sourceIndex = source + spectrumPlane * (z + workload.nz * field);
                    const auto targetIndex = target + spectrumPlane * (z + workload.nz * field);
                    output[targetIndex] = conjugate(output[sourceIndex]);
                }
            }
        }
    }

#if SKBENCH_HAVE_ACCELERATE
    void forwardMode(std::size_t modeIndex, const Complex* input, Complex* output) {
        const BlasComplex alpha{1.0, 0.0};
        const BlasComplex beta{0.0, 0.0};
        const auto matrixOffset = modeGroupIndices[modeIndex] * matrixElementsPerGroup;
        const auto frequency = frequencyOffsets[modeIndex];
        for (std::size_t field = 0; field < workload.fields; ++field) {
            const auto inputOffset = frequency + spectrumPlane * workload.nz * field;
            const auto outputOffset = modalIndex(frequency, 0, field);
            cblas_zgemv(CblasColMajor, CblasNoTrans, nj, nz,
                        &alpha, complexForwardMatrix.data() + matrixOffset, nj,
                        reinterpret_cast<const BlasComplex*>(input + inputOffset),
                        checkedBlasDimension(spectrumPlane, "plane-major input stride"),
                        &beta, reinterpret_cast<BlasComplex*>(output + outputOffset),
                        checkedBlasDimension(spectrumPlane, "plane-major output stride"));
        }
    }

    void inverseMode(std::size_t modeIndex, const Complex* input, Complex* output) {
        const BlasComplex alpha{1.0, 0.0};
        const BlasComplex beta{0.0, 0.0};
        const auto matrixOffset = modeGroupIndices[modeIndex] * matrixElementsPerGroup;
        const auto frequency = frequencyOffsets[modeIndex];
        for (std::size_t field = 0; field < workload.fields; ++field) {
            const auto inputOffset = modalIndex(frequency, 0, field);
            const auto outputOffset = frequency + spectrumPlane * workload.nz * field;
            cblas_zgemv(CblasColMajor, CblasNoTrans, nz, nj,
                        &alpha, complexInverseMatrix.data() + matrixOffset, nz,
                        reinterpret_cast<const BlasComplex*>(input + inputOffset),
                        checkedBlasDimension(spectrumPlane, "plane-major input stride"),
                        &beta, reinterpret_cast<BlasComplex*>(output + outputOffset),
                        checkedBlasDimension(spectrumPlane, "plane-major output stride"));
        }
    }

    static void forwardTask(void* opaque, std::size_t modeIndex) {
        auto* context = static_cast<ExecutionContext*>(opaque);
        context->self->forwardMode(modeIndex, context->input, context->output);
    }

    static void inverseTask(void* opaque, std::size_t modeIndex) {
        auto* context = static_cast<ExecutionContext*>(opaque);
        context->self->inverseMode(modeIndex, context->input, context->output);
    }
#endif
};

PlaneMajorDirectVerticalGemmProvider::PlaneMajorDirectVerticalGemmProvider(
    const Workload& workload, const std::vector<RetainedMode>& modes,
    const GroupedVerticalOperators& operators, VerticalGemmStrategy strategy)
    : impl_(std::make_unique<Impl>(workload, modes, operators, strategy)) {}

PlaneMajorDirectVerticalGemmProvider::~PlaneMajorDirectVerticalGemmProvider() = default;
PlaneMajorDirectVerticalGemmProvider::PlaneMajorDirectVerticalGemmProvider(
    PlaneMajorDirectVerticalGemmProvider&&) noexcept = default;
PlaneMajorDirectVerticalGemmProvider& PlaneMajorDirectVerticalGemmProvider::operator=(
    PlaneMajorDirectVerticalGemmProvider&&) noexcept = default;

bool PlaneMajorDirectVerticalGemmProvider::supported() const noexcept { return impl_->available; }
std::string PlaneMajorDirectVerticalGemmProvider::capability() const { return impl_->capabilityText; }

void PlaneMajorDirectVerticalGemmProvider::initializeModalOutput(Complex* output) const {
    impl_->requireAvailable();
    std::fill_n(output, impl_->modalSpectrumCount, Complex{});
}

void PlaneMajorDirectVerticalGemmProvider::initializeSpectrumOutput(Complex* output) const {
    impl_->requireAvailable();
    std::fill_n(output, impl_->workload.spectrumElements(), Complex{});
}

void PlaneMajorDirectVerticalGemmProvider::executeForward(
    const Complex* fullSpectrum, Complex* fullModalSpectrum) {
    impl_->requireAvailable();
#if SKBENCH_HAVE_ACCELERATE
    Impl::ExecutionContext context{impl_.get(), fullSpectrum, fullModalSpectrum};
    if (impl_->strategy.schedule == VerticalGemmSchedule::serial) {
        for (std::size_t mode = 0; mode < impl_->modes.size(); ++mode) {
            impl_->forwardMode(mode, fullSpectrum, fullModalSpectrum);
        }
    } else {
        impl_->executor->run(&context, &Impl::forwardTask);
    }
    impl_->repairModalBoundaries(fullModalSpectrum);
#endif
}

void PlaneMajorDirectVerticalGemmProvider::executeInverse(
    const Complex* fullModalSpectrum, Complex* fullSpectrum) {
    impl_->requireAvailable();
#if SKBENCH_HAVE_ACCELERATE
    Impl::ExecutionContext context{impl_.get(), fullModalSpectrum, fullSpectrum};
    if (impl_->strategy.schedule == VerticalGemmSchedule::serial) {
        for (std::size_t mode = 0; mode < impl_->modes.size(); ++mode) {
            impl_->inverseMode(mode, fullModalSpectrum, fullSpectrum);
        }
    } else {
        impl_->executor->run(&context, &Impl::inverseTask);
    }
    impl_->repairSpectrumBoundaries(fullSpectrum);
#endif
}

void PlaneMajorDirectVerticalGemmProvider::executeSchedulerNoop() {
    impl_->requireAvailable();
    if (impl_->strategy.schedule != VerticalGemmSchedule::serial) {
        impl_->executor->run(nullptr, &noopGroup);
    }
}

std::size_t PlaneMajorDirectVerticalGemmProvider::modalSpectrumElements() const noexcept {
    return impl_->modalSpectrumCount;
}
std::size_t PlaneMajorDirectVerticalGemmProvider::gemvCallsPerExecution() const noexcept {
    return impl_->modes.size() * impl_->workload.fields;
}
std::size_t PlaneMajorDirectVerticalGemmProvider::outerWorkers() const noexcept {
    return impl_->strategy.outerWorkers;
}
VerticalGemmStrategy PlaneMajorDirectVerticalGemmProvider::strategy() const noexcept {
    return impl_->strategy;
}
std::size_t PlaneMajorDirectVerticalGemmProvider::persistentBytes() const noexcept {
    return impl_->persistentBytes();
}
std::size_t PlaneMajorDirectVerticalGemmProvider::schedulerPersistentBytes() const noexcept {
    return impl_->executor == nullptr ? 0 : impl_->executor->explicitBytes();
}
std::size_t PlaneMajorDirectVerticalGemmProvider::matrixBytesPerDirection() const noexcept {
    return impl_->complexForwardMatrix.bytes();
}
double PlaneMajorDirectVerticalGemmProvider::allocationSeconds() const noexcept {
    return impl_->allocationTime;
}
double PlaneMajorDirectVerticalGemmProvider::matrixPreparationSeconds() const noexcept {
    return impl_->preparationTime;
}
double PlaneMajorDirectVerticalGemmProvider::schedulerSetupSeconds() const noexcept {
    return impl_->schedulerSetupTime;
}
bool PlaneMajorDirectVerticalGemmProvider::hasOpaqueSchedulerMemory() const noexcept {
    return impl_->strategy.outerWorkers > 1;
}
std::string PlaneMajorDirectVerticalGemmProvider::libraryIdentity() const {
#if SKBENCH_HAVE_ACCELERATE
    return "/System/Library/Frameworks/Accelerate.framework";
#else
    return "unavailable";
#endif
}

} // namespace skbench
