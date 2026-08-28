#include "skbench/skbench.hpp"

#include <fftw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#include <sys/utsname.h>
#include <sys/resource.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#ifndef SKBENCH_BUILD_FLAGS
#define SKBENCH_BUILD_FLAGS "unknown"
#endif
#ifndef SKBENCH_GIT_COMMIT
#define SKBENCH_GIT_COMMIT "unknown"
#endif
#ifndef SKBENCH_GIT_DIRTY
#define SKBENCH_GIT_DIRTY 1
#endif

namespace skbench {
namespace {

using Clock = std::chrono::steady_clock;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double tolerance = 1.0e-12;

std::uint64_t bytes(std::size_t count, std::size_t elementSize) {
    if (count != 0 && elementSize > std::numeric_limits<std::uint64_t>::max() / count) {
        throw std::overflow_error("byte count overflow");
    }
    return static_cast<std::uint64_t>(count) * static_cast<std::uint64_t>(elementSize);
}

std::uint64_t processHighWaterBytes() noexcept {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) return 0;
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
#endif
}

template <typename Value>
class FFTWArray {
public:
    explicit FFTWArray(std::size_t count) : count_(count) {
        storage_ = static_cast<Value*>(fftw_malloc(count_ * sizeof(Value)));
        if (storage_ == nullptr) throw std::bad_alloc();
    }

    ~FFTWArray() { fftw_free(storage_); }
    FFTWArray(const FFTWArray&) = delete;
    FFTWArray& operator=(const FFTWArray&) = delete;

    Value* data() noexcept { return storage_; }
    const Value* data() const noexcept { return storage_; }
    Value* begin() noexcept { return storage_; }
    const Value* begin() const noexcept { return storage_; }
    Value* end() noexcept { return storage_ + count_; }
    const Value* end() const noexcept { return storage_ + count_; }
    std::size_t size() const noexcept { return count_; }

private:
    Value* storage_ = nullptr;
    std::size_t count_ = 0;
};

template <typename Prepare, typename Action>
std::vector<double> measure(std::size_t warmups, std::size_t samples, Prepare prepare, Action action) {
    for (std::size_t index = 0; index < warmups; ++index) {
        prepare();
        action();
    }
    std::vector<double> result;
    result.reserve(samples);
    for (std::size_t index = 0; index < samples; ++index) {
        prepare();
        const auto start = Clock::now();
        action();
        result.push_back(std::chrono::duration<double>(Clock::now() - start).count());
    }
    return result;
}

template <typename Action>
std::vector<double> measure(std::size_t warmups, std::size_t samples, Action action) {
    return measure(warmups, samples, [] {}, std::move(action));
}

TimingSeries series(std::string scope, std::string stage, std::string direction, StageState state,
                    std::uint64_t bytesMoved, std::vector<double> samples = {}) {
    return {std::move(scope), std::move(stage), std::move(direction), state, bytesMoved, std::move(samples)};
}

CorrectnessMetric metric(std::string name, double error) {
    return {std::move(name), error, tolerance, error <= tolerance};
}

CorrectnessMetric metric(std::string name, const Complex* actual, const Complex* expected, std::size_t count) {
    const auto maximumError = maximumRelativeError(actual, expected, count);
    const auto l2Error = relativeL2Error(actual, expected, count);
    return {std::move(name), maximumError, tolerance,
            maximumError <= tolerance && l2Error <= tolerance, l2Error};
}

bool correctnessPassed(const ProviderRecord& provider) {
    return std::all_of(provider.correctness.begin(), provider.correctness.end(), [](const CorrectnessMetric& item) {
        return item.passed;
    });
}

std::vector<Complex> directRetained(const Workload& workload, const std::vector<RetainedMode>& modes, const double* input) {
    std::vector<Complex> result(modes.size() * workload.planes());
    for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        const auto& mode = modes[modeIndex];
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t z = 0; z < workload.nz; ++z) {
                Complex sum;
                for (std::size_t y = 0; y < workload.ny; ++y) {
                    for (std::size_t x = 0; x < workload.nx; ++x) {
                        const double angle = -2.0 * pi *
                            (static_cast<double>(mode.k) * static_cast<double>(x) / static_cast<double>(workload.nx) +
                             static_cast<double>(mode.l) * static_cast<double>(y) / static_cast<double>(workload.ny));
                        const double value = input[realIndex(workload, x, y, z, field)];
                        sum.real += value * std::cos(angle);
                        sum.imag += value * std::sin(angle);
                    }
                }
                result[retainedSpectrumIndex(workload, modeIndex, z, field)] = sum;
            }
        }
    }
    return result;
}

std::vector<Complex> directHorizontalVerticalProjection(const Workload& workload, const std::vector<RetainedMode>& modes,
                                                        const VerticalOperators& vertical, const double* input) {
    std::vector<Complex> result(modes.size() * workload.fields * vertical.nj);
    for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        const auto& mode = modes[modeIndex];
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t j = 0; j < vertical.nj; ++j) {
                Complex sum;
                for (std::size_t z = 0; z < workload.nz; ++z) {
                    const double verticalFactor = vertical.forward[j * workload.nz + z];
                    for (std::size_t y = 0; y < workload.ny; ++y) {
                        for (std::size_t x = 0; x < workload.nx; ++x) {
                            const double angle = -2.0 * pi *
                                (static_cast<double>(mode.k) * static_cast<double>(x) / static_cast<double>(workload.nx) +
                                 static_cast<double>(mode.l) * static_cast<double>(y) / static_cast<double>(workload.ny));
                            const double value = verticalFactor * input[realIndex(workload, x, y, z, field)];
                            sum.real += value * std::cos(angle);
                            sum.imag += value * std::sin(angle);
                        }
                    }
                }
                result[modalSpectrumIndex(workload, modeIndex, j, field)] = sum;
            }
        }
    }
    return result;
}

double deterministicSignedUnit(std::uint64_t& state) {
    state ^= state >> 12U;
    state ^= state << 25U;
    state ^= state >> 27U;
    const auto value = state * UINT64_C(2685821657736338717);
    const auto mantissa = value >> 11U;
    return 2.0 * static_cast<double>(mantissa) / static_cast<double>(UINT64_C(1) << 53U) - 1.0;
}

std::vector<Complex> verticalComplexFixture(std::size_t count, std::uint64_t seed) {
    std::uint64_t state = seed == 0 ? UINT64_C(0x9e3779b97f4a7c15) : seed;
    std::vector<Complex> result(count);
    for (auto& value : result) {
        value = {deterministicSignedUnit(state), deterministicSignedUnit(state)};
    }
    return result;
}

std::vector<std::size_t> verticalProbeColumns(std::size_t columns) {
    const auto count = std::min<std::size_t>(17, columns);
    std::vector<std::size_t> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto column = count == 1 ? 0 : index * (columns - 1) / (count - 1);
        if (result.empty() || result.back() != column) result.push_back(column);
    }
    return result;
}

std::size_t verticalGroupForColumn(const GroupedVerticalOperators& operators,
                                   const Workload& workload, std::size_t column) {
    const auto mode = column / workload.fields;
    const auto match = std::find_if(
        operators.groups.begin(), operators.groups.end(), [mode](const VerticalModeGroup& group) {
            return mode >= group.firstMode && mode < group.firstMode + group.modeCount;
        });
    if (match == operators.groups.end()) {
        throw std::logic_error("Vertical probe column is outside the matrix-family groups.");
    }
    return static_cast<std::size_t>(std::distance(operators.groups.begin(), match));
}

std::vector<Complex> directVerticalForwardProbes(const GroupedVerticalOperators& operators,
                                                 const Workload& workload,
                                                 const std::vector<std::size_t>& columns,
                                                 const Complex* physical) {
    std::vector<Complex> result(columns.size() * operators.nj);
    const auto matrixElements = operators.nj * operators.nz;
    for (std::size_t probe = 0; probe < columns.size(); ++probe) {
        const auto column = columns[probe];
        const auto matrixOffset = verticalGroupForColumn(operators, workload, column) * matrixElements;
        for (std::size_t j = 0; j < operators.nj; ++j) {
            Complex sum;
            for (std::size_t z = 0; z < operators.nz; ++z) {
                const auto value = physical[z + operators.nz * column];
                const auto factor = operators.forward[matrixOffset + j * operators.nz + z];
                sum.real += factor * value.real;
                sum.imag += factor * value.imag;
            }
            result[j + operators.nj * probe] = sum;
        }
    }
    return result;
}

std::vector<Complex> directVerticalInverseProbes(const GroupedVerticalOperators& operators,
                                                 const Workload& workload,
                                                 const std::vector<std::size_t>& columns,
                                                 const Complex* modal) {
    std::vector<Complex> result(columns.size() * operators.nz);
    const auto matrixElements = operators.nj * operators.nz;
    for (std::size_t probe = 0; probe < columns.size(); ++probe) {
        const auto column = columns[probe];
        const auto matrixOffset = verticalGroupForColumn(operators, workload, column) * matrixElements;
        for (std::size_t z = 0; z < operators.nz; ++z) {
            Complex sum;
            for (std::size_t j = 0; j < operators.nj; ++j) {
                const auto value = modal[j + operators.nj * column];
                const auto factor = operators.inverse[matrixOffset + z * operators.nj + j];
                sum.real += factor * value.real;
                sum.imag += factor * value.imag;
            }
            result[z + operators.nz * probe] = sum;
        }
    }
    return result;
}

std::vector<Complex> gatherVerticalProbes(const Complex* values, std::size_t rows,
                                          const std::vector<std::size_t>& columns) {
    std::vector<Complex> result(rows * columns.size());
    for (std::size_t probe = 0; probe < columns.size(); ++probe) {
        std::copy_n(values + rows * columns[probe], rows, result.data() + rows * probe);
    }
    return result;
}

std::size_t planeMajorModalIndex(const Workload& workload, std::size_t frequency,
                                 std::size_t j, std::size_t field) {
    return frequency + workload.halfRows() *
        (j + workload.retainedVerticalModes() * field);
}

void compactModalToPlaneMajor(const Workload& workload,
                              const std::vector<RetainedMode>& modes,
                              const Complex* compact, Complex* planeMajor) {
    const auto count = workload.halfRows() * workload.retainedVerticalModes() * workload.fields;
    std::fill_n(planeMajor, count, Complex{});
    for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        const auto& mode = modes[modeIndex];
        const auto frequency = mode.storedKx + workload.nxHalf() * mode.storedKy;
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t j = 0; j < workload.retainedVerticalModes(); ++j) {
                auto value = compact[modalSpectrumIndex(workload, modeIndex, j, field)];
                if (mode.conjugatesStoredValue) value = conjugate(value);
                planeMajor[planeMajorModalIndex(workload, frequency, j, field)] = value;
                if (mode.storedKx == 0 && mode.storedKy != 0 &&
                    2 * mode.storedKy != workload.ny) {
                    const auto conjugateFrequency = workload.nxHalf() *
                        ((workload.ny - mode.storedKy) % workload.ny);
                    planeMajor[planeMajorModalIndex(
                        workload, conjugateFrequency, j, field)] = conjugate(value);
                }
            }
        }
    }
}

void planeMajorModalToCompact(const Workload& workload,
                              const std::vector<RetainedMode>& modes,
                              const Complex* planeMajor, Complex* compact) {
    for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        const auto& mode = modes[modeIndex];
        const auto frequency = mode.storedKx + workload.nxHalf() * mode.storedKy;
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t j = 0; j < workload.retainedVerticalModes(); ++j) {
                auto value = planeMajor[planeMajorModalIndex(workload, frequency, j, field)];
                if (mode.conjugatesStoredValue) value = conjugate(value);
                compact[modalSpectrumIndex(workload, modeIndex, j, field)] = value;
            }
        }
    }
}

void planeMajorPhysicalToCompact(const Workload& workload,
                                 const std::vector<RetainedMode>& modes,
                                 const Complex* planeMajor, Complex* compact) {
    for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        const auto& mode = modes[modeIndex];
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t z = 0; z < workload.nz; ++z) {
                auto value = planeMajor[planeMajorSpectrumIndex(
                    workload, mode.storedKx, mode.storedKy, z, field)];
                if (mode.conjugatesStoredValue) value = conjugate(value);
                compact[retainedSpectrumIndex(workload, modeIndex, z, field)] = value;
            }
        }
    }
}

std::size_t adjacentEquivalentVerticalGroupPairs(const GroupedVerticalOperators& operators) {
    if (operators.groups.size() < 2) return 0;
    const auto matrixElements = operators.nz * operators.nj;
    std::size_t equivalent = 0;
    for (std::size_t group = 1; group < operators.groups.size(); ++group) {
        const auto previous = (group - 1) * matrixElements;
        const auto current = group * matrixElements;
        const bool forwardEqual = std::equal(
            operators.forward.begin() + static_cast<std::ptrdiff_t>(previous),
            operators.forward.begin() + static_cast<std::ptrdiff_t>(previous + matrixElements),
            operators.forward.begin() + static_cast<std::ptrdiff_t>(current));
        const bool inverseEqual = std::equal(
            operators.inverse.begin() + static_cast<std::ptrdiff_t>(previous),
            operators.inverse.begin() + static_cast<std::ptrdiff_t>(previous + matrixElements),
            operators.inverse.begin() + static_cast<std::ptrdiff_t>(current));
        if (forwardEqual && inverseEqual) ++equivalent;
    }
    return equivalent;
}

std::size_t configuredAccelerateThreads(const EnvironmentRecord& environment) {
    const char* value = std::getenv("VECLIB_MAXIMUM_THREADS");
    if (value == nullptr || *value == '\0') return std::max<std::size_t>(1, environment.totalCores);
    try {
        std::size_t consumed = 0;
        const auto parsed = std::stoull(value, &consumed);
        if (parsed == 0 || consumed != std::string_view(value).size()) {
            throw std::invalid_argument("not a positive integer");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::invalid_argument("VECLIB_MAXIMUM_THREADS must be a positive integer when set.");
    }
}

std::string accelerateSchedulingId() {
    const char* value = std::getenv("VECLIB_MAXIMUM_THREADS");
    return value == nullptr || *value == '\0'
        ? "accelerate-system-default"
        : "accelerate-veclib-maximum-threads-" + std::string(value);
}

std::string verticalGemmSchedulingId(VerticalGemmStrategy strategy) {
    return "vertical-gemm-" + std::string(verticalGemmScheduleName(strategy.schedule)) +
        "-outer-workers-" + std::to_string(strategy.outerWorkers) + ";" + accelerateSchedulingId();
}

ExecutionContract verticalGemmExecutionContract(const VerticalGemmProvider& provider,
                                                const Workload& workload) {
    const auto columns = provider.columns();
    const auto nj = workload.retainedVerticalModes();
    const auto split = provider.layout() == VerticalGemmLayout::split;
    const auto representation = split
        ? "vertical-columns-split-complex"
        : "vertical-columns-interleaved-complex";
    const std::string component = split ? "two disjoint real/imaginary arrays, each " : "";
    const auto physical = component + "[Nz=" + std::to_string(workload.nz) + "][K=" +
        std::to_string(columns) + "]";
    const auto modal = component + "[Nj=" + std::to_string(nj) + "][K=" +
        std::to_string(columns) + "]";
    const auto physicalStrides = "z=1,column=" + std::to_string(workload.nz) +
        ",column-key=field+fields*mode";
    const auto modalStrides = "j=1,column=" + std::to_string(nj) +
        ",column-key=field+fields*mode";
    const auto aliasing = split
        ? "real and imaginary components are disjoint; input and output components do not overlap"
        : "input and output complex matrices do not overlap";
    DirectionExecutionContract forward{
        "out-of-place", "out-of-place", false, true, false, false, false,
        representation, representation, representation, representation,
        "input=" + physical + "; output=" + modal,
        "input{" + physicalStrides + "}; output{" + modalStrides + "}",
        0, provider.minimumAlignmentBytes(), aliasing, 0, true};
    ExecutionContract contract;
    contract.forward = forward;
    contract.inverse = forward;
    contract.inverse.physicalExtents = "input=" + modal + "; output=" + physical;
    contract.inverse.stridesElements = "input{" + modalStrides + "}; output{" + physicalStrides + "}";
    return contract;
}

std::vector<LedgerEntry> verticalGemmLedger(VerticalGemmLayout layout, std::size_t groupCount,
                                            VerticalGemmStrategy strategy,
                                            std::size_t adjacentEquivalentPairs) {
    const auto split = layout == VerticalGemmLayout::split;
    const auto groupText = std::to_string(groupCount) + " contiguous matrix group(s)";
    const auto scheduleText = std::string(verticalGemmScheduleName(strategy.schedule)) +
        " over " + std::to_string(strategy.outerWorkers) + " outer worker(s)";
    return {
        {"setup/planning", StageState::setupOnly,
         split ? "transpose immutable real forward/inverse matrix families into BLAS column-major storage; " + groupText
               : "transpose and expand immutable real forward/inverse matrix families into complex BLAS column-major storage; " + groupText},
        {"raw forward FFT", StageState::unsupported, "outside this primitive vertical GEMM experiment"},
        {"horizontal retention", StageState::unsupported, "inputs already contain the retained horizontal columns"},
        {"representation conversion", StageState::elided, "operands are pre-arranged before primitive timing"},
        {"permutation/packing", StageState::elided, "excluded from issue #8 primitive timing and owned by issue #13"},
        {"raw forward vertical MM", StageState::executed,
         split ? "two cblas_dgemm calls per matrix group over split operands; " + scheduleText
               : "one cblas_zgemm call per matrix group with each real matrix expanded to complex; " + scheduleText},
        {"variable-size grouped BLAS batching", StageState::unsupported,
         "the installed public Accelerate CBLAS headers expose no variable-size grouped GEMM batch API"},
        {"exact group consolidation", StageState::unsupported,
         groupCount == 1
             ? "the common-matrix baseline is already one consolidated GEMM"
             : std::to_string(adjacentEquivalentPairs) +
                   " adjacent matrix pair(s) are exactly equal; nonadjacent consolidation requires reordering or block-diagonal expansion"},
        {"modal work", StageState::unsupported, "outside this primitive vertical GEMM experiment"},
        {"raw inverse vertical MM", StageState::executed,
         split ? "two cblas_dgemm calls per matrix group over split operands; " + scheduleText
               : "one cblas_zgemm call per matrix group with each real matrix expanded to complex; " + scheduleText},
        {"horizontal embedding", StageState::unsupported, "outside this primitive vertical GEMM experiment"},
        {"raw inverse FFT", StageState::unsupported, "outside this primitive vertical GEMM experiment"},
        {"uninstrumented total", StageState::unsupported, "complete spectral pipeline belongs to issue #9"}};
}

std::string utcTimestamp(bool compact) {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_r(&time, &utc);
    std::ostringstream stream;
    if (compact) {
        const auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1000000;
        stream << std::put_time(&utc, "%Y%m%dT%H%M%S") << std::setfill('0') << std::setw(6) << microseconds << 'Z';
    } else {
        stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    }
    return stream.str();
}

std::string hostName() {
    char buffer[256]{};
    if (gethostname(buffer, sizeof(buffer)) != 0) return "unknown";
    std::string value(buffer);
    const auto separator = value.find('.');
    if (separator != std::string::npos) value.resize(separator);
    return value;
}

#if defined(__APPLE__)
template <typename Value>
Value sysctlValue(const char* name, Value fallback = {}) {
    Value value{};
    std::size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, nullptr, 0) == 0 ? value : fallback;
}

std::string sysctlString(const char* name) {
    std::size_t size = 0;
    if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) return {};
    std::string value(size, '\0');
    if (sysctlbyname(name, value.data(), &size, nullptr, 0) != 0) return {};
    while (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}
#endif

std::string fftwAlgorithmId(const FFTWProvider& provider) {
    const auto strategy = provider.strategy();
    const auto order = strategy.spectrumOrder == FFTWSpectrumOrder::planeMajor
        ? "plane-major-guru64-" : "wvm-guru64-";
    const auto layout = strategy.layout == FFTWDataLayout::split ? "split-" : "interleaved-";
    return std::string(order) + layout + std::string(fftwPlanningModeName(strategy.planningMode)) + "-" +
        std::string(fftwAlignmentStrategyName(strategy.alignment)) + "-" +
        std::string(fftwWisdomStrategyName(strategy.wisdom));
}

std::string fftwSchedulingId(const FFTWProvider& provider) {
    if (provider.outerWorkers() == 1) return "fftw-internal-pthreads";
    if (provider.internalWorkers() == 1) return "persistent-outer-batch-sharding";
    return "persistent-outer-batch-sharding+fftw-internal-pthreads";
}

std::string prunedFftwSchedulingId(const FFTWPrunedProvider& provider) {
    if (provider.outerWorkers() == 1) return "fftw-internal-pthreads";
    if (provider.internalWorkers() == 1) return "persistent-outer-plane-sharding";
    return "persistent-outer-plane-sharding+fftw-internal-pthreads";
}

std::string fftwPlanningConfiguration(const FFTWProvider& provider) {
    const auto strategy = provider.strategy();
    std::ostringstream output;
    output << "FFTW_";
    switch (strategy.planningMode) {
        case FFTWPlanningMode::estimate: output << "ESTIMATE"; break;
        case FFTWPlanningMode::measure: output << "MEASURE"; break;
        case FFTWPlanningMode::patient: output << "PATIENT"; break;
        case FFTWPlanningMode::exhaustive: output << "EXHAUSTIVE"; break;
    }
    if (strategy.alignment == FFTWAlignmentStrategy::unaligned) output << "|FFTW_UNALIGNED";
    output << "; guru64; layout=" << fftwDataLayoutName(strategy.layout)
           << "; spectrum-order=" << fftwSpectrumOrderName(strategy.spectrumOrder)
           << "; wisdom=" << fftwWisdomStrategyName(strategy.wisdom)
           << "; internal-workers=" << provider.internalWorkers()
           << "; outer-workers=" << provider.outerWorkers();
    if (provider.planningTimeLimitSeconds() > 0.0) {
        output << "; per-plan-time-limit-seconds=" << provider.planningTimeLimitSeconds()
               << "; budget-exhausted=" << (provider.planningBudgetExhausted() ? "true" : "false");
    }
    return output.str();
}

std::vector<LedgerEntry> fftwLedger(const FFTWProvider& provider) {
    const auto setup = fftwPlanningConfiguration(provider);
    const auto split = provider.strategy().layout == FFTWDataLayout::split;
    const std::string scheduling = provider.outerWorkers() > 1
        ? "persistent outer batch sharding with separately measured empty-dispatch overhead"
        : "single guru64 batch plan with FFTW internal pthreads";
    return {
        {"setup/planning", StageState::setupOnly, setup},
        {"batch scheduling", provider.outerWorkers() > 1 ? StageState::executed : StageState::elided, scheduling},
        {"raw forward FFT", StageState::executed, split ? "provider-native WVM-strided split r2c" : "provider-native WVM-strided interleaved r2c"},
        {"horizontal retention", StageState::executed, split ? "radial two-thirds mode-keyed gather directly from split arrays" : "radial two-thirds mode-keyed gather"},
        {"representation conversion", split ? StageState::executed : StageState::elided,
         split ? "split/interleaved conversion is required only by the WVM-compatible adapter" : "FFTW writes the WVM frequency-major interleaved representation directly"},
        {"permutation/packing", StageState::elided, "no packing required for the FFTW primitive"},
        {"raw forward vertical MM", StageState::unsupported, "outside this FFT vertical slice"},
        {"modal work", StageState::unsupported, "outside this FFT vertical slice"},
        {"raw inverse vertical MM", StageState::unsupported, "outside this FFT vertical slice"},
        {"horizontal embedding", StageState::executed, split ? "mode-keyed retained split spectrum embedded directly into split half-spectrum arrays" : "mode-keyed retained spectrum embedded into the Hermitian half-spectrum"},
        {"raw inverse FFT", StageState::executed, split ? "provider-native WVM-strided split c2r" : "provider-native WVM-strided interleaved c2r"},
        {"uninstrumented total", StageState::executed, "retained horizontal forward or inverse operator"}};
}

std::vector<LedgerEntry> fftwFusedSplitRetainedLedger(
    const FFTWProvider& provider) {
    auto ledger = fftwLedger(provider);
    for (auto& entry : ledger) {
        if (entry.stage == "horizontal retention") {
            entry.detail = "radial selection, logical conjugation, interleaved-to-split conversion, and optional uniform normalization are one fused pass into compact retained storage";
        } else if (entry.stage == "representation conversion") {
            entry.state = StageState::fused;
            entry.detail = "interleaved full-spectrum to compact split conversion is fused with selection; no standalone conversion pass executes";
        } else if (entry.stage == "horizontal embedding") {
            entry.detail = "compact split coefficients are embedded directly into interleaved full-spectrum scratch with zero fill and Hermitian-boundary repair";
        } else if (entry.stage == "uninstrumented total") {
            entry.detail = "full FFTW primitive plus fused compact split retention or embedding";
        }
    }
    return ledger;
}

std::vector<LedgerEntry> fftwRetainedViewLedger(const FFTWProvider& provider) {
    auto ledger = fftwLedger(provider);
    for (auto& entry : ledger) {
        if (entry.stage == "horizontal retention") {
            entry.state = StageState::elided;
            entry.detail = "the logical retained result is an immutable mode-index view over the provider-order full half-spectrum; no coefficients move";
        } else if (entry.stage == "representation conversion") {
            entry.state = StageState::elided;
            entry.detail = "the provider-order full half-spectrum persists across the forward boundary";
        } else if (entry.stage == "horizontal embedding") {
            entry.state = StageState::elided;
            entry.detail = "the inverse boundary already contains a complete zero-padded provider-order half-spectrum; producing that representation is explicitly deferred to issue #13";
        } else if (entry.stage == "uninstrumented total") {
            entry.detail = "raw FFTW execution from or to a ready persistent provider-order view; inverse input is dead and may be destroyed";
        }
    }
    ledger.push_back({
        "multidimensional inverse input preservation", StageState::unsupported,
        "FFTW 3.3.11 documents no input-preserving algorithms for multidimensional c2r; requesting FFTW_PRESERVE_INPUT returns a null plan"});
    return ledger;
}

std::vector<LedgerEntry> prunedFftwLedger(
    const FFTWPrunedProvider& provider, bool splitRetained = false) {
    const auto columns = std::to_string(provider.columnTransformsPerDirection());
    const auto omitted = std::to_string(provider.omittedColumnTransformsPerDirection());
    const auto planCount = std::to_string(4 * provider.outerWorkers());
    const std::string scheduling = provider.outerWorkers() > 1
        ? "persistent outer plane/field sharding with single-threaded or explicitly declared hybrid FFTW plans"
        : "one separable batch plan set using FFTW internal pthreads";
    return {
        {"setup/planning", StageState::setupOnly,
         planCount + " reusable FFTW guru64 plans and one full-sized plane-major row-spectrum scratch allocation partitioned into disjoint worker slices"},
        {"batch scheduling", provider.outerWorkers() > 1 ? StageState::executed : StageState::elided,
         scheduling + "; maximum shard scratch=" + std::to_string(provider.maximumShardScratchBytes()) + " bytes"},
        {"raw forward FFT", StageState::fused,
         "the retained transform is decomposed into separately timed row and selected-column stages"},
        {"forward real row FFTs", StageState::executed,
         std::to_string(provider.rowTransformsPerDirection()) + " length-Nx out-of-place r2c transforms"},
        {"forward selected complex column FFTs", StageState::executed,
         columns + " length-Ny in-place c2c transforms; " + omitted + " high-kx transforms are elided"},
        {"horizontal retention", StageState::executed,
         splitRetained
             ? "radially retained logical modes are gathered directly from plane-major interleaved scratch into compact split storage"
             : "radially retained logical modes are gathered directly from plane-major row-spectrum scratch"},
        {"representation conversion", splitRetained ? StageState::fused : StageState::elided,
         splitRetained
             ? "interleaved scratch to split retained conversion is fused with radial selection"
             : "the candidate uses interleaved complex scratch and interleaved compact retained output"},
        {"permutation/packing", StageState::elided,
         "no full WVM-order half-spectrum or transpose is materialized"},
        {"raw forward vertical MM", StageState::unsupported, "outside issue #12"},
        {"modal work", StageState::unsupported, "outside issue #12"},
        {"raw inverse vertical MM", StageState::unsupported, "outside issue #12"},
        {"horizontal embedding", StageState::executed,
         splitRetained
             ? "zero full row-spectrum scratch and embed compact split values directly into interleaved scratch with Hermitian-boundary repair"
             : "zero full row-spectrum scratch and embed retained values with Hermitian-boundary repair"},
        {"inverse selected complex column FFTs", StageState::executed,
         columns + " length-Ny in-place c2c transforms; " + omitted + " high-kx transforms are elided"},
        {"raw inverse FFT", StageState::fused,
         "the retained inverse is decomposed into embedding, selected-column, and real-row stages"},
        {"inverse real row FFTs", StageState::executed,
         std::to_string(provider.rowTransformsPerDirection()) + " length-Nx out-of-place c2r transforms"},
        {"complete transformed half-spectrum output", StageState::elided,
         "only the compact retained output is exposed; full-sized first-pass scratch still exists and is reported"},
        {"in-place retained operator", StageState::unsupported,
         provider.inPlaceRetainedOperatorCapability()},
        {"uninstrumented total", StageState::executed,
         "complete partial-column-pruned retained forward or inverse operation"}};
}

ExecutionContract prunedFftwExecutionContract(const Workload& workload,
                                              const FFTWPrunedProvider& provider,
                                              bool splitRetained = false) {
    const auto planes = workload.planes();
    const auto retained = retainedHorizontalModes(workload).size();
    const auto realExtents = "[planes=" + std::to_string(planes) + "][Ny=" +
        std::to_string(workload.ny) + "][Nx=" + std::to_string(workload.nx) + "]";
    const auto scratchExtents = "[planes=" + std::to_string(planes) + "][Ny=" +
        std::to_string(workload.ny) + "][NxHalf=" + std::to_string(workload.nxHalf()) + "]";
    const auto retainedExtents = (splitRetained ? "two arrays " : "") +
        std::string("[mode=") + std::to_string(retained) + "][planes=" +
        std::to_string(planes) + "]";
    const auto retainedRepresentation = splitRetained
        ? "logical-radial-retained-split-complex"
        : "logical-radial-retained-interleaved";
    DirectionExecutionContract forward{
        "out-of-place",
        "out-of-place",
        false,
        true,
        false,
        false,
        false,
        "wvm-x-fastest-real-grid",
        retainedRepresentation,
        "wvm-x-fastest-real-grid",
        retainedRepresentation,
        "input=" + realExtents + "; scratch=" + scratchExtents + "; output=" + retainedExtents,
        "input{x=1,y=Nx,plane=Nx*Ny}; scratch{kx=1,ky=NxHalf,plane=NxHalf*Ny}; output{plane=1,mode=planes}",
        0,
        provider.minimumAlignmentBytes(),
        "caller input and compact output are disjoint; outer workers own disjoint plane shards and selected column transforms overwrite their reusable private scratch slices",
        provider.scratchBytes(),
        true};
    DirectionExecutionContract inverse{
        "out-of-place",
        "out-of-place",
        false,
        true,
        false,
        false,
        false,
        retainedRepresentation,
        "wvm-x-fastest-real-grid",
        retainedRepresentation,
        "wvm-x-fastest-real-grid",
        "input=" + retainedExtents + "; scratch=" + scratchExtents + "; output=" + realExtents,
        "input{plane=1,mode=planes}; scratch{kx=1,ky=NxHalf,plane=NxHalf*Ny}; output{x=1,y=Nx,plane=Nx*Ny}",
        0,
        provider.minimumAlignmentBytes(),
        "caller retained input and real output are disjoint; outer workers initialize disjoint plane shards and inverse transforms overwrite their reusable private scratch slices",
        provider.scratchBytes(),
        true};
    return {std::move(forward), std::move(inverse)};
}

std::string vdspDirectApiName(VDSPTransformStrategy strategy) {
    switch (strategy) {
        case VDSPTransformStrategy::inPlace: return "vDSP_fft2d_zripD";
        case VDSPTransformStrategy::inPlaceExplicitScratch: return "vDSP_fft2d_zriptD";
        case VDSPTransformStrategy::outOfPlace: return "vDSP_fft2d_zropD";
        case VDSPTransformStrategy::outOfPlaceExplicitScratch: return "vDSP_fft2d_zroptD";
    }
    return "unknown";
}

std::string vdspApiName(VDSPTransformStrategy strategy, VDSPBatchStrategy batchStrategy) {
    if (batchStrategy == VDSPBatchStrategy::separablePersistent ||
        batchStrategy == VDSPBatchStrategy::separableGcd) {
        return "vDSP_fft_zripD rows and vDSP_fft_zipD columns";
    }
    return vdspDirectApiName(strategy);
}

std::string vdspAlgorithmId(VDSPTransformStrategy strategy, VDSPBatchStrategy batchStrategy) {
    const auto decomposition = batchStrategy == VDSPBatchStrategy::separablePersistent ||
                               batchStrategy == VDSPBatchStrategy::separableGcd
        ? "separable-packed-real"
        : "native-2d";
    return "vdsp-radix2-" + std::string(decomposition) + "-" +
        std::string(vdspTransformStrategyName(strategy)) + "-" +
        std::string(vdspBatchStrategyName(batchStrategy));
}

std::string vdspSchedulingId(VDSPBatchStrategy batchStrategy) {
    return batchStrategy == VDSPBatchStrategy::directGcd || batchStrategy == VDSPBatchStrategy::separableGcd
        ? "gcd-global-user-initiated"
        : "persistent-thread-pool";
}

std::vector<LedgerEntry> vdspLedger(VDSPTransformStrategy strategy, VDSPBatchStrategy batchStrategy) {
    const auto api = vdspApiName(strategy, batchStrategy);
    const auto separable = batchStrategy == VDSPBatchStrategy::separablePersistent ||
                           batchStrategy == VDSPBatchStrategy::separableGcd;
    return {
        {"setup/planning", StageState::setupOnly, "one radix-2 setup per logical batch worker"},
        {"raw forward FFT", StageState::executed, "batched outer scheduling of " + api},
        {"batch scheduling", StageState::executed, "non-additive empty-dispatch diagnostic for " + vdspSchedulingId(batchStrategy)},
        {"row/column decomposition", separable ? StageState::executed : StageState::fused,
         separable ? "separately timed real-row and complex-column phases" : "native 2-D call owns decomposition"},
        {"horizontal retention", StageState::executed, "radial two-thirds mode-keyed gather"},
        {"representation conversion", StageState::fused, "split/interleaved conversion is fused with vDSP packing and WVM reordering"},
        {"permutation/packing", StageState::fused, "real packing and frequency-major WVM reordering are timed adapter components"},
        {"raw forward vertical MM", StageState::unsupported, "outside this FFT vertical slice"},
        {"modal work", StageState::unsupported, "outside this FFT vertical slice"},
        {"raw inverse vertical MM", StageState::unsupported, "outside this FFT vertical slice"},
        {"horizontal embedding", StageState::executed, "mode-keyed retained spectrum embedded before the inverse adapter"},
        {"raw inverse FFT", StageState::executed, "batched outer scheduling of " + api},
        {"uninstrumented total", StageState::executed, "retained horizontal forward or inverse operator"}};
}

std::vector<LedgerEntry> vdspNativeRetainedLedger(
    VDSPTransformStrategy strategy, VDSPBatchStrategy batchStrategy) {
    const auto api = vdspApiName(strategy, batchStrategy);
    return {
        {"setup/planning", StageState::setupOnly,
         "one radix-2 setup per logical batch worker plus persistent packed split buffers"},
        {"raw forward FFT", StageState::executed,
         "batched outer scheduling of " + api},
        {"batch scheduling", StageState::executed,
         "non-additive empty-dispatch diagnostic for " +
             vdspSchedulingId(batchStrategy)},
        {"horizontal retention", StageState::executed,
         "direct mode-keyed gather from vDSP packed split storage into plane-major compact split storage"},
        {"representation conversion", StageState::elided,
         "split complex persists across the retained-operator boundary"},
        {"permutation/packing", StageState::fused,
         "native packed boundary decoding and radial compact ordering are fused with retention"},
        {"raw forward vertical MM", StageState::unsupported,
         "outside this retained-horizontal experiment"},
        {"modal work", StageState::unsupported,
         "outside this retained-horizontal experiment"},
        {"raw inverse vertical MM", StageState::unsupported,
         "outside this retained-horizontal experiment"},
        {"horizontal embedding", StageState::executed,
         "parallel zero fill and direct mode-keyed embedding from compact split storage into vDSP packing"},
        {"raw inverse FFT", StageState::executed,
         "batched outer scheduling of " + api},
        {"uninstrumented total", StageState::executed,
         "persistent native split retained horizontal forward or inverse operation"}};
}

ExecutionContract fftwExecutionContract(const Workload& workload, const FFTWProvider& provider) {
    const auto planes = workload.planes();
    const auto planeMajor = provider.strategy().spectrumOrder == FFTWSpectrumOrder::planeMajor;
    const auto realExtents = "[planes=" + std::to_string(planes) + "][Ny=" + std::to_string(workload.ny) +
        "][Nx=" + std::to_string(workload.nx) + "]";
    const auto spectrumExtents = planeMajor
        ? "[planes=" + std::to_string(planes) + "][Ny=" + std::to_string(workload.ny) +
            "][NxHalf=" + std::to_string(workload.nxHalf()) + "]"
        : "[Ny=" + std::to_string(workload.ny) + "][NxHalf=" +
            std::to_string(workload.nxHalf()) + "][planes=" + std::to_string(planes) + "]";
    const auto realStrides = "x=1,y=" + std::to_string(workload.nx) + ",plane=" +
        std::to_string(workload.realPlaneElements());
    const auto spectrumStrides = planeMajor
        ? "kx=1,ky=" + std::to_string(workload.nxHalf()) + ",plane=" +
            std::to_string(workload.ny * workload.nxHalf())
        : "plane=1,kx=" + std::to_string(planes) + ",ky=" +
            std::to_string(planes * workload.nxHalf());
    const auto split = provider.strategy().layout == FFTWDataLayout::split;
    const auto orderName = planeMajor ? "plane-major" : "wvm-frequency-major";
    const auto nativeSpectrum = orderName + std::string(split
        ? "-split-half-spectrum" : "-interleaved-half-spectrum");
    const auto nativeSpectrumExtents = split
        ? "one contiguous allocation containing real then imaginary arrays, each " + spectrumExtents
        : spectrumExtents;
    const auto aliasing = split
        ? "input is disjoint from the contiguous [real][imaginary] output; the component separation is fixed at the full half-spectrum element count; multidimensional split in-place planning is unsupported"
        : (provider.minimumAlignmentBytes() > 1
            ? "input and output do not overlap; new-array execution uses the planning alignment classes"
            : "input and output do not overlap; FFTW_UNALIGNED accepts arbitrary scalar alignment");

    ExecutionContract contract;
    contract.forward = {
        "out-of-place",
        "out-of-place",
        false,
        true,
        false,
        false,
        false,
        "wvm-x-fastest-real-grid",
        nativeSpectrum,
        "wvm-x-fastest-real-grid",
        "wvm-frequency-major-interleaved-half-spectrum",
        "input=" + realExtents + "; output=" + nativeSpectrumExtents,
        "input{" + realStrides + "}; output{" + spectrumStrides + "}",
        0,
        provider.minimumAlignmentBytes(),
        aliasing,
        0,
        true};
    contract.inverse = {
        "out-of-place",
        "out-of-place",
        true,
        split,
        true,
        false,
        split,
        nativeSpectrum,
        "wvm-x-fastest-real-grid",
        "wvm-frequency-major-interleaved-half-spectrum",
        "wvm-x-fastest-real-grid",
        "input=" + nativeSpectrumExtents + "; output=" + realExtents,
        "input{" + spectrumStrides + "}; output{" + realStrides + "}",
        0,
        provider.minimumAlignmentBytes(),
        split
            ? "the contiguous [real][imaginary] input is disjoint from output and has the planning-time component separation; multidimensional FFTW split c2r may destroy both components; split in-place planning is unsupported"
            : (provider.minimumAlignmentBytes() > 1
                ? "input and output do not overlap and match planning alignment classes; multidimensional FFTW c2r may destroy its input"
                : "input and output do not overlap; multidimensional FFTW c2r may destroy its input"),
        0,
        true};
    return contract;
}

ExecutionContract fftwFusedSplitRetainedExecutionContract(
    const Workload& workload, const FFTWProvider& provider,
    std::size_t retainedModeCount) {
    auto contract = fftwExecutionContract(workload, provider);
    const auto retainedExtents = "two arrays [mode=" +
        std::to_string(retainedModeCount) + "][planes=" +
        std::to_string(workload.planes()) + "]";
    const auto retainedStrides = "plane=1,mode=" +
        std::to_string(workload.planes());
    contract.forward.adapterOutputRepresentationId =
        "logical-radial-retained-split-complex";
    contract.forward.physicalExtents += "; retained-output=" + retainedExtents;
    contract.forward.stridesElements += "; retained-output{" +
        retainedStrides + "}";
    contract.forward.reusableWorkBytes = workload.spectrumElements() * sizeof(Complex);
    contract.forward.outputCanFeedOppositeDirection = true;
    contract.inverse.adapterInputRepresentationId =
        "logical-radial-retained-split-complex";
    contract.inverse.adapterPreservesCallerInput = true;
    contract.inverse.requiresPreservationCopyForRepeatedExecution = false;
    contract.inverse.preservationIncludedInAdapterTiming = false;
    contract.inverse.physicalExtents = "retained-input=" + retainedExtents +
        "; " + contract.inverse.physicalExtents;
    contract.inverse.stridesElements = "retained-input{" + retainedStrides +
        "}; " + contract.inverse.stridesElements;
    contract.inverse.reusableWorkBytes = workload.spectrumElements() * sizeof(Complex);
    contract.inverse.outputCanFeedOppositeDirection = true;
    return contract;
}

ExecutionContract fftwRetainedViewExecutionContract(
    const Workload& workload, const FFTWProvider& provider,
    std::size_t retainedModeCount) {
    auto contract = fftwExecutionContract(workload, provider);
    const auto view = "immutable mode-index view with " +
        std::to_string(retainedModeCount) + " logical modes over plane-major full storage";
    contract.forward.adapterOutputRepresentationId =
        "plane-major-interleaved-half-spectrum+logical-retained-index-view";
    contract.forward.adapterPlacement = "out-of-place-view";
    contract.forward.physicalExtents += "; logical-output=" + view;
    contract.forward.stridesElements +=
        "; view{mode-keyed indirect frequency index,plane=full-spectrum plane}";
    contract.forward.outputCanFeedOppositeDirection = false;
    contract.inverse.adapterInputRepresentationId =
        "zero-padded-plane-major-interleaved-half-spectrum+logical-retained-index-view";
    contract.inverse.adapterPlacement = "out-of-place-view";
    contract.inverse.adapterPreservesCallerInput = false;
    contract.inverse.requiresPreservationCopyForRepeatedExecution = false;
    contract.inverse.preservationIncludedInAdapterTiming = false;
    contract.inverse.physicalExtents = "logical-input=" + view +
        "; ready zero-padded provider input; " + contract.inverse.physicalExtents;
    contract.inverse.aliasing +=
        "; the ready provider-order input is dead after the call; repeat-benchmark fixture restoration is outside the single-call boundary";
    contract.inverse.outputCanFeedOppositeDirection = false;
    return contract;
}

ExecutionContract vdspExecutionContract(const Workload& workload, const VDSPProvider& provider) {
    const auto half = workload.nx / 2;
    const auto planes = workload.planes();
    const auto strategy = provider.strategy();
    const bool outOfPlace = strategy == VDSPTransformStrategy::outOfPlace ||
                            strategy == VDSPTransformStrategy::outOfPlaceExplicitScratch;
    const auto nativeExtents = "two split arrays [planes=" + std::to_string(planes) + "][Ny=" +
        std::to_string(workload.ny) + "][Nx/2=" + std::to_string(half) + "]";
    const auto nativeStrides = "split-slot=1,row=" + std::to_string(half) + ",plane=" +
        std::to_string(half * workload.ny);

    const auto nativePlacement = outOfPlace ? "out-of-place" : "in-place";
    const auto physicalExtents = outOfPlace
        ? "input=" + nativeExtents + "; output=" + nativeExtents
        : nativeExtents;
    const auto aliasing = outOfPlace
        ? "input and output split arrays are disjoint; each real/imaginary pair is disjoint"
        : "real and imaginary split arrays are disjoint; native output overwrites native input";
    DirectionExecutionContract forward{
        nativePlacement,
        "out-of-place",
        !outOfPlace,
        true,
        !outOfPlace,
        false,
        !outOfPlace,
        "vdsp-packed-split-complex",
        "vdsp-packed-split-complex",
        "wvm-x-fastest-real-grid",
        "wvm-frequency-major-interleaved-half-spectrum",
        physicalExtents,
        nativeStrides,
        0,
        provider.minimumAlignmentBytes(),
        aliasing,
        provider.scratchBytes(),
        true};
    ExecutionContract contract;
    contract.forward = forward;
    contract.inverse = forward;
    contract.inverse.adapterInputRepresentationId = "wvm-frequency-major-interleaved-half-spectrum";
    contract.inverse.adapterOutputRepresentationId = "wvm-x-fastest-real-grid";
    return contract;
}

ExecutionContract vdspNativeRetainedExecutionContract(
    const Workload& workload, const VDSPProvider& provider,
    std::size_t retainedModeCount) {
    auto contract = vdspExecutionContract(workload, provider);
    const auto retainedExtents = "two split arrays [planes=" +
        std::to_string(workload.planes()) + "][retainedModes=" +
        std::to_string(retainedModeCount) + "]";
    const auto retainedStrides = "mode=1,plane=" +
        std::to_string(retainedModeCount);
    contract.forward.adapterOutputRepresentationId =
        "plane-major-radial-retained-split-complex";
    contract.forward.adapterPreservesCallerInput = true;
    contract.forward.requiresPreservationCopyForRepeatedExecution = false;
    contract.forward.preservationIncludedInAdapterTiming = false;
    contract.forward.physicalExtents += "; retained-output=" + retainedExtents;
    contract.forward.stridesElements += "; retained-output{" + retainedStrides + "}";
    contract.forward.reusableWorkBytes = provider.nativeBufferBytes();
    contract.inverse.adapterInputRepresentationId =
        "plane-major-radial-retained-split-complex";
    contract.inverse.adapterPreservesCallerInput = true;
    contract.inverse.requiresPreservationCopyForRepeatedExecution = false;
    contract.inverse.preservationIncludedInAdapterTiming = false;
    contract.inverse.physicalExtents = "retained-input=" + retainedExtents + "; " +
        contract.inverse.physicalExtents;
    contract.inverse.stridesElements = "retained-input{" + retainedStrides + "}; " +
        contract.inverse.stridesElements;
    contract.inverse.reusableWorkBytes = provider.nativeBufferBytes();
    return contract;
}

void appendUnsupportedVdspRecord(BenchmarkReport& report, const Profile& selected, const VDSPProvider& provider) {
    ProviderRecord record;
    record.id = "accelerate-vdsp";
    record.version = "system";
    record.libraryIdentity = provider.libraryIdentity();
    record.algorithmId = vdspAlgorithmId(provider.strategy(), provider.batchStrategy());
    record.nativeRepresentationId = "vdsp-packed-split-complex";
    record.modeOrderId = "vdsp-packed-special-boundaries";
    record.schedulingId = vdspSchedulingId(provider.batchStrategy());
    record.sourceIdentity = "Apple Accelerate system framework";
    record.planningConfiguration = "radix-2 setup per logical batch worker; " +
        std::string(vdspTransformStrategyName(provider.strategy())) + "; " +
        std::string(vdspBatchStrategyName(provider.batchStrategy()));
    record.workers = selected.defaultWorkers;
    record.internalWorkers = 1;
    record.outerWorkers = selected.defaultWorkers;
    record.execution = vdspExecutionContract(report.workload, provider);
    record.otherSetupSeconds = provider.otherSetupSeconds();
    record.allocationSeconds = provider.allocationSeconds();
    record.planningSeconds = provider.planningSeconds();
    record.ledger = vdspLedger(provider.strategy(), provider.batchStrategy());
    for (auto& entry : record.ledger) entry.state = StageState::unsupported;
    record.correctness.push_back({provider.capability(), std::numeric_limits<double>::infinity(), tolerance, false});
    report.providers.push_back(std::move(record));
}

} // namespace

std::vector<Profile> profiles() {
    const auto totalWorkers = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    return {
        {"smoke", "Small deterministic correctness and CLI exercise; performance is not meaningful.", {8, 8, 7, 2, 1.0, 1.0, true}, 1, 1, 2},
        {"quick", "First production FFT vertical slice from WVM issue #129.", {256, 256, 65, 3, 1.0, 1.0, true}, totalWorkers, 3, 15},
        {"exhaustive", "Initial larger reference shape from WVM issue #129.", {512, 512, 129, 4, 1.0, 1.0, true}, totalWorkers, 3, 20},
        {"wvm-historical-256-nz65-f3", "Historical issue #129 medium workload, three fields.", {256, 256, 65, 3, 1.0, 1.0, true}, totalWorkers, 3, 15},
        {"wvm-historical-256-nz65-f4", "Historical issue #129 medium workload, four fields.", {256, 256, 65, 4, 1.0, 1.0, true}, totalWorkers, 3, 15},
        {"wvm-historical-512-nz129-f3", "Historical issue #129 large workload, three fields.", {512, 512, 129, 3, 1.0, 1.0, true}, totalWorkers, 3, 20},
        {"wvm-historical-512-nz129-f4", "Historical issue #129 large workload, four fields.", {512, 512, 129, 4, 1.0, 1.0, true}, totalWorkers, 3, 20},
        {"wvm-current-256-nz129-f1", "Current WVM 256-cubed-class workload, one field.", {256, 256, 129, 1, 1.0, 1.0, true}, totalWorkers, 3, 15},
        {"wvm-current-256-nz129-f3", "Current WVM 256-cubed-class workload, three fields.", {256, 256, 129, 3, 1.0, 1.0, true}, totalWorkers, 3, 15},
        {"wvm-current-256-nz129-f4", "Current WVM 256-cubed-class workload, four fields.", {256, 256, 129, 4, 1.0, 1.0, true}, totalWorkers, 3, 15},
        {"wvm-current-512-nz257-f1", "Current WVM 512-cubed-class workload, one field.", {512, 512, 257, 1, 1.0, 1.0, true}, totalWorkers, 3, 20},
        {"wvm-current-512-nz257-f3", "Current WVM 512-cubed-class workload, three fields.", {512, 512, 257, 3, 1.0, 1.0, true}, totalWorkers, 3, 20},
        {"wvm-current-512-nz257-f4", "Current WVM 512-cubed-class workload, four fields.", {512, 512, 257, 4, 1.0, 1.0, true}, totalWorkers, 3, 20},
        {"wvm-large-1024-nz129-f4", "Large-horizontal nonhydrostatic decision workload, four fields.", {1024, 1024, 129, 4, 1.0, 1.0, true}, totalWorkers, 3, 20}};
}

Profile profileNamed(std::string_view name) {
    const auto available = profiles();
    const auto match = std::find_if(available.begin(), available.end(), [name](const Profile& profile) {
        return profile.name == name;
    });
    if (match == available.end()) throw std::invalid_argument("Unknown profile: " + std::string(name));
    return *match;
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
}

EnvironmentRecord environmentRecord() {
    EnvironmentRecord environment;
    environment.timestampUtc = utcTimestamp(false);
    environment.hostname = hostName();
    utsname system{};
    if (uname(&system) == 0) {
        environment.operatingSystem = std::string(system.sysname) + " " + system.release + " " + system.machine;
    }
#if defined(__APPLE__)
    environment.machineModel = sysctlString("hw.model");
    environment.cpuBrand = sysctlString("machdep.cpu.brand_string");
    environment.totalCores = sysctlValue<std::uint32_t>("hw.physicalcpu", static_cast<std::uint32_t>(std::thread::hardware_concurrency()));
    environment.performanceCores = sysctlValue<std::uint32_t>("hw.perflevel0.physicalcpu", 0);
    environment.efficiencyCores = sysctlValue<std::uint32_t>("hw.perflevel1.physicalcpu", 0);
    environment.physicalMemoryBytes = sysctlValue<std::uint64_t>("hw.memsize", 0);
#else
    environment.totalCores = std::thread::hardware_concurrency();
#endif
#if defined(__clang__)
#if defined(__apple_build_version__)
    environment.compiler = "AppleClang";
#else
    environment.compiler = "Clang";
#endif
    environment.compilerVersion = __clang_version__;
#elif defined(__GNUC__)
    environment.compiler = "GCC";
    environment.compilerVersion = __VERSION__;
#else
    environment.compiler = "unknown";
#endif
    environment.compilerFlags = SKBENCH_BUILD_FLAGS;
#ifdef NDEBUG
    environment.buildType = "Release";
#else
    environment.buildType = "Debug";
#endif
    environment.gitCommit = SKBENCH_GIT_COMMIT;
    environment.gitDirty = SKBENCH_GIT_DIRTY != 0;
    return environment;
}

BenchmarkReport runVerticalGemmBenchmark(const RunOptions& options) {
    if (options.providers != "both") {
        throw std::invalid_argument("vertical-gemm currently compares both Accelerate GEMM formulations; omit --providers.");
    }
    if (options.workers != 0) {
        throw std::invalid_argument("vertical-gemm thread limits are process settings; use VECLIB_MAXIMUM_THREADS in an isolated run.");
    }
    if (options.verticalGemmFamily != "common" && options.verticalGemmFamily != "k2-grouped") {
        throw std::invalid_argument("vertical-gemm family must be either 'common' or 'k2-grouped'.");
    }
    const VerticalGemmStrategy requestedStrategy{
        verticalGemmScheduleNamed(options.verticalGemmSchedule), options.verticalGemmOuterWorkers};
    if (requestedStrategy.schedule == VerticalGemmSchedule::serial && requestedStrategy.outerWorkers != 1) {
        throw std::invalid_argument("vertical-gemm serial scheduling requires --vertical-gemm-outer-workers 1.");
    }
    if (requestedStrategy.schedule != VerticalGemmSchedule::serial && options.verticalGemmFamily != "k2-grouped") {
        throw std::invalid_argument("vertical-gemm outer scheduling is defined only for the k2-grouped family.");
    }

    const auto selected = profileNamed(options.profile);
    const auto warmups = options.warmups == 0 ? selected.warmups : options.warmups;
    const auto sampleCount = options.samples == 0 ? selected.samples : options.samples;
    if (sampleCount == 0) throw std::invalid_argument("vertical-gemm requires at least one sample.");

    BenchmarkReport report;
    report.profile = selected.name;
    report.seed = options.seed;
    report.warmups = warmups;
    report.samples = sampleCount;
    report.workload = selected.workload;
    report.environment = environmentRecord();
    report.runId = utcTimestamp(true) + "-" + report.environment.hostname;
    const auto configuredThreads = configuredAccelerateThreads(report.environment);
    if (requestedStrategy.outerWorkers > 1 && configuredThreads != 1) {
        throw std::invalid_argument(
            "vertical-gemm outer scheduling requires VECLIB_MAXIMUM_THREADS=1 to avoid nested BLAS oversubscription.");
    }

    const auto& workload = report.workload;
    const auto modes = retainedHorizontalModes(workload);
    const auto fixtureStart = Clock::now();
    auto vertical = options.verticalGemmFamily == "common"
        ? commonVerticalFixture(
            modes.size(), orthonormalVerticalFixture(workload.nz, workload.retainedVerticalModes()))
        : squaredWavenumberVerticalFixture(workload, modes);
    const auto fixtureGenerationSeconds = std::chrono::duration<double>(Clock::now() - fixtureStart).count();
    const auto equivalenceScanStart = Clock::now();
    const auto adjacentEquivalentPairs = adjacentEquivalentVerticalGroupPairs(vertical);
    const auto equivalenceScanSeconds = std::chrono::duration<double>(Clock::now() - equivalenceScanStart).count();
    const auto columns = modes.size() * workload.fields;
    const auto physicalElements = workload.nz * columns;
    const auto modalElements = vertical.nj * columns;
    report.retainedHorizontalModeCount = modes.size();
    report.retainedModeOrderHash = modeOrderHash(modes);
    report.wvmFullSpectrumOrderHash = wvmSpectrumOrderHash(workload);
    report.fullRealBytes = bytes(workload.realElements(), sizeof(double));
    report.fullSpectrumBytes = bytes(workload.spectrumElements(), sizeof(Complex));
    report.retainedSpectrumBytes = bytes(physicalElements, sizeof(Complex));
    report.modalSpectrumBytes = bytes(modalElements, sizeof(Complex));
    report.verticalMatrixFamilySourceBytes = bytes(
        vertical.forward.size() + vertical.inverse.size(), sizeof(double));
    report.verticalMatrixFamilyId = vertical.id;
    report.verticalGroupCount = vertical.groups.size();
    report.verticalGroupOrderHash = verticalModeGroupHash(vertical.groups);
    std::vector<double> groupModes;
    std::vector<double> groupColumns;
    groupModes.reserve(vertical.groups.size());
    groupColumns.reserve(vertical.groups.size());
    for (const auto& group : vertical.groups) {
        groupModes.push_back(static_cast<double>(group.modeCount));
        groupColumns.push_back(static_cast<double>(group.modeCount * workload.fields));
    }
    report.minimumVerticalGroupModes = static_cast<std::size_t>(*std::min_element(groupModes.begin(), groupModes.end()));
    report.medianVerticalGroupModes = median(groupModes);
    report.maximumVerticalGroupModes = static_cast<std::size_t>(*std::max_element(groupModes.begin(), groupModes.end()));
    report.minimumVerticalGroupColumns = static_cast<std::size_t>(*std::min_element(groupColumns.begin(), groupColumns.end()));
    report.medianVerticalGroupColumns = median(groupColumns);
    report.maximumVerticalGroupColumns = static_cast<std::size_t>(*std::max_element(groupColumns.begin(), groupColumns.end()));

    const auto physicalInput = verticalComplexFixture(physicalElements, options.seed ^ UINT64_C(0x243f6a8885a308d3));
    const auto modalInput = verticalComplexFixture(modalElements, options.seed ^ UINT64_C(0x13198a2e03707344));
    const auto probes = verticalProbeColumns(columns);
    const auto forwardOracle = directVerticalForwardProbes(vertical, workload, probes, physicalInput.data());
    const auto inverseOracle = directVerticalInverseProbes(vertical, workload, probes, modalInput.data());

    VerticalGemmProvider complexProvider(
        workload, vertical, VerticalGemmLayout::complexInterleaved, requestedStrategy);
    VerticalGemmProvider splitProvider(workload, vertical, VerticalGemmLayout::split, requestedStrategy);
    if (!complexProvider.supported() || !splitProvider.supported()) {
        throw std::runtime_error(!complexProvider.supported() ? complexProvider.capability() : splitProvider.capability());
    }
    complexProvider.loadPhysicalInput(physicalInput.data());
    complexProvider.loadModalInput(modalInput.data());
    splitProvider.loadPhysicalInput(physicalInput.data());
    splitProvider.loadModalInput(modalInput.data());
    vertical.forward = {};
    vertical.inverse = {};

    complexProvider.executeForward();
    complexProvider.executeInverse();
    splitProvider.executeForward();
    splitProvider.executeInverse();
    std::vector<Complex> complexForward(modalElements);
    std::vector<Complex> complexInverse(physicalElements);
    std::vector<Complex> splitForward(modalElements);
    std::vector<Complex> splitInverse(physicalElements);
    complexProvider.copyForwardOutput(complexForward.data());
    complexProvider.copyInverseOutput(complexInverse.data());
    splitProvider.copyForwardOutput(splitForward.data());
    splitProvider.copyInverseOutput(splitInverse.data());
    const auto complexForwardProbes = gatherVerticalProbes(complexForward.data(), vertical.nj, probes);
    const auto complexInverseProbes = gatherVerticalProbes(complexInverse.data(), workload.nz, probes);
    const auto splitForwardProbes = gatherVerticalProbes(splitForward.data(), vertical.nj, probes);
    const auto splitInverseProbes = gatherVerticalProbes(splitInverse.data(), workload.nz, probes);

    const auto physicalBytes = bytes(physicalElements, sizeof(Complex));
    const auto modalBytes = bytes(modalElements, sizeof(Complex));
    const auto externalOperandBytes = physicalBytes + modalBytes;
    const auto providerPersistentBytes = static_cast<std::uint64_t>(
        complexProvider.persistentBytes() + splitProvider.persistentBytes());
    const auto constructionPeakBytes = report.verticalMatrixFamilySourceBytes +
        providerPersistentBytes + externalOperandBytes;
    const auto outputInspectionPeakBytes = providerPersistentBytes + 3 * externalOperandBytes;
    report.verticalBenchmarkEstimatedExplicitPeakBytes = std::max(
        constructionPeakBytes, outputInspectionPeakBytes);

    auto makeRecord = [&](VerticalGemmProvider& provider, std::string id, std::string algorithmId,
                          std::vector<CorrectnessMetric> correctness) {
        ProviderRecord record;
        record.id = std::move(id);
        record.version = "system";
        record.libraryIdentity = provider.libraryIdentity();
        record.algorithmId = std::move(algorithmId);
        record.nativeRepresentationId = provider.layout() == VerticalGemmLayout::split
            ? "vertical-columns-split-complex"
            : "vertical-columns-interleaved-complex";
        record.modeOrderId = options.verticalGemmFamily == "common"
            ? "vertical-contiguous;column=field+fields*radial-mode"
            : "vertical-contiguous;k2-group-contiguous;column=field+fields*radial-mode";
        const auto providerStrategy = provider.strategy();
        const auto schedulingId = verticalGemmSchedulingId(providerStrategy);
        record.schedulingId = schedulingId;
        record.sourceIdentity = "Apple Accelerate system framework";
        record.configureFlags = "system framework";
        record.compilerFlags = report.environment.compilerFlags;
        record.planningConfiguration = vertical.id + "; column-major; K=" +
            std::to_string(columns) + "; groups=" + std::to_string(vertical.groups.size()) +
            "; GEMM calls per direction=" + std::to_string(provider.gemmCallsPerExecution()) +
            "; public variable-size grouped BLAS batch API=unavailable" +
            "; exactly equivalent adjacent matrix pairs=" + std::to_string(adjacentEquivalentPairs) +
            "; nonadjacent consolidation=requires reordering or block-diagonal expansion" +
            "; " + schedulingId;
        record.workers = configuredThreads * provider.outerWorkers();
        record.internalWorkers = configuredThreads;
        record.outerWorkers = provider.outerWorkers();
        record.gemmCallsPerExecution = provider.gemmCallsPerExecution();
        record.execution = verticalGemmExecutionContract(provider, workload);
        record.explicitPersistentBytes = provider.persistentBytes();
        record.scratchBytes = 0;
        record.opaqueProviderMemory = provider.hasOpaqueSchedulerMemory();
        record.otherSetupSeconds = provider.matrixPreparationSeconds() + provider.schedulerSetupSeconds();
        record.allocationSeconds = provider.allocationSeconds();
        record.planningSeconds = 0.0;
        record.ledger = verticalGemmLedger(
            provider.layout(), vertical.groups.size(), providerStrategy, adjacentEquivalentPairs);
        record.correctness = std::move(correctness);

        const auto matrixBytes = static_cast<std::uint64_t>(provider.matrixBytesPerDirection());
        const auto matrixReads = provider.layout() == VerticalGemmLayout::split ? 2 * matrixBytes : matrixBytes;
        record.timings.push_back(series(
            "setup-shared-component", "logical matrix-family fixture generation", "shared", StageState::setupOnly,
            report.verticalMatrixFamilySourceBytes, {fixtureGenerationSeconds}));
        record.timings.push_back(series(
            "setup-shared-component", "adjacent matrix equivalence scan", "shared", StageState::setupOnly,
            report.verticalMatrixFamilySourceBytes, {equivalenceScanSeconds}));
        record.timings.push_back(series(
            "setup-component", "matrix preparation", "shared", StageState::setupOnly,
            2 * matrixBytes, {provider.matrixPreparationSeconds()}));
        record.timings.push_back(series(
            "setup-component", "persistent outer scheduler creation", "shared", StageState::setupOnly,
            provider.schedulerPersistentBytes(), {provider.schedulerSetupSeconds()}));
        record.timings.push_back(series(
            "primitive", "raw vertical GEMM", "forward", StageState::executed,
            matrixReads + physicalBytes + modalBytes,
            measure(warmups, sampleCount, [&] { provider.executeForward(); })));
        record.timings.push_back(series(
            "primitive", "raw vertical GEMM", "inverse", StageState::executed,
            matrixReads + modalBytes + physicalBytes,
            measure(warmups, sampleCount, [&] { provider.executeInverse(); })));
        if (provider.layout() == VerticalGemmLayout::split) {
            record.timings.push_back(series(
                "primitive-component", "real-part GEMM", "forward", StageState::executed,
                matrixBytes + physicalBytes / 2 + modalBytes / 2,
                measure(warmups, sampleCount, [&] { provider.executeForwardReal(); })));
            record.timings.push_back(series(
                "primitive-component", "imaginary-part GEMM", "forward", StageState::executed,
                matrixBytes + physicalBytes / 2 + modalBytes / 2,
                measure(warmups, sampleCount, [&] { provider.executeForwardImaginary(); })));
            record.timings.push_back(series(
                "primitive-component", "real-part GEMM", "inverse", StageState::executed,
                matrixBytes + modalBytes / 2 + physicalBytes / 2,
                measure(warmups, sampleCount, [&] { provider.executeInverseReal(); })));
            record.timings.push_back(series(
                "primitive-component", "imaginary-part GEMM", "inverse", StageState::executed,
                matrixBytes + modalBytes / 2 + physicalBytes / 2,
                measure(warmups, sampleCount, [&] { provider.executeInverseImaginary(); })));
        }
        record.timings.push_back(series(
            "primitive-diagnostic", "empty group dispatch", "shared",
            providerStrategy.schedule == VerticalGemmSchedule::serial
                ? StageState::elided : StageState::executed,
            0, providerStrategy.schedule == VerticalGemmSchedule::serial
                ? std::vector<double>{}
                : measure(warmups, sampleCount, [&] { provider.executeSchedulerNoop(); })));
        record.timings.push_back(series(
            "adapter-component", "packing and representation conversion", "shared", StageState::elided, 0));
        return record;
    };

    report.providers.push_back(makeRecord(
        complexProvider,
        "accelerate-zgemm",
        options.verticalGemmFamily == "common"
            ? "accelerate-zgemm-real-matrix-expanded-common"
            : "accelerate-zgemm-k2-group-" + options.verticalGemmSchedule + "-real-matrices-expanded",
        {
            metric("forward selected columns versus independent scalar oracle",
                   complexForwardProbes.data(), forwardOracle.data(), forwardOracle.size()),
            metric("inverse selected columns versus independent scalar oracle",
                   complexInverseProbes.data(), inverseOracle.data(), inverseOracle.size()),
            metric("forward full output versus split formulation",
                   complexForward.data(), splitForward.data(), complexForward.size()),
            metric("inverse full output versus split formulation",
                   complexInverse.data(), splitInverse.data(), complexInverse.size()),
        }));
    report.providers.push_back(makeRecord(
        splitProvider,
        "accelerate-split-dgemm",
        options.verticalGemmFamily == "common"
            ? "accelerate-two-dgemm-split-common"
            : "accelerate-two-dgemm-k2-group-" + options.verticalGemmSchedule + "-split",
        {
            metric("forward selected columns versus independent scalar oracle",
                   splitForwardProbes.data(), forwardOracle.data(), forwardOracle.size()),
            metric("inverse selected columns versus independent scalar oracle",
                   splitInverseProbes.data(), inverseOracle.data(), inverseOracle.size()),
            metric("forward full output versus complex formulation",
                   splitForward.data(), complexForward.data(), splitForward.size()),
            metric("inverse full output versus complex formulation",
                   splitInverse.data(), complexInverse.data(), splitInverse.size()),
        }));
    report.status = std::all_of(report.providers.begin(), report.providers.end(), correctnessPassed)
        ? "passed" : "failed";
    return report;
}

BenchmarkReport runOrderingPackingBenchmark(const RunOptions& options) {
    if (options.providers != "both") {
        throw std::invalid_argument(
            "ordering-packing currently compares both Accelerate GEMM representations; omit --providers.");
    }
    if (options.workers != 0) {
        throw std::invalid_argument(
            "ordering-packing thread limits are process settings; use VECLIB_MAXIMUM_THREADS in an isolated run.");
    }
    if (options.verticalGemmFamily != "k2-grouped") {
        throw std::invalid_argument(
            "the first ordering-packing increment requires --vertical-gemm-family k2-grouped.");
    }
    const VerticalGemmStrategy requestedStrategy{
        verticalGemmScheduleNamed(options.verticalGemmSchedule), options.verticalGemmOuterWorkers};
    if (requestedStrategy.schedule == VerticalGemmSchedule::serial && requestedStrategy.outerWorkers != 1) {
        throw std::invalid_argument(
            "ordering-packing serial scheduling requires --vertical-gemm-outer-workers 1.");
    }

    const auto selected = profileNamed(options.profile);
    const auto warmups = options.warmups == 0 ? selected.warmups : options.warmups;
    const auto sampleCount = options.samples == 0 ? selected.samples : options.samples;
    if (sampleCount == 0) throw std::invalid_argument("ordering-packing requires at least one sample.");

    BenchmarkReport report;
    report.profile = selected.name;
    report.seed = options.seed;
    report.warmups = warmups;
    report.samples = sampleCount;
    report.workload = selected.workload;
    report.environment = environmentRecord();
    report.runId = utcTimestamp(true) + "-" + report.environment.hostname;
    const auto configuredThreads = configuredAccelerateThreads(report.environment);
    if (requestedStrategy.outerWorkers > 1 && configuredThreads != 1) {
        throw std::invalid_argument(
            "ordering-packing outer scheduling requires VECLIB_MAXIMUM_THREADS=1 to avoid nested BLAS oversubscription.");
    }

    const auto& workload = report.workload;
    const auto modes = retainedHorizontalModes(workload);
    const auto fixtureStart = Clock::now();
    auto vertical = squaredWavenumberVerticalFixture(workload, modes);
    const auto fixtureGenerationSeconds =
        std::chrono::duration<double>(Clock::now() - fixtureStart).count();
    const auto equivalenceScanStart = Clock::now();
    const auto adjacentEquivalentPairs = adjacentEquivalentVerticalGroupPairs(vertical);
    const auto equivalenceScanSeconds =
        std::chrono::duration<double>(Clock::now() - equivalenceScanStart).count();
    const auto columns = modes.size() * workload.fields;
    const auto physicalElements = workload.nz * columns;
    const auto modalElements = vertical.nj * columns;

    report.retainedHorizontalModeCount = modes.size();
    report.retainedModeOrderHash = modeOrderHash(modes);
    report.wvmFullSpectrumOrderHash = wvmSpectrumOrderHash(workload);
    report.fullRealBytes = bytes(workload.realElements(), sizeof(double));
    report.fullSpectrumBytes = bytes(workload.spectrumElements(), sizeof(Complex));
    report.retainedSpectrumBytes = bytes(physicalElements, sizeof(Complex));
    report.modalSpectrumBytes = bytes(modalElements, sizeof(Complex));
    report.verticalMatrixFamilySourceBytes = bytes(
        vertical.forward.size() + vertical.inverse.size(), sizeof(double));
    report.verticalMatrixFamilyId = vertical.id;
    report.verticalGroupCount = vertical.groups.size();
    report.verticalGroupOrderHash = verticalModeGroupHash(vertical.groups);
    std::vector<double> groupModes;
    std::vector<double> groupColumns;
    groupModes.reserve(vertical.groups.size());
    groupColumns.reserve(vertical.groups.size());
    for (const auto& group : vertical.groups) {
        groupModes.push_back(static_cast<double>(group.modeCount));
        groupColumns.push_back(static_cast<double>(group.modeCount * workload.fields));
    }
    report.minimumVerticalGroupModes =
        static_cast<std::size_t>(*std::min_element(groupModes.begin(), groupModes.end()));
    report.medianVerticalGroupModes = median(groupModes);
    report.maximumVerticalGroupModes =
        static_cast<std::size_t>(*std::max_element(groupModes.begin(), groupModes.end()));
    report.minimumVerticalGroupColumns =
        static_cast<std::size_t>(*std::min_element(groupColumns.begin(), groupColumns.end()));
    report.medianVerticalGroupColumns = median(groupColumns);
    report.maximumVerticalGroupColumns =
        static_cast<std::size_t>(*std::max_element(groupColumns.begin(), groupColumns.end()));

    const auto physicalInput = verticalComplexFixture(
        physicalElements, options.seed ^ UINT64_C(0x243f6a8885a308d3));
    const auto modalInput = verticalComplexFixture(
        modalElements, options.seed ^ UINT64_C(0x13198a2e03707344));
    std::vector<Complex> fullInput(workload.spectrumElements());
    embedRetained(workload, modes, physicalInput.data(), fullInput.data());
    const auto probes = verticalProbeColumns(columns);
    const auto forwardOracle = directVerticalForwardProbes(
        vertical, workload, probes, physicalInput.data());
    const auto inverseOracle = directVerticalInverseProbes(
        vertical, workload, probes, modalInput.data());

    VerticalGemmProvider complexProvider(
        workload, vertical, VerticalGemmLayout::complexInterleaved, requestedStrategy);
    VerticalGemmProvider splitProvider(
        workload, vertical, VerticalGemmLayout::split, requestedStrategy);
    WvmDirectVerticalGemmProvider directProvider(
        workload, modes, vertical, requestedStrategy);
    if (!complexProvider.supported() || !splitProvider.supported() || !directProvider.supported()) {
        throw std::runtime_error(
            !complexProvider.supported() ? complexProvider.capability()
            : !splitProvider.supported() ? splitProvider.capability()
                                         : directProvider.capability());
    }
    complexProvider.packPhysicalInputFromWvm(modes, fullInput.data());
    splitProvider.packPhysicalInputFromWvm(modes, fullInput.data());
    complexProvider.loadModalInput(modalInput.data());
    splitProvider.loadModalInput(modalInput.data());
    std::vector<Complex> directModalInput(directProvider.modalSpectrumElements());
    std::vector<Complex> directModalOutput(directProvider.modalSpectrumElements());
    std::vector<Complex> directFull(workload.spectrumElements());
    embedRetainedModal(workload, modes, modalInput.data(), directModalInput.data());
    directProvider.initializeModalOutput(directModalOutput.data());
    directProvider.initializeSpectrumOutput(directFull.data());
    vertical.forward = {};
    vertical.inverse = {};

    complexProvider.executeForward();
    complexProvider.executeInverse();
    splitProvider.executeForward();
    splitProvider.executeInverse();
    directProvider.executeForward(fullInput.data(), directModalOutput.data());
    directProvider.executeInverse(directModalInput.data(), directFull.data());
    std::vector<Complex> complexForward(modalElements);
    std::vector<Complex> complexInverse(physicalElements);
    std::vector<Complex> splitForward(modalElements);
    std::vector<Complex> splitInverse(physicalElements);
    std::vector<Complex> directForward(modalElements);
    std::vector<Complex> directInverse(physicalElements);
    complexProvider.copyForwardOutput(complexForward.data());
    complexProvider.copyInverseOutput(complexInverse.data());
    splitProvider.copyForwardOutput(splitForward.data());
    splitProvider.copyInverseOutput(splitInverse.data());
    gatherRetainedModal(workload, modes, directModalOutput.data(), directForward.data());
    gatherRetained(workload, modes, directFull.data(), directInverse.data());
    const auto complexForwardProbes = gatherVerticalProbes(complexForward.data(), vertical.nj, probes);
    const auto complexInverseProbes = gatherVerticalProbes(complexInverse.data(), workload.nz, probes);
    const auto splitForwardProbes = gatherVerticalProbes(splitForward.data(), vertical.nj, probes);
    const auto splitInverseProbes = gatherVerticalProbes(splitInverse.data(), workload.nz, probes);
    const auto directForwardProbes = gatherVerticalProbes(directForward.data(), vertical.nj, probes);
    const auto directInverseProbes = gatherVerticalProbes(directInverse.data(), workload.nz, probes);

    std::vector<Complex> expectedFull(workload.spectrumElements());
    std::vector<Complex> complexFull(workload.spectrumElements());
    std::vector<Complex> splitFull(workload.spectrumElements());
    embedRetained(workload, modes, complexInverse.data(), expectedFull.data());
    complexProvider.embedPhysicalOutputToWvm(modes, complexFull.data());
    splitProvider.embedPhysicalOutputToWvm(modes, splitFull.data());
    std::vector<Complex> complexGathered(physicalElements);
    std::vector<Complex> splitGathered(physicalElements);
    std::vector<Complex> directGathered(physicalElements);
    gatherRetained(workload, modes, complexFull.data(), complexGathered.data());
    gatherRetained(workload, modes, splitFull.data(), splitGathered.data());
    gatherRetained(workload, modes, directFull.data(), directGathered.data());

    const auto physicalBytes = report.retainedSpectrumBytes;
    const auto modalBytes = report.modalSpectrumBytes;
    const auto fullBytes = report.fullSpectrumBytes;
    const auto fullModalBytes = bytes(directProvider.modalSpectrumElements(), sizeof(Complex));
    const auto conjugateBoundaryModes = static_cast<std::uint64_t>(std::count_if(
        modes.begin(), modes.end(), [&](const RetainedMode& mode) {
            return mode.storedKx == 0 && mode.storedKy != 0 &&
                2 * mode.storedKy != workload.ny;
        }));
    const auto conjugateBoundaryBytes = conjugateBoundaryModes *
        static_cast<std::uint64_t>(workload.planes()) * sizeof(Complex);
    const auto conjugateModalBoundaryBytes = conjugateBoundaryModes *
        static_cast<std::uint64_t>(workload.retainedVerticalModes() * workload.fields) *
        sizeof(Complex);
    const auto forwardPackingBytes = 2 * physicalBytes;
    const auto inverseEmbeddingBytes = fullBytes + 2 * physicalBytes + conjugateBoundaryBytes;
    const auto providerPersistentBytes = static_cast<std::uint64_t>(
        complexProvider.persistentBytes() + splitProvider.persistentBytes() +
        directProvider.persistentBytes());
    const auto constructionPeak = report.verticalMatrixFamilySourceBytes + providerPersistentBytes +
        physicalBytes + modalBytes + 2 * fullBytes + 2 * fullModalBytes;
    const auto inspectionPeak = providerPersistentBytes + 5 * physicalBytes + 3 * modalBytes +
        5 * fullBytes + 2 * fullModalBytes +
        bytes(forwardOracle.size() + inverseOracle.size(), sizeof(Complex));
    report.orderingPackingEstimatedExplicitPeakBytes = std::max(constructionPeak, inspectionPeak);

    auto makeRecord = [&](VerticalGemmProvider& provider, std::string id,
                          std::vector<CorrectnessMetric> correctness,
                          std::vector<Complex>& embeddedOutput) {
        ProviderRecord record;
        const auto split = provider.layout() == VerticalGemmLayout::split;
        const auto representation = split
            ? "vertical-columns-split-complex"
            : "vertical-columns-interleaved-complex";
        const auto providerStrategy = provider.strategy();
        const auto schedulingId = verticalGemmSchedulingId(providerStrategy);
        record.id = std::move(id);
        record.version = "system";
        record.libraryIdentity = provider.libraryIdentity();
        record.algorithmId = split
            ? "matlab-radial-gather-to-split-k2-" + options.verticalGemmSchedule
            : "matlab-radial-gather-to-interleaved-k2-" + options.verticalGemmSchedule;
        record.nativeRepresentationId = representation;
        record.modeOrderId = "WVM-frequency-major-to-k2-group-contiguous;column=field+fields*radial-mode";
        record.schedulingId = schedulingId;
        record.sourceIdentity = "Apple Accelerate system framework plus skbench movement adapter";
        record.configureFlags = "system framework";
        record.compilerFlags = report.environment.compilerFlags;
        record.planningConfiguration = vertical.id + "; MATLAB-style radial gather baseline; K=" +
            std::to_string(columns) + "; groups=" + std::to_string(vertical.groups.size()) +
            "; exactly equivalent adjacent matrix pairs=" +
            std::to_string(adjacentEquivalentPairs) + "; reuse counts=2,4,8; " + schedulingId;
        record.workers = configuredThreads * provider.outerWorkers();
        record.internalWorkers = configuredThreads;
        record.outerWorkers = provider.outerWorkers();
        record.gemmCallsPerExecution = provider.gemmCallsPerExecution();
        record.execution = verticalGemmExecutionContract(provider, workload);
        record.execution.forward.adapterInputRepresentationId =
            "wvm-frequency-major-interleaved-half-spectrum";
        record.execution.forward.adapterOutputRepresentationId = representation;
        record.execution.forward.adapterPreservesCallerInput = true;
        record.execution.forward.reusableWorkBytes = physicalBytes;
        record.execution.inverse.adapterInputRepresentationId = representation;
        record.execution.inverse.adapterOutputRepresentationId =
            "wvm-frequency-major-interleaved-half-spectrum";
        record.execution.inverse.adapterPreservesCallerInput = true;
        record.execution.inverse.reusableWorkBytes = fullBytes;
        record.explicitPersistentBytes = provider.persistentBytes();
        record.scratchBytes = 0;
        record.opaqueProviderMemory = provider.hasOpaqueSchedulerMemory();
        record.otherSetupSeconds = provider.matrixPreparationSeconds() + provider.schedulerSetupSeconds();
        record.allocationSeconds = provider.allocationSeconds();
        record.planningSeconds = 0.0;
        record.ledger = {
            {"setup/planning", StageState::setupOnly,
             "prepare immutable K-squared matrix family and persistent outer scheduler"},
            {"raw forward FFT", StageState::unsupported,
             "excluded from this first issue #13 movement increment"},
            {"horizontal retention", StageState::executed,
             "gather retained modes by logical mode key from WVM frequency-major storage"},
            {"representation conversion", split ? StageState::fused : StageState::elided,
             split ? "interleaved-to-split conversion is fused into the gather"
                   : "gather remains interleaved"},
            {"permutation/packing", StageState::executed,
             "MATLAB-style radial order and vertical-contiguous columns are materialized"},
            {"raw forward vertical MM", StageState::executed,
             "issue #8 K-squared grouped finalist primitive"},
            {"modal work", StageState::unsupported,
             "modal physics and nonlinear flux are excluded"},
            {"raw inverse vertical MM", StageState::executed,
             "issue #8 K-squared grouped finalist primitive"},
            {"horizontal embedding", StageState::executed,
             "zero full half-spectrum and scatter retained modes with Hermitian boundary repair"},
            {"raw inverse FFT", StageState::unsupported,
             "excluded from this first issue #13 movement increment"},
            {"uninstrumented total", StageState::executed,
             "synthetic movement-plus-vertical totals only; complete pipeline belongs to issue #9"}};
        record.correctness = std::move(correctness);

        const auto matrixBytes = static_cast<std::uint64_t>(provider.matrixBytesPerDirection());
        const auto matrixReads = split ? 2 * matrixBytes : matrixBytes;
        const auto rawForwardBytes = matrixReads + physicalBytes + modalBytes;
        const auto rawInverseBytes = matrixReads + modalBytes + physicalBytes;
        record.timings.push_back(series(
            "setup-shared-component", "logical matrix-family fixture generation", "shared",
            StageState::setupOnly, report.verticalMatrixFamilySourceBytes,
            {fixtureGenerationSeconds}));
        record.timings.push_back(series(
            "setup-shared-component", "adjacent matrix equivalence scan", "shared",
            StageState::setupOnly, report.verticalMatrixFamilySourceBytes,
            {equivalenceScanSeconds}));
        record.timings.push_back(series(
            "setup-component", "matrix preparation", "shared", StageState::setupOnly,
            2 * matrixBytes, {provider.matrixPreparationSeconds()}));
        record.timings.push_back(series(
            "setup-component", "persistent outer scheduler creation", "shared",
            StageState::setupOnly, provider.schedulerPersistentBytes(),
            {provider.schedulerSetupSeconds()}));
        record.timings.push_back(series(
            "primitive", "raw vertical GEMM", "forward", StageState::executed,
            rawForwardBytes,
            measure(warmups, sampleCount, [&] { provider.executeForward(); })));
        record.timings.push_back(series(
            "primitive", "raw vertical GEMM", "inverse", StageState::executed,
            rawInverseBytes,
            measure(warmups, sampleCount, [&] { provider.executeInverse(); })));
        record.timings.push_back(series(
            "adapter-component", "WVM retained gather and radial pack", "forward",
            StageState::executed, forwardPackingBytes,
            measure(warmups, sampleCount, [&] {
                provider.packPhysicalInputFromWvm(modes, fullInput.data());
            })));
        record.timings.push_back(series(
            "adapter-component", "WVM scatter and Hermitian embed", "inverse",
            StageState::executed, inverseEmbeddingBytes,
            measure(warmups, sampleCount, [&] {
                provider.embedPhysicalOutputToWvm(modes, embeddedOutput.data());
            })));
        record.timings.push_back(series(
            "adapter-total", "one-shot movement plus vertical GEMM", "forward",
            StageState::executed, forwardPackingBytes + rawForwardBytes,
            measure(warmups, sampleCount, [&] {
                provider.packPhysicalInputFromWvm(modes, fullInput.data());
                provider.executeForward();
            })));
        record.timings.push_back(series(
            "adapter-total", "one-shot movement plus vertical GEMM", "inverse",
            StageState::executed, rawInverseBytes + inverseEmbeddingBytes,
            measure(warmups, sampleCount, [&] {
                provider.executeInverse();
                provider.embedPhysicalOutputToWvm(modes, embeddedOutput.data());
            })));

        for (const std::size_t reuseCount : {2U, 4U, 8U}) {
            const auto suffix = "-r" + std::to_string(reuseCount);
            record.timings.push_back(series(
                "reuse-total", "boundary-movement-each-use", "forward" + suffix,
                StageState::executed,
                reuseCount * (forwardPackingBytes + rawForwardBytes),
                measure(warmups, sampleCount, [&] {
                    for (std::size_t use = 0; use < reuseCount; ++use) {
                        provider.packPhysicalInputFromWvm(modes, fullInput.data());
                        provider.executeForward();
                    }
                })));
            record.timings.push_back(series(
                "reuse-total", "persistent-compact-boundary-once", "forward" + suffix,
                StageState::executed,
                forwardPackingBytes + reuseCount * rawForwardBytes,
                measure(warmups, sampleCount, [&] {
                    provider.packPhysicalInputFromWvm(modes, fullInput.data());
                    for (std::size_t use = 0; use < reuseCount; ++use) {
                        provider.executeForward();
                    }
                })));
            record.timings.push_back(series(
                "reuse-total", "boundary-movement-each-use", "inverse" + suffix,
                StageState::executed,
                reuseCount * (rawInverseBytes + inverseEmbeddingBytes),
                measure(warmups, sampleCount, [&] {
                    for (std::size_t use = 0; use < reuseCount; ++use) {
                        provider.executeInverse();
                        provider.embedPhysicalOutputToWvm(modes, embeddedOutput.data());
                    }
                })));
            record.timings.push_back(series(
                "reuse-total", "persistent-compact-boundary-once", "inverse" + suffix,
                StageState::executed,
                reuseCount * rawInverseBytes + inverseEmbeddingBytes,
                measure(warmups, sampleCount, [&] {
                    for (std::size_t use = 0; use < reuseCount; ++use) {
                        provider.executeInverse();
                    }
                    provider.embedPhysicalOutputToWvm(modes, embeddedOutput.data());
                })));
        }
        return record;
    };

    report.providers.push_back(makeRecord(
        complexProvider,
        "ordering-pack-accelerate-zgemm",
        {
            metric("forward selected modes versus independent scalar oracle",
                   complexForwardProbes.data(), forwardOracle.data(), forwardOracle.size()),
            metric("inverse selected modes versus independent scalar oracle",
                   complexInverseProbes.data(), inverseOracle.data(), inverseOracle.size()),
            metric("forward full compact output versus split formulation",
                   complexForward.data(), splitForward.data(), complexForward.size()),
            metric("inverse full compact output versus split formulation",
                   complexInverse.data(), splitInverse.data(), complexInverse.size()),
            metric("reverse WVM embedding versus independent embedding",
                   complexFull.data(), expectedFull.data(), complexFull.size()),
            metric("reverse embedding mode-key round trip",
                   complexGathered.data(), complexInverse.data(), complexInverse.size()),
        },
        complexFull));
    report.providers.push_back(makeRecord(
        splitProvider,
        "ordering-pack-accelerate-split-dgemm",
        {
            metric("forward selected modes versus independent scalar oracle",
                   splitForwardProbes.data(), forwardOracle.data(), forwardOracle.size()),
            metric("inverse selected modes versus independent scalar oracle",
                   splitInverseProbes.data(), inverseOracle.data(), inverseOracle.size()),
            metric("forward full compact output versus complex formulation",
                   splitForward.data(), complexForward.data(), splitForward.size()),
            metric("inverse full compact output versus complex formulation",
                   splitInverse.data(), complexInverse.data(), splitInverse.size()),
            metric("reverse WVM embedding versus independent embedding",
                   splitFull.data(), expectedFull.data(), splitFull.size()),
            metric("reverse embedding mode-key round trip",
                   splitGathered.data(), splitInverse.data(), splitInverse.size()),
        },
        splitFull));

    ProviderRecord directRecord;
    const auto directStrategy = directProvider.strategy();
    const auto directSchedulingId = verticalGemmSchedulingId(directStrategy);
    directRecord.id = "ordering-no-reorder-accelerate-zgemm";
    directRecord.version = "system";
    directRecord.libraryIdentity = directProvider.libraryIdentity();
    directRecord.algorithmId =
        "wvm-frequency-major-direct-per-mode-zgemm-k2-" + options.verticalGemmSchedule;
    directRecord.nativeRepresentationId =
        "wvm-frequency-major-interleaved-antialiased-zero-padded";
    directRecord.modeOrderId =
        "WVM-frequency-major;vertical-contiguous;field-block-within-frequency";
    directRecord.schedulingId = directSchedulingId;
    directRecord.sourceIdentity = "Apple Accelerate system framework plus direct WVM-order kernel";
    directRecord.configureFlags = "system framework";
    directRecord.compilerFlags = report.environment.compilerFlags;
    directRecord.planningConfiguration = vertical.id +
        "; no gather, transpose, radial permutation, or split conversion" +
        "; retained frequencies=" + std::to_string(modes.size()) +
        "; matrix groups=" + std::to_string(vertical.groups.size()) +
        "; fields per GEMM=" + std::to_string(workload.fields) +
        "; GEMM calls per direction=" + std::to_string(directProvider.gemmCallsPerExecution()) +
        "; persistent zero padding initialized once; reuse counts=2,4,8; " +
        directSchedulingId;
    directRecord.workers = configuredThreads * directProvider.outerWorkers();
    directRecord.internalWorkers = configuredThreads;
    directRecord.outerWorkers = directProvider.outerWorkers();
    directRecord.gemmCallsPerExecution = directProvider.gemmCallsPerExecution();

    DirectionExecutionContract directForwardContract;
    directForwardContract.nativePlacement = "out-of-place";
    directForwardContract.adapterPlacement = "out-of-place";
    directForwardContract.adapterPreservesCallerInput = true;
    directForwardContract.nativeInputRepresentationId =
        "wvm-frequency-major-interleaved-half-spectrum";
    directForwardContract.nativeOutputRepresentationId =
        "wvm-frequency-major-interleaved-modal-half-spectrum";
    directForwardContract.adapterInputRepresentationId =
        directForwardContract.nativeInputRepresentationId;
    directForwardContract.adapterOutputRepresentationId =
        directForwardContract.nativeOutputRepresentationId;
    directForwardContract.physicalExtents =
        "input=[frequency=" + std::to_string(workload.halfRows()) + "][field=" +
        std::to_string(workload.fields) + "][Nz=" + std::to_string(workload.nz) +
        "]; output=[frequency=" + std::to_string(workload.halfRows()) + "][field=" +
        std::to_string(workload.fields) + "][Nj=" +
        std::to_string(workload.retainedVerticalModes()) + "]";
    directForwardContract.stridesElements =
        "input{z=1,field=Nz,frequency=Nz*fields}; "
        "output{j=1,field=Nj,frequency=Nj*fields}";
    directForwardContract.paddingElements =
        directProvider.modalSpectrumElements() - modalElements;
    directForwardContract.minimumAlignmentBytes = alignof(Complex);
    directForwardContract.aliasing = "input and output do not overlap";
    directForwardContract.reusableWorkBytes = fullModalBytes;
    directForwardContract.outputCanFeedOppositeDirection = true;
    directRecord.execution.forward = directForwardContract;
    directRecord.execution.inverse = directForwardContract;
    directRecord.execution.inverse.nativeInputRepresentationId =
        directForwardContract.nativeOutputRepresentationId;
    directRecord.execution.inverse.nativeOutputRepresentationId =
        directForwardContract.nativeInputRepresentationId;
    directRecord.execution.inverse.adapterInputRepresentationId =
        directRecord.execution.inverse.nativeInputRepresentationId;
    directRecord.execution.inverse.adapterOutputRepresentationId =
        directRecord.execution.inverse.nativeOutputRepresentationId;
    directRecord.execution.inverse.physicalExtents =
        "input=[frequency=" + std::to_string(workload.halfRows()) + "][field=" +
        std::to_string(workload.fields) + "][Nj=" +
        std::to_string(workload.retainedVerticalModes()) +
        "]; output=[frequency=" + std::to_string(workload.halfRows()) + "][field=" +
        std::to_string(workload.fields) + "][Nz=" + std::to_string(workload.nz) + "]";
    directRecord.execution.inverse.stridesElements =
        "input{j=1,field=Nj,frequency=Nj*fields}; "
        "output{z=1,field=Nz,frequency=Nz*fields}";
    directRecord.execution.inverse.paddingElements =
        workload.spectrumElements() - physicalElements;
    directRecord.execution.inverse.reusableWorkBytes = fullBytes;

    directRecord.explicitPersistentBytes = directProvider.persistentBytes();
    directRecord.scratchBytes = 0;
    directRecord.opaqueProviderMemory = directProvider.hasOpaqueSchedulerMemory();
    directRecord.otherSetupSeconds =
        directProvider.matrixPreparationSeconds() + directProvider.schedulerSetupSeconds();
    directRecord.allocationSeconds = directProvider.allocationSeconds();
    directRecord.planningSeconds = 0.0;
    directRecord.ledger = {
        {"setup/planning", StageState::setupOnly,
         "prepare immutable complex K-squared matrix family, persistent outer scheduler, and one-time zero padding"},
        {"raw forward FFT", StageState::unsupported,
         "excluded from this issue #13 vertical ordering comparison"},
        {"horizontal retention", StageState::elided,
         "the kernel indexes retained stored frequencies directly in WVM order"},
        {"representation conversion", StageState::elided,
         "the interleaved WVM representation persists across the vertical operator"},
        {"permutation/packing", StageState::elided,
         "no radial gather, transpose, or column pack is executed"},
        {"raw forward vertical MM", StageState::executed,
         std::to_string(directProvider.gemmCallsPerExecution()) +
             " small per-frequency zgemm calls; Hermitian boundary repair is fused"},
        {"modal work", StageState::unsupported,
         "modal physics and nonlinear flux are excluded"},
        {"raw inverse vertical MM", StageState::executed,
         std::to_string(directProvider.gemmCallsPerExecution()) +
             " small per-frequency zgemm calls; Hermitian boundary repair is fused"},
        {"horizontal embedding", StageState::elided,
         "retained frequencies are written directly into persistent zero-padded WVM storage"},
        {"raw inverse FFT", StageState::unsupported,
         "excluded from this issue #13 vertical ordering comparison"},
        {"uninstrumented total", StageState::executed,
         "synthetic no-reorder vertical total only; complete pipeline belongs to issue #9"}};
    std::vector<Complex> directModalInputRoundTrip(modalElements);
    gatherRetainedModal(
        workload, modes, directModalInput.data(), directModalInputRoundTrip.data());
    directRecord.correctness = {
        metric("forward selected modes versus independent scalar oracle",
               directForwardProbes.data(), forwardOracle.data(), forwardOracle.size()),
        metric("inverse selected modes versus independent scalar oracle",
               directInverseProbes.data(), inverseOracle.data(), inverseOracle.size()),
        metric("forward full compact output versus packed split formulation",
               directForward.data(), splitForward.data(), directForward.size()),
        metric("inverse full compact output versus packed split formulation",
               directInverse.data(), splitInverse.data(), directInverse.size()),
        metric("direct WVM reconstruction versus independent embedding",
               directFull.data(), expectedFull.data(), directFull.size()),
        metric("frequency-major modal representation mode-key round trip",
               directModalInputRoundTrip.data(), modalInput.data(), modalInput.size()),
    };

    const auto directMatrixBytes = static_cast<std::uint64_t>(
        directProvider.matrixBytesPerDirection());
    const auto matrixBytesPerMode = bytes(
        workload.nz * workload.retainedVerticalModes(), sizeof(Complex));
    const auto directMatrixTraffic =
        static_cast<std::uint64_t>(directProvider.gemmCallsPerExecution()) * matrixBytesPerMode;
    const auto directForwardBytes = directMatrixTraffic + physicalBytes + modalBytes +
        2 * conjugateModalBoundaryBytes;
    const auto directInverseBytes = directMatrixTraffic + modalBytes + physicalBytes +
        2 * conjugateBoundaryBytes;
    directRecord.timings.push_back(series(
        "setup-shared-component", "logical matrix-family fixture generation", "shared",
        StageState::setupOnly, report.verticalMatrixFamilySourceBytes,
        {fixtureGenerationSeconds}));
    directRecord.timings.push_back(series(
        "setup-shared-component", "adjacent matrix equivalence scan", "shared",
        StageState::setupOnly, report.verticalMatrixFamilySourceBytes,
        {equivalenceScanSeconds}));
    directRecord.timings.push_back(series(
        "setup-component", "matrix preparation", "shared", StageState::setupOnly,
        2 * directMatrixBytes, {directProvider.matrixPreparationSeconds()}));
    directRecord.timings.push_back(series(
        "setup-component", "persistent outer scheduler creation", "shared",
        StageState::setupOnly, directProvider.schedulerPersistentBytes(),
        {directProvider.schedulerSetupSeconds()}));
    directRecord.timings.push_back(series(
        "setup-component", "persistent modal zero-padding initialization", "forward",
        StageState::setupOnly, fullModalBytes,
        measure(0, 1, [&] { directProvider.initializeModalOutput(directModalOutput.data()); })));
    directRecord.timings.push_back(series(
        "setup-component", "persistent spectrum zero-padding initialization", "inverse",
        StageState::setupOnly, fullBytes,
        measure(0, 1, [&] { directProvider.initializeSpectrumOutput(directFull.data()); })));
    directRecord.timings.push_back(series(
        "primitive", "direct frequency-major per-mode vertical GEMM", "forward",
        StageState::executed, directForwardBytes,
        measure(warmups, sampleCount, [&] {
            directProvider.executeForward(fullInput.data(), directModalOutput.data());
        })));
    directRecord.timings.push_back(series(
        "primitive", "direct frequency-major per-mode vertical GEMM", "inverse",
        StageState::executed, directInverseBytes,
        measure(warmups, sampleCount, [&] {
            directProvider.executeInverse(directModalInput.data(), directFull.data());
        })));
    directRecord.timings.push_back(series(
        "adapter-component", "WVM retained gather and radial pack", "forward",
        StageState::elided, 0));
    directRecord.timings.push_back(series(
        "adapter-component", "WVM scatter and Hermitian embed", "inverse",
        StageState::elided, 0));
    directRecord.timings.push_back(series(
        "adapter-total", "one-shot no-reorder vertical projection", "forward",
        StageState::executed, directForwardBytes,
        measure(warmups, sampleCount, [&] {
            directProvider.executeForward(fullInput.data(), directModalOutput.data());
        })));
    directRecord.timings.push_back(series(
        "adapter-total", "one-shot no-reorder vertical projection", "inverse",
        StageState::executed, directInverseBytes,
        measure(warmups, sampleCount, [&] {
            directProvider.executeInverse(directModalInput.data(), directFull.data());
        })));
    for (const std::size_t reuseCount : {2U, 4U, 8U}) {
        const auto suffix = "-r" + std::to_string(reuseCount);
        directRecord.timings.push_back(series(
            "reuse-total", "persistent-provider-order-no-movement", "forward" + suffix,
            StageState::executed, reuseCount * directForwardBytes,
            measure(warmups, sampleCount, [&] {
                for (std::size_t use = 0; use < reuseCount; ++use) {
                    directProvider.executeForward(fullInput.data(), directModalOutput.data());
                }
            })));
        directRecord.timings.push_back(series(
            "reuse-total", "persistent-provider-order-no-movement", "inverse" + suffix,
            StageState::executed, reuseCount * directInverseBytes,
            measure(warmups, sampleCount, [&] {
                for (std::size_t use = 0; use < reuseCount; ++use) {
                    directProvider.executeInverse(directModalInput.data(), directFull.data());
                }
            })));
    }
    directRecord.timings.push_back(series(
        "primitive-diagnostic", "empty frequency dispatch", "shared",
        directStrategy.schedule == VerticalGemmSchedule::serial
            ? StageState::elided : StageState::executed,
        0, directStrategy.schedule == VerticalGemmSchedule::serial
            ? std::vector<double>{}
            : measure(warmups, sampleCount, [&] { directProvider.executeSchedulerNoop(); })));
    report.providers.push_back(std::move(directRecord));
    report.status = std::all_of(report.providers.begin(), report.providers.end(), correctnessPassed)
        ? "passed" : "failed";
    return report;
}

BenchmarkReport runSpectralBoundaryBenchmark(const RunOptions& options) {
    static const std::vector<std::string> policies{
        "wvm-direct", "wvm-packed-split", "pruned-compact-interleaved",
        "plane-major-fused-split", "plane-major-view"};
    if (std::find(policies.begin(), policies.end(), options.boundaryPolicy) == policies.end()) {
        throw std::invalid_argument(
            "boundary-policy must be wvm-direct, wvm-packed-split, "
            "pruned-compact-interleaved, plane-major-fused-split, or plane-major-view.");
    }
    if (options.verticalGemmFamily != "k2-grouped") {
        throw std::invalid_argument(
            "spectral-boundary requires --vertical-gemm-family k2-grouped.");
    }
    if (options.workers != 0) {
        throw std::invalid_argument(
            "spectral-boundary uses independent FFT and vertical worker controls; omit --workers.");
    }

    const auto selected = profileNamed(options.profile);
    const auto warmups = options.warmups == 0 ? selected.warmups : options.warmups;
    const auto sampleCount = options.samples == 0 ? selected.samples : options.samples;
    if (sampleCount == 0) {
        throw std::invalid_argument("spectral-boundary requires at least one sample.");
    }
    const VerticalGemmStrategy verticalStrategy{
        verticalGemmScheduleNamed(options.verticalGemmSchedule),
        options.verticalGemmOuterWorkers};
    if (verticalStrategy.schedule == VerticalGemmSchedule::serial &&
        verticalStrategy.outerWorkers != 1) {
        throw std::invalid_argument(
            "spectral-boundary serial vertical scheduling requires one outer worker.");
    }

    BenchmarkReport report;
    report.profile = selected.name;
    report.seed = options.seed;
    report.warmups = warmups;
    report.samples = sampleCount;
    report.workload = selected.workload;
    report.environment = environmentRecord();
    report.runId = utcTimestamp(true) + "-" + report.environment.hostname;
    const auto configuredThreads = configuredAccelerateThreads(report.environment);
    if (verticalStrategy.outerWorkers > 1 && configuredThreads != 1) {
        throw std::invalid_argument(
            "spectral-boundary outer vertical scheduling requires VECLIB_MAXIMUM_THREADS=1.");
    }

    const auto& workload = report.workload;
    const auto modes = retainedHorizontalModes(workload);
    const auto fixtureStart = Clock::now();
    auto vertical = squaredWavenumberVerticalFixture(workload, modes);
    const auto fixtureGenerationSeconds =
        std::chrono::duration<double>(Clock::now() - fixtureStart).count();
    const auto columns = modes.size() * workload.fields;
    const auto physicalElements = workload.nz * columns;
    const auto modalElements = workload.retainedVerticalModes() * columns;
    const auto fullModalElements =
        workload.halfRows() * workload.retainedVerticalModes() * workload.fields;
    report.retainedHorizontalModeCount = modes.size();
    report.retainedModeOrderHash = modeOrderHash(modes);
    report.wvmFullSpectrumOrderHash = wvmSpectrumOrderHash(workload);
    report.fullRealBytes = bytes(workload.realElements(), sizeof(double));
    report.fullSpectrumBytes = bytes(workload.spectrumElements(), sizeof(Complex));
    report.retainedSpectrumBytes = bytes(physicalElements, sizeof(Complex));
    report.modalSpectrumBytes = bytes(modalElements, sizeof(Complex));
    report.verticalMatrixFamilySourceBytes = bytes(
        vertical.forward.size() + vertical.inverse.size(), sizeof(double));
    report.verticalMatrixFamilyId = vertical.id;
    report.verticalGroupCount = vertical.groups.size();
    report.verticalGroupOrderHash = verticalModeGroupHash(vertical.groups);
    std::vector<double> groupModes;
    std::vector<double> groupColumns;
    groupModes.reserve(vertical.groups.size());
    groupColumns.reserve(vertical.groups.size());
    for (const auto& group : vertical.groups) {
        groupModes.push_back(static_cast<double>(group.modeCount));
        groupColumns.push_back(static_cast<double>(group.modeCount * workload.fields));
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

    const auto input = makeFixture(workload, FixtureKind::random, options.seed);
    std::vector<Complex> referenceSpectrum(workload.spectrumElements());
    std::vector<Complex> referenceRetained(physicalElements);
    std::vector<double> referenceInverseOutput(workload.realElements());
    std::vector<Complex> sparseModalInput(modalElements, Complex{});
    const auto probes = verticalProbeColumns(columns);
    const auto sparseValues = verticalComplexFixture(
        probes.size() * workload.retainedVerticalModes(),
        options.seed ^ UINT64_C(0xa4093822299f31d0));
    for (std::size_t probe = 0; probe < probes.size(); ++probe) {
        std::copy_n(
            sparseValues.data() + probe * workload.retainedVerticalModes(),
            workload.retainedVerticalModes(),
            sparseModalInput.data() + probes[probe] * workload.retainedVerticalModes());
    }

    {
        FFTWProvider reference(workload, FFTWStrategy{
            FFTWPlanningMode::estimate, FFTWAlignmentStrategy::unaligned,
            FFTWWisdomStrategy::cold, 1, 1, 0.0,
            FFTWDataLayout::interleaved, FFTWSpectrumOrder::wvmFrequencyMajor});
        reference.forward(input.data(), referenceSpectrum.data());
        gatherRetained(
            workload, modes, referenceSpectrum.data(), referenceRetained.data());
    }
    const auto forwardOracle = directVerticalForwardProbes(
        vertical, workload, probes, referenceRetained.data());
    const auto inverseOracle = directVerticalInverseProbes(
        vertical, workload, probes, sparseModalInput.data());
    std::vector<Complex> expectedInversePhysical(physicalElements, Complex{});
    for (std::size_t probe = 0; probe < probes.size(); ++probe) {
        std::copy_n(
            inverseOracle.data() + probe * workload.nz, workload.nz,
            expectedInversePhysical.data() + probes[probe] * workload.nz);
    }
    std::vector<Complex> expectedInverseSpectrum(workload.spectrumElements());
    embedRetained(
        workload, modes, expectedInversePhysical.data(),
        expectedInverseSpectrum.data());
    {
        FFTWProvider reference(workload, FFTWStrategy{
            FFTWPlanningMode::estimate, FFTWAlignmentStrategy::unaligned,
            FFTWWisdomStrategy::cold, 1, 1, 0.0,
            FFTWDataLayout::interleaved, FFTWSpectrumOrder::wvmFrequencyMajor});
        reference.inverse(expectedInverseSpectrum.data(), referenceInverseOutput.data());
    }

    auto makeRecord = [&](std::string id, std::string algorithm,
                          std::string representation) {
        ProviderRecord record;
        record.id = std::move(id);
        record.version = "FFTW 3.3.11 + Apple Accelerate";
        record.libraryIdentity =
            "pinned FFTW 3.3.11 and /System/Library/Frameworks/Accelerate.framework";
        record.algorithmId = std::move(algorithm);
        record.nativeRepresentationId = std::move(representation);
        record.modeOrderId = "logical-radial-retained-modes-keyed-by-k-l-j-field";
        record.schedulingId = "horizontal-outer-workers-" +
            std::to_string(options.fftwOuterWorkers) +
            "-internal-workers-" +
            std::to_string(options.fftwInternalWorkers == 0 ? 1 : options.fftwInternalWorkers) +
            ";" + verticalGemmSchedulingId(verticalStrategy);
        record.sourceIdentity =
            "https://fftw.org/pub/fftw/fftw-3.3.11.tar.gz + Apple Accelerate system framework";
        record.sourceSha256 =
            "5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1";
        record.configureFlags =
            "FFTW --host=aarch64-apple-darwin --enable-neon --enable-threads; Accelerate system framework";
        record.compilerFlags = report.environment.compilerFlags;
        record.planningConfiguration = vertical.id +
            "; Float64; antialiased horizontal radial two-thirds retention; Nj=" +
            std::to_string(workload.retainedVerticalModes()) +
            "; modal work and nonlinear flux excluded";
        record.workers = (options.fftwInternalWorkers == 0 ? 1 : options.fftwInternalWorkers) *
            options.fftwOuterWorkers + configuredThreads * verticalStrategy.outerWorkers;
        record.internalWorkers = options.fftwInternalWorkers == 0 ? 1 : options.fftwInternalWorkers;
        record.outerWorkers = options.fftwOuterWorkers;
        record.opaqueProviderMemory = true;
        return record;
    };

    auto addCorrectness = [&](ProviderRecord& record,
                              const std::vector<Complex>& horizontal,
                              const std::vector<Complex>& forward,
                              const std::vector<Complex>& inverse,
                              const std::vector<double>& inverseOutput) {
        const auto forwardProbes = gatherVerticalProbes(
            forward.data(), workload.retainedVerticalModes(), probes);
        const auto inverseProbes = gatherVerticalProbes(
            inverse.data(), workload.nz, probes);
        record.correctness = {
            metric("retained horizontal coefficients versus mode-keyed FFTW oracle",
                   horizontal.data(), referenceRetained.data(), horizontal.size()),
            metric("forward vertical projection probes versus independent scalar oracle",
                   forwardProbes.data(), forwardOracle.data(), forwardOracle.size()),
            metric("inverse vertical projection probes versus independent scalar oracle",
                   inverseProbes.data(), inverseOracle.data(), inverseOracle.size()),
            metric("complete inverse boundary for sparse arbitrary modal input",
                   maximumRelativeError(
                       inverseOutput.data(), referenceInverseOutput.data(),
                       inverseOutput.size()))};
    };

    if (options.boundaryPolicy == "wvm-direct") {
        const auto fftwInternalWorkers = options.fftwInternalWorkers == 0
            ? 1 : options.fftwInternalWorkers;
        FFTWProvider fftw(workload, FFTWStrategy{
            fftwPlanningModeNamed(options.fftwPlanning),
            fftwAlignmentStrategyNamed(options.fftwAlignment),
            fftwWisdomStrategyNamed(options.fftwWisdom),
            fftwInternalWorkers, options.fftwOuterWorkers,
            options.fftwPlanningTimeLimitSeconds,
            FFTWDataLayout::interleaved,
            FFTWSpectrumOrder::wvmFrequencyMajor});
        WvmDirectVerticalGemmProvider provider(
            workload, modes, vertical, verticalStrategy);
        if (!provider.supported()) throw std::runtime_error(provider.capability());

        std::vector<Complex> fullSpectrum(workload.spectrumElements());
        std::vector<Complex> fullModalOutput(provider.modalSpectrumElements());
        std::vector<Complex> fullModalInput(provider.modalSpectrumElements());
        std::vector<double> inverseOutput(workload.realElements());
        std::vector<Complex> horizontal(physicalElements);
        std::vector<Complex> forward(modalElements);
        std::vector<Complex> inverse(physicalElements);
        embedRetainedModal(
            workload, modes, sparseModalInput.data(), fullModalInput.data());
        provider.initializeModalOutput(fullModalOutput.data());
        provider.initializeSpectrumOutput(fullSpectrum.data());
        vertical.forward = {};
        vertical.inverse = {};

        fftw.forward(input.data(), fullSpectrum.data());
        gatherRetained(
            workload, modes, fullSpectrum.data(), horizontal.data());
        provider.executeForward(fullSpectrum.data(), fullModalOutput.data());
        gatherRetainedModal(
            workload, modes, fullModalOutput.data(), forward.data());
        provider.initializeSpectrumOutput(fullSpectrum.data());
        provider.executeInverse(fullModalInput.data(), fullSpectrum.data());
        gatherRetained(workload, modes, fullSpectrum.data(), inverse.data());
        fftw.inverse(fullSpectrum.data(), inverseOutput.data());

        auto record = makeRecord(
            "boundary-wvm-direct",
            "full-wvm-order-fftw+direct-per-mode-zgemm-v1",
            "persistent-wvm-frequency-major-interleaved-full-spectrum-and-modal-view");
        record.version = fftw.version() + " + Apple Accelerate";
        record.gemmCallsPerExecution = provider.gemmCallsPerExecution();
        record.explicitPersistentBytes = provider.persistentBytes();
        record.opaquePlanningBytes = fftw.planningBytes();
        record.otherSetupSeconds = fftw.otherSetupSeconds() +
            provider.matrixPreparationSeconds() + provider.schedulerSetupSeconds();
        record.allocationSeconds = fftw.allocationSeconds() + provider.allocationSeconds();
        record.planningSeconds = fftw.planningSeconds();
        record.wisdomGenerationSeconds = fftw.wisdomGenerationSeconds();
        record.wisdomImportSeconds = fftw.wisdomImportSeconds();
        record.planningTimeLimitSeconds = fftw.planningTimeLimitSeconds();
        record.planningBudgetExhausted = fftw.planningBudgetExhausted();
        record.wisdomBytes = fftw.wisdomBytes();
        record.execution.forward = {
            "out-of-place", "out-of-place", false, true, false, false, false,
            "real-grid", "wvm-frequency-major-interleaved-modal-view",
            "real-grid", "wvm-frequency-major-interleaved-modal-view",
            "real grid -> full WVM half-spectrum -> retained frequency zgemm -> full modal view",
            "frequency-major; z/j contiguous within each field and retained frequency",
            provider.modalSpectrumElements() - modalElements, alignof(Complex),
            "full spectrum and modal view do not overlap",
            provider.modalSpectrumElements() * sizeof(Complex), true};
        record.execution.inverse = record.execution.forward;
        record.execution.inverse.destroysNativeInput = true;
        record.execution.inverse.nativeInputRepresentationId =
            "wvm-frequency-major-interleaved-modal-view";
        record.execution.inverse.nativeOutputRepresentationId = "real-grid";
        record.execution.inverse.adapterInputRepresentationId =
            record.execution.inverse.nativeInputRepresentationId;
        record.execution.inverse.adapterOutputRepresentationId = "real-grid";
        record.execution.inverse.physicalExtents =
            "full modal view -> retained frequency zgemm -> rebuilt zero-padded WVM spectrum -> real grid";
        record.execution.inverse.paddingElements =
            workload.spectrumElements() - physicalElements;
        record.execution.inverse.reusableWorkBytes = report.fullSpectrumBytes;
        record.ledger = {
            {"setup/planning", StageState::setupOnly,
             "FFTW plans, immutable expanded complex matrices, and persistent schedulers"},
            {"raw forward FFT", StageState::executed, "full WVM-order FFTW r2c"},
            {"horizontal retention", StageState::elided,
             "retained frequencies remain an indexed view of the full spectrum"},
            {"representation conversion", StageState::elided,
             "interleaved WVM order persists across the boundary"},
            {"permutation/packing", StageState::elided,
             "no gather, transpose, radial ordering, or split conversion"},
            {"raw forward vertical MM", StageState::executed,
             "one small zgemm per retained frequency; fields share each call"},
            {"modal work", StageState::unsupported,
             "modal physics and nonlinear flux are outside this boundary experiment"},
            {"raw inverse vertical MM", StageState::executed,
             "one small zgemm per retained frequency; fields share each call"},
            {"horizontal embedding", StageState::executed,
             "full spectrum must be zeroed on every inverse because FFTW destroys it"},
            {"raw inverse FFT", StageState::executed,
             "full WVM-order FFTW c2r destroys the rebuilt spectrum"},
            {"uninstrumented total", StageState::executed,
             "horizontal and vertical boundary only; no modal work or nonlinear flux"}};
        addCorrectness(record, horizontal, forward, inverse, inverseOutput);

        const auto matrixBytes = provider.matrixBytesPerDirection();
        const auto verticalBytes = matrixBytes +
            report.retainedSpectrumBytes + report.modalSpectrumBytes;
        record.timings = {
            series("setup-shared-component", "logical matrix-family fixture generation", "shared",
                   StageState::setupOnly, report.verticalMatrixFamilySourceBytes,
                   {fixtureGenerationSeconds}),
            series("setup-component", "FFTW planning", "shared", StageState::setupOnly,
                   fftw.planningBytes(), {fftw.planningSeconds()}),
            series("setup-component", "vertical matrix preparation", "shared",
                   StageState::setupOnly, 2 * matrixBytes,
                   {provider.matrixPreparationSeconds()}),
            series("primitive", "raw FFT", "forward", StageState::executed,
                   report.fullRealBytes + report.fullSpectrumBytes,
                   measure(warmups, sampleCount, [&] {
                       fftw.forward(input.data(), fullSpectrum.data());
                   })),
            series("primitive", "raw FFT", "inverse", StageState::executed,
                   report.fullSpectrumBytes + report.fullRealBytes,
                   measure(warmups, sampleCount,
                           [&] {
                               std::copy(referenceSpectrum.begin(), referenceSpectrum.end(),
                                         fullSpectrum.begin());
                           },
                           [&] { fftw.inverse(fullSpectrum.data(), inverseOutput.data()); })),
            series("adapter-component", "logical retained provider-order view", "forward",
                   StageState::elided, 0),
            series("primitive", "raw vertical MM", "forward", StageState::executed,
                   verticalBytes,
                   measure(warmups, sampleCount, [&] {
                       provider.executeForward(fullSpectrum.data(), fullModalOutput.data());
                   })),
            series("primitive", "raw vertical MM", "inverse", StageState::executed,
                   verticalBytes,
                   measure(warmups, sampleCount, [&] {
                       provider.executeInverse(fullModalInput.data(), fullSpectrum.data());
                   })),
            series("adapter-component", "rebuild zero-padded inverse spectrum", "inverse",
                   StageState::executed, report.fullSpectrumBytes,
                   measure(warmups, sampleCount, [&] {
                       provider.initializeSpectrumOutput(fullSpectrum.data());
                   })),
            series("uninstrumented-total", "composed horizontal-vertical boundary", "forward",
                   StageState::executed,
                   report.fullRealBytes + report.fullSpectrumBytes + verticalBytes,
                   measure(warmups, sampleCount, [&] {
                       fftw.forward(input.data(), fullSpectrum.data());
                       provider.executeForward(fullSpectrum.data(), fullModalOutput.data());
                   })),
            series("uninstrumented-total", "composed horizontal-vertical boundary", "inverse",
                   StageState::executed,
                   report.fullSpectrumBytes + verticalBytes +
                       report.fullSpectrumBytes + report.fullRealBytes,
                   measure(warmups, sampleCount, [&] {
                       provider.initializeSpectrumOutput(fullSpectrum.data());
                       provider.executeInverse(fullModalInput.data(), fullSpectrum.data());
                       fftw.inverse(fullSpectrum.data(), inverseOutput.data());
                   }))};
        report.orderingPackingEstimatedExplicitPeakBytes =
            report.verticalMatrixFamilySourceBytes + provider.persistentBytes() +
            2 * report.fullSpectrumBytes +
            bytes(2 * fullModalElements, sizeof(Complex)) + report.fullRealBytes;
        report.providers.push_back(std::move(record));
        report.status = correctnessPassed(report.providers.front()) ? "passed" : "failed";
        return report;
    }

    if (options.boundaryPolicy == "pruned-compact-interleaved") {
        const auto fftwInternalWorkers = options.fftwInternalWorkers == 0
            ? 1 : options.fftwInternalWorkers;
        FFTWPrunedProvider fftw(
            workload, modes, fftwPlanningModeNamed(options.fftwPlanning),
            fftwInternalWorkers, options.fftwOuterWorkers);
        VerticalGemmProvider provider(
            workload, vertical, VerticalGemmLayout::complexInterleaved,
            verticalStrategy);
        if (!provider.supported()) throw std::runtime_error(provider.capability());
        provider.loadModalInput(sparseModalInput.data());
        vertical.forward = {};
        vertical.inverse = {};

        std::vector<double> inverseOutput(workload.realElements());
        std::vector<Complex> horizontal(physicalElements);
        std::vector<Complex> forward(modalElements);
        std::vector<Complex> inverse(physicalElements);

        fftw.forward(input.data(), provider.interleavedPhysicalInputData());
        std::copy_n(
            provider.interleavedPhysicalInputData(), physicalElements,
            horizontal.data());
        provider.executeForward();
        provider.copyForwardOutput(forward.data());
        provider.executeInverse();
        provider.copyInverseOutput(inverse.data());
        fftw.inverse(provider.interleavedPhysicalOutputData(), inverseOutput.data());

        auto record = makeRecord(
            "boundary-pruned-compact-interleaved",
            "partial-column-pruned-fftw+compact-interleaved-k2-zgemm-v1",
            "radial-compact-interleaved-vertical-columns");
        record.version = fftw.version() + " + Apple Accelerate";
        record.gemmCallsPerExecution = provider.gemmCallsPerExecution();
        record.explicitPersistentBytes = provider.persistentBytes();
        record.scratchBytes = fftw.scratchBytes();
        record.opaquePlanningBytes = fftw.planningBytes();
        record.otherSetupSeconds = fftw.otherSetupSeconds() +
            provider.matrixPreparationSeconds() + provider.schedulerSetupSeconds();
        record.allocationSeconds = fftw.allocationSeconds() + provider.allocationSeconds();
        record.planningSeconds = fftw.planningSeconds();
        record.execution.forward = {
            "out-of-place", "out-of-place", false, true, false, false, false,
            "real-grid", "radial-compact-interleaved-modal",
            "real-grid", "radial-compact-interleaved-modal",
            "real grid -> pruned internal row/column scratch -> retained [Nz][K] -> modal [Nj][K]",
            "compact z/j-contiguous columns keyed by field+fields*mode",
            0, std::max<std::size_t>(64, fftw.minimumAlignmentBytes()),
            "pruned FFT scratch and compact vertical operands do not overlap",
            fftw.scratchBytes() + provider.physicalElements() * sizeof(Complex), true};
        record.execution.inverse = record.execution.forward;
        record.execution.inverse.nativeInputRepresentationId =
            "radial-compact-interleaved-modal";
        record.execution.inverse.nativeOutputRepresentationId = "real-grid";
        record.execution.inverse.adapterInputRepresentationId =
            record.execution.inverse.nativeInputRepresentationId;
        record.execution.inverse.adapterOutputRepresentationId = "real-grid";
        record.execution.inverse.physicalExtents =
            "modal [Nj][K] -> physical [Nz][K] -> pruned zero-padded scratch -> real grid";
        record.execution.inverse.destroysNativeInput = false;
        record.execution.inverse.reusableWorkBytes = fftw.scratchBytes();
        record.ledger = {
            {"setup/planning", StageState::setupOnly,
             "separable pruned FFTW row/column plans, expanded complex grouped matrices, and persistent schedulers"},
            {"raw forward FFT", StageState::executed,
             "all row r2c transforms plus only retained-support column transforms"},
            {"horizontal retention", StageState::executed,
             "compact mode-keyed gather from internal pruned scratch"},
            {"representation conversion", StageState::elided,
             "compact interleaved output is the vertical zgemm input representation"},
            {"permutation/packing", StageState::elided,
             "no additional copy between compact horizontal output and vertical input"},
            {"raw forward vertical MM", StageState::executed,
             "one grouped zgemm per K-squared matrix group"},
            {"modal work", StageState::unsupported,
             "modal physics and nonlinear flux are outside this boundary experiment"},
            {"raw inverse vertical MM", StageState::executed,
             "one grouped zgemm per K-squared matrix group"},
            {"horizontal embedding", StageState::executed,
             "zero and embed compact coefficients into persistent pruned scratch"},
            {"raw inverse FFT", StageState::executed,
             "retained-support inverse columns plus all inverse rows"},
            {"uninstrumented total", StageState::executed,
             "horizontal and vertical boundary only; no modal work or nonlinear flux"}};
        addCorrectness(record, horizontal, forward, inverse, inverseOutput);

        const auto matrixBytes = provider.matrixBytesPerDirection();
        const auto verticalBytes = matrixBytes +
            report.retainedSpectrumBytes + report.modalSpectrumBytes;
        const auto retainedMovementBytes = 2 * report.retainedSpectrumBytes;
        const auto prunedScratchBytes = static_cast<std::uint64_t>(fftw.scratchBytes());
        record.timings = {
            series("setup-shared-component", "logical matrix-family fixture generation", "shared",
                   StageState::setupOnly, report.verticalMatrixFamilySourceBytes,
                   {fixtureGenerationSeconds}),
            series("setup-component", "pruned FFTW planning", "shared",
                   StageState::setupOnly, fftw.planningBytes(),
                   {fftw.planningSeconds()}),
            series("setup-component", "vertical matrix preparation", "shared",
                   StageState::setupOnly, 2 * matrixBytes,
                   {provider.matrixPreparationSeconds()}),
            series("primitive-component", "pruned row FFT batch", "forward",
                   StageState::executed, report.fullRealBytes + prunedScratchBytes,
                   measure(warmups, sampleCount, [&] {
                       fftw.executeForwardRows(input.data());
                   })),
            series("primitive-component", "retained-support column FFT batch", "forward",
                   StageState::executed, 2 * prunedScratchBytes,
                   measure(warmups, sampleCount,
                           [&] { fftw.executeForwardRows(input.data()); },
                           [&] { fftw.executeForwardColumns(); })),
            series("primitive", "raw pruned FFT", "forward", StageState::executed,
                   report.fullRealBytes + 3 * prunedScratchBytes,
                   measure(warmups, sampleCount, [&] {
                       fftw.executeForwardRows(input.data());
                       fftw.executeForwardColumns();
                   })),
            series("adapter-component", "compact retained gather", "forward",
                   StageState::executed, retainedMovementBytes,
                   measure(warmups, sampleCount,
                           [&] {
                               fftw.executeForwardRows(input.data());
                               fftw.executeForwardColumns();
                           },
                           [&] {
                               fftw.gatherForward(
                                   provider.interleavedPhysicalInputData());
                           })),
            series("primitive", "raw vertical MM", "forward", StageState::executed,
                   verticalBytes,
                   measure(warmups, sampleCount, [&] { provider.executeForward(); })),
            series("primitive", "raw vertical MM", "inverse", StageState::executed,
                   verticalBytes,
                   measure(warmups, sampleCount, [&] { provider.executeInverse(); })),
            series("adapter-component", "zero-pad and compact retained embed", "inverse",
                   StageState::executed, prunedScratchBytes + retainedMovementBytes,
                   measure(warmups, sampleCount, [&] {
                       fftw.embedInverse(provider.interleavedPhysicalOutputData());
                   })),
            series("primitive-component", "retained-support inverse column FFT batch", "inverse",
                   StageState::executed, 2 * prunedScratchBytes,
                   measure(warmups, sampleCount,
                           [&] {
                               fftw.embedInverse(
                                   provider.interleavedPhysicalOutputData());
                           },
                           [&] { fftw.executeInverseColumns(); })),
            series("primitive-component", "pruned inverse row FFT batch", "inverse",
                   StageState::executed, prunedScratchBytes + report.fullRealBytes,
                   measure(warmups, sampleCount,
                           [&] {
                               fftw.embedInverse(
                                   provider.interleavedPhysicalOutputData());
                               fftw.executeInverseColumns();
                           },
                           [&] { fftw.executeInverseRows(inverseOutput.data()); })),
            series("primitive", "raw pruned FFT", "inverse", StageState::executed,
                   3 * prunedScratchBytes + report.fullRealBytes,
                   measure(warmups, sampleCount,
                           [&] {
                               fftw.embedInverse(
                                   provider.interleavedPhysicalOutputData());
                           },
                           [&] {
                               fftw.executeInverseColumns();
                               fftw.executeInverseRows(inverseOutput.data());
                           })),
            series("uninstrumented-total", "composed horizontal-vertical boundary", "forward",
                   StageState::executed,
                   report.fullRealBytes + 3 * prunedScratchBytes +
                       retainedMovementBytes + verticalBytes,
                   measure(warmups, sampleCount, [&] {
                       fftw.forward(
                           input.data(), provider.interleavedPhysicalInputData());
                       provider.executeForward();
                   })),
            series("uninstrumented-total", "composed horizontal-vertical boundary", "inverse",
                   StageState::executed,
                   verticalBytes + prunedScratchBytes + retainedMovementBytes +
                       3 * prunedScratchBytes + report.fullRealBytes,
                   measure(warmups, sampleCount, [&] {
                       provider.executeInverse();
                       fftw.inverse(
                           provider.interleavedPhysicalOutputData(),
                           inverseOutput.data());
                   }))};
        report.orderingPackingEstimatedExplicitPeakBytes =
            report.verticalMatrixFamilySourceBytes + provider.persistentBytes() +
            fftw.scratchBytes() + report.fullRealBytes +
            report.retainedSpectrumBytes + report.modalSpectrumBytes;
        report.providers.push_back(std::move(record));
        report.status = correctnessPassed(report.providers.front()) ? "passed" : "failed";
        return report;
    }

    if (options.boundaryPolicy == "plane-major-view") {
        const auto fftwInternalWorkers = options.fftwInternalWorkers == 0
            ? 1 : options.fftwInternalWorkers;
        FFTWProvider fftw(workload, FFTWStrategy{
            fftwPlanningModeNamed(options.fftwPlanning),
            fftwAlignmentStrategyNamed(options.fftwAlignment),
            fftwWisdomStrategyNamed(options.fftwWisdom),
            fftwInternalWorkers, options.fftwOuterWorkers,
            options.fftwPlanningTimeLimitSeconds,
            FFTWDataLayout::interleaved,
            FFTWSpectrumOrder::planeMajor});
        PlaneMajorDirectVerticalGemmProvider provider(
            workload, modes, vertical, verticalStrategy);
        if (!provider.supported()) throw std::runtime_error(provider.capability());

        std::vector<Complex> fullSpectrum(workload.spectrumElements());
        std::vector<Complex> fullModalOutput(provider.modalSpectrumElements());
        std::vector<Complex> fullModalInput(provider.modalSpectrumElements());
        std::vector<double> inverseOutput(workload.realElements());
        std::vector<Complex> horizontal(physicalElements);
        std::vector<Complex> forward(modalElements);
        std::vector<Complex> inverse(physicalElements);
        compactModalToPlaneMajor(
            workload, modes, sparseModalInput.data(), fullModalInput.data());
        provider.initializeModalOutput(fullModalOutput.data());
        provider.initializeSpectrumOutput(fullSpectrum.data());
        vertical.forward = {};
        vertical.inverse = {};

        fftw.forward(input.data(), fullSpectrum.data());
        planeMajorPhysicalToCompact(
            workload, modes, fullSpectrum.data(), horizontal.data());
        provider.executeForward(fullSpectrum.data(), fullModalOutput.data());
        planeMajorModalToCompact(
            workload, modes, fullModalOutput.data(), forward.data());
        provider.initializeSpectrumOutput(fullSpectrum.data());
        provider.executeInverse(fullModalInput.data(), fullSpectrum.data());
        planeMajorPhysicalToCompact(
            workload, modes, fullSpectrum.data(), inverse.data());
        fftw.inverse(fullSpectrum.data(), inverseOutput.data());

        auto record = makeRecord(
            "boundary-plane-major-view",
            "full-plane-major-fftw+strided-retained-zgemv-view-v1",
            "persistent-plane-major-interleaved-full-spectrum-and-modal-view");
        record.version = fftw.version() + " + Apple Accelerate";
        record.gemmCallsPerExecution = provider.gemvCallsPerExecution();
        record.explicitPersistentBytes = provider.persistentBytes();
        record.opaquePlanningBytes = fftw.planningBytes();
        record.otherSetupSeconds = fftw.otherSetupSeconds() +
            provider.matrixPreparationSeconds() + provider.schedulerSetupSeconds();
        record.allocationSeconds = fftw.allocationSeconds() + provider.allocationSeconds();
        record.planningSeconds = fftw.planningSeconds();
        record.wisdomGenerationSeconds = fftw.wisdomGenerationSeconds();
        record.wisdomImportSeconds = fftw.wisdomImportSeconds();
        record.planningTimeLimitSeconds = fftw.planningTimeLimitSeconds();
        record.planningBudgetExhausted = fftw.planningBudgetExhausted();
        record.wisdomBytes = fftw.wisdomBytes();
        record.execution.forward = {
            "out-of-place", "out-of-place", false, true, false, false, false,
            "real-grid", "plane-major-interleaved-modal-index-view",
            "real-grid", "plane-major-interleaved-modal-index-view",
            "real grid -> full plane-major half-spectrum -> strided retained zgemv -> full modal view",
            "frequency=1; z/j=halfRows; field=halfRows*Nz-or-Nj",
            provider.modalSpectrumElements() - modalElements, alignof(Complex),
            "full spectrum and modal view do not overlap",
            provider.modalSpectrumElements() * sizeof(Complex), true};
        record.execution.inverse = record.execution.forward;
        record.execution.inverse.destroysNativeInput = true;
        record.execution.inverse.nativeInputRepresentationId =
            "plane-major-interleaved-modal-index-view";
        record.execution.inverse.nativeOutputRepresentationId = "real-grid";
        record.execution.inverse.adapterInputRepresentationId =
            record.execution.inverse.nativeInputRepresentationId;
        record.execution.inverse.adapterOutputRepresentationId = "real-grid";
        record.execution.inverse.physicalExtents =
            "full modal view -> strided retained zgemv -> rebuilt zero-padded plane-major spectrum -> real grid";
        record.execution.inverse.paddingElements =
            workload.spectrumElements() - physicalElements;
        record.execution.inverse.reusableWorkBytes = report.fullSpectrumBytes;
        record.ledger = {
            {"setup/planning", StageState::setupOnly,
             "plane-major FFTW plans, immutable expanded complex matrices, and persistent schedulers"},
            {"raw forward FFT", StageState::executed,
             "full plane-major FFTW r2c"},
            {"horizontal retention", StageState::elided,
             "retained frequencies remain an indexed view of the full spectrum"},
            {"representation conversion", StageState::elided,
             "plane-major interleaved storage persists across the boundary"},
            {"permutation/packing", StageState::elided,
             "no data movement; z/j operands are consumed with halfRows stride"},
            {"raw forward vertical MM", StageState::executed,
             "one strided zgemv per retained frequency and field"},
            {"modal work", StageState::unsupported,
             "modal physics and nonlinear flux are outside this boundary experiment"},
            {"raw inverse vertical MM", StageState::executed,
             "one strided zgemv per retained frequency and field"},
            {"horizontal embedding", StageState::executed,
             "full spectrum must be zeroed on every inverse because FFTW destroys it"},
            {"raw inverse FFT", StageState::executed,
             "full plane-major FFTW c2r destroys the rebuilt spectrum"},
            {"uninstrumented total", StageState::executed,
             "horizontal and vertical boundary only; no modal work or nonlinear flux"}};
        addCorrectness(record, horizontal, forward, inverse, inverseOutput);

        const auto matrixBytes = provider.matrixBytesPerDirection();
        const auto verticalBytes = matrixBytes +
            report.retainedSpectrumBytes + report.modalSpectrumBytes;
        record.timings = {
            series("setup-shared-component", "logical matrix-family fixture generation", "shared",
                   StageState::setupOnly, report.verticalMatrixFamilySourceBytes,
                   {fixtureGenerationSeconds}),
            series("setup-component", "FFTW planning", "shared", StageState::setupOnly,
                   fftw.planningBytes(), {fftw.planningSeconds()}),
            series("setup-component", "vertical matrix preparation", "shared",
                   StageState::setupOnly, 2 * matrixBytes,
                   {provider.matrixPreparationSeconds()}),
            series("primitive", "raw FFT", "forward", StageState::executed,
                   report.fullRealBytes + report.fullSpectrumBytes,
                   measure(warmups, sampleCount, [&] {
                       fftw.forward(input.data(), fullSpectrum.data());
                   })),
            series("primitive", "raw FFT", "inverse", StageState::executed,
                   report.fullSpectrumBytes + report.fullRealBytes,
                   measure(warmups, sampleCount,
                           [&] {
                               wvmToPlaneMajor(
                                   workload, referenceSpectrum.data(),
                                   fullSpectrum.data());
                           },
                           [&] { fftw.inverse(fullSpectrum.data(), inverseOutput.data()); })),
            series("adapter-component", "logical retained provider-order view", "forward",
                   StageState::elided, 0),
            series("primitive", "raw vertical MM", "forward", StageState::executed,
                   verticalBytes,
                   measure(warmups, sampleCount, [&] {
                       provider.executeForward(
                           fullSpectrum.data(), fullModalOutput.data());
                   })),
            series("primitive", "raw vertical MM", "inverse", StageState::executed,
                   verticalBytes,
                   measure(warmups, sampleCount, [&] {
                       provider.executeInverse(
                           fullModalInput.data(), fullSpectrum.data());
                   })),
            series("adapter-component", "rebuild zero-padded inverse view", "inverse",
                   StageState::executed, report.fullSpectrumBytes,
                   measure(warmups, sampleCount, [&] {
                       provider.initializeSpectrumOutput(fullSpectrum.data());
                   })),
            series("uninstrumented-total", "composed horizontal-vertical boundary", "forward",
                   StageState::executed,
                   report.fullRealBytes + report.fullSpectrumBytes + verticalBytes,
                   measure(warmups, sampleCount, [&] {
                       fftw.forward(input.data(), fullSpectrum.data());
                       provider.executeForward(
                           fullSpectrum.data(), fullModalOutput.data());
                   })),
            series("uninstrumented-total", "composed horizontal-vertical boundary", "inverse",
                   StageState::executed,
                   report.fullSpectrumBytes + verticalBytes +
                       report.fullSpectrumBytes + report.fullRealBytes,
                   measure(warmups, sampleCount, [&] {
                       provider.initializeSpectrumOutput(fullSpectrum.data());
                       provider.executeInverse(
                           fullModalInput.data(), fullSpectrum.data());
                       fftw.inverse(fullSpectrum.data(), inverseOutput.data());
                   }))};
        report.orderingPackingEstimatedExplicitPeakBytes =
            report.verticalMatrixFamilySourceBytes + provider.persistentBytes() +
            2 * report.fullSpectrumBytes +
            bytes(2 * fullModalElements, sizeof(Complex)) + report.fullRealBytes;
        report.providers.push_back(std::move(record));
        report.status = correctnessPassed(report.providers.front()) ? "passed" : "failed";
        return report;
    }

    if (options.boundaryPolicy == "wvm-packed-split" ||
        options.boundaryPolicy == "plane-major-fused-split") {
        const bool planeFused = options.boundaryPolicy == "plane-major-fused-split";
        const auto fftwInternalWorkers = options.fftwInternalWorkers == 0
            ? 1 : options.fftwInternalWorkers;
        FFTWProvider fftw(workload, FFTWStrategy{
            fftwPlanningModeNamed(options.fftwPlanning),
            fftwAlignmentStrategyNamed(options.fftwAlignment),
            fftwWisdomStrategyNamed(options.fftwWisdom),
            fftwInternalWorkers, options.fftwOuterWorkers,
            options.fftwPlanningTimeLimitSeconds,
            FFTWDataLayout::interleaved,
            planeFused ? FFTWSpectrumOrder::planeMajor
                       : FFTWSpectrumOrder::wvmFrequencyMajor});
        VerticalGemmProvider provider(
            workload, vertical, VerticalGemmLayout::split, verticalStrategy);
        if (!provider.supported()) throw std::runtime_error(provider.capability());
        provider.loadModalInput(sparseModalInput.data());
        vertical.forward = {};
        vertical.inverse = {};

        std::vector<Complex> fullSpectrum(workload.spectrumElements());
        std::vector<double> inverseOutput(workload.realElements());
        std::vector<Complex> horizontal(physicalElements);
        std::vector<Complex> forward(modalElements);
        std::vector<Complex> inverse(physicalElements);

        fftw.forward(input.data(), fullSpectrum.data());
        if (planeFused) {
            fftw.gatherRetainedToSplitOuter(
                modes, fullSpectrum.data(),
                provider.splitPhysicalInputRealData(),
                provider.splitPhysicalInputImaginaryData());
        } else {
            provider.packPhysicalInputFromWvm(modes, fullSpectrum.data());
        }
        splitToInterleaved(
            physicalElements, provider.splitPhysicalInputRealData(),
            provider.splitPhysicalInputImaginaryData(), horizontal.data());
        provider.executeForward();
        provider.copyForwardOutput(forward.data());
        provider.executeInverse();
        provider.copyInverseOutput(inverse.data());
        if (planeFused) {
            fftw.embedRetainedFromSplitOuter(
                modes, provider.splitPhysicalOutputRealData(),
                provider.splitPhysicalOutputImaginaryData(),
                fullSpectrum.data());
        } else {
            provider.embedPhysicalOutputToWvm(modes, fullSpectrum.data());
        }
        fftw.inverse(fullSpectrum.data(), inverseOutput.data());

        auto record = makeRecord(
            planeFused ? "boundary-plane-major-fused-split"
                       : "boundary-wvm-packed-split",
            planeFused
                ? "full-plane-major-fftw+fused-retained-split+split-k2-dgemm-v1"
                : "full-wvm-order-fftw+radial-pack+split-k2-dgemm-v1",
            planeFused
                ? "plane-major-full-interleaved-to-radial-compact-split"
                : "wvm-frequency-major-full-interleaved-to-radial-compact-split");
        record.version = fftw.version() + " + Apple Accelerate";
        record.gemmCallsPerExecution = provider.gemmCallsPerExecution();
        record.explicitPersistentBytes = provider.persistentBytes();
        record.scratchBytes = 0;
        record.opaquePlanningBytes = fftw.planningBytes();
        record.otherSetupSeconds = fftw.otherSetupSeconds() +
            provider.matrixPreparationSeconds() + provider.schedulerSetupSeconds();
        record.allocationSeconds = fftw.allocationSeconds() + provider.allocationSeconds();
        record.planningSeconds = fftw.planningSeconds();
        record.wisdomGenerationSeconds = fftw.wisdomGenerationSeconds();
        record.wisdomImportSeconds = fftw.wisdomImportSeconds();
        record.planningTimeLimitSeconds = fftw.planningTimeLimitSeconds();
        record.planningBudgetExhausted = fftw.planningBudgetExhausted();
        record.wisdomBytes = fftw.wisdomBytes();
        record.execution.forward = {
            "out-of-place", "out-of-place", false, true, false, false, false,
            "real-grid", "radial-compact-split-modal",
            "real-grid", "radial-compact-split-modal",
            planeFused
                ? "real grid -> full plane-major half-spectrum -> retained [Nz][K] -> modal [Nj][K]"
                : "real grid -> full WVM half-spectrum -> retained [Nz][K] -> modal [Nj][K]",
            planeFused
                ? "plane-major full spectrum, then z/j contiguous compact columns"
                : "WVM frequency-major, then z/j contiguous compact columns",
            workload.spectrumElements() - physicalElements, 64,
            "FFT, split vertical input, and split modal output are disjoint",
            workload.spectrumElements() * sizeof(Complex), true};
        record.execution.inverse = record.execution.forward;
        record.execution.inverse.destroysNativeInput = true;
        record.execution.inverse.nativeInputRepresentationId =
            "radial-compact-split-modal";
        record.execution.inverse.nativeOutputRepresentationId = "real-grid";
        record.execution.inverse.adapterInputRepresentationId =
            record.execution.inverse.nativeInputRepresentationId;
        record.execution.inverse.adapterOutputRepresentationId = "real-grid";
        record.execution.inverse.physicalExtents =
            planeFused
                ? "modal [Nj][K] -> physical [Nz][K] -> zero-padded plane-major half-spectrum -> real grid"
                : "modal [Nj][K] -> physical [Nz][K] -> zero-padded WVM half-spectrum -> real grid";
        record.execution.inverse.requiresPreservationCopyForRepeatedExecution = false;
        record.execution.inverse.reusableWorkBytes = report.fullSpectrumBytes;
        record.ledger = {
            {"setup/planning", StageState::setupOnly,
             "FFTW plans, grouped real matrix family, split operands, and persistent schedulers"},
            {"raw forward FFT", StageState::executed,
             planeFused ? "full plane-major FFTW r2c" : "full WVM-order FFTW r2c"},
            {"horizontal retention", StageState::fused,
             "retained selection is fused with radial split packing"},
            {"representation conversion", StageState::fused,
             "interleaved-to-split conversion is fused with retained selection"},
            {"permutation/packing", StageState::executed,
             planeFused
                 ? "outer-sharded fused retained selection and split conversion"
                 : "historical MATLAB-style WVM gather into radial contiguous vertical columns"},
            {"raw forward vertical MM", StageState::executed,
             "two grouped dgemm calls per K-squared matrix group"},
            {"modal work", StageState::unsupported,
             "modal physics and nonlinear flux are outside this boundary experiment"},
            {"raw inverse vertical MM", StageState::executed,
             "two grouped dgemm calls per K-squared matrix group"},
            {"horizontal embedding", StageState::executed,
             planeFused
                 ? "outer-sharded fused split scatter into a zeroed plane-major spectrum"
                 : "zero full WVM spectrum and scatter split retained coefficients"},
            {"raw inverse FFT", StageState::executed,
             planeFused
                 ? "full plane-major FFTW c2r; FFTW destroys its spectrum input"
                 : "full WVM-order FFTW c2r; FFTW destroys its spectrum input"},
            {"uninstrumented total", StageState::executed,
             "horizontal and vertical boundary only; no modal work or nonlinear flux"}};
        addCorrectness(record, horizontal, forward, inverse, inverseOutput);

        const auto matrixBytes = provider.matrixBytesPerDirection();
        const auto verticalBytes = 2 * matrixBytes +
            report.retainedSpectrumBytes + report.modalSpectrumBytes;
        const auto packBytes = 2 * report.retainedSpectrumBytes;
        const auto embedBytes = report.fullSpectrumBytes +
            2 * report.retainedSpectrumBytes;
        record.timings = {
            series("setup-shared-component", "logical matrix-family fixture generation", "shared",
                   StageState::setupOnly, report.verticalMatrixFamilySourceBytes,
                   {fixtureGenerationSeconds}),
            series("setup-component", "FFTW planning", "shared", StageState::setupOnly,
                   fftw.planningBytes(), {fftw.planningSeconds()}),
            series("setup-component", "vertical matrix preparation", "shared",
                   StageState::setupOnly, 2 * matrixBytes,
                   {provider.matrixPreparationSeconds()}),
            series("primitive", "raw FFT", "forward", StageState::executed,
                   report.fullRealBytes + report.fullSpectrumBytes,
                   measure(warmups, sampleCount, [&] {
                       fftw.forward(input.data(), fullSpectrum.data());
                   })),
            series("primitive", "raw FFT", "inverse", StageState::executed,
                   report.fullSpectrumBytes + report.fullRealBytes,
                   measure(warmups, sampleCount,
                           [&] {
                               if (planeFused) {
                                   wvmToPlaneMajor(
                                       workload, referenceSpectrum.data(),
                                       fullSpectrum.data());
                               } else {
                                   std::copy(referenceSpectrum.begin(), referenceSpectrum.end(),
                                             fullSpectrum.begin());
                               }
                           },
                           [&] { fftw.inverse(fullSpectrum.data(), inverseOutput.data()); })),
            series("adapter-component",
                   planeFused ? "fused retained selection and split pack"
                              : "WVM retained gather and radial split pack",
                   "forward",
                   StageState::executed, packBytes,
                   measure(warmups, sampleCount, [&] {
                           if (planeFused) {
                               fftw.gatherRetainedToSplitOuter(
                                   modes, fullSpectrum.data(),
                                   provider.splitPhysicalInputRealData(),
                                   provider.splitPhysicalInputImaginaryData());
                           } else {
                               provider.packPhysicalInputFromWvm(
                                   modes, referenceSpectrum.data());
                           }
                   })),
            series("primitive", "raw vertical MM", "forward", StageState::executed,
                   verticalBytes,
                   measure(warmups, sampleCount, [&] { provider.executeForward(); })),
            series("primitive", "raw vertical MM", "inverse", StageState::executed,
                   verticalBytes,
                   measure(warmups, sampleCount, [&] { provider.executeInverse(); })),
            series("adapter-component",
                   planeFused ? "fused split embed into zeroed plane-major spectrum"
                              : "zero-pad and WVM retained embed",
                   "inverse",
                   StageState::executed, embedBytes,
                   measure(warmups, sampleCount, [&] {
                       if (planeFused) {
                           fftw.embedRetainedFromSplitOuter(
                               modes, provider.splitPhysicalOutputRealData(),
                               provider.splitPhysicalOutputImaginaryData(),
                               fullSpectrum.data());
                       } else {
                           provider.embedPhysicalOutputToWvm(
                               modes, fullSpectrum.data());
                       }
                   })),
            series("uninstrumented-total", "composed horizontal-vertical boundary", "forward",
                   StageState::executed,
                   report.fullRealBytes + report.fullSpectrumBytes + packBytes + verticalBytes,
                   measure(warmups, sampleCount, [&] {
                       fftw.forward(input.data(), fullSpectrum.data());
                       if (planeFused) {
                           fftw.gatherRetainedToSplitOuter(
                               modes, fullSpectrum.data(),
                               provider.splitPhysicalInputRealData(),
                               provider.splitPhysicalInputImaginaryData());
                       } else {
                           provider.packPhysicalInputFromWvm(
                               modes, fullSpectrum.data());
                       }
                       provider.executeForward();
                   })),
            series("uninstrumented-total", "composed horizontal-vertical boundary", "inverse",
                   StageState::executed,
                   verticalBytes + embedBytes + report.fullSpectrumBytes + report.fullRealBytes,
                   measure(warmups, sampleCount, [&] {
                       provider.executeInverse();
                       if (planeFused) {
                           fftw.embedRetainedFromSplitOuter(
                               modes, provider.splitPhysicalOutputRealData(),
                               provider.splitPhysicalOutputImaginaryData(),
                               fullSpectrum.data());
                       } else {
                           provider.embedPhysicalOutputToWvm(
                               modes, fullSpectrum.data());
                       }
                       fftw.inverse(fullSpectrum.data(), inverseOutput.data());
                   }))};
        report.orderingPackingEstimatedExplicitPeakBytes =
            report.verticalMatrixFamilySourceBytes + provider.persistentBytes() +
            2 * report.fullSpectrumBytes + report.fullRealBytes +
            report.retainedSpectrumBytes + report.modalSpectrumBytes;
        report.providers.push_back(std::move(record));
        report.status = correctnessPassed(report.providers.front()) ? "passed" : "failed";
        return report;
    }

    throw std::logic_error(
        "The requested spectral-boundary policy is registered but not implemented.");
}

BenchmarkReport runSpectralPipelineBenchmark(const RunOptions& options) {
    if (options.boundaryPolicy != "wvm-direct" &&
        options.boundaryPolicy != "plane-major-fused-split") {
        throw std::invalid_argument(
            "spectral-pipeline boundary-policy must be wvm-direct or "
            "plane-major-fused-split.");
    }
    if (options.verticalGemmFamily != "k2-grouped") {
        throw std::invalid_argument(
            "spectral-pipeline requires --vertical-gemm-family k2-grouped.");
    }
    if (options.workers != 0) {
        throw std::invalid_argument(
            "spectral-pipeline uses independent FFT and vertical worker controls; "
            "omit --workers.");
    }

    const auto selected = profileNamed(options.profile);
    const auto warmups = options.warmups == 0 ? selected.warmups : options.warmups;
    const auto sampleCount = options.samples == 0 ? selected.samples : options.samples;
    if (sampleCount == 0) {
        throw std::invalid_argument("spectral-pipeline requires at least one sample.");
    }
    const VerticalGemmStrategy verticalStrategy{
        verticalGemmScheduleNamed(options.verticalGemmSchedule),
        options.verticalGemmOuterWorkers};
    if (verticalStrategy.schedule == VerticalGemmSchedule::serial &&
        verticalStrategy.outerWorkers != 1) {
        throw std::invalid_argument(
            "spectral-pipeline serial vertical scheduling requires one outer worker.");
    }

    BenchmarkReport report;
    report.profile = selected.name;
    report.seed = options.seed;
    report.warmups = warmups;
    report.samples = sampleCount;
    report.workload = selected.workload;
    report.environment = environmentRecord();
    report.runId = utcTimestamp(true) + "-" + report.environment.hostname;
    const auto configuredThreads = configuredAccelerateThreads(report.environment);
    if (verticalStrategy.outerWorkers > 1 && configuredThreads != 1) {
        throw std::invalid_argument(
            "spectral-pipeline outer vertical scheduling requires "
            "VECLIB_MAXIMUM_THREADS=1.");
    }

    const auto& workload = report.workload;
    const auto modes = retainedHorizontalModes(workload);
    const auto fixtureStart = Clock::now();
    auto vertical = squaredWavenumberVerticalFixture(workload, modes);
    const auto fixtureGenerationSeconds =
        std::chrono::duration<double>(Clock::now() - fixtureStart).count();
    const auto weightStart = Clock::now();
    const auto modalWeights = syntheticModalWorkWeights(workload, modes);
    const auto weightGenerationSeconds =
        std::chrono::duration<double>(Clock::now() - weightStart).count();
    const auto columns = modes.size() * workload.fields;
    const auto physicalElements = workload.nz * columns;
    const auto modalElements = workload.retainedVerticalModes() * columns;
    report.retainedHorizontalModeCount = modes.size();
    report.retainedModeOrderHash = modeOrderHash(modes);
    report.wvmFullSpectrumOrderHash = wvmSpectrumOrderHash(workload);
    report.fullRealBytes = bytes(workload.realElements(), sizeof(double));
    report.fullSpectrumBytes = bytes(workload.spectrumElements(), sizeof(Complex));
    report.retainedSpectrumBytes = bytes(physicalElements, sizeof(Complex));
    report.modalSpectrumBytes = bytes(modalElements, sizeof(Complex));
    report.verticalMatrixFamilySourceBytes = bytes(
        vertical.forward.size() + vertical.inverse.size(), sizeof(double));
    report.verticalMatrixFamilyId = vertical.id;
    report.verticalGroupCount = vertical.groups.size();
    report.verticalGroupOrderHash = verticalModeGroupHash(vertical.groups);
    std::vector<double> groupModes;
    std::vector<double> groupColumns;
    groupModes.reserve(vertical.groups.size());
    groupColumns.reserve(vertical.groups.size());
    for (const auto& group : vertical.groups) {
        groupModes.push_back(static_cast<double>(group.modeCount));
        groupColumns.push_back(static_cast<double>(group.modeCount * workload.fields));
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

    const auto input = makeFixture(workload, FixtureKind::random, options.seed);
    std::vector<Complex> referenceSpectrum(workload.spectrumElements());
    std::vector<Complex> referenceRetained(physicalElements);
    std::vector<Complex> referenceForwardModal(modalElements);
    std::vector<Complex> referencePostWorkModal(modalElements);
    std::vector<Complex> referenceInversePhysical(physicalElements);
    std::vector<Complex> referenceInverseSpectrum(workload.spectrumElements());
    std::vector<double> referencePipelineOutput(workload.realElements());
    {
        FFTWProvider referenceFftw(workload, FFTWStrategy{
            FFTWPlanningMode::estimate, FFTWAlignmentStrategy::unaligned,
            FFTWWisdomStrategy::cold, 1, 1, 0.0,
            FFTWDataLayout::interleaved,
            FFTWSpectrumOrder::wvmFrequencyMajor});
        referenceFftw.forward(input.data(), referenceSpectrum.data());
        gatherRetained(
            workload, modes, referenceSpectrum.data(), referenceRetained.data());

        VerticalGemmProvider referenceVertical(
            workload, vertical, VerticalGemmLayout::complexInterleaved,
            {VerticalGemmSchedule::serial, 1});
        if (!referenceVertical.supported()) {
            throw std::runtime_error(referenceVertical.capability());
        }
        referenceVertical.loadPhysicalInput(referenceRetained.data());
        referenceVertical.executeForward();
        referenceVertical.copyForwardOutput(referenceForwardModal.data());
        applySyntheticModalWorkInterleaved(
            modalElements, modalWeights.data(), referenceForwardModal.data(),
            referencePostWorkModal.data());
        referenceVertical.loadModalInput(referencePostWorkModal.data());
        referenceVertical.executeInverse();
        referenceVertical.copyInverseOutput(referenceInversePhysical.data());
        embedRetained(
            workload, modes, referenceInversePhysical.data(),
            referenceInverseSpectrum.data());
        referenceFftw.inverse(
            referenceInverseSpectrum.data(), referencePipelineOutput.data());
    }

    const auto probes = verticalProbeColumns(columns);
    const auto forwardProbeOracle = directVerticalForwardProbes(
        vertical, workload, probes, referenceRetained.data());
    const auto inverseProbeOracle = directVerticalInverseProbes(
        vertical, workload, probes, referencePostWorkModal.data());

    auto makeRecord = [&](std::string id, std::string algorithm,
                          std::string representation) {
        ProviderRecord record;
        record.id = std::move(id);
        record.version = "FFTW 3.3.11 + Apple Accelerate";
        record.libraryIdentity =
            "pinned FFTW 3.3.11 and /System/Library/Frameworks/Accelerate.framework";
        record.algorithmId = std::move(algorithm);
        record.nativeRepresentationId = std::move(representation);
        record.modeOrderId = "logical-radial-retained-modes-keyed-by-k-l-j-field";
        record.schedulingId = "horizontal-outer-workers-" +
            std::to_string(options.fftwOuterWorkers) + "-internal-workers-" +
            std::to_string(options.fftwInternalWorkers == 0 ? 1 : options.fftwInternalWorkers) +
            ";" + verticalGemmSchedulingId(verticalStrategy) +
            ";modal-work-single-thread-auto-vectorized";
        record.sourceIdentity =
            "https://fftw.org/pub/fftw/fftw-3.3.11.tar.gz + Apple Accelerate system framework";
        record.sourceSha256 =
            "5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1";
        record.configureFlags =
            "FFTW --host=aarch64-apple-darwin --enable-neon --enable-threads; "
            "Accelerate system framework";
        record.compilerFlags = report.environment.compilerFlags;
        record.planningConfiguration = vertical.id +
            "; Float64; antialiased horizontal radial two-thirds retention; Nj=" +
            std::to_string(workload.retainedVerticalModes()) +
            "; synthetic real-diagonal-mode-keyed-v1 modal work; nonlinear flux excluded";
        record.workers =
            (options.fftwInternalWorkers == 0 ? 1 : options.fftwInternalWorkers) *
                options.fftwOuterWorkers +
            configuredThreads * verticalStrategy.outerWorkers;
        record.internalWorkers = options.fftwInternalWorkers == 0
            ? 1 : options.fftwInternalWorkers;
        record.outerWorkers = options.fftwOuterWorkers;
        record.opaqueProviderMemory = true;
        return record;
    };

    auto addCorrectness = [&](ProviderRecord& record,
                              const std::vector<Complex>& horizontal,
                              const std::vector<Complex>& forward,
                              const std::vector<Complex>& postWork,
                              const std::vector<Complex>& inverse,
                              const std::vector<double>& output) {
        const auto forwardProbes = gatherVerticalProbes(
            forward.data(), workload.retainedVerticalModes(), probes);
        const auto inverseProbes = gatherVerticalProbes(
            inverse.data(), workload.nz, probes);
        record.correctness = {
            metric("retained horizontal coefficients versus mode-keyed FFTW oracle",
                   horizontal.data(), referenceRetained.data(), horizontal.size()),
            metric("forward vertical projection probes versus independent scalar oracle",
                   forwardProbes.data(), forwardProbeOracle.data(), forwardProbeOracle.size()),
            metric("deterministic mode-keyed modal work versus canonical compact oracle",
                   postWork.data(), referencePostWorkModal.data(), postWork.size()),
            metric("inverse vertical projection probes versus independent scalar oracle",
                   inverseProbes.data(), inverseProbeOracle.data(), inverseProbeOracle.size()),
            metric("complete synthetic spectral pipeline versus independent canonical oracle",
                   maximumRelativeError(
                       output.data(), referencePipelineOutput.data(), output.size()))};
    };

    const auto modalWorkBytes = bytes(modalElements, sizeof(double)) +
        2 * report.modalSpectrumBytes;
    const auto fftwInternalWorkers = options.fftwInternalWorkers == 0
        ? 1 : options.fftwInternalWorkers;

    if (options.boundaryPolicy == "wvm-direct") {
        FFTWProvider fftw(workload, FFTWStrategy{
            fftwPlanningModeNamed(options.fftwPlanning),
            fftwAlignmentStrategyNamed(options.fftwAlignment),
            fftwWisdomStrategyNamed(options.fftwWisdom),
            fftwInternalWorkers, options.fftwOuterWorkers,
            options.fftwPlanningTimeLimitSeconds,
            FFTWDataLayout::interleaved,
            FFTWSpectrumOrder::wvmFrequencyMajor});
        WvmDirectVerticalGemmProvider provider(
            workload, modes, vertical, verticalStrategy);
        if (!provider.supported()) throw std::runtime_error(provider.capability());

        std::vector<Complex> fullSpectrum(workload.spectrumElements());
        std::vector<Complex> fullModalOutput(provider.modalSpectrumElements());
        std::vector<Complex> fullModalInput(provider.modalSpectrumElements());
        std::vector<double> output(workload.realElements());
        std::vector<Complex> horizontal(physicalElements);
        std::vector<Complex> forward(modalElements);
        std::vector<Complex> postWork(modalElements);
        std::vector<Complex> inverse(physicalElements);
        provider.initializeModalOutput(fullModalOutput.data());
        provider.initializeModalOutput(fullModalInput.data());
        vertical.forward = {};
        vertical.inverse = {};

        fftw.forward(input.data(), fullSpectrum.data());
        gatherRetained(workload, modes, fullSpectrum.data(), horizontal.data());
        provider.executeForward(fullSpectrum.data(), fullModalOutput.data());
        gatherRetainedModal(workload, modes, fullModalOutput.data(), forward.data());
        applySyntheticModalWorkWvm(
            workload, modes, modalWeights.data(), fullModalOutput.data(),
            fullModalInput.data());
        gatherRetainedModal(workload, modes, fullModalInput.data(), postWork.data());
        provider.initializeSpectrumOutput(fullSpectrum.data());
        provider.executeInverse(fullModalInput.data(), fullSpectrum.data());
        gatherRetained(workload, modes, fullSpectrum.data(), inverse.data());
        fftw.inverse(fullSpectrum.data(), output.data());

        auto record = makeRecord(
            "pipeline-wvm-direct",
            "full-wvm-order-fftw+direct-per-mode-zgemm+modal-diagonal+inverse-v1",
            "persistent-wvm-frequency-major-interleaved-full-spectrum-and-modal-view");
        record.version = fftw.version() + " + Apple Accelerate";
        record.gemmCallsPerExecution = 2 * provider.gemmCallsPerExecution();
        record.opaquePlanningBytes = fftw.planningBytes();
        record.otherSetupSeconds = fftw.otherSetupSeconds() +
            provider.matrixPreparationSeconds() + provider.schedulerSetupSeconds() +
            weightGenerationSeconds;
        record.allocationSeconds = fftw.allocationSeconds() +
            provider.allocationSeconds();
        record.planningSeconds = fftw.planningSeconds();
        record.wisdomGenerationSeconds = fftw.wisdomGenerationSeconds();
        record.wisdomImportSeconds = fftw.wisdomImportSeconds();
        record.planningTimeLimitSeconds = fftw.planningTimeLimitSeconds();
        record.planningBudgetExhausted = fftw.planningBudgetExhausted();
        record.wisdomBytes = fftw.wisdomBytes();
        record.explicitPersistentBytes = provider.persistentBytes() +
            bytes(modalWeights.size(), sizeof(double)) + report.fullSpectrumBytes +
            bytes(2 * provider.modalSpectrumElements(), sizeof(Complex)) +
            report.fullRealBytes;
        record.scratchBytes = 0;
        record.execution.forward = {
            "out-of-place", "out-of-place", false, true, false, false, false,
            "real-grid", "wvm-frequency-major-interleaved-modal-view-after-real-diagonal-work",
            "real-grid", "wvm-frequency-major-interleaved-modal-view-after-real-diagonal-work",
            "real grid -> full WVM half-spectrum -> direct zgemm -> modal work",
            "frequency-major; j contiguous within each field and retained frequency",
            provider.modalSpectrumElements() - modalElements, alignof(Complex),
            "FFT, forward modal, and post-work modal buffers do not overlap",
            bytes(2 * provider.modalSpectrumElements(), sizeof(Complex)), true};
        record.execution.inverse = record.execution.forward;
        record.execution.inverse.destroysNativeInput = false;
        record.execution.inverse.nativeInputRepresentationId =
            "wvm-frequency-major-interleaved-modal-view-after-real-diagonal-work";
        record.execution.inverse.nativeOutputRepresentationId = "real-grid";
        record.execution.inverse.adapterInputRepresentationId =
            record.execution.inverse.nativeInputRepresentationId;
        record.execution.inverse.adapterOutputRepresentationId = "real-grid";
        record.execution.inverse.physicalExtents =
            "post-work modal view -> direct zgemm -> rebuilt zero-padded WVM spectrum -> real grid";
        record.execution.inverse.paddingElements =
            workload.spectrumElements() - physicalElements;
        record.execution.inverse.reusableWorkBytes = report.fullSpectrumBytes;
        record.ledger = {
            {"setup/planning", StageState::setupOnly,
             "FFTW plans, immutable expanded complex matrices, modal weights, and persistent schedulers"},
            {"raw forward FFT", StageState::executed, "full WVM-order FFTW r2c"},
            {"horizontal retention", StageState::elided,
             "retained frequencies remain an indexed view of the full spectrum"},
            {"representation conversion", StageState::elided,
             "interleaved WVM order persists through modal work"},
            {"permutation/packing", StageState::elided,
             "no gather, transpose, radial ordering, or split conversion"},
            {"raw forward vertical MM", StageState::executed,
             "one small zgemm per retained frequency; fields share each call"},
            {"modal work", StageState::executed,
             "out-of-place real diagonal scaling over retained logical (k,l,j,field) values"},
            {"raw inverse vertical MM", StageState::executed,
             "one small zgemm per retained frequency; fields share each call"},
            {"horizontal embedding", StageState::executed,
             "full spectrum is zeroed every invocation before inverse vertical reconstruction"},
            {"raw inverse FFT", StageState::executed,
             "full WVM-order FFTW c2r destroys the rebuilt spectrum"},
            {"uninstrumented total", StageState::executed,
             "complete synthetic spectral operator; nonlinear flux excluded"}};
        addCorrectness(record, horizontal, forward, postWork, inverse, output);

        const auto matrixBytes = provider.matrixBytesPerDirection();
        const auto verticalBytes = matrixBytes +
            report.retainedSpectrumBytes + report.modalSpectrumBytes;
        record.timings = {
            series("setup-shared-component", "logical matrix-family fixture generation", "shared",
                   StageState::setupOnly, report.verticalMatrixFamilySourceBytes,
                   {fixtureGenerationSeconds}),
            series("setup-shared-component", "mode-keyed modal weight generation", "shared",
                   StageState::setupOnly, bytes(modalWeights.size(), sizeof(double)),
                   {weightGenerationSeconds}),
            series("setup-component", "FFTW planning", "shared", StageState::setupOnly,
                   fftw.planningBytes(), {fftw.planningSeconds()}),
            series("setup-component", "vertical matrix preparation", "shared",
                   StageState::setupOnly, 2 * matrixBytes,
                   {provider.matrixPreparationSeconds()}),
            series("primitive", "raw FFT", "forward", StageState::executed,
                   report.fullRealBytes + report.fullSpectrumBytes,
                   measure(warmups, sampleCount, [&] {
                       fftw.forward(input.data(), fullSpectrum.data());
                   })),
            series("primitive", "raw FFT", "inverse", StageState::executed,
                   report.fullSpectrumBytes + report.fullRealBytes,
                   measure(warmups, sampleCount,
                           [&] {
                               std::copy(referenceSpectrum.begin(), referenceSpectrum.end(),
                                         fullSpectrum.begin());
                           },
                           [&] { fftw.inverse(fullSpectrum.data(), output.data()); })),
            series("adapter-component", "logical retained provider-order view", "forward",
                   StageState::elided, 0),
            series("primitive", "raw vertical MM", "forward", StageState::executed,
                   verticalBytes,
                   measure(warmups, sampleCount,
                           [&] {
                               std::copy(referenceSpectrum.begin(), referenceSpectrum.end(),
                                         fullSpectrum.begin());
                           },
                           [&] {
                               provider.executeForward(
                                   fullSpectrum.data(), fullModalOutput.data());
                           })),
            series("component", "mode-keyed modal work", "modal", StageState::executed,
                   modalWorkBytes,
                   measure(warmups, sampleCount, [&] {
                       applySyntheticModalWorkWvm(
                           workload, modes, modalWeights.data(), fullModalOutput.data(),
                           fullModalInput.data());
                   })),
            series("primitive", "raw vertical MM", "inverse", StageState::executed,
                   verticalBytes,
                   measure(warmups, sampleCount, [&] {
                       provider.executeInverse(
                           fullModalInput.data(), fullSpectrum.data());
                   })),
            series("adapter-component", "rebuild zero-padded inverse spectrum", "inverse",
                   StageState::executed, report.fullSpectrumBytes,
                   measure(warmups, sampleCount, [&] {
                       provider.initializeSpectrumOutput(fullSpectrum.data());
                   })),
            series("uninstrumented-total", "synthetic antialiased spectral pipeline", "round-trip",
                   StageState::executed,
                   2 * report.fullRealBytes + 2 * report.fullSpectrumBytes +
                       2 * verticalBytes + modalWorkBytes + report.fullSpectrumBytes,
                   measure(warmups, sampleCount, [&] {
                       fftw.forward(input.data(), fullSpectrum.data());
                       provider.executeForward(
                           fullSpectrum.data(), fullModalOutput.data());
                       applySyntheticModalWorkWvm(
                           workload, modes, modalWeights.data(), fullModalOutput.data(),
                           fullModalInput.data());
                       provider.initializeSpectrumOutput(fullSpectrum.data());
                       provider.executeInverse(
                           fullModalInput.data(), fullSpectrum.data());
                       fftw.inverse(fullSpectrum.data(), output.data());
                   }))};
        report.spectralPipelineEstimatedExplicitPeakBytes =
            report.verticalMatrixFamilySourceBytes + record.explicitPersistentBytes +
            2 * report.fullSpectrumBytes + 2 * report.fullRealBytes +
            2 * report.retainedSpectrumBytes + 2 * report.modalSpectrumBytes;
        record.algorithmResidentBytes = record.explicitPersistentBytes +
            report.fullRealBytes + record.scratchBytes;
        record.estimatedProcessPeakBytes =
            report.spectralPipelineEstimatedExplicitPeakBytes;
        record.benchmarkHarnessBytes = record.estimatedProcessPeakBytes -
            record.algorithmResidentBytes;
        record.observedProcessHighWaterBytes = processHighWaterBytes();
        report.providers.push_back(std::move(record));
        report.status = correctnessPassed(report.providers.front()) ? "passed" : "failed";
        return report;
    }

    FFTWProvider fftw(workload, FFTWStrategy{
        fftwPlanningModeNamed(options.fftwPlanning),
        fftwAlignmentStrategyNamed(options.fftwAlignment),
        fftwWisdomStrategyNamed(options.fftwWisdom),
        fftwInternalWorkers, options.fftwOuterWorkers,
        options.fftwPlanningTimeLimitSeconds,
        FFTWDataLayout::interleaved,
        FFTWSpectrumOrder::planeMajor});
    VerticalGemmProvider provider(
        workload, vertical, VerticalGemmLayout::split, verticalStrategy);
    if (!provider.supported()) throw std::runtime_error(provider.capability());
    vertical.forward = {};
    vertical.inverse = {};

    std::vector<Complex> fullSpectrum(workload.spectrumElements());
    std::vector<Complex> referencePlaneMajor(workload.spectrumElements());
    std::vector<double> output(workload.realElements());
    std::vector<Complex> horizontal(physicalElements);
    std::vector<Complex> forward(modalElements);
    std::vector<Complex> postWork(modalElements);
    std::vector<Complex> inverse(physicalElements);
    wvmToPlaneMajor(workload, referenceSpectrum.data(), referencePlaneMajor.data());

    fftw.forward(input.data(), fullSpectrum.data());
    fftw.gatherRetainedToSplitOuter(
        modes, fullSpectrum.data(),
        provider.splitPhysicalInputRealData(),
        provider.splitPhysicalInputImaginaryData());
    splitToInterleaved(
        physicalElements, provider.splitPhysicalInputRealData(),
        provider.splitPhysicalInputImaginaryData(), horizontal.data());
    provider.executeForward();
    provider.copyForwardOutput(forward.data());
    applySyntheticModalWorkSplit(
        modalElements, modalWeights.data(),
        provider.splitModalOutputRealData(),
        provider.splitModalOutputImaginaryData(),
        provider.splitModalInputRealData(),
        provider.splitModalInputImaginaryData());
    splitToInterleaved(
        modalElements, provider.splitModalInputRealData(),
        provider.splitModalInputImaginaryData(), postWork.data());
    provider.executeInverse();
    provider.copyInverseOutput(inverse.data());
    fftw.embedRetainedFromSplitOuter(
        modes, provider.splitPhysicalOutputRealData(),
        provider.splitPhysicalOutputImaginaryData(), fullSpectrum.data());
    fftw.inverse(fullSpectrum.data(), output.data());

    auto record = makeRecord(
        "pipeline-plane-major-fused-split",
        "full-plane-major-fftw+fused-retained-split+split-k2-dgemm+modal-diagonal+inverse-v1",
        "plane-major-full-interleaved-to-persistent-radial-compact-split");
    record.version = fftw.version() + " + Apple Accelerate";
    record.gemmCallsPerExecution = 2 * provider.gemmCallsPerExecution();
    record.opaquePlanningBytes = fftw.planningBytes();
    record.otherSetupSeconds = fftw.otherSetupSeconds() +
        provider.matrixPreparationSeconds() + provider.schedulerSetupSeconds() +
        weightGenerationSeconds;
    record.allocationSeconds = fftw.allocationSeconds() + provider.allocationSeconds();
    record.planningSeconds = fftw.planningSeconds();
    record.wisdomGenerationSeconds = fftw.wisdomGenerationSeconds();
    record.wisdomImportSeconds = fftw.wisdomImportSeconds();
    record.planningTimeLimitSeconds = fftw.planningTimeLimitSeconds();
    record.planningBudgetExhausted = fftw.planningBudgetExhausted();
    record.wisdomBytes = fftw.wisdomBytes();
    record.explicitPersistentBytes = provider.persistentBytes() +
        bytes(modalWeights.size(), sizeof(double)) + report.fullSpectrumBytes +
        report.fullRealBytes;
    record.scratchBytes = 0;
    record.execution.forward = {
        "out-of-place", "out-of-place", false, true, false, false, false,
        "real-grid", "radial-compact-split-modal-after-real-diagonal-work",
        "real-grid", "radial-compact-split-modal-after-real-diagonal-work",
        "real grid -> full plane-major half-spectrum -> fused retained split -> grouped dgemm -> modal work",
        "plane-major full spectrum; z/j contiguous compact radial split columns",
        workload.spectrumElements() - physicalElements, 64,
        "FFT, split physical, split forward-modal, and split post-work buffers do not overlap",
        provider.persistentBytes(), true};
    record.execution.inverse = record.execution.forward;
    record.execution.inverse.destroysNativeInput = false;
    record.execution.inverse.nativeInputRepresentationId =
        "radial-compact-split-modal-after-real-diagonal-work";
    record.execution.inverse.nativeOutputRepresentationId = "real-grid";
    record.execution.inverse.adapterInputRepresentationId =
        record.execution.inverse.nativeInputRepresentationId;
    record.execution.inverse.adapterOutputRepresentationId = "real-grid";
    record.execution.inverse.physicalExtents =
        "post-work split modal -> grouped dgemm -> zero-padded plane-major spectrum -> real grid";
    record.execution.inverse.reusableWorkBytes = report.fullSpectrumBytes;
    record.ledger = {
        {"setup/planning", StageState::setupOnly,
         "FFTW plans, grouped real matrices, modal weights, split operands, and persistent schedulers"},
        {"raw forward FFT", StageState::executed, "full plane-major FFTW r2c"},
        {"horizontal retention", StageState::fused,
         "retained selection is fused with radial split conversion"},
        {"representation conversion", StageState::fused,
         "interleaved-to-split conversion is fused with retained selection"},
        {"permutation/packing", StageState::executed,
         "outer-sharded fused retained selection and split conversion"},
        {"raw forward vertical MM", StageState::executed,
         "two grouped dgemm calls per K-squared matrix group"},
        {"modal work", StageState::executed,
         "out-of-place real diagonal scaling over contiguous split (k,l,j,field) values"},
        {"raw inverse vertical MM", StageState::executed,
         "two grouped dgemm calls per K-squared matrix group"},
        {"horizontal embedding", StageState::executed,
         "outer-sharded fused split scatter into a zeroed plane-major spectrum"},
        {"raw inverse FFT", StageState::executed,
         "full plane-major FFTW c2r destroys the rebuilt spectrum"},
        {"uninstrumented total", StageState::executed,
         "complete synthetic spectral operator; nonlinear flux excluded"}};
    addCorrectness(record, horizontal, forward, postWork, inverse, output);

    const auto matrixBytes = provider.matrixBytesPerDirection();
    const auto verticalBytes = 2 * matrixBytes +
        report.retainedSpectrumBytes + report.modalSpectrumBytes;
    const auto packBytes = 2 * report.retainedSpectrumBytes;
    const auto embedBytes = report.fullSpectrumBytes +
        2 * report.retainedSpectrumBytes;
    record.timings = {
        series("setup-shared-component", "logical matrix-family fixture generation", "shared",
               StageState::setupOnly, report.verticalMatrixFamilySourceBytes,
               {fixtureGenerationSeconds}),
        series("setup-shared-component", "mode-keyed modal weight generation", "shared",
               StageState::setupOnly, bytes(modalWeights.size(), sizeof(double)),
               {weightGenerationSeconds}),
        series("setup-component", "FFTW planning", "shared", StageState::setupOnly,
               fftw.planningBytes(), {fftw.planningSeconds()}),
        series("setup-component", "vertical matrix preparation", "shared",
               StageState::setupOnly, 2 * matrixBytes,
               {provider.matrixPreparationSeconds()}),
        series("primitive", "raw FFT", "forward", StageState::executed,
               report.fullRealBytes + report.fullSpectrumBytes,
               measure(warmups, sampleCount, [&] {
                   fftw.forward(input.data(), fullSpectrum.data());
               })),
        series("primitive", "raw FFT", "inverse", StageState::executed,
               report.fullSpectrumBytes + report.fullRealBytes,
               measure(warmups, sampleCount,
                       [&] {
                           std::copy(referencePlaneMajor.begin(), referencePlaneMajor.end(),
                                     fullSpectrum.begin());
                       },
                       [&] { fftw.inverse(fullSpectrum.data(), output.data()); })),
        series("adapter-component", "fused retained selection and split pack", "forward",
               StageState::executed, packBytes,
               measure(warmups, sampleCount,
                       [&] {
                           std::copy(referencePlaneMajor.begin(), referencePlaneMajor.end(),
                                     fullSpectrum.begin());
                       },
                       [&] {
                           fftw.gatherRetainedToSplitOuter(
                               modes, fullSpectrum.data(),
                               provider.splitPhysicalInputRealData(),
                               provider.splitPhysicalInputImaginaryData());
                       })),
        series("primitive", "raw vertical MM", "forward", StageState::executed,
               verticalBytes,
               measure(warmups, sampleCount, [&] { provider.executeForward(); })),
        series("component", "mode-keyed modal work", "modal", StageState::executed,
               modalWorkBytes,
               measure(warmups, sampleCount, [&] {
                   applySyntheticModalWorkSplit(
                       modalElements, modalWeights.data(),
                       provider.splitModalOutputRealData(),
                       provider.splitModalOutputImaginaryData(),
                       provider.splitModalInputRealData(),
                       provider.splitModalInputImaginaryData());
               })),
        series("primitive", "raw vertical MM", "inverse", StageState::executed,
               verticalBytes,
               measure(warmups, sampleCount, [&] { provider.executeInverse(); })),
        series("adapter-component", "fused split embed into zeroed plane-major spectrum", "inverse",
               StageState::executed, embedBytes,
               measure(warmups, sampleCount, [&] {
                   fftw.embedRetainedFromSplitOuter(
                       modes, provider.splitPhysicalOutputRealData(),
                       provider.splitPhysicalOutputImaginaryData(),
                       fullSpectrum.data());
               })),
        series("uninstrumented-total", "synthetic antialiased spectral pipeline", "round-trip",
               StageState::executed,
               2 * report.fullRealBytes + 2 * report.fullSpectrumBytes +
                   packBytes + 2 * verticalBytes + modalWorkBytes + embedBytes,
               measure(warmups, sampleCount, [&] {
                   fftw.forward(input.data(), fullSpectrum.data());
                   fftw.gatherRetainedToSplitOuter(
                       modes, fullSpectrum.data(),
                       provider.splitPhysicalInputRealData(),
                       provider.splitPhysicalInputImaginaryData());
                   provider.executeForward();
                   applySyntheticModalWorkSplit(
                       modalElements, modalWeights.data(),
                       provider.splitModalOutputRealData(),
                       provider.splitModalOutputImaginaryData(),
                       provider.splitModalInputRealData(),
                       provider.splitModalInputImaginaryData());
                   provider.executeInverse();
                   fftw.embedRetainedFromSplitOuter(
                       modes, provider.splitPhysicalOutputRealData(),
                       provider.splitPhysicalOutputImaginaryData(),
                       fullSpectrum.data());
                   fftw.inverse(fullSpectrum.data(), output.data());
               }))};
    report.spectralPipelineEstimatedExplicitPeakBytes =
        report.verticalMatrixFamilySourceBytes + record.explicitPersistentBytes +
        2 * report.fullSpectrumBytes + 2 * report.fullRealBytes +
        2 * report.retainedSpectrumBytes + 2 * report.modalSpectrumBytes;
    record.algorithmResidentBytes = record.explicitPersistentBytes +
        report.fullRealBytes + record.scratchBytes;
    record.estimatedProcessPeakBytes =
        report.spectralPipelineEstimatedExplicitPeakBytes;
    record.benchmarkHarnessBytes = record.estimatedProcessPeakBytes -
        record.algorithmResidentBytes;
    record.observedProcessHighWaterBytes = processHighWaterBytes();
    report.providers.push_back(std::move(record));
    report.status = correctnessPassed(report.providers.front()) ? "passed" : "failed";
    return report;
}

ValidationReport validateBenchmark(std::string_view profileName) {
    const auto requested = profileNamed(profileName);
    Workload workload = requested.name == "smoke" ? requested.workload : Workload{8, 8, 7, 2, 1.0, 1.0, true};
    ValidationReport report;
    const auto modes = retainedHorizontalModes(workload);
    const auto vertical = orthonormalVerticalFixture(workload.nz, workload.retainedVerticalModes());
    const std::vector<FixtureKind> fixtures = {
        FixtureKind::impulse, FixtureKind::sinusoid, FixtureKind::random, FixtureKind::dc, FixtureKind::nyquist};

    FFTWProvider fftw(workload, 1);
    std::vector<Complex> oracle(workload.spectrumElements());
    std::vector<Complex> actual(workload.spectrumElements());
    std::vector<Complex> inverseInput(workload.spectrumElements());
    std::vector<Complex> planeMajor(workload.spectrumElements());
    std::vector<Complex> layoutRoundTrip(workload.spectrumElements());
    std::vector<Complex> retained(modes.size() * workload.planes());
    std::vector<Complex> embedded(workload.spectrumElements());
    std::vector<Complex> modal(modes.size() * workload.fields * vertical.nj);
    std::vector<Complex> modalAgain(modal.size());
    std::vector<Complex> verticalPhysical(retained.size());
    std::vector<double> output(workload.realElements());

    bool passed = true;
    for (std::size_t fixtureIndex = 0; fixtureIndex < fixtures.size(); ++fixtureIndex) {
        const auto fixture = fixtures[fixtureIndex];
        const auto input = makeFixture(workload, fixture, 129 + fixtureIndex);
        directR2C(workload, input.data(), oracle.data());

        fftw.forward(input.data(), actual.data());
        const auto fftwForwardError = maximumRelativeError(actual.data(), oracle.data(), oracle.size());
        passed = passed && fftwForwardError <= tolerance;
        report.messages.push_back("fftw/" + std::string(fixtureName(fixture)) + "/forward=" + std::to_string(fftwForwardError));

        inverseInput = oracle;
        fftw.inverse(inverseInput.data(), output.data());
        const auto fftwInverseError = maximumRelativeError(output.data(), input.data(), input.size(), 1.0 / static_cast<double>(workload.nx * workload.ny));
        passed = passed && fftwInverseError <= tolerance;
        report.messages.push_back("fftw/" + std::string(fixtureName(fixture)) + "/inverse=" + std::to_string(fftwInverseError));

        const std::vector<std::pair<VDSPTransformStrategy, VDSPBatchStrategy>> vdspCandidates = {
            {VDSPTransformStrategy::inPlace, VDSPBatchStrategy::directPersistent},
            {VDSPTransformStrategy::inPlaceExplicitScratch, VDSPBatchStrategy::directPersistent},
            {VDSPTransformStrategy::outOfPlace, VDSPBatchStrategy::directPersistent},
            {VDSPTransformStrategy::outOfPlaceExplicitScratch, VDSPBatchStrategy::directPersistent},
            {VDSPTransformStrategy::inPlace, VDSPBatchStrategy::directGcd},
            {VDSPTransformStrategy::inPlace, VDSPBatchStrategy::separablePersistent},
            {VDSPTransformStrategy::inPlace, VDSPBatchStrategy::separableGcd}};
        for (const auto [strategy, batchStrategy] : vdspCandidates) {
            VDSPProvider vdsp(workload, 1, strategy, batchStrategy);
            if (!vdsp.supported()) {
                passed = false;
                report.messages.push_back(vdsp.capability());
                continue;
            }
            vdsp.forwardAdapter(input.data(), actual.data());
            const auto vdspForwardError = maximumRelativeError(actual.data(), oracle.data(), oracle.size());
            passed = passed && vdspForwardError <= tolerance;
            const auto candidate = std::string(vdspTransformStrategyName(strategy)) + "/" +
                std::string(vdspBatchStrategyName(batchStrategy));
            report.messages.push_back("vdsp/" + candidate + "/" +
                                      std::string(fixtureName(fixture)) + "/forward=" + std::to_string(vdspForwardError));

            vdsp.inverseAdapter(oracle.data(), output.data());
            const auto vdspInverseError = maximumRelativeError(
                output.data(), input.data(), input.size(), 1.0 / static_cast<double>(workload.nx * workload.ny));
            passed = passed && vdspInverseError <= tolerance;
            report.messages.push_back("vdsp/" + candidate + "/" +
                                      std::string(fixtureName(fixture)) + "/inverse=" + std::to_string(vdspInverseError));
        }

        wvmToPlaneMajor(workload, oracle.data(), planeMajor.data());
        planeMajorToWvm(workload, planeMajor.data(), layoutRoundTrip.data());
        const auto layoutError = maximumRelativeError(layoutRoundTrip.data(), oracle.data(), oracle.size());
        passed = passed && layoutError == 0.0;

        const auto retainedOracle = directRetained(workload, modes, input.data());
        gatherRetained(workload, modes, oracle.data(), retained.data());
        const auto retainedError = maximumRelativeError(retained.data(), retainedOracle.data(), retained.size());
        passed = passed && retainedError <= tolerance;
        report.messages.push_back("operator/" + std::string(fixtureName(fixture)) + "/retained=" + std::to_string(retainedError));

        embedRetained(workload, modes, retained.data(), embedded.data());
        std::vector<Complex> gatheredAgain(retained.size());
        gatherRetained(workload, modes, embedded.data(), gatheredAgain.data());
        const auto permutationError = maximumRelativeError(gatheredAgain.data(), retained.data(), retained.size());
        passed = passed && permutationError <= tolerance;

        verticalForward(workload, modes.size(), vertical, retained.data(), modal.data());
        const auto combinedOracle = directHorizontalVerticalProjection(workload, modes, vertical, input.data());
        const auto combinedError = maximumRelativeError(modal.data(), combinedOracle.data(), modal.size());
        passed = passed && combinedError <= tolerance;
        report.messages.push_back("operator/" + std::string(fixtureName(fixture)) + "/horizontal-vertical=" + std::to_string(combinedError));

        verticalInverse(workload, modes.size(), vertical, modal.data(), verticalPhysical.data());
        verticalForward(workload, modes.size(), vertical, verticalPhysical.data(), modalAgain.data());
        const auto modalRoundTripError = maximumRelativeError(modalAgain.data(), modal.data(), modal.size());
        passed = passed && modalRoundTripError <= tolerance;
        report.messages.push_back("operator/" + std::string(fixtureName(fixture)) + "/vertical-modal-round-trip=" + std::to_string(modalRoundTripError));
    }

    report.passed = passed;
    report.messages.push_back("retained-mode-count=" + std::to_string(modes.size()));
    report.messages.push_back("retained-mode-order-hash=" + modeOrderHash(modes));
    report.messages.push_back(passed ? "validation=passed" : "validation=failed");
    return report;
}

BenchmarkReport runPrunedHorizontalBenchmark(const RunOptions& options) {
    if (options.providers != "both" && options.providers != "fftw") {
        throw std::invalid_argument("The pruned-horizontal kernel compares two FFTW algorithms; providers must be 'both' or 'fftw'.");
    }
    if (options.fftwLayout != "interleaved") {
        throw std::invalid_argument("The initial pruned-horizontal candidate requires --fftw-layout interleaved.");
    }
    if (options.retainedRepresentation != "interleaved" &&
        options.retainedRepresentation != "split") {
        throw std::invalid_argument(
            "The pruned-horizontal kernel supports retained-representation interleaved or split.");
    }
    const bool splitRetained = options.retainedRepresentation == "split";
    if (options.fftwAlignment != "unaligned") {
        throw std::invalid_argument("The initial pruned-horizontal candidate requires --fftw-alignment unaligned.");
    }
    if (options.fftwWisdom != "cold") {
        throw std::invalid_argument("The initial pruned-horizontal candidate requires --fftw-wisdom cold.");
    }
    if (options.fftwPlanningTimeLimitSeconds != 0.0) {
        throw std::invalid_argument("The initial pruned-horizontal candidate does not yet support a planning time limit.");
    }

    auto selected = profileNamed(options.profile);
    const auto planningMode = fftwPlanningModeNamed(options.fftwPlanning);
    const auto workers = options.workers == 0 ? selected.defaultWorkers : options.workers;
    const auto internalWorkers = options.fftwInternalWorkers == 0 ? workers : options.fftwInternalWorkers;
    const auto outerWorkers = options.fftwOuterWorkers;
    const auto warmups = options.warmups == 0 ? selected.warmups : options.warmups;
    const auto sampleCount = options.samples == 0 ? selected.samples : options.samples;

    BenchmarkReport report;
    report.profile = selected.name;
    report.seed = options.seed;
    report.warmups = warmups;
    report.samples = sampleCount;
    report.workload = selected.workload;
    report.environment = environmentRecord();
    report.runId = utcTimestamp(true) + "-" + report.environment.hostname;

    const auto& workload = report.workload;
    const auto modes = retainedHorizontalModes(workload);
    report.retainedHorizontalModeCount = modes.size();
    report.retainedModeOrderHash = modeOrderHash(modes);
    report.wvmFullSpectrumOrderHash = wvmSpectrumOrderHash(workload);
    report.fullRealBytes = bytes(workload.realElements(), sizeof(double));
    report.fullSpectrumBytes = bytes(workload.spectrumElements(), sizeof(Complex));
    report.retainedSpectrumBytes = bytes(modes.size() * workload.planes(), sizeof(Complex));
    report.modalSpectrumBytes = bytes(
        modes.size() * workload.fields * workload.retainedVerticalModes(), sizeof(Complex));

    const auto inputFixture = makeFixture(workload, FixtureKind::random, options.seed);
    FFTWArray<double> input(workload.realElements());
    std::copy(inputFixture.begin(), inputFixture.end(), input.begin());
    FFTWArray<Complex> referenceSpectrum(workload.spectrumElements());
    FFTWArray<Complex> fullWorkingSpectrum(workload.spectrumElements());
    FFTWArray<Complex> inverseFullSpectrum(workload.spectrumElements());
    FFTWArray<double> referenceProjectedOutput(workload.realElements());
    FFTWArray<double> fullOutput(workload.realElements());
    FFTWArray<double> prunedOutput(workload.realElements());
    std::vector<Complex> retainedReference(modes.size() * workload.planes());
    std::vector<Complex> retainedFull(modes.size() * workload.planes());
    std::vector<Complex> retainedPruned(modes.size() * workload.planes());
    std::vector<double> retainedReferenceReal(retainedReference.size());
    std::vector<double> retainedReferenceImag(retainedReference.size());
    std::vector<double> retainedPrunedReal(retainedReference.size());
    std::vector<double> retainedPrunedImag(retainedReference.size());

    FFTWProvider fixedReference(workload, FFTWStrategy{
        FFTWPlanningMode::estimate,
        FFTWAlignmentStrategy::unaligned,
        FFTWWisdomStrategy::cold,
        1,
        1,
        0.0,
        FFTWDataLayout::interleaved});
    fixedReference.forward(input.data(), referenceSpectrum.data());
    gatherRetained(workload, modes, referenceSpectrum.data(), retainedReference.data());
    interleavedToSplit(
        retainedReference.size(), retainedReference.data(),
        retainedReferenceReal.data(), retainedReferenceImag.data());
    embedRetained(workload, modes, retainedReference.data(), inverseFullSpectrum.data());
    fixedReference.inverse(inverseFullSpectrum.data(), referenceProjectedOutput.data());

    const FFTWStrategy fullStrategy{
        planningMode,
        FFTWAlignmentStrategy::unaligned,
        FFTWWisdomStrategy::cold,
        internalWorkers,
        outerWorkers,
        0.0,
        FFTWDataLayout::interleaved};
    FFTWProvider full(workload, fullStrategy);
    full.forward(input.data(), fullWorkingSpectrum.data());
    full.gatherRetainedOuter(modes, fullWorkingSpectrum.data(), retainedFull.data());
    full.embedRetainedOuter(modes, retainedReference.data(), inverseFullSpectrum.data());
    full.inverse(inverseFullSpectrum.data(), fullOutput.data());

    FFTWPrunedProvider pruned(
        workload, modes, planningMode, internalWorkers, outerWorkers);
    if (splitRetained) {
        pruned.forwardSplit(
            input.data(), retainedPrunedReal.data(), retainedPrunedImag.data());
        splitToInterleaved(
            retainedPruned.size(), retainedPrunedReal.data(),
            retainedPrunedImag.data(), retainedPruned.data());
        pruned.inverseSplit(
            retainedReferenceReal.data(), retainedReferenceImag.data(),
            prunedOutput.data());
    } else {
        pruned.forward(input.data(), retainedPruned.data());
        pruned.inverse(retainedReference.data(), prunedOutput.data());
    }

    ProviderRecord fullRecord;
    fullRecord.id = "fftw-full-2d-retained-reference";
    fullRecord.version = full.version();
    fullRecord.libraryIdentity = full.libraryIdentity();
    fullRecord.algorithmId = "full-2d-plus-radial-selection-" + fftwAlgorithmId(full);
    fullRecord.nativeRepresentationId = "wvm-frequency-major-interleaved-half-spectrum";
    fullRecord.modeOrderId = "full-r2c-kx-nonnegative-ky-wrapped";
    fullRecord.schedulingId = fftwSchedulingId(full);
    fullRecord.sourceIdentity = "https://fftw.org/pub/fftw/fftw-3.3.11.tar.gz";
    fullRecord.sourceSha256 = "5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1";
    fullRecord.configureFlags = "--host=aarch64-apple-darwin --enable-neon --enable-threads --disable-fortran --disable-openmp --enable-shared --disable-static";
    fullRecord.compilerFlags = "-O3 -mcpu=native -mmacosx-version-min=13.3";
    fullRecord.planningConfiguration = fftwPlanningConfiguration(full);
    fullRecord.workers = full.totalLogicalWorkers();
    fullRecord.internalWorkers = full.internalWorkers();
    fullRecord.outerWorkers = full.outerWorkers();
    fullRecord.execution = fftwExecutionContract(workload, full);
    fullRecord.otherSetupSeconds = full.otherSetupSeconds();
    fullRecord.allocationSeconds = full.allocationSeconds();
    fullRecord.planningSeconds = full.planningSeconds();
    fullRecord.opaquePlanningBytes = full.planningBytes();
    fullRecord.ledger = fftwLedger(full);
    fullRecord.correctness = {
        metric("full forward versus fixed FFTW ESTIMATE reference",
               fullWorkingSpectrum.data(), referenceSpectrum.data(), referenceSpectrum.size()),
        metric("retained forward versus mode-keyed full FFTW reference",
               retainedFull.data(), retainedReference.data(), retainedReference.size()),
        metric("retained inverse versus fixed FFTW ESTIMATE reference",
               maximumRelativeError(fullOutput.data(), referenceProjectedOutput.data(),
                                    fullOutput.size()))};
    fullRecord.timings.push_back(series(
        "primitive", "raw full 2-D FFT", "forward", StageState::executed,
        report.fullRealBytes + report.fullSpectrumBytes,
        measure(warmups, sampleCount, [&] {
            full.forward(input.data(), fullWorkingSpectrum.data());
        })));
    fullRecord.timings.push_back(series(
        "primitive", "raw full 2-D FFT", "inverse", StageState::executed,
        report.fullSpectrumBytes + report.fullRealBytes,
        measure(warmups, sampleCount,
                [&] {
                    std::copy(referenceSpectrum.begin(), referenceSpectrum.end(),
                              inverseFullSpectrum.begin());
                },
                [&] { full.inverse(inverseFullSpectrum.data(), fullOutput.data()); })));
    fullRecord.timings.push_back(series(
        "operator-component", "radial retention", "forward", StageState::executed,
        report.fullSpectrumBytes + report.retainedSpectrumBytes,
        measure(warmups, sampleCount, [&] {
            full.gatherRetainedOuter(
                modes, referenceSpectrum.data(), retainedFull.data());
        })));
    fullRecord.timings.push_back(series(
        "operator-component", "radial embedding and full-spectrum zero fill", "inverse",
        StageState::executed, report.retainedSpectrumBytes + report.fullSpectrumBytes,
        measure(warmups, sampleCount, [&] {
            full.embedRetainedOuter(
                modes, retainedReference.data(), inverseFullSpectrum.data());
        })));
    fullRecord.timings.push_back(series(
        "diagnostic-component", "batch scheduler empty dispatch", "shared",
        full.outerWorkers() > 1 ? StageState::executed : StageState::elided, 0,
        full.outerWorkers() > 1
            ? measure(warmups, sampleCount, [&] { full.executeSchedulerNoop(); })
            : std::vector<double>{}));
    fullRecord.timings.push_back(series(
        "uninstrumented-total", "retained horizontal operator", "forward", StageState::executed,
        report.fullRealBytes + report.fullSpectrumBytes + report.retainedSpectrumBytes,
        measure(warmups, sampleCount, [&] {
            full.forward(input.data(), fullWorkingSpectrum.data());
            full.gatherRetainedOuter(
                modes, fullWorkingSpectrum.data(), retainedFull.data());
        })));
    fullRecord.timings.push_back(series(
        "uninstrumented-total", "retained horizontal operator", "inverse", StageState::executed,
        report.retainedSpectrumBytes + report.fullSpectrumBytes + report.fullRealBytes,
        measure(warmups, sampleCount, [&] {
            full.embedRetainedOuter(
                modes, retainedReference.data(), inverseFullSpectrum.data());
            full.inverse(inverseFullSpectrum.data(), fullOutput.data());
        })));
    report.providers.push_back(std::move(fullRecord));

    ProviderRecord prunedRecord;
    prunedRecord.id = splitRetained
        ? "fftw-partial-column-pruned-fused-split"
        : "fftw-partial-column-pruned";
    prunedRecord.version = pruned.version();
    prunedRecord.libraryIdentity = pruned.libraryIdentity();
    prunedRecord.algorithmId = splitRetained
        ? "separable-row-r2c-selected-kx-column-c2c-fused-retained-split-v1"
        : "separable-row-r2c-selected-kx-column-c2c-v1";
    prunedRecord.nativeRepresentationId =
        splitRetained
            ? "plane-major-full-row-spectrum-scratch+logical-radial-retained-split-complex"
            : "plane-major-full-row-spectrum-scratch+logical-radial-retained-interleaved";
    prunedRecord.modeOrderId = "logical-radial-retained-mode-order";
    prunedRecord.schedulingId = prunedFftwSchedulingId(pruned);
    prunedRecord.sourceIdentity = "https://fftw.org/pub/fftw/fftw-3.3.11.tar.gz";
    prunedRecord.sourceSha256 = "5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1";
    prunedRecord.configureFlags = fullStrategy.layout == FFTWDataLayout::interleaved
        ? "--host=aarch64-apple-darwin --enable-neon --enable-threads --disable-fortran --disable-openmp --enable-shared --disable-static"
        : "unsupported";
    prunedRecord.compilerFlags = "-O3 -mcpu=native -mmacosx-version-min=13.3";
    prunedRecord.planningConfiguration =
        "FFTW_" + std::string(fftwPlanningModeName(pruned.planningMode())) +
        "|FFTW_UNALIGNED; guru64 1-D row r2c/c2r plus in-place selected-kx column c2c; active-kx=" +
        std::to_string(pruned.activeKxCount()) + "/" + std::to_string(pruned.fullKxCount()) +
        "; internal-workers=" + std::to_string(pruned.internalWorkers()) +
        "; outer-workers=" + std::to_string(pruned.outerWorkers()) +
        "; maximum-shard-scratch-bytes=" +
        std::to_string(pruned.maximumShardScratchBytes());
    prunedRecord.workers = pruned.totalLogicalWorkers();
    prunedRecord.internalWorkers = pruned.internalWorkers();
    prunedRecord.outerWorkers = pruned.outerWorkers();
    prunedRecord.execution = prunedFftwExecutionContract(
        workload, pruned, splitRetained);
    prunedRecord.explicitPersistentBytes = 0;
    prunedRecord.scratchBytes = pruned.scratchBytes();
    prunedRecord.otherSetupSeconds = pruned.otherSetupSeconds();
    prunedRecord.allocationSeconds = pruned.allocationSeconds();
    prunedRecord.planningSeconds = pruned.planningSeconds();
    prunedRecord.opaquePlanningBytes = pruned.planningBytes();
    prunedRecord.ledger = prunedFftwLedger(pruned, splitRetained);
    prunedRecord.correctness = {
        metric("retained forward versus mode-keyed full FFTW reference",
               retainedPruned.data(), retainedReference.data(), retainedReference.size()),
        metric("retained inverse versus full FFTW embed and inverse",
               maximumRelativeError(prunedOutput.data(), referenceProjectedOutput.data(),
                                    prunedOutput.size()))};

    const auto activeColumnBytes = bytes(
        pruned.activeKxCount() * workload.ny * workload.planes(), sizeof(Complex));
    prunedRecord.timings.push_back(series(
        "primitive-component", "real row FFTs", "forward", StageState::executed,
        report.fullRealBytes + report.fullSpectrumBytes,
        measure(warmups, sampleCount, [&] { pruned.executeForwardRows(input.data()); })));
    prunedRecord.timings.push_back(series(
        "primitive-component", "selected-kx complex column FFTs", "forward",
        StageState::executed, 2 * activeColumnBytes,
        measure(warmups, sampleCount,
                [&] { pruned.executeForwardRows(input.data()); },
                [&] { pruned.executeForwardColumns(); })));
    prunedRecord.timings.push_back(series(
        "operator-component",
        splitRetained
            ? "fused radial retention and split conversion from plane-major scratch"
            : "direct radial retention from plane-major scratch",
        "forward",
        StageState::executed, 2 * report.retainedSpectrumBytes,
        measure(warmups, sampleCount,
                [&] {
                    pruned.executeForwardRows(input.data());
                    pruned.executeForwardColumns();
                },
                [&] {
                    if (splitRetained) {
                        pruned.gatherForwardSplit(
                            retainedPrunedReal.data(), retainedPrunedImag.data());
                    } else {
                        pruned.gatherForward(retainedPruned.data());
                    }
                })));
    prunedRecord.timings.push_back(series(
        "operator-component",
        splitRetained
            ? "fused split conversion, radial embedding, and row-spectrum zero fill"
            : "radial embedding and row-spectrum zero fill",
        "inverse",
        StageState::executed, report.retainedSpectrumBytes + report.fullSpectrumBytes,
        measure(warmups, sampleCount, [&] {
            if (splitRetained) {
                pruned.embedInverseSplit(
                    retainedReferenceReal.data(), retainedReferenceImag.data());
            } else {
                pruned.embedInverse(retainedReference.data());
            }
        })));
    prunedRecord.timings.push_back(series(
        "primitive-component", "selected-kx complex column FFTs", "inverse",
        StageState::executed, 2 * activeColumnBytes,
        measure(warmups, sampleCount,
                [&] {
                    if (splitRetained) {
                        pruned.embedInverseSplit(
                            retainedReferenceReal.data(), retainedReferenceImag.data());
                    } else {
                        pruned.embedInverse(retainedReference.data());
                    }
                },
                [&] { pruned.executeInverseColumns(); })));
    prunedRecord.timings.push_back(series(
        "primitive-component", "real row FFTs", "inverse", StageState::executed,
        report.fullSpectrumBytes + report.fullRealBytes,
        measure(warmups, sampleCount,
                [&] {
                    if (splitRetained) {
                        pruned.embedInverseSplit(
                            retainedReferenceReal.data(), retainedReferenceImag.data());
                    } else {
                        pruned.embedInverse(retainedReference.data());
                    }
                    pruned.executeInverseColumns();
                },
                [&] { pruned.executeInverseRows(prunedOutput.data()); })));
    prunedRecord.timings.push_back(series(
        "algorithm-component", "omitted high-kx complex column FFTs", "forward",
        StageState::elided, 0));
    prunedRecord.timings.push_back(series(
        "algorithm-component", "omitted high-kx complex column FFTs", "inverse",
        StageState::elided, 0));
    prunedRecord.timings.push_back(series(
        "diagnostic-component", "batch scheduler empty dispatch", "shared",
        pruned.outerWorkers() > 1 ? StageState::executed : StageState::elided, 0,
        pruned.outerWorkers() > 1
            ? measure(warmups, sampleCount, [&] { pruned.executeSchedulerNoop(); })
            : std::vector<double>{}));
    prunedRecord.timings.push_back(series(
        "uninstrumented-total", "partial-column-pruned retained horizontal operator", "forward",
        StageState::executed,
        report.fullRealBytes + report.fullSpectrumBytes + 2 * activeColumnBytes +
            2 * report.retainedSpectrumBytes,
        measure(warmups, sampleCount, [&] {
            if (splitRetained) {
                pruned.forwardSplit(
                    input.data(), retainedPrunedReal.data(), retainedPrunedImag.data());
            } else {
                pruned.forward(input.data(), retainedPruned.data());
            }
        })));
    prunedRecord.timings.push_back(series(
        "uninstrumented-total", "partial-column-pruned retained horizontal operator", "inverse",
        StageState::executed,
        report.retainedSpectrumBytes + 2 * report.fullSpectrumBytes +
            2 * activeColumnBytes + report.fullRealBytes,
        measure(warmups, sampleCount, [&] {
            if (splitRetained) {
                pruned.inverseSplit(
                    retainedReferenceReal.data(), retainedReferenceImag.data(),
                    prunedOutput.data());
            } else {
                pruned.inverse(retainedReference.data(), prunedOutput.data());
            }
        })));
    prunedRecord.timings.push_back(series(
        "capability", "complete transformed half-spectrum output", "shared",
        StageState::elided, 0));
    prunedRecord.timings.push_back(series(
        "capability", "in-place retained operator", "shared", StageState::unsupported, 0));
    report.providers.push_back(std::move(prunedRecord));

    report.status = std::all_of(report.providers.begin(), report.providers.end(), correctnessPassed)
        ? "passed" : "failed";
    return report;
}

BenchmarkReport runBenchmark(const RunOptions& options) {
    if (options.kernel == "pruned-horizontal") return runPrunedHorizontalBenchmark(options);
    if (options.kernel == "vertical-gemm") return runVerticalGemmBenchmark(options);
    if (options.kernel == "ordering-packing") return runOrderingPackingBenchmark(options);
    if (options.kernel == "spectral-boundary") return runSpectralBoundaryBenchmark(options);
    if (options.kernel == "spectral-pipeline") return runSpectralPipelineBenchmark(options);
    if (options.kernel != "fft") {
        throw std::invalid_argument(
            "kernel must be 'fft', 'pruned-horizontal', 'vertical-gemm', "
            "'ordering-packing', 'spectral-boundary', or 'spectral-pipeline'.");
    }
    auto selected = profileNamed(options.profile);
    if (options.providers != "both" && options.providers != "fftw") {
        throw std::invalid_argument("providers must be either 'both' or 'fftw'.");
    }
    if (options.retainedRepresentation != "interleaved" &&
        options.retainedRepresentation != "split" &&
        options.retainedRepresentation != "view") {
        throw std::invalid_argument(
            "retained-representation must be 'interleaved', 'split', or 'view'.");
    }
    if (options.fftwLayout != "interleaved" && options.fftwLayout != "split" && options.fftwLayout != "paired") {
        throw std::invalid_argument("fftw-layout must be 'interleaved', 'split', or 'paired'.");
    }
    if (options.providers == "both" && options.fftwLayout != "interleaved") {
        throw std::invalid_argument("fftw-layout split/paired is currently an FFTW-only experiment; use --providers fftw.");
    }
    const auto fftwSpectrumOrder = fftwSpectrumOrderNamed(options.fftwSpectrumOrder);
    if (options.providers == "both" && fftwSpectrumOrder != FFTWSpectrumOrder::wvmFrequencyMajor) {
        throw std::invalid_argument(
            "plane-major FFTW output is currently an FFTW-only experiment; use --providers fftw.");
    }
    const bool fusedSplitRetained = options.retainedRepresentation == "split";
    const bool retainedView = options.retainedRepresentation == "view";
    if ((fusedSplitRetained || retainedView) &&
        (options.providers != "fftw" || options.fftwLayout != "interleaved" ||
         fftwSpectrumOrder != FFTWSpectrumOrder::planeMajor)) {
        throw std::invalid_argument(
            "retained-representation split/view requires --providers fftw "
            "--fftw-layout interleaved --fftw-spectrum-order plane-major.");
    }
    const auto fftwPlanningMode = fftwPlanningModeNamed(options.fftwPlanning);
    const auto fftwAlignment = fftwAlignmentStrategyNamed(options.fftwAlignment);
    const auto fftwWisdom = fftwWisdomStrategyNamed(options.fftwWisdom);
    const auto vdspStrategy = vdspTransformStrategyNamed(options.vdspStrategy);
    const auto vdspBatchStrategy = vdspBatchStrategyNamed(options.vdspBatchStrategy);
    const auto workers = options.workers == 0 ? selected.defaultWorkers : options.workers;
    const auto fftwInternalWorkers = options.fftwInternalWorkers == 0 ? workers : options.fftwInternalWorkers;
    const FFTWStrategy fftwStrategy{
        fftwPlanningMode,
        fftwAlignment,
        fftwWisdom,
        fftwInternalWorkers,
        options.fftwOuterWorkers,
        options.fftwPlanningTimeLimitSeconds,
        FFTWDataLayout::interleaved,
        fftwSpectrumOrder};
    const auto warmups = options.warmups == 0 ? selected.warmups : options.warmups;
    const auto sampleCount = options.samples == 0 ? selected.samples : options.samples;
    selected.defaultWorkers = workers;
    selected.warmups = warmups;
    selected.samples = sampleCount;

    BenchmarkReport report;
    report.profile = selected.name;
    report.seed = options.seed;
    report.warmups = warmups;
    report.samples = sampleCount;
    report.workload = selected.workload;
    report.environment = environmentRecord();
    report.runId = utcTimestamp(true) + "-" + report.environment.hostname;

    const auto& workload = report.workload;
    const auto modes = retainedHorizontalModes(workload);
    report.retainedHorizontalModeCount = modes.size();
    report.retainedModeOrderHash = modeOrderHash(modes);
    report.wvmFullSpectrumOrderHash = wvmSpectrumOrderHash(workload);
    report.fullRealBytes = bytes(workload.realElements(), sizeof(double));
    report.fullSpectrumBytes = bytes(workload.spectrumElements(), sizeof(Complex));
    report.retainedSpectrumBytes = bytes(modes.size() * workload.planes(), sizeof(Complex));
    report.modalSpectrumBytes = bytes(modes.size() * workload.fields * workload.retainedVerticalModes(), sizeof(Complex));

    const auto inputFixture = makeFixture(workload, FixtureKind::random, options.seed);
    FFTWArray<double> input(workload.realElements());
    std::copy(inputFixture.begin(), inputFixture.end(), input.begin());
    FFTWArray<Complex> referenceSpectrum(workload.spectrumElements());
    FFTWArray<Complex> nativeReferenceSpectrum(workload.spectrumElements());
    FFTWArray<Complex> nativeWorkingSpectrum(workload.spectrumElements());
    FFTWArray<Complex> nativeInverseSpectrum(workload.spectrumElements());
    FFTWArray<Complex> nativeRetainedInverseReady(workload.spectrumElements());
    FFTWArray<Complex> workingSpectrum(workload.spectrumElements());
    FFTWArray<Complex> inverseSpectrum(workload.spectrumElements());
    std::vector<Complex> retainedSpectrum(modes.size() * workload.planes());
    std::vector<Complex> retainedWorking(modes.size() * workload.planes());
    std::vector<double> retainedReferenceReal(retainedSpectrum.size());
    std::vector<double> retainedReferenceImag(retainedSpectrum.size());
    std::vector<double> retainedWorkingReal(retainedSpectrum.size());
    std::vector<double> retainedWorkingImag(retainedSpectrum.size());
    std::vector<double> retainedNormalizedReal(retainedSpectrum.size());
    std::vector<double> retainedNormalizedImag(retainedSpectrum.size());
    std::vector<double> retainedSeparatelyNormalizedReal(retainedSpectrum.size());
    std::vector<double> retainedSeparatelyNormalizedImag(retainedSpectrum.size());
    FFTWArray<double> referenceOutput(workload.realElements());
    FFTWArray<double> fftwOutput(workload.realElements());
    FFTWArray<double> output(workload.realElements());

    FFTWProvider referenceFftw(workload, FFTWStrategy{
        FFTWPlanningMode::estimate,
        FFTWAlignmentStrategy::unaligned,
        FFTWWisdomStrategy::cold,
        1,
        1,
        0.0});
    referenceFftw.forward(input.data(), referenceSpectrum.data());
    std::copy(referenceSpectrum.begin(), referenceSpectrum.end(), inverseSpectrum.begin());
    referenceFftw.inverse(inverseSpectrum.data(), referenceOutput.data());
    gatherRetained(workload, modes, referenceSpectrum.data(), retainedSpectrum.data());
    interleavedToSplit(
        retainedSpectrum.size(), retainedSpectrum.data(),
        retainedReferenceReal.data(), retainedReferenceImag.data());

    const auto nativeToWvm = [&](const Complex* native, Complex* wvm) {
        if (fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor) {
            planeMajorToWvm(workload, native, wvm);
        } else {
            std::copy_n(native, workload.spectrumElements(), wvm);
        }
    };
    const auto wvmToNative = [&](const Complex* wvm, Complex* native) {
        if (fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor) {
            wvmToPlaneMajor(workload, wvm, native);
        } else {
            std::copy_n(wvm, workload.spectrumElements(), native);
        }
    };
    wvmToNative(referenceSpectrum.data(), nativeReferenceSpectrum.data());

    FFTWProvider fftw(workload, fftwStrategy);
    fftw.forward(input.data(), nativeWorkingSpectrum.data());
    nativeToWvm(nativeWorkingSpectrum.data(), workingSpectrum.data());
    const auto fftwForwardReferenceError = maximumRelativeError(
        workingSpectrum.data(), referenceSpectrum.data(), referenceSpectrum.size());
    std::copy(nativeReferenceSpectrum.begin(), nativeReferenceSpectrum.end(), nativeInverseSpectrum.begin());
    fftw.inverse(nativeInverseSpectrum.data(), fftwOutput.data());
    const auto fftwInverseReferenceError = maximumRelativeError(
        fftwOutput.data(), referenceOutput.data(), referenceOutput.size());
    const auto fftwRoundTripError = maximumRelativeError(
        fftwOutput.data(), input.data(), input.size(), 1.0 / static_cast<double>(workload.nx * workload.ny));

    fftw.gatherRetainedOuter(modes, nativeWorkingSpectrum.data(), retainedWorking.data());
    const auto fftwRetainedForwardError = maximumRelativeError(
        retainedWorking.data(), retainedSpectrum.data(), retainedSpectrum.size());
    fftw.embedRetainedOuter(modes, retainedSpectrum.data(), nativeInverseSpectrum.data());
    std::copy(
        nativeInverseSpectrum.begin(), nativeInverseSpectrum.end(),
        nativeRetainedInverseReady.begin());
    fftw.inverse(nativeInverseSpectrum.data(), output.data());
    embedRetained(workload, modes, retainedSpectrum.data(), inverseSpectrum.data());
    referenceFftw.inverse(inverseSpectrum.data(), referenceOutput.data());
    const auto fftwRetainedInverseError = maximumRelativeError(
        output.data(), referenceOutput.data(), referenceOutput.size());

    double fusedSplitForwardError = 0.0;
    double fusedSplitInverseError = 0.0;
    double fusedNormalizationError = 0.0;
    double separateNormalizationError = 0.0;
    if (fusedSplitRetained) {
        fftw.gatherRetainedToSplitOuter(
            modes, nativeWorkingSpectrum.data(),
            retainedWorkingReal.data(), retainedWorkingImag.data());
        splitToInterleaved(
            retainedSpectrum.size(), retainedWorkingReal.data(),
            retainedWorkingImag.data(), retainedWorking.data());
        fusedSplitForwardError = maximumRelativeError(
            retainedWorking.data(), retainedSpectrum.data(), retainedSpectrum.size());

        fftw.embedRetainedFromSplitOuter(
            modes, retainedReferenceReal.data(), retainedReferenceImag.data(),
            nativeInverseSpectrum.data());
        fftw.inverse(nativeInverseSpectrum.data(), output.data());
        fusedSplitInverseError = maximumRelativeError(
            output.data(), referenceOutput.data(), referenceOutput.size());

        const auto normalization =
            1.0 / static_cast<double>(workload.nx * workload.ny);
        fftw.gatherRetainedToSplitOuter(
            modes, nativeReferenceSpectrum.data(),
            retainedNormalizedReal.data(), retainedNormalizedImag.data(),
            normalization);
        for (std::size_t index = 0; index < retainedSpectrum.size(); ++index) {
            retainedWorkingReal[index] = retainedReferenceReal[index] * normalization;
            retainedWorkingImag[index] = retainedReferenceImag[index] * normalization;
        }
        fusedNormalizationError = std::max(
            maximumRelativeError(
                retainedNormalizedReal.data(), retainedWorkingReal.data(),
                retainedSpectrum.size()),
            maximumRelativeError(
                retainedNormalizedImag.data(), retainedWorkingImag.data(),
                retainedSpectrum.size()));

        fftw.gatherRetainedToSplitOuter(
            modes, nativeReferenceSpectrum.data(),
            retainedSeparatelyNormalizedReal.data(),
            retainedSeparatelyNormalizedImag.data());
        fftw.scaleRetainedSplitOuter(
            modes, retainedSeparatelyNormalizedReal.data(),
            retainedSeparatelyNormalizedImag.data(), normalization);
        separateNormalizationError = std::max(
            maximumRelativeError(
                retainedSeparatelyNormalizedReal.data(),
                retainedNormalizedReal.data(), retainedSpectrum.size()),
            maximumRelativeError(
                retainedSeparatelyNormalizedImag.data(),
                retainedNormalizedImag.data(), retainedSpectrum.size()));
    }

    ProviderRecord fftwRecord;
    fftwRecord.id = fusedSplitRetained
        ? "fftw-plane-major-fused-retained-split"
        : (retainedView ? "fftw-plane-major-retained-view" : "fftw");
    fftwRecord.version = fftw.version();
    fftwRecord.libraryIdentity = fftw.libraryIdentity();
    fftwRecord.algorithmId = fftwAlgorithmId(fftw) +
        (fusedSplitRetained ? "-fused-radial-retained-split-v1"
                           : (retainedView ? "-persistent-retained-index-view-v1" : ""));
    fftwRecord.nativeRepresentationId = fusedSplitRetained
        ? "plane-major-interleaved-half-spectrum-scratch+logical-radial-retained-split-complex"
        : (retainedView
            ? "plane-major-interleaved-half-spectrum+logical-retained-index-view"
            : (fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor
                ? "plane-major-interleaved-half-spectrum"
                : "wvm-frequency-major-interleaved-half-spectrum"));
    fftwRecord.modeOrderId = "full-r2c-kx-nonnegative-ky-wrapped";
    fftwRecord.schedulingId = fftwSchedulingId(fftw);
    fftwRecord.sourceIdentity = "https://fftw.org/pub/fftw/fftw-3.3.11.tar.gz";
    fftwRecord.sourceSha256 = "5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1";
    fftwRecord.configureFlags = "--host=aarch64-apple-darwin --enable-neon --enable-threads --disable-fortran --disable-openmp --enable-shared --disable-static";
    fftwRecord.compilerFlags = "-O3 -mcpu=native -mmacosx-version-min=13.3";
    fftwRecord.planningConfiguration = fftwPlanningConfiguration(fftw);
    fftwRecord.workers = fftw.totalLogicalWorkers();
    fftwRecord.internalWorkers = fftw.internalWorkers();
    fftwRecord.outerWorkers = fftw.outerWorkers();
    fftwRecord.execution = fusedSplitRetained
        ? fftwFusedSplitRetainedExecutionContract(workload, fftw, modes.size())
        : (retainedView
            ? fftwRetainedViewExecutionContract(workload, fftw, modes.size())
            : fftwExecutionContract(workload, fftw));
    fftwRecord.otherSetupSeconds = fftw.otherSetupSeconds();
    fftwRecord.allocationSeconds = fftw.allocationSeconds();
    fftwRecord.planningSeconds = fftw.planningSeconds();
    fftwRecord.wisdomGenerationSeconds = fftw.wisdomGenerationSeconds();
    fftwRecord.wisdomImportSeconds = fftw.wisdomImportSeconds();
    fftwRecord.planningTimeLimitSeconds = fftw.planningTimeLimitSeconds();
    fftwRecord.planningBudgetExhausted = fftw.planningBudgetExhausted();
    fftwRecord.wisdomBytes = fftw.wisdomBytes();
    fftwRecord.opaquePlanningBytes = fftw.planningBytes();
    fftwRecord.ledger = fusedSplitRetained
        ? fftwFusedSplitRetainedLedger(fftw)
        : (retainedView ? fftwRetainedViewLedger(fftw) : fftwLedger(fftw));
    fftwRecord.correctness = {
        metric("full forward versus fixed FFTW ESTIMATE reference", fftwForwardReferenceError),
        metric("full inverse versus fixed FFTW ESTIMATE reference", fftwInverseReferenceError),
        metric("full inverse round trip", fftwRoundTripError),
        metric("native-order retained forward versus mode-keyed oracle", fftwRetainedForwardError),
        metric("native-order retained inverse versus mode-keyed oracle", fftwRetainedInverseError)};
    if (fusedSplitRetained) {
        fftwRecord.correctness.push_back(metric(
            "fused compact split retained forward versus mode-keyed oracle",
            fusedSplitForwardError));
        fftwRecord.correctness.push_back(metric(
            "fused compact split retained inverse versus mode-keyed oracle",
            fusedSplitInverseError));
        fftwRecord.correctness.push_back(metric(
            "fused horizontal normalization versus scaled oracle",
            fusedNormalizationError));
        fftwRecord.correctness.push_back(metric(
            "separate normalization versus fused normalization",
            separateNormalizationError));
    }

    fftwRecord.timings.push_back(series("primitive", "raw FFT", "forward", StageState::executed,
        report.fullRealBytes + report.fullSpectrumBytes,
        measure(warmups, sampleCount, [&] { fftw.forward(input.data(), nativeWorkingSpectrum.data()); })));
    fftwRecord.timings.push_back(series("primitive", "raw FFT", "inverse", StageState::executed,
        report.fullSpectrumBytes + report.fullRealBytes,
        measure(warmups, sampleCount,
            [&] {
                std::copy(nativeReferenceSpectrum.begin(), nativeReferenceSpectrum.end(),
                          nativeInverseSpectrum.begin());
            },
            [&] { fftw.inverse(nativeInverseSpectrum.data(), output.data()); })));
    fftwRecord.timings.push_back(series(
        "diagnostic-component", "batch scheduler empty dispatch", "shared",
        fftw.outerWorkers() > 1 ? StageState::executed : StageState::elided, 0,
        fftw.outerWorkers() > 1
            ? measure(warmups, sampleCount, [&] { fftw.executeSchedulerNoop(); })
            : std::vector<double>{}));
    const auto orderConversionState = fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor
        ? StageState::executed : StageState::elided;
    fftwRecord.timings.push_back(series(
        "adapter-component", "spectrum-order permutation", "forward", orderConversionState,
        fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor ? 2 * report.fullSpectrumBytes : 0,
        fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor
            ? measure(warmups, sampleCount, [&] {
                planeMajorToWvm(
                    workload, nativeReferenceSpectrum.data(), workingSpectrum.data());
            })
            : std::vector<double>{}));
    fftwRecord.timings.push_back(series(
        "adapter-component", "spectrum-order permutation", "inverse", orderConversionState,
        fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor ? 2 * report.fullSpectrumBytes : 0,
        fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor
            ? measure(warmups, sampleCount, [&] {
                wvmToPlaneMajor(
                    workload, referenceSpectrum.data(), nativeInverseSpectrum.data());
            })
            : std::vector<double>{}));
    fftwRecord.timings.push_back(series("adapter-total", "WVM-compatible full-spectrum adapter", "forward", StageState::executed,
        report.fullRealBytes + report.fullSpectrumBytes +
            (fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor ? 2 * report.fullSpectrumBytes : 0),
        measure(warmups, sampleCount, [&] {
            fftw.forward(input.data(), nativeWorkingSpectrum.data());
            if (fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor) {
                planeMajorToWvm(workload, nativeWorkingSpectrum.data(), workingSpectrum.data());
            }
        })));
    fftwRecord.timings.push_back(series("adapter-total", "WVM-compatible full-spectrum adapter", "inverse", StageState::executed,
        report.fullSpectrumBytes + report.fullRealBytes +
            (fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor ? 2 * report.fullSpectrumBytes : 0),
        measure(warmups, sampleCount,
            [&] {
                if (fftwSpectrumOrder == FFTWSpectrumOrder::wvmFrequencyMajor) {
                    std::copy(referenceSpectrum.begin(), referenceSpectrum.end(),
                              nativeInverseSpectrum.begin());
                }
            },
            [&] {
                if (fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor) {
                    wvmToPlaneMajor(
                        workload, referenceSpectrum.data(), nativeInverseSpectrum.data());
                }
                fftw.inverse(nativeInverseSpectrum.data(), output.data());
            })));
    if (retainedView) {
        fftwRecord.timings.push_back(series(
            "operator-component", "logical retained index view", "forward",
            StageState::elided, 0));
        fftwRecord.timings.push_back(series(
            "operator-component", "ready zero-padded provider-order view", "inverse",
            StageState::elided, 0));
        fftwRecord.timings.push_back(series(
            "uninstrumented-total", "retained horizontal operator over persistent view",
            "forward", StageState::executed,
            report.fullRealBytes + report.fullSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                fftw.forward(input.data(), nativeWorkingSpectrum.data());
            })));
        fftwRecord.timings.push_back(series(
            "uninstrumented-total", "retained horizontal operator over ready dead view",
            "inverse", StageState::executed,
            report.fullSpectrumBytes + report.fullRealBytes,
            measure(warmups, sampleCount,
                [&] {
                    std::copy(
                        nativeRetainedInverseReady.begin(),
                        nativeRetainedInverseReady.end(),
                        nativeInverseSpectrum.begin());
                },
                [&] { fftw.inverse(nativeInverseSpectrum.data(), output.data()); })));
    } else if (fusedSplitRetained) {
        fftwRecord.timings.push_back(series(
            "operator-component", "fused horizontal retention and split conversion",
            "forward", StageState::executed, 2 * report.retainedSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                fftw.gatherRetainedToSplitOuter(
                    modes, nativeReferenceSpectrum.data(),
                    retainedWorkingReal.data(), retainedWorkingImag.data());
            })));
        fftwRecord.timings.push_back(series(
            "operator-component",
            "fused split conversion, horizontal embedding, and zero fill",
            "inverse", StageState::executed,
            report.retainedSpectrumBytes + report.fullSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                fftw.embedRetainedFromSplitOuter(
                    modes, retainedReferenceReal.data(), retainedReferenceImag.data(),
                    nativeInverseSpectrum.data());
            })));
        fftwRecord.timings.push_back(series(
            "uninstrumented-total", "retained horizontal operator with compact split output",
            "forward", StageState::executed,
            report.fullRealBytes + report.fullSpectrumBytes + 2 * report.retainedSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                fftw.forward(input.data(), nativeWorkingSpectrum.data());
                fftw.gatherRetainedToSplitOuter(
                    modes, nativeWorkingSpectrum.data(),
                    retainedWorkingReal.data(), retainedWorkingImag.data());
            })));
        fftwRecord.timings.push_back(series(
            "uninstrumented-total", "retained horizontal operator with compact split input",
            "inverse", StageState::executed,
            report.retainedSpectrumBytes + report.fullSpectrumBytes + report.fullRealBytes,
            measure(warmups, sampleCount, [&] {
                fftw.embedRetainedFromSplitOuter(
                    modes, retainedReferenceReal.data(), retainedReferenceImag.data(),
                    nativeInverseSpectrum.data());
                fftw.inverse(nativeInverseSpectrum.data(), output.data());
            })));

        const auto normalization =
            1.0 / static_cast<double>(workload.nx * workload.ny);
        fftwRecord.timings.push_back(series(
            "diagnostic-component", "separate compact split normalization pass",
            "forward", StageState::executed, 2 * report.retainedSpectrumBytes,
            measure(warmups, sampleCount,
                [&] {
                    std::copy(
                        retainedReferenceReal.begin(), retainedReferenceReal.end(),
                        retainedSeparatelyNormalizedReal.begin());
                    std::copy(
                        retainedReferenceImag.begin(), retainedReferenceImag.end(),
                        retainedSeparatelyNormalizedImag.begin());
                },
                [&] {
                    fftw.scaleRetainedSplitOuter(
                        modes, retainedSeparatelyNormalizedReal.data(),
                        retainedSeparatelyNormalizedImag.data(), normalization);
                })));
        fftwRecord.timings.push_back(series(
            "diagnostic-total", "retained operator with fused horizontal normalization",
            "forward", StageState::executed,
            report.fullRealBytes + report.fullSpectrumBytes +
                2 * report.retainedSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                fftw.forward(input.data(), nativeWorkingSpectrum.data());
                fftw.gatherRetainedToSplitOuter(
                    modes, nativeWorkingSpectrum.data(),
                    retainedWorkingReal.data(), retainedWorkingImag.data(),
                    normalization);
            })));
        fftwRecord.timings.push_back(series(
            "diagnostic-total", "retained operator with separate horizontal normalization",
            "forward", StageState::executed,
            report.fullRealBytes + report.fullSpectrumBytes +
                4 * report.retainedSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                fftw.forward(input.data(), nativeWorkingSpectrum.data());
                fftw.gatherRetainedToSplitOuter(
                    modes, nativeWorkingSpectrum.data(),
                    retainedWorkingReal.data(), retainedWorkingImag.data());
                fftw.scaleRetainedSplitOuter(
                    modes, retainedWorkingReal.data(), retainedWorkingImag.data(),
                    normalization);
            })));
    } else {
        fftwRecord.timings.push_back(series("operator-component", "horizontal retention", "forward", StageState::executed,
            report.fullSpectrumBytes + report.retainedSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                fftw.gatherRetainedOuter(
                    modes, nativeReferenceSpectrum.data(), retainedWorking.data());
            })));
        fftwRecord.timings.push_back(series("operator-component", "horizontal embedding", "inverse", StageState::executed,
            report.retainedSpectrumBytes + report.fullSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                fftw.embedRetainedOuter(
                    modes, retainedSpectrum.data(), nativeInverseSpectrum.data());
            })));
        fftwRecord.timings.push_back(series("uninstrumented-total", "retained horizontal operator", "forward", StageState::executed,
            report.fullRealBytes + report.fullSpectrumBytes + report.retainedSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                fftw.forward(input.data(), nativeWorkingSpectrum.data());
                fftw.gatherRetainedOuter(
                    modes, nativeWorkingSpectrum.data(), retainedWorking.data());
            })));
        fftwRecord.timings.push_back(series("uninstrumented-total", "retained horizontal operator", "inverse", StageState::executed,
            report.retainedSpectrumBytes + report.fullSpectrumBytes + report.fullRealBytes,
            measure(warmups, sampleCount, [&] {
                fftw.embedRetainedOuter(
                    modes, retainedSpectrum.data(), nativeInverseSpectrum.data());
                fftw.inverse(nativeInverseSpectrum.data(), output.data());
            })));
    }
    if (options.fftwLayout == "interleaved" || options.fftwLayout == "paired") {
        report.providers.push_back(std::move(fftwRecord));
    }

    if (options.fftwLayout == "split" || options.fftwLayout == "paired") {
        auto splitStrategy = fftwStrategy;
        splitStrategy.layout = FFTWDataLayout::split;
        FFTWProvider splitFftw(workload, splitStrategy);
        const auto splitCount = workload.spectrumElements();
        FFTWArray<double> splitStorage(2 * splitCount);
        FFTWArray<double> referenceSplitStorage(2 * splitCount);
        FFTWArray<double> inverseSplitStorage(2 * splitCount);
        auto* splitReal = splitStorage.data();
        auto* splitImag = splitStorage.data() + splitCount;
        auto* referenceSplitReal = referenceSplitStorage.data();
        auto* referenceSplitImag = referenceSplitStorage.data() + splitCount;
        auto* inverseSplitReal = inverseSplitStorage.data();
        auto* inverseSplitImag = inverseSplitStorage.data() + splitCount;
        std::vector<double> retainedSplitReal(retainedSpectrum.size());
        std::vector<double> retainedSplitImag(retainedSpectrum.size());
        std::vector<double> retainedReferenceReal(retainedSpectrum.size());
        std::vector<double> retainedReferenceImag(retainedSpectrum.size());

        interleavedToSplit(referenceSpectrum.size(), nativeReferenceSpectrum.data(),
                           referenceSplitReal, referenceSplitImag);
        interleavedToSplit(retainedSpectrum.size(), retainedSpectrum.data(),
                           retainedReferenceReal.data(), retainedReferenceImag.data());

        splitFftw.forwardSplit(input.data(), splitReal, splitImag);
        splitToInterleaved(
            referenceSpectrum.size(), splitReal, splitImag, nativeWorkingSpectrum.data());
        nativeToWvm(nativeWorkingSpectrum.data(), workingSpectrum.data());
        const auto splitForwardReferenceError = maximumRelativeError(
            workingSpectrum.data(), referenceSpectrum.data(), referenceSpectrum.size());
        std::copy_n(referenceSplitReal, splitCount, inverseSplitReal);
        std::copy_n(referenceSplitImag, splitCount, inverseSplitImag);
        splitFftw.inverseSplit(inverseSplitReal, inverseSplitImag, output.data());
        const auto splitInverseReferenceError = maximumRelativeError(
            output.data(), fftwOutput.data(), fftwOutput.size());
        const auto splitRoundTripError = maximumRelativeError(
            output.data(), input.data(), input.size(), 1.0 / static_cast<double>(workload.nx * workload.ny));

        splitFftw.gatherRetainedSplitOuter(
            modes, splitReal, splitImag, retainedSplitReal.data(), retainedSplitImag.data());
        splitToInterleaved(retainedSpectrum.size(), retainedSplitReal.data(), retainedSplitImag.data(), retainedWorking.data());
        const auto splitRetainedForwardError = maximumRelativeError(
            retainedWorking.data(), retainedSpectrum.data(), retainedSpectrum.size());

        splitFftw.embedRetainedSplitOuter(
            modes, retainedReferenceReal.data(), retainedReferenceImag.data(),
            inverseSplitReal, inverseSplitImag);
        splitFftw.inverseSplit(inverseSplitReal, inverseSplitImag, output.data());
        embedRetained(workload, modes, retainedSpectrum.data(), inverseSpectrum.data());
        referenceFftw.inverse(inverseSpectrum.data(), fftwOutput.data());
        const auto splitRetainedInverseError = maximumRelativeError(output.data(), fftwOutput.data(), output.size());

        splitToInterleaved(
            referenceSpectrum.size(), referenceSplitReal, referenceSplitImag,
            nativeWorkingSpectrum.data());
        const auto splitToInterleavedError = maximumRelativeError(
            nativeWorkingSpectrum.data(), nativeReferenceSpectrum.data(), referenceSpectrum.size());
        nativeToWvm(nativeWorkingSpectrum.data(), workingSpectrum.data());
        const auto splitNativeToWvmError = maximumRelativeError(
            workingSpectrum.data(), referenceSpectrum.data(), referenceSpectrum.size());
        interleavedToSplit(
            referenceSpectrum.size(), nativeReferenceSpectrum.data(), splitReal, splitImag);
        const auto interleavedToSplitRealError = maximumRelativeError(
            splitReal, referenceSplitReal, referenceSpectrum.size());
        const auto interleavedToSplitImagError = maximumRelativeError(
            splitImag, referenceSplitImag, referenceSpectrum.size());

        ProviderRecord splitRecord;
        splitRecord.id = "fftw-split";
        splitRecord.version = splitFftw.version();
        splitRecord.libraryIdentity = splitFftw.libraryIdentity();
        splitRecord.algorithmId = fftwAlgorithmId(splitFftw);
        splitRecord.nativeRepresentationId = fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor
            ? "plane-major-split-half-spectrum"
            : "wvm-frequency-major-split-half-spectrum";
        splitRecord.modeOrderId = "full-r2c-kx-nonnegative-ky-wrapped";
        splitRecord.schedulingId = fftwSchedulingId(splitFftw);
        splitRecord.sourceIdentity = "https://fftw.org/pub/fftw/fftw-3.3.11.tar.gz";
        splitRecord.sourceSha256 = "5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1";
        splitRecord.configureFlags = "--host=aarch64-apple-darwin --enable-neon --enable-threads --disable-fortran --disable-openmp --enable-shared --disable-static";
        splitRecord.compilerFlags = "-O3 -mcpu=native -mmacosx-version-min=13.3";
        splitRecord.planningConfiguration = fftwPlanningConfiguration(splitFftw);
        splitRecord.workers = splitFftw.totalLogicalWorkers();
        splitRecord.internalWorkers = splitFftw.internalWorkers();
        splitRecord.outerWorkers = splitFftw.outerWorkers();
        splitRecord.execution = fftwExecutionContract(workload, splitFftw);
        splitRecord.otherSetupSeconds = splitFftw.otherSetupSeconds();
        splitRecord.allocationSeconds = splitFftw.allocationSeconds();
        splitRecord.planningSeconds = splitFftw.planningSeconds();
        splitRecord.wisdomGenerationSeconds = splitFftw.wisdomGenerationSeconds();
        splitRecord.wisdomImportSeconds = splitFftw.wisdomImportSeconds();
        splitRecord.planningTimeLimitSeconds = splitFftw.planningTimeLimitSeconds();
        splitRecord.planningBudgetExhausted = splitFftw.planningBudgetExhausted();
        splitRecord.wisdomBytes = splitFftw.wisdomBytes();
        splitRecord.opaquePlanningBytes = splitFftw.planningBytes();
        splitRecord.ledger = fftwLedger(splitFftw);
        splitRecord.ledger.push_back({"multidimensional split in-place", StageState::unsupported,
                                      splitFftw.splitInPlaceWvmOrderCapability()});
        splitRecord.correctness = {
            metric("full split forward versus fixed FFTW ESTIMATE reference", splitForwardReferenceError),
            metric("full split inverse versus fixed FFTW ESTIMATE reference", splitInverseReferenceError),
            metric("full split inverse round trip", splitRoundTripError),
            metric("direct split retained forward versus interleaved oracle", splitRetainedForwardError),
            metric("direct split retained inverse versus interleaved oracle", splitRetainedInverseError),
            metric("split-to-interleaved conversion", splitToInterleavedError),
            metric("native-order-to-WVM permutation", splitNativeToWvmError),
            metric("interleaved-to-split real conversion", interleavedToSplitRealError),
            metric("interleaved-to-split imaginary conversion", interleavedToSplitImagError)};

        splitRecord.timings.push_back(series("primitive", "raw FFT", "forward", StageState::executed,
            report.fullRealBytes + report.fullSpectrumBytes,
            measure(warmups, sampleCount, [&] { splitFftw.forwardSplit(input.data(), splitReal, splitImag); })));
        splitRecord.timings.push_back(series("primitive", "raw FFT", "inverse", StageState::executed,
            report.fullSpectrumBytes + report.fullRealBytes,
            measure(warmups, sampleCount,
                [&] {
                    std::copy_n(referenceSplitReal, splitCount, inverseSplitReal);
                    std::copy_n(referenceSplitImag, splitCount, inverseSplitImag);
                },
                [&] { splitFftw.inverseSplit(inverseSplitReal, inverseSplitImag, output.data()); })));
        splitRecord.timings.push_back(series(
            "diagnostic-component", "batch scheduler empty dispatch", "shared",
            splitFftw.outerWorkers() > 1 ? StageState::executed : StageState::elided, 0,
            splitFftw.outerWorkers() > 1
                ? measure(warmups, sampleCount, [&] { splitFftw.executeSchedulerNoop(); })
                : std::vector<double>{}));
        splitRecord.timings.push_back(series("adapter-component", "split-to-interleaved conversion", "forward", StageState::executed,
            2 * report.fullSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                splitToInterleaved(
                    referenceSpectrum.size(), referenceSplitReal, referenceSplitImag,
                    nativeWorkingSpectrum.data());
            })));
        splitRecord.timings.push_back(series(
            "adapter-component", "spectrum-order permutation", "forward", orderConversionState,
            fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor ? 2 * report.fullSpectrumBytes : 0,
            fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor
                ? measure(warmups, sampleCount, [&] {
                    planeMajorToWvm(
                        workload, nativeReferenceSpectrum.data(), workingSpectrum.data());
                })
                : std::vector<double>{}));
        splitRecord.timings.push_back(series(
            "adapter-component", "spectrum-order permutation", "inverse", orderConversionState,
            fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor ? 2 * report.fullSpectrumBytes : 0,
            fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor
                ? measure(warmups, sampleCount, [&] {
                    wvmToPlaneMajor(
                        workload, referenceSpectrum.data(), nativeInverseSpectrum.data());
                })
                : std::vector<double>{}));
        splitRecord.timings.push_back(series("adapter-component", "interleaved-to-split conversion", "inverse", StageState::executed,
            2 * report.fullSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                interleavedToSplit(
                    referenceSpectrum.size(), nativeReferenceSpectrum.data(),
                    inverseSplitReal, inverseSplitImag);
            })));
        splitRecord.timings.push_back(series("adapter-total", "WVM-compatible full-spectrum adapter", "forward", StageState::executed,
            report.fullRealBytes + 3 * report.fullSpectrumBytes +
                (fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor ? 2 * report.fullSpectrumBytes : 0),
            measure(warmups, sampleCount, [&] {
                splitFftw.forwardSplit(input.data(), splitReal, splitImag);
                splitToInterleaved(
                    referenceSpectrum.size(), splitReal, splitImag,
                    nativeWorkingSpectrum.data());
                if (fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor) {
                    planeMajorToWvm(
                        workload, nativeWorkingSpectrum.data(), workingSpectrum.data());
                }
            })));
        splitRecord.timings.push_back(series("adapter-total", "WVM-compatible full-spectrum adapter", "inverse", StageState::executed,
            3 * report.fullSpectrumBytes + report.fullRealBytes +
                (fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor ? 2 * report.fullSpectrumBytes : 0),
            measure(warmups, sampleCount, [&] {
                if (fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor) {
                    wvmToPlaneMajor(
                        workload, referenceSpectrum.data(), nativeInverseSpectrum.data());
                }
                interleavedToSplit(
                    referenceSpectrum.size(),
                    fftwSpectrumOrder == FFTWSpectrumOrder::planeMajor
                        ? nativeInverseSpectrum.data() : referenceSpectrum.data(),
                    inverseSplitReal, inverseSplitImag);
                splitFftw.inverseSplit(inverseSplitReal, inverseSplitImag, output.data());
            })));
        splitRecord.timings.push_back(series("operator-component", "direct split horizontal retention", "forward", StageState::executed,
            report.fullSpectrumBytes + report.retainedSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                splitFftw.gatherRetainedSplitOuter(
                    modes, referenceSplitReal, referenceSplitImag,
                    retainedSplitReal.data(), retainedSplitImag.data());
            })));
        splitRecord.timings.push_back(series("operator-component", "direct split horizontal embedding", "inverse", StageState::executed,
            report.retainedSpectrumBytes + report.fullSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                splitFftw.embedRetainedSplitOuter(
                    modes, retainedReferenceReal.data(), retainedReferenceImag.data(),
                    inverseSplitReal, inverseSplitImag);
            })));
        splitRecord.timings.push_back(series("uninstrumented-total", "persistent split retained horizontal operator", "forward", StageState::executed,
            report.fullRealBytes + report.fullSpectrumBytes + report.retainedSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                splitFftw.forwardSplit(input.data(), splitReal, splitImag);
                splitFftw.gatherRetainedSplitOuter(
                    modes, splitReal, splitImag,
                    retainedSplitReal.data(), retainedSplitImag.data());
            })));
        splitRecord.timings.push_back(series("uninstrumented-total", "persistent split retained horizontal operator", "inverse", StageState::executed,
            report.retainedSpectrumBytes + report.fullSpectrumBytes + report.fullRealBytes,
            measure(warmups, sampleCount, [&] {
                splitFftw.embedRetainedSplitOuter(
                    modes, retainedReferenceReal.data(), retainedReferenceImag.data(),
                    inverseSplitReal, inverseSplitImag);
                splitFftw.inverseSplit(inverseSplitReal, inverseSplitImag, output.data());
            })));
        splitRecord.timings.push_back(series(
            "capability", "multidimensional split in-place", "shared",
            StageState::unsupported, 0));
        report.providers.push_back(std::move(splitRecord));
    }

    if (options.providers == "fftw") {
        report.status = std::all_of(report.providers.begin(), report.providers.end(), correctnessPassed) ? "passed" : "failed";
        return report;
    }

    VDSPProvider vdsp(workload, workers, vdspStrategy, vdspBatchStrategy);
    if (!vdsp.supported()) {
        appendUnsupportedVdspRecord(report, selected, vdsp);
        report.status = "failed";
        return report;
    }

    vdsp.forwardAdapter(input.data(), workingSpectrum.data());
    const auto vdspForwardError = maximumRelativeError(workingSpectrum.data(), referenceSpectrum.data(), referenceSpectrum.size());
    gatherRetained(workload, modes, workingSpectrum.data(), retainedWorking.data());
    const auto vdspRetainedError = maximumRelativeError(retainedWorking.data(), retainedSpectrum.data(), retainedSpectrum.size());
    vdsp.inverseAdapter(referenceSpectrum.data(), output.data());
    const auto vdspInverseReferenceError = maximumRelativeError(output.data(), fftwOutput.data(), output.size());
    const auto vdspRoundTripError = maximumRelativeError(output.data(), input.data(), input.size(), 1.0 / static_cast<double>(workload.nx * workload.ny));

    embedRetained(workload, modes, retainedSpectrum.data(), inverseSpectrum.data());
    vdsp.inverseAdapter(inverseSpectrum.data(), output.data());
    embedRetained(workload, modes, retainedSpectrum.data(), workingSpectrum.data());
    fftw.inverse(workingSpectrum.data(), fftwOutput.data());
    const auto vdspRetainedInverseError = maximumRelativeError(output.data(), fftwOutput.data(), output.size());

    const auto retainedElements = retainedSpectrum.size();
    std::vector<double> vdspRetainedReferenceReal(retainedElements);
    std::vector<double> vdspRetainedReferenceImag(retainedElements);
    std::vector<double> vdspRetainedWorkingReal(retainedElements);
    std::vector<double> vdspRetainedWorkingImag(retainedElements);
    std::vector<Complex> vdspRetainedCanonical(retainedElements);
    FFTWArray<double> vdspNativeRetainedOutput(workload.realElements());
    for (std::size_t plane = 0; plane < workload.planes(); ++plane) {
        const auto z = plane % workload.nz;
        const auto field = plane / workload.nz;
        for (std::size_t mode = 0; mode < modes.size(); ++mode) {
            const auto canonical = retainedSpectrumIndex(workload, mode, z, field);
            const auto native = mode + modes.size() * plane;
            vdspRetainedReferenceReal[native] = retainedSpectrum[canonical].real;
            vdspRetainedReferenceImag[native] = retainedSpectrum[canonical].imag;
        }
    }
    vdsp.forwardRetainedNativeSplit(
        input.data(), modes, vdspRetainedWorkingReal.data(),
        vdspRetainedWorkingImag.data());
    for (std::size_t plane = 0; plane < workload.planes(); ++plane) {
        const auto z = plane % workload.nz;
        const auto field = plane / workload.nz;
        for (std::size_t mode = 0; mode < modes.size(); ++mode) {
            const auto canonical = retainedSpectrumIndex(workload, mode, z, field);
            const auto native = mode + modes.size() * plane;
            vdspRetainedCanonical[canonical] = {
                vdspRetainedWorkingReal[native],
                vdspRetainedWorkingImag[native]};
        }
    }
    const auto vdspNativeRetainedForwardError = maximumRelativeError(
        vdspRetainedCanonical.data(), retainedSpectrum.data(), retainedElements);
    vdsp.inverseRetainedNativeSplit(
        modes, vdspRetainedReferenceReal.data(), vdspRetainedReferenceImag.data(),
        vdspNativeRetainedOutput.data());
    const auto vdspNativeRetainedInverseError = maximumRelativeError(
        vdspNativeRetainedOutput.data(), fftwOutput.data(),
        vdspNativeRetainedOutput.size());

    ProviderRecord vdspRecord;
    vdspRecord.id = "accelerate-vdsp";
    vdspRecord.version = "system";
    vdspRecord.libraryIdentity = vdsp.libraryIdentity();
    vdspRecord.algorithmId = vdspAlgorithmId(vdsp.strategy(), vdsp.batchStrategy());
    vdspRecord.nativeRepresentationId = "vdsp-packed-split-complex";
    vdspRecord.modeOrderId = "vdsp-packed-special-boundaries";
    vdspRecord.schedulingId = vdspSchedulingId(vdsp.batchStrategy());
    vdspRecord.sourceIdentity = "Apple Accelerate system framework";
    vdspRecord.configureFlags = "system framework";
    vdspRecord.compilerFlags = report.environment.compilerFlags;
    vdspRecord.planningConfiguration = "radix-2 setup per logical batch worker; " +
        std::string(vdspTransformStrategyName(vdsp.strategy())) + "; " +
        std::string(vdspBatchStrategyName(vdsp.batchStrategy()));
    vdspRecord.workers = workers;
    vdspRecord.internalWorkers = 1;
    vdspRecord.outerWorkers = workers;
    vdspRecord.execution = vdspExecutionContract(workload, vdsp);
    vdspRecord.explicitPersistentBytes = vdsp.explicitPersistentBytes();
    vdspRecord.scratchBytes = vdsp.scratchBytes();
    vdspRecord.otherSetupSeconds = vdsp.otherSetupSeconds();
    vdspRecord.allocationSeconds = vdsp.allocationSeconds();
    vdspRecord.planningSeconds = vdsp.planningSeconds();
    vdspRecord.ledger = vdspLedger(vdsp.strategy(), vdsp.batchStrategy());
    vdspRecord.correctness = {
        metric("full forward versus FFTW", vdspForwardError),
        metric("full inverse versus FFTW", vdspInverseReferenceError),
        metric("full inverse round trip", vdspRoundTripError),
        metric("retained forward versus FFTW", vdspRetainedError),
        metric("retained inverse versus FFTW", vdspRetainedInverseError)};

    vdspRecord.timings.push_back(series("primitive", "raw FFT", "forward", StageState::executed,
        vdsp.nativeOperandBytes() * 2,
        measure(warmups, sampleCount,
            [&] { vdsp.packForwardInput(input.data()); },
            [&] { vdsp.executeForwardNative(); })));
    vdspRecord.timings.push_back(series("primitive", "raw FFT", "inverse", StageState::executed,
        vdsp.nativeOperandBytes() * 2,
        measure(warmups, sampleCount,
            [&] { vdsp.packInverseInput(referenceSpectrum.data()); },
            [&] { vdsp.executeInverseNative(); })));
    vdspRecord.timings.push_back(series("diagnostic-component", "batch scheduler empty dispatch", "shared",
        StageState::executed, 0,
        measure(warmups, sampleCount, [&] { vdsp.executeSchedulerNoop(); })));

    if (vdsp.separable()) {
        vdspRecord.timings.push_back(series("primitive-component", "real row FFTs", "forward", StageState::executed,
            vdsp.nativeOperandBytes() * 2,
            measure(warmups, sampleCount,
                [&] { vdsp.packForwardInput(input.data()); },
                [&] { vdsp.executeForwardRowsNative(); })));
        vdspRecord.timings.push_back(series("primitive-component", "complex column FFTs and Hermitian boundaries", "forward",
            StageState::executed, vdsp.nativeOperandBytes() * 2,
            measure(warmups, sampleCount,
                [&] {
                    vdsp.packForwardInput(input.data());
                    vdsp.executeForwardRowsNative();
                },
                [&] { vdsp.executeForwardColumnsNative(); })));
        vdspRecord.timings.push_back(series("primitive-component", "complex column FFTs and Hermitian boundaries", "inverse",
            StageState::executed, vdsp.nativeOperandBytes() * 2,
            measure(warmups, sampleCount,
                [&] { vdsp.packInverseInput(referenceSpectrum.data()); },
                [&] { vdsp.executeInverseColumnsNative(); })));
        vdspRecord.timings.push_back(series("primitive-component", "real row FFTs", "inverse", StageState::executed,
            vdsp.nativeOperandBytes() * 2,
            measure(warmups, sampleCount,
                [&] {
                    vdsp.packInverseInput(referenceSpectrum.data());
                    vdsp.executeInverseColumnsNative();
                },
                [&] { vdsp.executeInverseRowsNative(); })));
    } else {
        vdspRecord.timings.push_back(series("primitive-component", "real row FFTs", "forward", StageState::fused, 0));
        vdspRecord.timings.push_back(series("primitive-component", "complex column FFTs and Hermitian boundaries", "forward", StageState::fused, 0));
        vdspRecord.timings.push_back(series("primitive-component", "complex column FFTs and Hermitian boundaries", "inverse", StageState::fused, 0));
        vdspRecord.timings.push_back(series("primitive-component", "real row FFTs", "inverse", StageState::fused, 0));
    }
    vdspRecord.timings.push_back(series("algorithm-component", "column transposition", "shared", StageState::elided, 0));

    vdspRecord.timings.push_back(series("adapter-component", "real-to-vDSP packing", "forward", StageState::executed,
        report.fullRealBytes + vdsp.nativeOperandBytes(),
        measure(warmups, sampleCount, [&] { vdsp.packForwardInput(input.data()); })));
    vdsp.packForwardInput(input.data());
    vdsp.executeForwardNative();
    vdspRecord.timings.push_back(series("adapter-component", "vDSP-to-WVM conversion and permutation", "forward", StageState::executed,
        vdsp.nativeOperandBytes() + report.fullSpectrumBytes,
        measure(warmups, sampleCount, [&] { vdsp.unpackForwardOutput(workingSpectrum.data()); })));
    vdspRecord.timings.push_back(series("adapter-component", "WVM-to-vDSP conversion and permutation", "inverse", StageState::executed,
        report.fullSpectrumBytes + vdsp.nativeOperandBytes(),
        measure(warmups, sampleCount, [&] { vdsp.packInverseInput(referenceSpectrum.data()); })));
    vdsp.packInverseInput(referenceSpectrum.data());
    vdsp.executeInverseNative();
    vdspRecord.timings.push_back(series("adapter-component", "vDSP-to-real unpacking", "inverse", StageState::executed,
        vdsp.nativeOperandBytes() + report.fullRealBytes,
        measure(warmups, sampleCount, [&] { vdsp.unpackInverseOutput(output.data()); })));

    vdspRecord.timings.push_back(series("adapter-total", "WVM-compatible full-spectrum adapter", "forward", StageState::executed,
        report.fullRealBytes + 2 * vdsp.nativeOperandBytes() + report.fullSpectrumBytes,
        measure(warmups, sampleCount, [&] { vdsp.forwardAdapter(input.data(), workingSpectrum.data()); })));
    vdspRecord.timings.push_back(series("adapter-total", "WVM-compatible full-spectrum adapter", "inverse", StageState::executed,
        report.fullSpectrumBytes + 2 * vdsp.nativeOperandBytes() + report.fullRealBytes,
        measure(warmups, sampleCount, [&] { vdsp.inverseAdapter(referenceSpectrum.data(), output.data()); })));
    vdspRecord.timings.push_back(series("operator-component", "horizontal retention", "forward", StageState::executed,
        report.fullSpectrumBytes + report.retainedSpectrumBytes,
        measure(warmups, sampleCount, [&] { gatherRetained(workload, modes, referenceSpectrum.data(), retainedWorking.data()); })));
    vdspRecord.timings.push_back(series("operator-component", "horizontal embedding", "inverse", StageState::executed,
        report.retainedSpectrumBytes + report.fullSpectrumBytes,
        measure(warmups, sampleCount, [&] { embedRetained(workload, modes, retainedSpectrum.data(), inverseSpectrum.data()); })));
    vdspRecord.timings.push_back(series("uninstrumented-total", "retained horizontal operator", "forward", StageState::executed,
        report.fullRealBytes + 2 * vdsp.nativeOperandBytes() + report.fullSpectrumBytes + report.retainedSpectrumBytes,
        measure(warmups, sampleCount, [&] {
            vdsp.forwardAdapter(input.data(), workingSpectrum.data());
            gatherRetained(workload, modes, workingSpectrum.data(), retainedWorking.data());
        })));
    vdspRecord.timings.push_back(series("uninstrumented-total", "retained horizontal operator", "inverse", StageState::executed,
        report.retainedSpectrumBytes + report.fullSpectrumBytes + 2 * vdsp.nativeOperandBytes() + report.fullRealBytes,
        measure(warmups, sampleCount, [&] {
            embedRetained(workload, modes, retainedSpectrum.data(), inverseSpectrum.data());
            vdsp.inverseAdapter(inverseSpectrum.data(), output.data());
        })));
    report.providers.push_back(std::move(vdspRecord));

    ProviderRecord vdspNativeRetainedRecord;
    vdspNativeRetainedRecord.id = "accelerate-vdsp-native-retained";
    vdspNativeRetainedRecord.version = "system";
    vdspNativeRetainedRecord.libraryIdentity = vdsp.libraryIdentity();
    vdspNativeRetainedRecord.algorithmId =
        vdspAlgorithmId(vdsp.strategy(), vdsp.batchStrategy()) +
        "-direct-native-retained-split";
    vdspNativeRetainedRecord.nativeRepresentationId =
        "vdsp-packed-split-complex+plane-major-radial-retained-split-complex";
    vdspNativeRetainedRecord.modeOrderId =
        "retained logical radial mode contiguous within plane; plane=z+Nz*field";
    vdspNativeRetainedRecord.schedulingId = vdspSchedulingId(vdsp.batchStrategy());
    vdspNativeRetainedRecord.sourceIdentity = "Apple Accelerate system framework";
    vdspNativeRetainedRecord.configureFlags = "system framework";
    vdspNativeRetainedRecord.compilerFlags = report.environment.compilerFlags;
    vdspNativeRetainedRecord.planningConfiguration =
        "radix-2 setup per logical batch worker; direct native packed split retention; " +
        std::string(vdspTransformStrategyName(vdsp.strategy())) + "; " +
        std::string(vdspBatchStrategyName(vdsp.batchStrategy()));
    vdspNativeRetainedRecord.workers = workers;
    vdspNativeRetainedRecord.internalWorkers = 1;
    vdspNativeRetainedRecord.outerWorkers = workers;
    vdspNativeRetainedRecord.execution = vdspNativeRetainedExecutionContract(
        workload, vdsp, modes.size());
    vdspNativeRetainedRecord.explicitPersistentBytes = vdsp.explicitPersistentBytes();
    vdspNativeRetainedRecord.scratchBytes = vdsp.scratchBytes();
    vdspNativeRetainedRecord.otherSetupSeconds = vdsp.otherSetupSeconds();
    vdspNativeRetainedRecord.allocationSeconds = vdsp.allocationSeconds();
    vdspNativeRetainedRecord.planningSeconds = vdsp.planningSeconds();
    vdspNativeRetainedRecord.ledger = vdspNativeRetainedLedger(
        vdsp.strategy(), vdsp.batchStrategy());
    vdspNativeRetainedRecord.correctness = {
        metric("native packed split retained forward versus mode-keyed FFTW oracle",
               vdspNativeRetainedForwardError),
        metric("native packed split retained inverse versus FFTW projection",
               vdspNativeRetainedInverseError)};
    vdspNativeRetainedRecord.timings.push_back(series(
        "primitive", "raw FFT", "forward", StageState::executed,
        vdsp.nativeOperandBytes() * 2,
        measure(warmups, sampleCount,
                [&] { vdsp.packForwardInput(input.data()); },
                [&] { vdsp.executeForwardNative(); })));
    vdspNativeRetainedRecord.timings.push_back(series(
        "primitive", "raw FFT", "inverse", StageState::executed,
        vdsp.nativeOperandBytes() * 2,
        measure(warmups, sampleCount,
                [&] {
                    vdsp.embedRetainedNativeSplit(
                        modes, vdspRetainedReferenceReal.data(),
                        vdspRetainedReferenceImag.data());
                },
                [&] { vdsp.executeInverseNative(); })));
    vdspNativeRetainedRecord.timings.push_back(series(
        "diagnostic-component", "batch scheduler empty dispatch", "shared",
        StageState::executed, 0,
        measure(warmups, sampleCount, [&] { vdsp.executeSchedulerNoop(); })));
    vdspNativeRetainedRecord.timings.push_back(series(
        "adapter-component", "real-to-vDSP packing", "forward",
        StageState::executed,
        report.fullRealBytes + vdsp.nativeOperandBytes(),
        measure(warmups, sampleCount,
                [&] { vdsp.packForwardInput(input.data()); })));
    vdsp.packForwardInput(input.data());
    vdsp.executeForwardNative();
    vdspNativeRetainedRecord.timings.push_back(series(
        "operator-component", "direct native packed split horizontal retention",
        "forward", StageState::executed,
        vdsp.nativeOperandBytes() + report.retainedSpectrumBytes,
        measure(warmups, sampleCount, [&] {
            vdsp.gatherRetainedNativeSplit(
                modes, vdspRetainedWorkingReal.data(),
                vdspRetainedWorkingImag.data());
        })));
    vdspNativeRetainedRecord.timings.push_back(series(
        "operator-component", "direct native packed split horizontal embedding",
        "inverse", StageState::executed,
        report.retainedSpectrumBytes + vdsp.nativeOperandBytes(),
        measure(warmups, sampleCount, [&] {
            vdsp.embedRetainedNativeSplit(
                modes, vdspRetainedReferenceReal.data(),
                vdspRetainedReferenceImag.data());
        })));
    vdsp.embedRetainedNativeSplit(
        modes, vdspRetainedReferenceReal.data(),
        vdspRetainedReferenceImag.data());
    vdsp.executeInverseNative();
    vdspNativeRetainedRecord.timings.push_back(series(
        "adapter-component", "vDSP-to-real unpacking", "inverse",
        StageState::executed,
        vdsp.nativeOperandBytes() + report.fullRealBytes,
        measure(warmups, sampleCount,
                [&] { vdsp.unpackInverseOutput(vdspNativeRetainedOutput.data()); })));
    vdspNativeRetainedRecord.timings.push_back(series(
        "uninstrumented-total", "persistent native split retained horizontal operator",
        "forward", StageState::executed,
        report.fullRealBytes + 3 * vdsp.nativeOperandBytes() +
            report.retainedSpectrumBytes,
        measure(warmups, sampleCount, [&] {
            vdsp.forwardRetainedNativeSplit(
                input.data(), modes, vdspRetainedWorkingReal.data(),
                vdspRetainedWorkingImag.data());
        })));
    vdspNativeRetainedRecord.timings.push_back(series(
        "uninstrumented-total", "persistent native split retained horizontal operator",
        "inverse", StageState::executed,
        report.retainedSpectrumBytes + 3 * vdsp.nativeOperandBytes() +
            report.fullRealBytes,
        measure(warmups, sampleCount, [&] {
            vdsp.inverseRetainedNativeSplit(
                modes, vdspRetainedReferenceReal.data(),
                vdspRetainedReferenceImag.data(),
                vdspNativeRetainedOutput.data());
        })));
    report.providers.push_back(std::move(vdspNativeRetainedRecord));

    report.status = std::all_of(report.providers.begin(), report.providers.end(), correctnessPassed) ? "passed" : "failed";
    return report;
}

} // namespace skbench
