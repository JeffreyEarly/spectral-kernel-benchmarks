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
    const auto layout = strategy.layout == FFTWDataLayout::split ? "wvm-guru64-split-" : "wvm-guru64-";
    return std::string(layout) + std::string(fftwPlanningModeName(strategy.planningMode)) + "-" +
        std::string(fftwAlignmentStrategyName(strategy.alignment)) + "-" +
        std::string(fftwWisdomStrategyName(strategy.wisdom));
}

std::string fftwSchedulingId(const FFTWProvider& provider) {
    if (provider.outerWorkers() == 1) return "fftw-internal-pthreads";
    if (provider.internalWorkers() == 1) return "persistent-outer-batch-sharding";
    return "persistent-outer-batch-sharding+fftw-internal-pthreads";
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
    const auto scheduling = provider.outerWorkers() > 1
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

ExecutionContract fftwExecutionContract(const Workload& workload, const FFTWProvider& provider) {
    const auto planes = workload.planes();
    const auto realExtents = "[planes=" + std::to_string(planes) + "][Ny=" + std::to_string(workload.ny) +
        "][Nx=" + std::to_string(workload.nx) + "]";
    const auto spectrumExtents = "[Ny=" + std::to_string(workload.ny) + "][NxHalf=" +
        std::to_string(workload.nxHalf()) + "][planes=" + std::to_string(planes) + "]";
    const auto realStrides = "x=1,y=" + std::to_string(workload.nx) + ",plane=" +
        std::to_string(workload.realPlaneElements());
    const auto spectrumStrides = "plane=1,kx=" + std::to_string(planes) + ",ky=" +
        std::to_string(planes * workload.nxHalf());
    const auto split = provider.strategy().layout == FFTWDataLayout::split;
    const auto nativeSpectrum = split
        ? "wvm-frequency-major-split-half-spectrum"
        : "wvm-frequency-major-interleaved-half-spectrum";
    const auto nativeSpectrumExtents = split
        ? "one contiguous allocation containing real then imaginary arrays, each " + spectrumExtents
        : spectrumExtents;
    const auto aliasing = split
        ? "input is disjoint from the contiguous [real][imaginary] output; the component separation is fixed at the full half-spectrum element count; exact WVM-order split in-place planning is unsupported"
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
            ? "the contiguous [real][imaginary] input is disjoint from output and has the planning-time component separation; multidimensional FFTW split c2r may destroy both components; exact WVM-order split in-place planning is unsupported"
            : (provider.minimumAlignmentBytes() > 1
                ? "input and output do not overlap and match planning alignment classes; multidimensional FFTW c2r may destroy its input"
                : "input and output do not overlap; multidimensional FFTW c2r may destroy its input"),
        0,
        true};
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
        {"wvm-current-512-nz257-f4", "Current WVM 512-cubed-class workload, four fields.", {512, 512, 257, 4, 1.0, 1.0, true}, totalWorkers, 3, 20}};
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

BenchmarkReport runBenchmark(const RunOptions& options) {
    if (options.kernel == "vertical-gemm") return runVerticalGemmBenchmark(options);
    if (options.kernel == "ordering-packing") return runOrderingPackingBenchmark(options);
    if (options.kernel != "fft") {
        throw std::invalid_argument("kernel must be 'fft', 'vertical-gemm', or 'ordering-packing'.");
    }
    auto selected = profileNamed(options.profile);
    if (options.providers != "both" && options.providers != "fftw") {
        throw std::invalid_argument("providers must be either 'both' or 'fftw'.");
    }
    if (options.fftwLayout != "interleaved" && options.fftwLayout != "split" && options.fftwLayout != "paired") {
        throw std::invalid_argument("fftw-layout must be 'interleaved', 'split', or 'paired'.");
    }
    if (options.providers == "both" && options.fftwLayout != "interleaved") {
        throw std::invalid_argument("fftw-layout split/paired is currently an FFTW-only experiment; use --providers fftw.");
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
        FFTWDataLayout::interleaved};
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
    FFTWArray<Complex> workingSpectrum(workload.spectrumElements());
    FFTWArray<Complex> inverseSpectrum(workload.spectrumElements());
    std::vector<Complex> retainedSpectrum(modes.size() * workload.planes());
    std::vector<Complex> retainedWorking(modes.size() * workload.planes());
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

    FFTWProvider fftw(workload, fftwStrategy);
    fftw.forward(input.data(), workingSpectrum.data());
    const auto fftwForwardReferenceError = maximumRelativeError(
        workingSpectrum.data(), referenceSpectrum.data(), referenceSpectrum.size());
    std::copy(referenceSpectrum.begin(), referenceSpectrum.end(), inverseSpectrum.begin());
    fftw.inverse(inverseSpectrum.data(), fftwOutput.data());
    const auto fftwInverseReferenceError = maximumRelativeError(
        fftwOutput.data(), referenceOutput.data(), referenceOutput.size());
    const auto fftwRoundTripError = maximumRelativeError(
        fftwOutput.data(), input.data(), input.size(), 1.0 / static_cast<double>(workload.nx * workload.ny));

    ProviderRecord fftwRecord;
    fftwRecord.id = "fftw";
    fftwRecord.version = fftw.version();
    fftwRecord.libraryIdentity = fftw.libraryIdentity();
    fftwRecord.algorithmId = fftwAlgorithmId(fftw);
    fftwRecord.nativeRepresentationId = "wvm-frequency-major-interleaved-half-spectrum";
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
    fftwRecord.execution = fftwExecutionContract(workload, fftw);
    fftwRecord.otherSetupSeconds = fftw.otherSetupSeconds();
    fftwRecord.allocationSeconds = fftw.allocationSeconds();
    fftwRecord.planningSeconds = fftw.planningSeconds();
    fftwRecord.wisdomGenerationSeconds = fftw.wisdomGenerationSeconds();
    fftwRecord.wisdomImportSeconds = fftw.wisdomImportSeconds();
    fftwRecord.planningTimeLimitSeconds = fftw.planningTimeLimitSeconds();
    fftwRecord.planningBudgetExhausted = fftw.planningBudgetExhausted();
    fftwRecord.wisdomBytes = fftw.wisdomBytes();
    fftwRecord.opaquePlanningBytes = fftw.planningBytes();
    fftwRecord.ledger = fftwLedger(fftw);
    fftwRecord.correctness = {
        metric("full forward versus fixed FFTW ESTIMATE reference", fftwForwardReferenceError),
        metric("full inverse versus fixed FFTW ESTIMATE reference", fftwInverseReferenceError),
        metric("full inverse round trip", fftwRoundTripError)};

    fftwRecord.timings.push_back(series("primitive", "raw FFT", "forward", StageState::executed,
        report.fullRealBytes + report.fullSpectrumBytes,
        measure(warmups, sampleCount, [&] { fftw.forward(input.data(), workingSpectrum.data()); })));
    fftwRecord.timings.push_back(series("primitive", "raw FFT", "inverse", StageState::executed,
        report.fullSpectrumBytes + report.fullRealBytes,
        measure(warmups, sampleCount,
            [&] { std::copy(referenceSpectrum.begin(), referenceSpectrum.end(), inverseSpectrum.begin()); },
            [&] { fftw.inverse(inverseSpectrum.data(), output.data()); })));
    fftwRecord.timings.push_back(series(
        "diagnostic-component", "batch scheduler empty dispatch", "shared",
        fftw.outerWorkers() > 1 ? StageState::executed : StageState::elided, 0,
        fftw.outerWorkers() > 1
            ? measure(warmups, sampleCount, [&] { fftw.executeSchedulerNoop(); })
            : std::vector<double>{}));
    fftwRecord.timings.push_back(series("adapter-component", "representation conversion", "forward", StageState::elided, 0));
    fftwRecord.timings.push_back(series("adapter-component", "representation conversion", "inverse", StageState::elided, 0));
    fftwRecord.timings.push_back(series("adapter-total", "WVM-compatible full-spectrum adapter", "forward", StageState::executed,
        report.fullRealBytes + report.fullSpectrumBytes,
        measure(warmups, sampleCount, [&] { fftw.forward(input.data(), workingSpectrum.data()); })));
    fftwRecord.timings.push_back(series("adapter-total", "WVM-compatible full-spectrum adapter", "inverse", StageState::executed,
        report.fullSpectrumBytes + report.fullRealBytes,
        measure(warmups, sampleCount,
            [&] { std::copy(referenceSpectrum.begin(), referenceSpectrum.end(), inverseSpectrum.begin()); },
            [&] { fftw.inverse(inverseSpectrum.data(), output.data()); })));
    fftwRecord.timings.push_back(series("operator-component", "horizontal retention", "forward", StageState::executed,
        report.fullSpectrumBytes + report.retainedSpectrumBytes,
        measure(warmups, sampleCount, [&] { gatherRetained(workload, modes, referenceSpectrum.data(), retainedWorking.data()); })));
    fftwRecord.timings.push_back(series("operator-component", "horizontal embedding", "inverse", StageState::executed,
        report.retainedSpectrumBytes + report.fullSpectrumBytes,
        measure(warmups, sampleCount, [&] { embedRetained(workload, modes, retainedSpectrum.data(), inverseSpectrum.data()); })));
    fftwRecord.timings.push_back(series("uninstrumented-total", "retained horizontal operator", "forward", StageState::executed,
        report.fullRealBytes + report.fullSpectrumBytes + report.retainedSpectrumBytes,
        measure(warmups, sampleCount, [&] {
            fftw.forward(input.data(), workingSpectrum.data());
            gatherRetained(workload, modes, workingSpectrum.data(), retainedWorking.data());
        })));
    fftwRecord.timings.push_back(series("uninstrumented-total", "retained horizontal operator", "inverse", StageState::executed,
        report.retainedSpectrumBytes + report.fullSpectrumBytes + report.fullRealBytes,
        measure(warmups, sampleCount, [&] {
            embedRetained(workload, modes, retainedSpectrum.data(), inverseSpectrum.data());
            fftw.inverse(inverseSpectrum.data(), output.data());
        })));
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

        interleavedToSplit(referenceSpectrum.size(), referenceSpectrum.data(),
                           referenceSplitReal, referenceSplitImag);
        interleavedToSplit(retainedSpectrum.size(), retainedSpectrum.data(),
                           retainedReferenceReal.data(), retainedReferenceImag.data());

        splitFftw.forwardSplit(input.data(), splitReal, splitImag);
        splitToInterleaved(referenceSpectrum.size(), splitReal, splitImag, workingSpectrum.data());
        const auto splitForwardReferenceError = maximumRelativeError(
            workingSpectrum.data(), referenceSpectrum.data(), referenceSpectrum.size());
        std::copy_n(referenceSplitReal, splitCount, inverseSplitReal);
        std::copy_n(referenceSplitImag, splitCount, inverseSplitImag);
        splitFftw.inverseSplit(inverseSplitReal, inverseSplitImag, output.data());
        const auto splitInverseReferenceError = maximumRelativeError(
            output.data(), referenceOutput.data(), referenceOutput.size());
        const auto splitRoundTripError = maximumRelativeError(
            output.data(), input.data(), input.size(), 1.0 / static_cast<double>(workload.nx * workload.ny));

        gatherRetainedSplit(workload, modes, splitReal, splitImag,
                            retainedSplitReal.data(), retainedSplitImag.data());
        splitToInterleaved(retainedSpectrum.size(), retainedSplitReal.data(), retainedSplitImag.data(), retainedWorking.data());
        const auto splitRetainedForwardError = maximumRelativeError(
            retainedWorking.data(), retainedSpectrum.data(), retainedSpectrum.size());

        embedRetainedSplit(workload, modes, retainedReferenceReal.data(), retainedReferenceImag.data(),
                           inverseSplitReal, inverseSplitImag);
        splitFftw.inverseSplit(inverseSplitReal, inverseSplitImag, output.data());
        embedRetained(workload, modes, retainedSpectrum.data(), inverseSpectrum.data());
        referenceFftw.inverse(inverseSpectrum.data(), fftwOutput.data());
        const auto splitRetainedInverseError = maximumRelativeError(output.data(), fftwOutput.data(), output.size());

        splitToInterleaved(referenceSpectrum.size(), referenceSplitReal, referenceSplitImag, workingSpectrum.data());
        const auto splitToInterleavedError = maximumRelativeError(
            workingSpectrum.data(), referenceSpectrum.data(), referenceSpectrum.size());
        interleavedToSplit(referenceSpectrum.size(), referenceSpectrum.data(), splitReal, splitImag);
        const auto interleavedToSplitRealError = maximumRelativeError(
            splitReal, referenceSplitReal, referenceSpectrum.size());
        const auto interleavedToSplitImagError = maximumRelativeError(
            splitImag, referenceSplitImag, referenceSpectrum.size());

        ProviderRecord splitRecord;
        splitRecord.id = "fftw-split";
        splitRecord.version = splitFftw.version();
        splitRecord.libraryIdentity = splitFftw.libraryIdentity();
        splitRecord.algorithmId = fftwAlgorithmId(splitFftw);
        splitRecord.nativeRepresentationId = "wvm-frequency-major-split-half-spectrum";
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
        splitRecord.ledger.push_back({"exact WVM-order split in-place", StageState::unsupported,
                                      splitFftw.splitInPlaceWvmOrderCapability()});
        splitRecord.correctness = {
            metric("full split forward versus fixed FFTW ESTIMATE reference", splitForwardReferenceError),
            metric("full split inverse versus fixed FFTW ESTIMATE reference", splitInverseReferenceError),
            metric("full split inverse round trip", splitRoundTripError),
            metric("direct split retained forward versus interleaved oracle", splitRetainedForwardError),
            metric("direct split retained inverse versus interleaved oracle", splitRetainedInverseError),
            metric("split-to-interleaved conversion", splitToInterleavedError),
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
                splitToInterleaved(referenceSpectrum.size(), referenceSplitReal, referenceSplitImag, workingSpectrum.data());
            })));
        splitRecord.timings.push_back(series("adapter-component", "interleaved-to-split conversion", "inverse", StageState::executed,
            2 * report.fullSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                interleavedToSplit(referenceSpectrum.size(), referenceSpectrum.data(), inverseSplitReal, inverseSplitImag);
            })));
        splitRecord.timings.push_back(series("adapter-total", "WVM-compatible full-spectrum adapter", "forward", StageState::executed,
            report.fullRealBytes + 3 * report.fullSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                splitFftw.forwardSplit(input.data(), splitReal, splitImag);
                splitToInterleaved(referenceSpectrum.size(), splitReal, splitImag, workingSpectrum.data());
            })));
        splitRecord.timings.push_back(series("adapter-total", "WVM-compatible full-spectrum adapter", "inverse", StageState::executed,
            3 * report.fullSpectrumBytes + report.fullRealBytes,
            measure(warmups, sampleCount, [&] {
                interleavedToSplit(referenceSpectrum.size(), referenceSpectrum.data(), inverseSplitReal, inverseSplitImag);
                splitFftw.inverseSplit(inverseSplitReal, inverseSplitImag, output.data());
            })));
        splitRecord.timings.push_back(series("operator-component", "direct split horizontal retention", "forward", StageState::executed,
            report.fullSpectrumBytes + report.retainedSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                gatherRetainedSplit(workload, modes, referenceSplitReal, referenceSplitImag,
                                    retainedSplitReal.data(), retainedSplitImag.data());
            })));
        splitRecord.timings.push_back(series("operator-component", "direct split horizontal embedding", "inverse", StageState::executed,
            report.retainedSpectrumBytes + report.fullSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                embedRetainedSplit(workload, modes, retainedReferenceReal.data(), retainedReferenceImag.data(),
                                   inverseSplitReal, inverseSplitImag);
            })));
        splitRecord.timings.push_back(series("uninstrumented-total", "persistent split retained horizontal operator", "forward", StageState::executed,
            report.fullRealBytes + report.fullSpectrumBytes + report.retainedSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                splitFftw.forwardSplit(input.data(), splitReal, splitImag);
                gatherRetainedSplit(workload, modes, splitReal, splitImag,
                                    retainedSplitReal.data(), retainedSplitImag.data());
            })));
        splitRecord.timings.push_back(series("uninstrumented-total", "persistent split retained horizontal operator", "inverse", StageState::executed,
            report.retainedSpectrumBytes + report.fullSpectrumBytes + report.fullRealBytes,
            measure(warmups, sampleCount, [&] {
                embedRetainedSplit(workload, modes, retainedReferenceReal.data(), retainedReferenceImag.data(),
                                   inverseSplitReal, inverseSplitImag);
                splitFftw.inverseSplit(inverseSplitReal, inverseSplitImag, output.data());
            })));
        splitRecord.timings.push_back(series("capability", "exact WVM-order split in-place", "shared", StageState::unsupported, 0));
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

    report.status = std::all_of(report.providers.begin(), report.providers.end(), correctnessPassed) ? "passed" : "failed";
    return report;
}

} // namespace skbench
