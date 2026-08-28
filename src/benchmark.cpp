#include "skbench/skbench.hpp"

#include <fftw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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
        ? "two disjoint arrays " + spectrumExtents
        : spectrumExtents;
    const auto aliasing = split
        ? "input and both output arrays are mutually disjoint; exact WVM-order split in-place planning is unsupported"
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
            ? "both split input arrays and output are mutually disjoint; multidimensional FFTW split c2r may destroy both inputs; exact WVM-order split in-place planning is unsupported"
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
        FFTWArray<double> splitReal(workload.spectrumElements());
        FFTWArray<double> splitImag(workload.spectrumElements());
        FFTWArray<double> referenceSplitReal(workload.spectrumElements());
        FFTWArray<double> referenceSplitImag(workload.spectrumElements());
        FFTWArray<double> inverseSplitReal(workload.spectrumElements());
        FFTWArray<double> inverseSplitImag(workload.spectrumElements());
        std::vector<double> retainedSplitReal(retainedSpectrum.size());
        std::vector<double> retainedSplitImag(retainedSpectrum.size());
        std::vector<double> retainedReferenceReal(retainedSpectrum.size());
        std::vector<double> retainedReferenceImag(retainedSpectrum.size());

        interleavedToSplit(referenceSpectrum.size(), referenceSpectrum.data(),
                           referenceSplitReal.data(), referenceSplitImag.data());
        interleavedToSplit(retainedSpectrum.size(), retainedSpectrum.data(),
                           retainedReferenceReal.data(), retainedReferenceImag.data());

        splitFftw.forwardSplit(input.data(), splitReal.data(), splitImag.data());
        splitToInterleaved(referenceSpectrum.size(), splitReal.data(), splitImag.data(), workingSpectrum.data());
        const auto splitForwardReferenceError = maximumRelativeError(
            workingSpectrum.data(), referenceSpectrum.data(), referenceSpectrum.size());
        std::copy(referenceSplitReal.begin(), referenceSplitReal.end(), inverseSplitReal.begin());
        std::copy(referenceSplitImag.begin(), referenceSplitImag.end(), inverseSplitImag.begin());
        splitFftw.inverseSplit(inverseSplitReal.data(), inverseSplitImag.data(), output.data());
        const auto splitInverseReferenceError = maximumRelativeError(
            output.data(), referenceOutput.data(), referenceOutput.size());
        const auto splitRoundTripError = maximumRelativeError(
            output.data(), input.data(), input.size(), 1.0 / static_cast<double>(workload.nx * workload.ny));

        gatherRetainedSplit(workload, modes, splitReal.data(), splitImag.data(),
                            retainedSplitReal.data(), retainedSplitImag.data());
        splitToInterleaved(retainedSpectrum.size(), retainedSplitReal.data(), retainedSplitImag.data(), retainedWorking.data());
        const auto splitRetainedForwardError = maximumRelativeError(
            retainedWorking.data(), retainedSpectrum.data(), retainedSpectrum.size());

        embedRetainedSplit(workload, modes, retainedReferenceReal.data(), retainedReferenceImag.data(),
                           inverseSplitReal.data(), inverseSplitImag.data());
        splitFftw.inverseSplit(inverseSplitReal.data(), inverseSplitImag.data(), output.data());
        embedRetained(workload, modes, retainedSpectrum.data(), inverseSpectrum.data());
        referenceFftw.inverse(inverseSpectrum.data(), fftwOutput.data());
        const auto splitRetainedInverseError = maximumRelativeError(output.data(), fftwOutput.data(), output.size());

        splitToInterleaved(referenceSpectrum.size(), referenceSplitReal.data(), referenceSplitImag.data(), workingSpectrum.data());
        const auto splitToInterleavedError = maximumRelativeError(
            workingSpectrum.data(), referenceSpectrum.data(), referenceSpectrum.size());
        interleavedToSplit(referenceSpectrum.size(), referenceSpectrum.data(), splitReal.data(), splitImag.data());
        const auto interleavedToSplitRealError = maximumRelativeError(
            splitReal.data(), referenceSplitReal.data(), referenceSpectrum.size());
        const auto interleavedToSplitImagError = maximumRelativeError(
            splitImag.data(), referenceSplitImag.data(), referenceSpectrum.size());

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
            measure(warmups, sampleCount, [&] { splitFftw.forwardSplit(input.data(), splitReal.data(), splitImag.data()); })));
        splitRecord.timings.push_back(series("primitive", "raw FFT", "inverse", StageState::executed,
            report.fullSpectrumBytes + report.fullRealBytes,
            measure(warmups, sampleCount,
                [&] {
                    std::copy(referenceSplitReal.begin(), referenceSplitReal.end(), inverseSplitReal.begin());
                    std::copy(referenceSplitImag.begin(), referenceSplitImag.end(), inverseSplitImag.begin());
                },
                [&] { splitFftw.inverseSplit(inverseSplitReal.data(), inverseSplitImag.data(), output.data()); })));
        splitRecord.timings.push_back(series(
            "diagnostic-component", "batch scheduler empty dispatch", "shared",
            splitFftw.outerWorkers() > 1 ? StageState::executed : StageState::elided, 0,
            splitFftw.outerWorkers() > 1
                ? measure(warmups, sampleCount, [&] { splitFftw.executeSchedulerNoop(); })
                : std::vector<double>{}));
        splitRecord.timings.push_back(series("adapter-component", "split-to-interleaved conversion", "forward", StageState::executed,
            2 * report.fullSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                splitToInterleaved(referenceSpectrum.size(), referenceSplitReal.data(), referenceSplitImag.data(), workingSpectrum.data());
            })));
        splitRecord.timings.push_back(series("adapter-component", "interleaved-to-split conversion", "inverse", StageState::executed,
            2 * report.fullSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                interleavedToSplit(referenceSpectrum.size(), referenceSpectrum.data(), inverseSplitReal.data(), inverseSplitImag.data());
            })));
        splitRecord.timings.push_back(series("adapter-total", "WVM-compatible full-spectrum adapter", "forward", StageState::executed,
            report.fullRealBytes + 3 * report.fullSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                splitFftw.forwardSplit(input.data(), splitReal.data(), splitImag.data());
                splitToInterleaved(referenceSpectrum.size(), splitReal.data(), splitImag.data(), workingSpectrum.data());
            })));
        splitRecord.timings.push_back(series("adapter-total", "WVM-compatible full-spectrum adapter", "inverse", StageState::executed,
            3 * report.fullSpectrumBytes + report.fullRealBytes,
            measure(warmups, sampleCount, [&] {
                interleavedToSplit(referenceSpectrum.size(), referenceSpectrum.data(), inverseSplitReal.data(), inverseSplitImag.data());
                splitFftw.inverseSplit(inverseSplitReal.data(), inverseSplitImag.data(), output.data());
            })));
        splitRecord.timings.push_back(series("operator-component", "direct split horizontal retention", "forward", StageState::executed,
            report.fullSpectrumBytes + report.retainedSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                gatherRetainedSplit(workload, modes, referenceSplitReal.data(), referenceSplitImag.data(),
                                    retainedSplitReal.data(), retainedSplitImag.data());
            })));
        splitRecord.timings.push_back(series("operator-component", "direct split horizontal embedding", "inverse", StageState::executed,
            report.retainedSpectrumBytes + report.fullSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                embedRetainedSplit(workload, modes, retainedReferenceReal.data(), retainedReferenceImag.data(),
                                   inverseSplitReal.data(), inverseSplitImag.data());
            })));
        splitRecord.timings.push_back(series("uninstrumented-total", "persistent split retained horizontal operator", "forward", StageState::executed,
            report.fullRealBytes + report.fullSpectrumBytes + report.retainedSpectrumBytes,
            measure(warmups, sampleCount, [&] {
                splitFftw.forwardSplit(input.data(), splitReal.data(), splitImag.data());
                gatherRetainedSplit(workload, modes, splitReal.data(), splitImag.data(),
                                    retainedSplitReal.data(), retainedSplitImag.data());
            })));
        splitRecord.timings.push_back(series("uninstrumented-total", "persistent split retained horizontal operator", "inverse", StageState::executed,
            report.retainedSpectrumBytes + report.fullSpectrumBytes + report.fullRealBytes,
            measure(warmups, sampleCount, [&] {
                embedRetainedSplit(workload, modes, retainedReferenceReal.data(), retainedReferenceImag.data(),
                                   inverseSplitReal.data(), inverseSplitImag.data());
                splitFftw.inverseSplit(inverseSplitReal.data(), inverseSplitImag.data(), output.data());
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
