#include "skbench/skbench.hpp"
#include "constant_stratification_coefficients.hpp"
#include "pointwise_advection.hpp"

#include <fftw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/resource.h>

namespace skbench {
namespace {

using Clock = std::chrono::steady_clock;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double tolerance = 1.0e-12;
constexpr std::string_view auditedWvmCommit =
    "6ad254fb9756ac918bb72e036020d004879df1f2";
constexpr std::string_view auditedKernelSha256 =
    "3e8f618fa813ca274b0c33ed3a34c023fc891ef79b0a062abdc99a967de4a3a9";

std::uint64_t byteCount(std::size_t count, std::size_t elementSize) {
    if (count != 0 && elementSize >
            std::numeric_limits<std::uint64_t>::max() / count) {
        throw std::overflow_error("constant-stratification byte count overflow");
    }
    return static_cast<std::uint64_t>(count) *
        static_cast<std::uint64_t>(elementSize);
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

std::size_t elementCount(std::size_t rows, std::size_t channels,
                         std::size_t nz) {
    if (rows != 0 && channels > std::numeric_limits<std::size_t>::max() / rows)
        throw std::overflow_error("constant-stratification element count overflow");
    const auto rowChannels = rows * channels;
    if (rowChannels != 0 &&
        nz > std::numeric_limits<std::size_t>::max() / rowChannels)
        throw std::overflow_error("constant-stratification element count overflow");
    return rowChannels * nz;
}

template <typename Value>
class FFTWArray {
public:
    explicit FFTWArray(std::size_t count) : count_(count) {
        storage_ = static_cast<Value*>(fftw_malloc(count * sizeof(Value)));
        if (storage_ == nullptr) throw std::bad_alloc();
    }
    ~FFTWArray() { fftw_free(storage_); }
    FFTWArray(const FFTWArray&) = delete;
    FFTWArray& operator=(const FFTWArray&) = delete;
    Value* data() noexcept { return storage_; }
    const Value* data() const noexcept { return storage_; }
    std::size_t size() const noexcept { return count_; }
private:
    Value* storage_ = nullptr;
    std::size_t count_ = 0;
};

unsigned planningFlags(FFTWPlanningMode mode) {
    switch (mode) {
        case FFTWPlanningMode::estimate: return FFTW_ESTIMATE | FFTW_UNALIGNED;
        case FFTWPlanningMode::measure: return FFTW_MEASURE | FFTW_UNALIGNED;
        case FFTWPlanningMode::patient: return FFTW_PATIENT | FFTW_UNALIGNED;
        case FFTWPlanningMode::exhaustive:
            return FFTW_EXHAUSTIVE | FFTW_UNALIGNED;
    }
    throw std::invalid_argument("Unknown FFTW planning mode.");
}

class R2RPlan {
public:
    R2RPlan(std::size_t rows, std::size_t nz, std::size_t storageChannels,
            std::size_t firstChannel, std::size_t transformedChannels,
            bool sine, FFTWPlanningMode planningMode,
            std::size_t internalWorkers) {
        if (rows == 0 || nz < 3 || storageChannels == 0 ||
            transformedChannels == 0 ||
            firstChannel + transformedChannels > storageChannels)
            throw std::invalid_argument("Invalid constant-stratification R2R plan shape.");
        if (fftw_init_threads() == 0)
            throw std::runtime_error("FFTW thread initialization failed.");
        fftw_plan_with_nthreads(static_cast<int>(internalWorkers));
        FFTWArray<Complex> surrogate(elementCount(rows, storageChannels, nz));
        const auto offset = firstChannel * nz + (sine ? 1U : 0U);
        auto* start = reinterpret_cast<double*>(surrogate.data() + offset);
        fftw_iodim64 transform{
            static_cast<ptrdiff_t>(sine ? nz - 2 : nz), 2, 2};
        fftw_iodim64 batches[] = {
            {2, 1, 1},
            {static_cast<ptrdiff_t>(rows),
             static_cast<ptrdiff_t>(2 * nz * storageChannels),
             static_cast<ptrdiff_t>(2 * nz * storageChannels)},
            {static_cast<ptrdiff_t>(transformedChannels),
             static_cast<ptrdiff_t>(2 * nz),
             static_cast<ptrdiff_t>(2 * nz)}};
        const fftw_r2r_kind kind = sine ? FFTW_RODFT00 : FFTW_REDFT00;
        const auto started = Clock::now();
        plan_ = fftw_plan_guru64_r2r(
            1, &transform, 3, batches, start, start, &kind,
            planningFlags(planningMode));
        planningSeconds_ =
            std::chrono::duration<double>(Clock::now() - started).count();
        if (plan_ == nullptr)
            throw std::runtime_error("FFTW could not create a WVM-compatible R2R plan.");
        offset_ = offset;
    }

    ~R2RPlan() { if (plan_ != nullptr) fftw_destroy_plan(plan_); }
    R2RPlan(const R2RPlan&) = delete;
    R2RPlan& operator=(const R2RPlan&) = delete;

    void execute(Complex* storage) const {
        auto* start = reinterpret_cast<double*>(storage + offset_);
        fftw_execute_r2r(plan_, start, start);
    }
    double planningSeconds() const noexcept { return planningSeconds_; }

private:
    fftw_plan plan_ = nullptr;
    std::size_t offset_ = 0;
    double planningSeconds_ = 0.0;
};

struct ProductionPlans {
    R2RPlan dct2Storage3;
    R2RPlan dst1Storage3;
    R2RPlan dst2Storage3;
    R2RPlan dct1Storage3;
    R2RPlan dct1Storage1;
    R2RPlan dst1Storage1;

    ProductionPlans(std::size_t rows, std::size_t nz,
                    FFTWPlanningMode planningMode,
                    std::size_t internalWorkers)
        : dct2Storage3(rows, nz, 3, 0, 2, false, planningMode,
                      internalWorkers),
          dst1Storage3(rows, nz, 3, 2, 1, true, planningMode,
                      internalWorkers),
          dst2Storage3(rows, nz, 3, 0, 2, true, planningMode,
                      internalWorkers),
          dct1Storage3(rows, nz, 3, 2, 1, false, planningMode,
                      internalWorkers),
          dct1Storage1(rows, nz, 1, 0, 1, false, planningMode,
                      internalWorkers),
          dst1Storage1(rows, nz, 1, 0, 1, true, planningMode,
                      internalWorkers) {}

    double planningSeconds() const noexcept {
        return dct2Storage3.planningSeconds() +
            dst1Storage3.planningSeconds() +
            dst2Storage3.planningSeconds() +
            dct1Storage3.planningSeconds() +
            dct1Storage1.planningSeconds() +
            dst1Storage1.planningSeconds();
    }
};

std::size_t index(std::size_t row, std::size_t channel, std::size_t z,
                  std::size_t channels, std::size_t nz) {
    return z + nz * channel + nz * channels * row;
}

Complex scaled(Complex value, double factor) noexcept {
    return {factor * value.real, factor * value.imag};
}

Complex conjugated(Complex value) noexcept {
    return {value.real, -value.imag};
}

void normalizeForwardDct(Complex* data, std::size_t rows, std::size_t nz,
                         std::size_t storageChannels,
                         std::size_t firstChannel, std::size_t channels) {
    const double scale = 1.0 / static_cast<double>(nz - 1);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t channel = firstChannel;
             channel < firstChannel + channels; ++channel) {
            const auto base = index(row, channel, 0, storageChannels, nz);
            for (std::size_t z = 0; z < nz; ++z)
                data[base + z] = scaled(data[base + z], scale);
            data[base + nz - 1] = scaled(data[base + nz - 1], 0.5);
        }
    }
}

void normalizeForwardDst(Complex* data, std::size_t rows, std::size_t nz,
                         std::size_t storageChannels,
                         std::size_t firstChannel, std::size_t channels) {
    const double scale = 1.0 / static_cast<double>(nz - 1);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t channel = firstChannel;
             channel < firstChannel + channels; ++channel) {
            const auto base = index(row, channel, 0, storageChannels, nz);
            data[base] = {};
            for (std::size_t z = 1; z + 1 < nz; ++z)
                data[base + z] = scaled(data[base + z], scale);
            data[base + nz - 1] = {};
        }
    }
}

Complex deterministicValue(std::size_t row, std::size_t channel,
                           std::size_t z) {
    const double argument = 0.017 * static_cast<double>(1 + row % 97) +
        0.113 * static_cast<double>(1 + channel) +
        0.071 * static_cast<double>(1 + z);
    return {std::sin(argument), std::cos(1.7 * argument)};
}

void fillFull(Complex* data, std::size_t rows, std::size_t channels,
              std::size_t nz) {
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t channel = 0; channel < channels; ++channel)
            for (std::size_t z = 0; z < nz; ++z)
                data[index(row, channel, z, channels, nz)] =
                    deterministicValue(row, channel, z);
}

void fillCompact(Complex* data, const std::vector<RetainedMode>& modes,
                 std::size_t nxHalf, std::size_t channels, std::size_t nz) {
    for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        const auto& mode = modes[modeIndex];
        const auto row = mode.storedKx + nxHalf * mode.storedKy;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            for (std::size_t z = 0; z < nz; ++z) {
                auto value = deterministicValue(row, channel, z);
                if (mode.conjugatesStoredValue) value = conjugated(value);
                data[index(modeIndex, channel, z, channels, nz)] = value;
            }
        }
    }
}

std::vector<Complex> rowValues(const Complex* data, std::size_t row,
                               std::size_t channel, std::size_t channels,
                               std::size_t nz) {
    const auto base = index(row, channel, 0, channels, nz);
    return {data + base, data + base + nz};
}

std::vector<Complex> directForwardDct(const std::vector<Complex>& input) {
    const auto nz = input.size();
    std::vector<Complex> output(nz);
    for (std::size_t j = 0; j < nz; ++j) {
        Complex value = input.front();
        const double endpointSign = j % 2 == 0 ? 1.0 : -1.0;
        value.real += endpointSign * input.back().real;
        value.imag += endpointSign * input.back().imag;
        for (std::size_t z = 1; z + 1 < nz; ++z) {
            const double factor = 2.0 * std::cos(
                pi * static_cast<double>(j * z) /
                static_cast<double>(nz - 1));
            value.real += factor * input[z].real;
            value.imag += factor * input[z].imag;
        }
        value = scaled(value, 1.0 / static_cast<double>(nz - 1));
        if (j + 1 == nz) value = scaled(value, 0.5);
        output[j] = value;
    }
    return output;
}

std::vector<Complex> directForwardDst(const std::vector<Complex>& input) {
    const auto nz = input.size();
    std::vector<Complex> output(nz);
    for (std::size_t j = 1; j + 1 < nz; ++j) {
        Complex value;
        for (std::size_t z = 1; z + 1 < nz; ++z) {
            const double factor = 2.0 * std::sin(
                pi * static_cast<double>(j * z) /
                static_cast<double>(nz - 1));
            value.real += factor * input[z].real;
            value.imag += factor * input[z].imag;
        }
        output[j] = scaled(value, 1.0 / static_cast<double>(nz - 1));
    }
    return output;
}

void executeCompleteVerticalGraph(ProductionPlans& plans, Complex* arena3,
                                  Complex* arena1, std::size_t rows,
                                  std::size_t nz) {
    // Production nonlinearFlux: one velocity reconstruction plus two
    // cosine-family and two sine-family derivative reconstructions.
    for (std::size_t group = 0; group < 3; ++group) {
        plans.dct2Storage3.execute(arena3);
        plans.dst1Storage3.execute(arena3);
    }
    for (std::size_t group = 0; group < 2; ++group) {
        plans.dst2Storage3.execute(arena3);
        plans.dct1Storage3.execute(arena3);
    }
    // Four projected flux targets: two cosine and two sine families.
    for (std::size_t group = 0; group < 2; ++group) {
        plans.dct1Storage1.execute(arena1);
        normalizeForwardDct(arena1, rows, nz, 1, 0, 1);
    }
    for (std::size_t group = 0; group < 2; ++group) {
        plans.dst1Storage1.execute(arena1);
        normalizeForwardDst(arena1, rows, nz, 1, 0, 1);
    }
}

template <typename Action>
std::vector<double> measure(std::size_t warmups, std::size_t samples,
                            Action action) {
    for (std::size_t index = 0; index < warmups; ++index) action();
    std::vector<double> result;
    result.reserve(samples);
    for (std::size_t index = 0; index < samples; ++index) {
        const auto start = Clock::now();
        action();
        result.push_back(
            std::chrono::duration<double>(Clock::now() - start).count());
    }
    return result;
}

TimingSeries timing(std::string scope, std::string stage,
                    std::string direction, std::uint64_t bytesMoved,
                    std::vector<double> samples) {
    return {std::move(scope), std::move(stage), std::move(direction),
            StageState::executed, bytesMoved, std::move(samples)};
}

CorrectnessMetric correctness(std::string name, const Complex* actual,
                              const Complex* expected, std::size_t count) {
    const auto maximum = maximumRelativeError(actual, expected, count);
    const auto l2 = relativeL2Error(actual, expected, count);
    return {std::move(name), maximum, tolerance,
            maximum <= tolerance && l2 <= tolerance, l2};
}

CorrectnessMetric compactFullCorrectness(
    std::string name, const Complex* compact, const Complex* full,
    const std::vector<RetainedMode>& modes, std::size_t nxHalf,
    std::size_t channels, std::size_t nz) {
    long double squaredError = 0.0;
    long double squaredReference = 0.0;
    double maximumError = 0.0;
    double maximumReference = 0.0;
    for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        const auto& mode = modes[modeIndex];
        const auto fullRow = mode.storedKx + nxHalf * mode.storedKy;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            for (std::size_t z = 0; z < nz; ++z) {
                auto reference =
                    full[index(fullRow, channel, z, channels, nz)];
                if (mode.conjugatesStoredValue)
                    reference = conjugated(reference);
                const auto actual =
                    compact[index(modeIndex, channel, z, channels, nz)];
                const long double realError = actual.real - reference.real;
                const long double imagError = actual.imag - reference.imag;
                const double error = std::hypot(
                    static_cast<double>(realError),
                    static_cast<double>(imagError));
                const double referenceMagnitude =
                    std::hypot(reference.real, reference.imag);
                maximumError = std::max(maximumError, error);
                maximumReference =
                    std::max(maximumReference, referenceMagnitude);
                squaredError += realError * realError + imagError * imagError;
                squaredReference +=
                    static_cast<long double>(reference.real) * reference.real +
                    static_cast<long double>(reference.imag) * reference.imag;
            }
        }
    }
    const double maximum = maximumError / std::max(maximumReference, 1.0);
    const double l2 = squaredReference == 0.0
        ? std::sqrt(static_cast<double>(squaredError))
        : std::sqrt(static_cast<double>(squaredError / squaredReference));
    return {std::move(name), maximum, tolerance,
            maximum <= tolerance && l2 <= tolerance, l2};
}

std::string timestampId(std::string value) {
    value.erase(std::remove(value.begin(), value.end(), '-'), value.end());
    value.erase(std::remove(value.begin(), value.end(), ':'), value.end());
    const auto marker = value.find('.');
    if (marker != std::string::npos) value.erase(marker);
    return value;
}

ProviderRecord providerRecord(
    std::string id, std::string algorithmId, std::string representation,
    std::size_t rows, std::size_t nz, std::size_t internalWorkers,
    FFTWPlanningMode planningMode, double planningSeconds) {
    ProviderRecord record;
    record.id = std::move(id);
    record.version = fftw_version;
    record.libraryIdentity = "pinned FFTW 3.3.11 threads";
    record.algorithmId = std::move(algorithmId);
    record.nativeRepresentationId = std::move(representation);
    record.modeOrderId = record.nativeRepresentationId.find("compact") !=
            std::string::npos
        ? "logical-radial-retained-mode-order"
        : "wvm-half-spectrum-row-order";
    record.schedulingId = "fftw-internal-" +
        std::to_string(internalWorkers);
    record.sourceIdentity =
        "JeffreyEarly/wave-vortex-model@" + std::string(auditedWvmCommit) +
        ":CompiledKernel/src/WVTransformConstantStratificationKernel.cpp";
    record.sourceSha256 = std::string(auditedKernelSha256);
    record.configureFlags =
        "FFTW 3.3.11 --enable-neon --enable-threads";
    record.compilerFlags = "-O3 -mcpu=native";
    record.planningConfiguration = std::string(fftwPlanningModeName(planningMode)) +
        " | FFTW_UNALIGNED; six guru64 in-place REDFT00/RODFT00 plans";
    record.workers = internalWorkers;
    record.internalWorkers = internalWorkers;
    record.outerWorkers = 1;
    record.planningSeconds = planningSeconds;
    record.explicitPersistentBytes = static_cast<std::size_t>(
        byteCount(elementCount(rows, 4, nz), sizeof(Complex)));
    record.algorithmResidentBytes = record.explicitPersistentBytes;
    record.estimatedProcessPeakBytes = record.explicitPersistentBytes;
    record.opaqueProviderMemory = true;
    record.execution.forward = {
        "in-place", "in-place", true, false, true, false, false,
        record.nativeRepresentationId, record.nativeRepresentationId,
        record.nativeRepresentationId, record.nativeRepresentationId,
        "[rows,channels,Nz] with Nz adjacent",
        "complex: z=1, channel=Nz, row=Nz*channels; real/imag stride=1 scalar",
        0, 16, "exact in-place alias", 0, false};
    record.execution.inverse = record.execution.forward;
    record.ledger = {
        {"setup/planning", StageState::setupOnly,
         "six reusable FFTW guru64 REDFT00/RODFT00 plans"},
        {"raw inverse vertical transforms", StageState::executed,
         "15 complex channels: velocity plus four derivative reconstructions"},
        {"raw forward vertical transforms", StageState::executed,
         "four complex flux-target projections"},
        {"forward normalization", StageState::executed,
         "exact WVM type-I endpoint and 1/(Nz-1) scaling"},
        {"steady-state application allocation", StageState::executed,
         "the benchmark action allocates no application storage; FFTW-owned execution allocation remains opaque"},
        {"horizontal retained transforms", StageState::unsupported,
         "isolated vertical call-graph benchmark; composed horizontal timing is deferred"},
        {"coefficient assembly and pointwise products", StageState::unsupported,
         "not part of this attributable vertical component benchmark"},
        {"complete nonlinear flux", StageState::unsupported,
         "must be measured in the production WVM integration experiment"}};
    return record;
}

bool passed(const ProviderRecord& record) {
    return std::all_of(record.correctness.begin(), record.correctness.end(),
        [](const CorrectnessMetric& metric) { return metric.passed; });
}

} // namespace

BenchmarkReport runConstantStratificationVerticalBenchmark(
    const RunOptions& options) {
    auto selected = profileNamed(options.profile);
    auto workload = selected.workload;
    workload.fields = 4;
    if (workload.nx != workload.ny || workload.nx % 2 != 0 ||
        workload.nz < 3)
        throw std::invalid_argument(
            "constant-stratification vertical benchmark requires an even square horizontal grid and Nz >= 3.");
    const auto warmups = options.warmups == 0 ? selected.warmups : options.warmups;
    const auto samples = options.samples == 0 ? selected.samples : options.samples;
    const auto internalWorkers = options.fftwInternalWorkers == 0
        ? std::max<std::size_t>(1, selected.defaultWorkers)
        : options.fftwInternalWorkers;
    if (internalWorkers > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("FFTW internal worker count is too large.");
    const auto planningMode = fftwPlanningModeNamed(options.fftwPlanning);
    const auto modes = retainedHorizontalModes(workload);
    const auto fullRows = workload.halfRows();
    const auto compactRows = modes.size();
    const auto nz = workload.nz;

    const auto allocationStart = Clock::now();
    FFTWArray<Complex> full3(elementCount(fullRows, 3, nz));
    FFTWArray<Complex> full1(elementCount(fullRows, 1, nz));
    FFTWArray<Complex> compact3(elementCount(compactRows, 3, nz));
    FFTWArray<Complex> compact1(elementCount(compactRows, 1, nz));
    const auto allocationSeconds =
        std::chrono::duration<double>(Clock::now() - allocationStart).count();

    ProductionPlans fullPlans(fullRows, nz, planningMode, internalWorkers);
    ProductionPlans compactPlans(
        compactRows, nz, planningMode, internalWorkers);

    BenchmarkReport report;
    report.environment = environmentRecord();
    report.runId = timestampId(report.environment.timestampUtc) +
        "-issue20-constant-stratification-vertical-" +
        report.environment.hostname;
    report.profile = options.profile;
    report.seed = options.seed;
    report.warmups = warmups;
    report.samples = samples;
    report.workload = workload;
    report.retainedHorizontalModeCount = compactRows;
    report.retainedModeOrderHash = modeOrderHash(modes);
    report.wvmFullSpectrumOrderHash = wvmSpectrumOrderHash(workload);
    report.fullRealBytes = byteCount(workload.realElements(), sizeof(double));
    report.fullSpectrumBytes =
        byteCount(elementCount(fullRows, 4, nz), sizeof(Complex));
    report.retainedSpectrumBytes =
        byteCount(elementCount(compactRows, 4, nz), sizeof(Complex));
    report.modalSpectrumBytes = byteCount(
        compactRows * workload.retainedVerticalModes() * workload.fields,
        sizeof(Complex));
    report.verticalMatrixFamilyId =
        "wvm-constant-stratification-fftw-type1-v1";

    fillFull(full1.data(), fullRows, 1, nz);
    fillCompact(compact1.data(), modes, workload.nxHalf(), 1, nz);
    const auto dctInput = rowValues(compact1.data(), 0, 0, 1, nz);
    const auto dctOracle = directForwardDct(dctInput);
    compactPlans.dct1Storage1.execute(compact1.data());
    normalizeForwardDct(compact1.data(), compactRows, nz, 1, 0, 1);
    const auto dctActual = rowValues(compact1.data(), 0, 0, 1, nz);

    fillCompact(compact1.data(), modes, workload.nxHalf(), 1, nz);
    const auto dstInput = rowValues(compact1.data(), 0, 0, 1, nz);
    const auto dstOracle = directForwardDst(dstInput);
    compactPlans.dst1Storage1.execute(compact1.data());
    normalizeForwardDst(compact1.data(), compactRows, nz, 1, 0, 1);
    const auto dstActual = rowValues(compact1.data(), 0, 0, 1, nz);

    fillFull(full3.data(), fullRows, 3, nz);
    fillFull(full1.data(), fullRows, 1, nz);
    fillCompact(compact3.data(), modes, workload.nxHalf(), 3, nz);
    fillCompact(compact1.data(), modes, workload.nxHalf(), 1, nz);
    executeCompleteVerticalGraph(
        fullPlans, full3.data(), full1.data(), fullRows, nz);
    executeCompleteVerticalGraph(
        compactPlans, compact3.data(), compact1.data(), compactRows, nz);
    const auto compactThreeError = compactFullCorrectness(
        "retained three-channel vertical call schedule versus full-half control",
        compact3.data(), full3.data(), modes, workload.nxHalf(), 3, nz);
    const auto compactOneError = compactFullCorrectness(
        "retained one-channel projection schedule versus full-half control",
        compact1.data(), full1.data(), modes, workload.nxHalf(), 1, nz);

    auto makeRecord = [&](bool compact) {
        const auto rows = compact ? compactRows : fullRows;
        auto& plans = compact ? compactPlans : fullPlans;
        auto* arena3 = compact ? compact3.data() : full3.data();
        auto* arena1 = compact ? compact1.data() : full1.data();
        fillFull(full3.data(), fullRows, 3, nz);
        fillFull(full1.data(), fullRows, 1, nz);
        fillCompact(compact3.data(), modes, workload.nxHalf(), 3, nz);
        fillCompact(compact1.data(), modes, workload.nxHalf(), 1, nz);
        auto record = providerRecord(
            compact ? "fftw-wvm-type1-retained-compact" :
                      "fftw-wvm-type1-full-half",
            compact ? "retained-rows-production-type1-call-graph" :
                      "wvm-full-half-production-type1-call-graph",
            compact ? "retained-radial-interleaved-complex-z-adjacent" :
                      "wvm-full-half-interleaved-complex-z-adjacent",
            rows, nz, internalWorkers, planningMode,
            plans.planningSeconds());
        record.allocationSeconds = allocationSeconds / 2.0;
        const auto oneChannelBytes =
            byteCount(elementCount(rows, 1, nz), sizeof(Complex));
        record.timings.push_back(timing(
            "primitive", "raw DCT-I one complex channel", "inverse",
            2 * oneChannelBytes,
            measure(warmups, samples,
                [&] { plans.dct1Storage1.execute(arena1); })));
        record.timings.push_back(timing(
            "primitive", "raw DST-I one complex interior channel", "inverse",
            2 * oneChannelBytes,
            measure(warmups, samples,
                [&] { plans.dst1Storage1.execute(arena1); })));
        record.timings.push_back(timing(
            "primitive", "raw DCT-I one complex channel", "forward",
            2 * oneChannelBytes,
            measure(warmups, samples,
                [&] { plans.dct1Storage1.execute(arena1); })));
        record.timings.push_back(timing(
            "primitive", "raw DST-I one complex interior channel", "forward",
            2 * oneChannelBytes,
            measure(warmups, samples,
                [&] { plans.dst1Storage1.execute(arena1); })));
        record.timings.push_back(timing(
            "adapter-component", "WVM forward DCT-I normalization", "forward",
            2 * oneChannelBytes,
            measure(warmups, samples, [&] {
                normalizeForwardDct(arena1, rows, nz, 1, 0, 1);
            })));
        record.timings.push_back(timing(
            "adapter-component", "WVM forward DST-I normalization and endpoints",
            "forward", 2 * oneChannelBytes,
            measure(warmups, samples, [&] {
                normalizeForwardDst(arena1, rows, nz, 1, 0, 1);
            })));
        record.timings.push_back(timing(
            "uninstrumented-total",
            "production nonlinear-flux vertical transform schedule (15 inverse + 4 forward complex channels)",
            "complete", 38 * oneChannelBytes,
            measure(warmups, samples, [&] {
                executeCompleteVerticalGraph(
                    plans, arena3, arena1, rows, nz);
            })));
        record.correctness.push_back(correctness(
            "WVM normalized DCT-I versus direct oracle", dctActual.data(),
            dctOracle.data(), nz));
        record.correctness.push_back(correctness(
            "WVM normalized DST-I versus direct oracle", dstActual.data(),
            dstOracle.data(), nz));
        if (compact) {
            record.correctness.push_back(compactThreeError);
            record.correctness.push_back(compactOneError);
        }
        return record;
    };

    report.providers.push_back(makeRecord(false));
    report.providers.push_back(makeRecord(true));
    const auto observedHighWater = processHighWaterBytes();
    for (auto& provider : report.providers)
        provider.observedProcessHighWaterBytes = observedHighWater;
    report.status = std::all_of(
        report.providers.begin(), report.providers.end(), passed)
        ? "passed" : "failed";
    return report;
}

namespace {

class SplitR2RPlan {
public:
    SplitR2RPlan(std::size_t rows, std::size_t nz,
                 std::size_t storageChannels, std::size_t firstChannel,
                 std::size_t transformedChannels, bool sine,
                 FFTWPlanningMode planningMode, std::size_t internalWorkers,
                 std::size_t componentStride) {
        if (rows == 0 || nz < 3 || storageChannels == 0 ||
            transformedChannels == 0 ||
            firstChannel + transformedChannels > storageChannels ||
            componentStride < rows * storageChannels * nz) {
            throw std::invalid_argument(
                "Invalid split constant-stratification R2R plan shape.");
        }
        if (fftw_init_threads() == 0)
            throw std::runtime_error("FFTW thread initialization failed.");
        fftw_plan_with_nthreads(static_cast<int>(internalWorkers));
        FFTWArray<double> surrogate(2 * componentStride);
        offset_ = firstChannel * nz + (sine ? 1U : 0U);
        auto* start = surrogate.data() + offset_;
        fftw_iodim64 transform{
            static_cast<ptrdiff_t>(sine ? nz - 2 : nz), 1, 1};
        fftw_iodim64 batches[] = {
            {2, static_cast<ptrdiff_t>(componentStride),
             static_cast<ptrdiff_t>(componentStride)},
            {static_cast<ptrdiff_t>(rows),
             static_cast<ptrdiff_t>(nz * storageChannels),
             static_cast<ptrdiff_t>(nz * storageChannels)},
            {static_cast<ptrdiff_t>(transformedChannels),
             static_cast<ptrdiff_t>(nz), static_cast<ptrdiff_t>(nz)}};
        const fftw_r2r_kind kind = sine ? FFTW_RODFT00 : FFTW_REDFT00;
        const auto started = Clock::now();
        plan_ = fftw_plan_guru64_r2r(
            1, &transform, 3, batches, start, start, &kind,
            planningFlags(planningMode));
        planningSeconds_ =
            std::chrono::duration<double>(Clock::now() - started).count();
        if (plan_ == nullptr)
            throw std::runtime_error(
                "FFTW could not create a split type-I plan.");
    }

    ~SplitR2RPlan() { if (plan_ != nullptr) fftw_destroy_plan(plan_); }
    SplitR2RPlan(const SplitR2RPlan&) = delete;
    SplitR2RPlan& operator=(const SplitR2RPlan&) = delete;

    void execute(double* storage) const {
        fftw_execute_r2r(plan_, storage + offset_, storage + offset_);
    }
    double planningSeconds() const noexcept { return planningSeconds_; }

private:
    fftw_plan plan_ = nullptr;
    std::size_t offset_ = 0;
    double planningSeconds_ = 0.0;
};

struct SplitProductionPlans {
    SplitR2RPlan dct2Storage3;
    SplitR2RPlan dst1Storage3;
    SplitR2RPlan dst2Storage3;
    SplitR2RPlan dct1Storage3;
    SplitR2RPlan dct1Storage1;
    SplitR2RPlan dst1Storage1;

    SplitProductionPlans(std::size_t rows, std::size_t nz,
                         FFTWPlanningMode planningMode,
                         std::size_t internalWorkers,
                         std::size_t componentStride)
        : dct2Storage3(rows, nz, 3, 0, 2, false, planningMode,
                      internalWorkers, componentStride),
          dst1Storage3(rows, nz, 3, 2, 1, true, planningMode,
                      internalWorkers, componentStride),
          dst2Storage3(rows, nz, 3, 0, 2, true, planningMode,
                      internalWorkers, componentStride),
          dct1Storage3(rows, nz, 3, 2, 1, false, planningMode,
                      internalWorkers, componentStride),
          dct1Storage1(rows, nz, 1, 0, 1, false, planningMode,
                      internalWorkers, componentStride),
          dst1Storage1(rows, nz, 1, 0, 1, true, planningMode,
                      internalWorkers, componentStride) {}

    double planningSeconds() const noexcept {
        return dct2Storage3.planningSeconds() +
            dst1Storage3.planningSeconds() +
            dst2Storage3.planningSeconds() +
            dct1Storage3.planningSeconds() +
            dct1Storage1.planningSeconds() +
            dst1Storage1.planningSeconds();
    }
};

void normalizeForwardSplitDct(
    double* storage, std::size_t componentStride, std::size_t rows,
    std::size_t nz, std::size_t storageChannels,
    std::size_t firstChannel, std::size_t channels) {
    const double scale = 1.0 / static_cast<double>(nz - 1);
    for (std::size_t component = 0; component < 2; ++component) {
        auto* values = storage + component * componentStride;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t channel = firstChannel;
                 channel < firstChannel + channels; ++channel) {
                const auto base =
                    nz * channel + nz * storageChannels * row;
                for (std::size_t z = 0; z < nz; ++z)
                    values[base + z] *= scale;
                values[base + nz - 1] *= 0.5;
            }
        }
    }
}

void normalizeForwardSplitDst(
    double* storage, std::size_t componentStride, std::size_t rows,
    std::size_t nz, std::size_t storageChannels,
    std::size_t firstChannel, std::size_t channels) {
    const double scale = 1.0 / static_cast<double>(nz - 1);
    for (std::size_t component = 0; component < 2; ++component) {
        auto* values = storage + component * componentStride;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t channel = firstChannel;
                 channel < firstChannel + channels; ++channel) {
                const auto base =
                    nz * channel + nz * storageChannels * row;
                values[base] = 0.0;
                for (std::size_t z = 1; z + 1 < nz; ++z)
                    values[base + z] *= scale;
                values[base + nz - 1] = 0.0;
            }
        }
    }
}

Complex addComplex(Complex first, Complex second) noexcept {
    return {first.real + second.real, first.imag + second.imag};
}

Complex multiplyImaginary(Complex value, double factor) noexcept {
    return {-factor * value.imag, factor * value.real};
}

Complex baseCoefficient(const Workload& workload,
                        const std::vector<RetainedMode>& modes,
                        std::size_t mode, std::size_t j,
                        std::size_t field) {
    const auto& retained = modes[mode];
    const auto storedRow =
        retained.storedKx + workload.nxHalf() * retained.storedKy;
    auto value = deterministicValue(storedRow + 13 * field, field, j);
    value = scaled(value, 2.0e-6 / static_cast<double>(1 + j));
    if (retained.conjugatesStoredValue) value = conjugated(value);
    if (retained.k == 0 && retained.l == 0) value.imag = 0.0;
    return value;
}

std::vector<Complex> makeCoefficientFixture(
    const Workload& workload, const std::vector<RetainedMode>& modes) {
    const auto nj = workload.retainedVerticalModes();
    std::vector<Complex> result(modes.size() * nj * 3);
    auto coefficientWorkload = workload;
    coefficientWorkload.fields = 3;
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
        for (std::size_t field = 0; field < 3; ++field) {
            for (std::size_t j = 0; j < nj; ++j) {
                result[modalSpectrumIndex(
                    coefficientWorkload, mode, j, field)] =
                    baseCoefficient(workload, modes, mode, j, field);
            }
        }
    }
    return result;
}

bool sineChannel(std::size_t group, std::size_t channel) noexcept {
    if (group <= 2) return channel == 2;
    return channel < 2;
}

Complex groupValue(const Workload& coefficientWorkload,
                   const std::vector<Complex>& coefficients,
                   const RetainedMode& mode, std::size_t modeIndex,
                   std::size_t j, std::size_t group,
                   std::size_t channel) {
    const auto a = coefficients[modalSpectrumIndex(
        coefficientWorkload, modeIndex, j, 0)];
    const auto b = coefficients[modalSpectrumIndex(
        coefficientWorkload, modeIndex, j, 1)];
    const auto c = coefficients[modalSpectrumIndex(
        coefficientWorkload, modeIndex, j, 2)];
    Complex value;
    if (group == 0) {
        if (channel == 0)
            value = addComplex(a, scaled(b, 0.25));
        else if (channel == 1)
            value = addComplex(b, scaled(c, -0.20));
        else
            value = addComplex(scaled(c, 0.50), scaled(a, 0.10));
    } else {
        const auto target = group - 1;
        auto q = addComplex(
            scaled(a, 0.45 + 0.05 * static_cast<double>(target)),
            scaled(b, -0.20 + 0.03 * static_cast<double>(target)));
        q = addComplex(q, scaled(
            c, 0.30 - 0.02 * static_cast<double>(target)));
        const auto horizontalScale =
            1.0 / static_cast<double>(coefficientWorkload.nx);
        if (channel == 0)
            value = multiplyImaginary(
                q, horizontalScale * static_cast<double>(mode.k));
        else if (channel == 1)
            value = multiplyImaginary(
                q, horizontalScale * static_cast<double>(mode.l));
        else
            value = scaled(
                q, static_cast<double>(j) /
                    static_cast<double>(coefficientWorkload.nz - 1));
    }
    if (sineChannel(group, channel) &&
        (j == 0 || j + 1 == coefficientWorkload.nz))
        return {};
    return scaled(value, 0.5);
}

void assembleCompactGroup(
    double* real, double* imaginary, const Workload& coefficientWorkload,
    const Workload& threeWorkload, const std::vector<RetainedMode>& modes,
    const std::vector<Complex>& coefficients, std::size_t group) {
    const auto activeElements = modes.size() * threeWorkload.planes();
    std::fill_n(real, activeElements, 0.0);
    std::fill_n(imaginary, activeElements, 0.0);
    const auto nj = coefficientWorkload.retainedVerticalModes();
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
            for (std::size_t j = 0; j < nj; ++j) {
                const auto value = groupValue(
                    coefficientWorkload, coefficients, modes[mode], mode, j,
                    group, channel);
                const auto destination = retainedSpectrumIndex(
                    threeWorkload, mode, j, channel);
                real[destination] = value.real;
                imaginary[destination] = value.imag;
            }
        }
    }
}

void assembleFullGroup(
    Complex* full, const Workload& coefficientWorkload,
    const Workload& threeWorkload, const std::vector<RetainedMode>& modes,
    const std::vector<Complex>& coefficients, std::size_t group) {
    std::fill_n(full, threeWorkload.spectrumElements(), Complex{});
    const auto nj = coefficientWorkload.retainedVerticalModes();
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
        const auto& retained = modes[mode];
        for (std::size_t channel = 0; channel < 3; ++channel) {
            for (std::size_t j = 0; j < nj; ++j) {
                const auto logical = groupValue(
                    coefficientWorkload, coefficients, retained, mode, j,
                    group, channel);
                const auto stored = retained.conjugatesStoredValue
                    ? conjugated(logical) : logical;
                full[wvmSpectrumIndex(
                    threeWorkload, retained.storedKx, retained.storedKy, j,
                    channel)] = stored;
                if (retained.storedKx == 0 && retained.storedKy != 0 &&
                    2 * retained.storedKy != threeWorkload.ny) {
                    const auto conjugateKy =
                        (threeWorkload.ny - retained.storedKy) %
                        threeWorkload.ny;
                    full[wvmSpectrumIndex(
                        threeWorkload, 0, conjugateKy, j, channel)] =
                        conjugated(stored);
                }
            }
        }
    }
}

void executeInverseGroup(ProductionPlans& plans, Complex* storage,
                         std::size_t group) {
    if (group <= 2) {
        plans.dct2Storage3.execute(storage);
        plans.dst1Storage3.execute(storage);
    } else {
        plans.dst2Storage3.execute(storage);
        plans.dct1Storage3.execute(storage);
    }
}

void executeInverseGroup(SplitProductionPlans& plans, double* storage,
                         std::size_t group) {
    if (group <= 2) {
        plans.dct2Storage3.execute(storage);
        plans.dst1Storage3.execute(storage);
    } else {
        plans.dst2Storage3.execute(storage);
        plans.dct1Storage3.execute(storage);
    }
}

void executeForwardTarget(ProductionPlans& plans, Complex* storage,
                          std::size_t rows, std::size_t nz,
                          std::size_t target) {
    if (target < 2) {
        plans.dct1Storage1.execute(storage);
        normalizeForwardDct(storage, rows, nz, 1, 0, 1);
    } else {
        plans.dst1Storage1.execute(storage);
        normalizeForwardDst(storage, rows, nz, 1, 0, 1);
    }
}

void executeForwardTarget(SplitProductionPlans& plans, double* storage,
                          std::size_t componentStride, std::size_t rows,
                          std::size_t nz, std::size_t target) {
    if (target < 2) {
        plans.dct1Storage1.execute(storage);
        normalizeForwardSplitDct(
            storage, componentStride, rows, nz, 1, 0, 1);
    } else {
        plans.dst1Storage1.execute(storage);
        normalizeForwardSplitDst(
            storage, componentStride, rows, nz, 1, 0, 1);
    }
}

void accumulateProjectedValue(std::vector<Complex>& output,
                              const Workload& coefficientWorkload,
                              std::size_t mode, std::size_t j,
                              std::size_t target, Complex value,
                              double horizontalScale) {
    value = scaled(value, horizontalScale);
    const std::array<double, 3> weights = {
        0.35 + 0.04 * static_cast<double>(target),
        -0.22 + 0.03 * static_cast<double>(target),
        0.18 - 0.02 * static_cast<double>(target)};
    for (std::size_t field = 0; field < 3; ++field) {
        const auto destination = modalSpectrumIndex(
            coefficientWorkload, mode, j, field);
        output[destination] = addComplex(
            output[destination], scaled(value, weights[field]));
    }
}

void projectFullTarget(
    const Complex* full, const Workload& scalarWorkload,
    const Workload& coefficientWorkload,
    const std::vector<RetainedMode>& modes, std::size_t target,
    std::vector<Complex>& output) {
    const auto nj = coefficientWorkload.retainedVerticalModes();
    const auto horizontalScale = 1.0 / static_cast<double>(
        coefficientWorkload.nx * coefficientWorkload.ny);
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
        const auto& retained = modes[mode];
        for (std::size_t j = 0; j < nj; ++j) {
            auto value = full[wvmSpectrumIndex(
                scalarWorkload, retained.storedKx, retained.storedKy, j, 0)];
            if (retained.conjugatesStoredValue) value = conjugated(value);
            accumulateProjectedValue(
                output, coefficientWorkload, mode, j, target, value,
                horizontalScale);
        }
    }
}

void projectCompactTarget(
    const double* real, const double* imaginary,
    const Workload& scalarWorkload,
    const Workload& coefficientWorkload,
    const std::vector<RetainedMode>& modes, std::size_t target,
    std::vector<Complex>& output) {
    const auto nj = coefficientWorkload.retainedVerticalModes();
    const auto horizontalScale = 1.0 / static_cast<double>(
        coefficientWorkload.nx * coefficientWorkload.ny);
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
        for (std::size_t j = 0; j < nj; ++j) {
            const auto source = retainedSpectrumIndex(
                scalarWorkload, mode, j, 0);
            accumulateProjectedValue(
                output, coefficientWorkload, mode, j, target,
                {real[source], imaginary[source]}, horizontalScale);
        }
    }
}

template <typename Operation>
void forEachCoefficientModeBlock(std::size_t modeCount,
                                 Operation&& operation) {
    constexpr std::size_t requestedWorkers = 2;
    const auto workerCount = std::min(requestedWorkers, modeCount);
    if (workerCount <= 1) {
        operation(0, modeCount);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(workerCount - 1);
    const auto blockSize = (modeCount + workerCount - 1) / workerCount;
    for (std::size_t worker = 1; worker < workerCount; ++worker) {
        const auto begin = std::min(worker * blockSize, modeCount);
        const auto end = std::min(begin + blockSize, modeCount);
        workers.emplace_back([begin, end, &operation] {
            operation(begin, end);
        });
    }
    operation(0, std::min(blockSize, modeCount));
    for (auto& worker : workers) worker.join();
}

std::array<Complex, 3> authoritativeGroupValues(
    const ConstantStratificationModeTable& table,
    const std::vector<Complex>& state, const std::vector<Complex>& phases,
    std::size_t mode, std::size_t j, std::size_t group,
    std::size_t nz) {
    if (group == 0)
        return constantStratificationVelocitySpectrum(
            table, state, phases, mode, j);
    return constantStratificationDerivativeSpectrum(
        table, state, phases, mode, j, group - 1, nz);
}

void assembleAuthoritativeCompactGroup(
    double* real, double* imaginary, const Workload& threeWorkload,
    const ConstantStratificationModeTable& table,
    const std::vector<Complex>& state, const std::vector<Complex>& phases,
    std::size_t group) {
    const auto activeElements = table.horizontal.size() *
        threeWorkload.planes();
    std::fill_n(real, activeElements, 0.0);
    std::fill_n(imaginary, activeElements, 0.0);
    forEachCoefficientModeBlock(
        table.horizontal.size(), [&](std::size_t begin, std::size_t end) {
            for (std::size_t mode = begin; mode < end; ++mode) {
                for (std::size_t j = 0; j < table.nj; ++j) {
                    const auto values = authoritativeGroupValues(
                        table, state, phases, mode, j, group,
                        threeWorkload.nz);
                    for (std::size_t channel = 0; channel < 3; ++channel) {
                        const auto destination = retainedSpectrumIndex(
                            threeWorkload, mode, j, channel);
                        real[destination] = values[channel].real;
                        imaginary[destination] = values[channel].imag;
                    }
                }
            }
        });
}

void assembleAuthoritativeFullGroup(
    Complex* full, const Workload& threeWorkload,
    const std::vector<RetainedMode>& modes,
    const ConstantStratificationModeTable& table,
    const std::vector<Complex>& state, const std::vector<Complex>& phases,
    std::size_t group) {
    std::fill_n(full, threeWorkload.spectrumElements(), Complex{});
    forEachCoefficientModeBlock(
        modes.size(), [&](std::size_t begin, std::size_t end) {
            for (std::size_t mode = begin; mode < end; ++mode) {
                const auto& retained = modes[mode];
                for (std::size_t j = 0; j < table.nj; ++j) {
                    const auto values = authoritativeGroupValues(
                        table, state, phases, mode, j, group,
                        threeWorkload.nz);
                    for (std::size_t channel = 0; channel < 3; ++channel) {
                        const auto stored = retained.conjugatesStoredValue
                            ? conjugated(values[channel]) : values[channel];
                        full[wvmSpectrumIndex(
                            threeWorkload, retained.storedKx,
                            retained.storedKy, j, channel)] = stored;
                        if (retained.storedKx == 0 &&
                            retained.storedKy != 0 &&
                            2 * retained.storedKy != threeWorkload.ny) {
                            const auto conjugateKy =
                                (threeWorkload.ny - retained.storedKy) %
                                threeWorkload.ny;
                            full[wvmSpectrumIndex(
                                threeWorkload, 0, conjugateKy, j,
                                channel)] = conjugated(stored);
                        }
                    }
                }
            }
        });
}

void projectAuthoritativeFullTarget(
    const Complex* full, const Workload& scalarWorkload,
    const std::vector<RetainedMode>& modes,
    const ConstantStratificationModeTable& table,
    const std::vector<Complex>& phases, std::size_t target,
    std::vector<Complex>& output) {
    const auto horizontalScale = 1.0 / static_cast<double>(
        scalarWorkload.nx * scalarWorkload.ny);
    forEachCoefficientModeBlock(
        modes.size(), [&](std::size_t begin, std::size_t end) {
            for (std::size_t mode = begin; mode < end; ++mode) {
                const auto& retained = modes[mode];
                for (std::size_t j = 0; j < table.nj; ++j) {
                    auto value = full[wvmSpectrumIndex(
                        scalarWorkload, retained.storedKx,
                        retained.storedKy, j, 0)];
                    if (retained.conjugatesStoredValue)
                        value = conjugated(value);
                    accumulateConstantStratificationFluxTarget(
                        table, phases, output, mode, j, target, value,
                        horizontalScale);
                }
            }
        });
}

void projectAuthoritativeCompactTarget(
    const double* real, const double* imaginary,
    const Workload& scalarWorkload,
    const ConstantStratificationModeTable& table,
    const std::vector<Complex>& phases, std::size_t target,
    std::vector<Complex>& output) {
    const auto horizontalScale = 1.0 / static_cast<double>(
        scalarWorkload.nx * scalarWorkload.ny);
    forEachCoefficientModeBlock(
        table.horizontal.size(), [&](std::size_t begin, std::size_t end) {
            for (std::size_t mode = begin; mode < end; ++mode) {
                for (std::size_t j = 0; j < table.nj; ++j) {
                    const auto source = retainedSpectrumIndex(
                        scalarWorkload, mode, j, 0);
                    accumulateConstantStratificationFluxTarget(
                        table, phases, output, mode, j, target,
                        {real[source], imaginary[source]}, horizontalScale);
                }
            }
        });
}

struct ComposedStageDurations {
    double phaseEvaluation = 0.0;
    double coefficientAssembly = 0.0;
    double verticalInverse = 0.0;
    double horizontalInverse = 0.0;
    double pointwise = 0.0;
    double horizontalForward = 0.0;
    double verticalForward = 0.0;
    double coefficientProjection = 0.0;
};

struct ComposedStageSamples {
    std::vector<double> phaseEvaluation;
    std::vector<double> coefficientAssembly;
    std::vector<double> verticalInverse;
    std::vector<double> horizontalInverse;
    std::vector<double> pointwise;
    std::vector<double> horizontalForward;
    std::vector<double> verticalForward;
    std::vector<double> coefficientProjection;

    void append(const ComposedStageDurations& values) {
        phaseEvaluation.push_back(values.phaseEvaluation);
        coefficientAssembly.push_back(values.coefficientAssembly);
        verticalInverse.push_back(values.verticalInverse);
        horizontalInverse.push_back(values.horizontalInverse);
        pointwise.push_back(values.pointwise);
        horizontalForward.push_back(values.horizontalForward);
        verticalForward.push_back(values.verticalForward);
        coefficientProjection.push_back(values.coefficientProjection);
    }
};

template <typename Action>
void measuredStage(double* elapsed, Action action) {
    if (elapsed == nullptr) {
        action();
        return;
    }
    const auto started = Clock::now();
    action();
    *elapsed += std::chrono::duration<double>(Clock::now() - started).count();
}

template <typename Execute>
ComposedStageSamples measureComposedStages(
    std::size_t warmups, std::size_t samples, Execute execute) {
    for (std::size_t warmup = 0; warmup < warmups; ++warmup)
        execute(nullptr);
    ComposedStageSamples result;
    for (std::size_t sample = 0; sample < samples; ++sample) {
        ComposedStageDurations durations;
        execute(&durations);
        result.append(durations);
    }
    return result;
}

CorrectnessMetric realCorrectness(std::string name, const double* actual,
                                  const double* expected,
                                  std::size_t count) {
    long double squaredError = 0.0;
    long double squaredReference = 0.0;
    double maximumError = 0.0;
    double maximumReference = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const auto error = std::abs(actual[index] - expected[index]);
        maximumError = std::max(maximumError, error);
        maximumReference = std::max(maximumReference, std::abs(expected[index]));
        squaredError += static_cast<long double>(error) * error;
        squaredReference +=
            static_cast<long double>(expected[index]) * expected[index];
    }
    const auto maximum = maximumError / std::max(maximumReference, 1.0);
    const auto l2 = squaredReference == 0.0
        ? std::sqrt(static_cast<double>(squaredError))
        : std::sqrt(static_cast<double>(squaredError / squaredReference));
    return {std::move(name), maximum, tolerance,
            maximum <= tolerance && l2 <= tolerance, l2};
}

CorrectnessMetric scalarCorrectness(std::string name, double error) {
    return {std::move(name), error, tolerance, error <= tolerance, error};
}

double dcImaginaryError(const std::vector<Complex>& output,
                        const Workload& coefficientWorkload,
                        const std::vector<RetainedMode>& modes) {
    double maximum = 0.0;
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
        if (modes[mode].k != 0 || modes[mode].l != 0) continue;
        for (std::size_t field = 0; field < 3; ++field) {
            for (std::size_t j = 0;
                 j < coefficientWorkload.retainedVerticalModes(); ++j) {
                maximum = std::max(maximum, std::abs(output[
                    modalSpectrumIndex(
                        coefficientWorkload, mode, j, field)].imag));
            }
        }
    }
    return maximum;
}

double dcWaveVortexConstraintError(
    const std::vector<Complex>& output,
    const Workload& coefficientWorkload,
    const std::vector<RetainedMode>& modes) {
    double maximum = 0.0;
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
        if (modes[mode].k != 0 || modes[mode].l != 0) continue;
        for (std::size_t j = 0;
             j < coefficientWorkload.retainedVerticalModes(); ++j) {
            const auto fp = output[modalSpectrumIndex(
                coefficientWorkload, mode, j, 0)];
            const auto fm = output[modalSpectrumIndex(
                coefficientWorkload, mode, j, 1)];
            const auto f0 = output[modalSpectrumIndex(
                coefficientWorkload, mode, j, 2)];
            maximum = std::max(maximum, std::hypot(
                fm.real - fp.real, fm.imag + fp.imag));
            maximum = std::max(maximum, std::abs(f0.imag));
        }
    }
    return maximum;
}

} // namespace

BenchmarkReport runConstantStratificationFluxBenchmark(
    const RunOptions& options) {
    auto selected = profileNamed(options.profile);
    auto coefficientWorkload = selected.workload;
    coefficientWorkload.fields = 3;
    std::optional<ConstantStratificationFluxFixture> fixture;
    if (!options.constantStratificationFluxFixture.empty()) {
        fixture = loadPreparedConstantStratificationFluxFixture(
            options.constantStratificationFluxFixture);
        if (fixture->workload.nx != coefficientWorkload.nx ||
            fixture->workload.ny != coefficientWorkload.ny ||
            fixture->workload.nz != coefficientWorkload.nz)
            throw std::invalid_argument(
                "The constant-stratification fixture dimensions do not match the selected profile.");
        coefficientWorkload = fixture->workload;
        coefficientWorkload.fields = 3;
    }
    const bool authoritative = fixture.has_value();
    if (coefficientWorkload.nx != coefficientWorkload.ny ||
        coefficientWorkload.nx % 2 != 0 || coefficientWorkload.nz < 3)
        throw std::invalid_argument(
            "constant-stratification flux benchmark requires an even square horizontal grid and Nz >= 3.");
    if (options.streamingTileWidth != 1 && options.streamingTileWidth != 16)
        throw std::invalid_argument(
            "The composed constant-stratification candidate freezes tile width 16.");
    const auto warmups = options.warmups == 0 ? selected.warmups : options.warmups;
    const auto samples = options.samples == 0 ? selected.samples : options.samples;
    const auto verticalWorkers = options.fftwInternalWorkers == 0
        ? std::max<std::size_t>(1, selected.defaultWorkers)
        : options.fftwInternalWorkers;
    const auto horizontalWorkers = options.fftwOuterWorkers == 1
        ? std::max<std::size_t>(1, selected.defaultWorkers)
        : options.fftwOuterWorkers;
    const auto pointwisePolicy =
        pointwiseAdvectionPolicyNamed(options.pointwisePolicy);
    const auto pointwiseWorkers = options.pointwiseWorkers == 0
        ? (pointwisePolicy == PointwiseAdvectionPolicy::spatialStatic
               ? horizontalWorkers : std::size_t{1})
        : options.pointwiseWorkers;
    if (pointwisePolicy != PointwiseAdvectionPolicy::spatialStatic &&
        pointwiseWorkers != 1)
        throw std::invalid_argument(
            "Serial pointwise policies require one worker.");
    const auto planningMode = fftwPlanningModeNamed(options.fftwPlanning);
    const auto modes = authoritative
        ? fixture->modes : retainedHorizontalModes(coefficientWorkload);
    const auto fullRows = coefficientWorkload.halfRows();
    const auto compactRows = modes.size();
    const auto nz = coefficientWorkload.nz;
    const auto nj = coefficientWorkload.retainedVerticalModes();
    auto threeWorkload = coefficientWorkload;
    threeWorkload.fields = 3;
    auto scalarWorkload = coefficientWorkload;
    scalarWorkload.fields = 1;
    const auto volume = coefficientWorkload.nx * coefficientWorkload.ny * nz;
    const auto fullCapacity = elementCount(fullRows, 4, nz);
    const auto compactCapacity = elementCount(compactRows, 4, nz);

    const auto setupStarted = Clock::now();
    std::optional<ConstantStratificationModeTable> modeTable;
    std::vector<Complex> authoritativeExpected;
    std::vector<Complex> phases;
    std::vector<Complex> coefficients;
    if (authoritative) {
        modeTable = makeConstantStratificationModeTable(*fixture);
        coefficients = std::move(fixture->modalState);
        authoritativeExpected = std::move(fixture->expectedModalFlux);
        phases.resize(modeTable->omega.size());
    } else {
        coefficients = makeCoefficientFixture(coefficientWorkload, modes);
    }
    std::vector<Complex> fullOutput(coefficients.size());
    std::vector<Complex> compactOutput(coefficients.size());
    FFTWArray<Complex> fullArena(fullCapacity);
    FFTWArray<double> compactArena(2 * compactCapacity);
    auto* compactReal = compactArena.data();
    auto* compactImaginary = compactArena.data() + compactCapacity;
    std::vector<double> fullShared(3 * volume);
    std::vector<double> fullDerivative(3 * volume);
    std::vector<double> fullTarget(volume);
    std::vector<double> compactShared(3 * volume);
    std::vector<double> compactDerivative(3 * volume);
    std::vector<double> compactTarget(volume);

    ProductionPlans fullVertical(
        fullRows, nz, planningMode, verticalWorkers);
    SplitProductionPlans compactVertical(
        compactRows, nz, planningMode, verticalWorkers, compactCapacity);
    FFTWStrategy horizontalStrategy;
    horizontalStrategy.planningMode = planningMode;
    horizontalStrategy.alignment = FFTWAlignmentStrategy::unaligned;
    horizontalStrategy.wisdom = FFTWWisdomStrategy::cold;
    horizontalStrategy.internalWorkers = 1;
    horizontalStrategy.outerWorkers = horizontalWorkers;
    horizontalStrategy.layout = FFTWDataLayout::interleaved;
    horizontalStrategy.spectrumOrder = FFTWSpectrumOrder::wvmFrequencyMajor;
    FFTWProvider fullInverse(threeWorkload, horizontalStrategy);
    FFTWProvider fullForward(scalarWorkload, horizontalStrategy);
    FFTWStreamingPrunedSplitProvider compactInverse(
        threeWorkload, modes, planningMode, 1, horizontalWorkers, 16,
        StreamingInversePreparationPolicy::fullZero);
    FFTWStreamingPrunedSplitProvider compactForward(
        scalarWorkload, modes, planningMode, 1, horizontalWorkers, 16,
        StreamingInversePreparationPolicy::fullZero);
    const auto pointwiseScale = authoritative
        ? fixture->pointwiseScale
        : 1.0 / static_cast<double>(
              coefficientWorkload.nx * coefficientWorkload.ny) /
              static_cast<double>(
                  coefficientWorkload.nx * coefficientWorkload.ny);
    PointwiseAdvectionExecutor fullPointwise(
        pointwisePolicy, pointwiseWorkers, volume, pointwiseScale);
    PointwiseAdvectionExecutor compactPointwise(
        pointwisePolicy, pointwiseWorkers, volume, pointwiseScale);
    const auto setupSeconds =
        std::chrono::duration<double>(Clock::now() - setupStarted).count();

    auto executeFull = [&](ComposedStageDurations* durations) {
        if (authoritative) {
            measuredStage(
                durations ? &durations->phaseEvaluation : nullptr,
                [&] {
                    evaluateConstantStratificationPhases(
                        *modeTable, fixture->elapsedTime, phases);
                    std::fill(fullOutput.begin(), fullOutput.end(), Complex{});
                });
        } else {
            measuredStage(
                durations ? &durations->coefficientProjection : nullptr,
                [&] {
                    std::fill(
                        fullOutput.begin(), fullOutput.end(), Complex{});
                });
        }
        for (std::size_t group = 0; group < 5; ++group) {
            measuredStage(
                durations ? &durations->coefficientAssembly : nullptr,
                [&] {
                    if (authoritative)
                        assembleAuthoritativeFullGroup(
                            fullArena.data(), threeWorkload, modes,
                            *modeTable, coefficients, phases, group);
                    else
                        assembleFullGroup(
                            fullArena.data(), coefficientWorkload,
                            threeWorkload, modes, coefficients, group);
                });
            measuredStage(
                durations ? &durations->verticalInverse : nullptr,
                [&] { executeInverseGroup(fullVertical, fullArena.data(), group); });
            auto& destination = group == 0 ? fullShared : fullDerivative;
            measuredStage(
                durations ? &durations->horizontalInverse : nullptr,
                [&] { fullInverse.inverse(fullArena.data(), destination.data()); });
            if (group == 0) continue;
            const auto target = group - 1;
            measuredStage(
                durations ? &durations->pointwise : nullptr,
                [&] {
                    fullPointwise.execute(
                        fullShared.data(), fullDerivative.data(),
                        fullTarget.data());
                });
            measuredStage(
                durations ? &durations->horizontalForward : nullptr,
                [&] { fullForward.forward(fullTarget.data(), fullArena.data()); });
            measuredStage(
                durations ? &durations->verticalForward : nullptr,
                [&] {
                    executeForwardTarget(
                        fullVertical, fullArena.data(), fullRows, nz, target);
                });
            measuredStage(
                durations ? &durations->coefficientProjection : nullptr,
                [&] {
                    if (authoritative)
                        projectAuthoritativeFullTarget(
                            fullArena.data(), scalarWorkload, modes,
                            *modeTable, phases, target, fullOutput);
                    else
                        projectFullTarget(
                            fullArena.data(), scalarWorkload,
                            coefficientWorkload, modes, target, fullOutput);
                });
        }
    };

    auto executeCompact = [&](ComposedStageDurations* durations) {
        if (authoritative) {
            measuredStage(
                durations ? &durations->phaseEvaluation : nullptr,
                [&] {
                    evaluateConstantStratificationPhases(
                        *modeTable, fixture->elapsedTime, phases);
                    std::fill(
                        compactOutput.begin(), compactOutput.end(), Complex{});
                });
        } else {
            measuredStage(
                durations ? &durations->coefficientProjection : nullptr,
                [&] {
                    std::fill(
                        compactOutput.begin(), compactOutput.end(), Complex{});
                });
        }
        for (std::size_t group = 0; group < 5; ++group) {
            measuredStage(
                durations ? &durations->coefficientAssembly : nullptr,
                [&] {
                    if (authoritative)
                        assembleAuthoritativeCompactGroup(
                            compactReal, compactImaginary, threeWorkload,
                            *modeTable, coefficients, phases, group);
                    else
                        assembleCompactGroup(
                            compactReal, compactImaginary,
                            coefficientWorkload, threeWorkload, modes,
                            coefficients, group);
                });
            measuredStage(
                durations ? &durations->verticalInverse : nullptr,
                [&] {
                    executeInverseGroup(
                        compactVertical, compactArena.data(), group);
                });
            auto& destination = group == 0 ? compactShared : compactDerivative;
            measuredStage(
                durations ? &durations->horizontalInverse : nullptr,
                [&] {
                    compactInverse.inverseSplit(
                        compactReal, compactImaginary, destination.data());
                });
            if (group == 0) continue;
            const auto target = group - 1;
            measuredStage(
                durations ? &durations->pointwise : nullptr,
                [&] {
                    compactPointwise.execute(
                        compactShared.data(), compactDerivative.data(),
                        compactTarget.data());
                });
            measuredStage(
                durations ? &durations->horizontalForward : nullptr,
                [&] {
                    compactForward.forwardSplit(
                        compactTarget.data(), compactReal, compactImaginary);
                });
            measuredStage(
                durations ? &durations->verticalForward : nullptr,
                [&] {
                    executeForwardTarget(
                        compactVertical, compactArena.data(), compactCapacity,
                        compactRows, nz, target);
                });
            measuredStage(
                durations ? &durations->coefficientProjection : nullptr,
                [&] {
                    if (authoritative)
                        projectAuthoritativeCompactTarget(
                            compactReal, compactImaginary, scalarWorkload,
                            *modeTable, phases, target, compactOutput);
                    else
                        projectCompactTarget(
                            compactReal, compactImaginary, scalarWorkload,
                            coefficientWorkload, modes, target,
                            compactOutput);
                });
        }
    };

    executeFull(nullptr);
    auto expectedOutput = authoritative
        ? std::move(authoritativeExpected) : fullOutput;
    const auto expectedShared = fullShared;
    const auto expectedDerivative = fullDerivative;
    const auto expectedTarget = fullTarget;
    executeCompact(nullptr);

    const auto fullStages = measureComposedStages(
        warmups, samples, executeFull);
    const auto compactStages = measureComposedStages(
        warmups, samples, executeCompact);
    const auto fullTotals = measure(
        warmups, samples, [&] { executeFull(nullptr); });
    const auto compactTotals = measure(
        warmups, samples, [&] { executeCompact(nullptr); });

    BenchmarkReport report;
    report.environment = environmentRecord();
    report.runId = timestampId(report.environment.timestampUtc) +
        (authoritative
             ? "-issue20-constant-stratification-authoritative-"
             : "-issue20-constant-stratification-composed-") +
        report.environment.hostname;
    report.profile = options.profile;
    report.seed = options.seed;
    report.warmups = warmups;
    report.samples = samples;
    report.workload = coefficientWorkload;
    report.workload.fields = 4;
    report.retainedHorizontalModeCount = compactRows;
    report.retainedModeOrderHash = modeOrderHash(modes);
    report.wvmFullSpectrumOrderHash = wvmSpectrumOrderHash(coefficientWorkload);
    report.fullRealBytes = byteCount(4 * volume, sizeof(double));
    report.fullSpectrumBytes = byteCount(
        elementCount(fullRows, 4, nz), sizeof(Complex));
    report.retainedSpectrumBytes = byteCount(
        elementCount(compactRows, 4, nz), sizeof(Complex));
    report.modalSpectrumBytes = byteCount(
        compactRows * nj * 3, sizeof(Complex));
    report.verticalMatrixFamilyId = authoritative
        ? "wvm-constant-stratification-natural-dimensional-prescaled-v1"
        : "wvm-constant-stratification-fftw-type1-composed-synthetic-map-v1";
    if (authoritative) {
        report.fixtureProvenance = {
            "authoritative-wvm-export",
            "constant-stratification-flux-fixture-v1",
            fixture->waveVortexModelRepository,
            fixture->waveVortexModelCommit,
            fixture->generatorIdentity,
            fixture->fixtureHash,
            fixture->normalization,
            fixture->modeMapping,
            fixture->coefficientContract,
            true};
    } else {
        report.fixtureProvenance = {
            "provider-independent-synthetic-development",
            "constant-stratification-composed-fixture-v1",
            "JeffreyEarly/wave-vortex-model",
            std::string(auditedWvmCommit),
            "skbench deterministic symmetry-preserving mode-keyed coefficient map",
            modeOrderHash(modes),
            "FFTW inverse horizontal scale represented by pointwise 1/(Nx*Ny)^2; "
            "forward horizontal 1/(Nx*Ny); exact WVM type-I vertical normalization",
            "logical radial (k,l,j); full WVM mapping versus compact split mapping",
            "cosine targets 0/1, sine targets 2/3; x/y imaginary multipliers and vertical family swap",
            false};
    }

    const auto realBufferBytes = byteCount(7 * volume, sizeof(double));
    const auto fullArenaBytes = byteCount(fullCapacity, sizeof(Complex));
    const auto compactArenaBytes = byteCount(2 * compactCapacity, sizeof(double));
    const auto coefficientBytes = byteCount(coefficients.size(), sizeof(Complex));
    const auto phaseBytes = authoritative
        ? byteCount(phases.size(), sizeof(Complex)) : std::uint64_t{0};
    const auto modeTableBytes = authoritative
        ? constantStratificationModeTableBytes(*modeTable) : std::uint64_t{0};
    const auto totalStage = authoritative
        ? "authoritative WVM constant-stratification nonlinear-flux composition"
        : "production-shaped constant-stratification spectral-flux composition";

    auto makeRecord = [&](bool compact) {
        auto record = providerRecord(
            compact
                ? (authoritative
                       ? "pipeline-constant-stratification-streaming-pruned-tile16-authoritative"
                       : "pipeline-constant-stratification-streaming-pruned-tile16")
                : (authoritative
                       ? "pipeline-constant-stratification-wvm-full-half-authoritative"
                       : "pipeline-constant-stratification-wvm-full-half"),
            compact
                ? (authoritative
                       ? "compact-split-type1+partial-column-pruned-tile16+"
                         "streamed-3-shared-3-derivative+pointwise-4-target+"
                         "wvm-coefficients-v1"
                       : "compact-split-type1+partial-column-pruned-tile16+"
                         "streamed-3-shared-3-derivative+pointwise-4-target-v1")
                : (authoritative
                       ? "wvm-full-half-type1+full-horizontal+streamed-3-shared-3-"
                         "derivative+pointwise-4-target+wvm-coefficients-v1"
                       : "wvm-full-half-type1+full-horizontal+streamed-3-shared-3-"
                         "derivative+pointwise-4-target-v1"),
            compact
                ? "persistent radial compact split type-I rows and tile-16 horizontal scratch"
                : "reusable WVM-order full-half interleaved type-I arena",
            compact ? compactRows : fullRows, nz, verticalWorkers,
            planningMode,
            compact ? compactVertical.planningSeconds()
                    : fullVertical.planningSeconds());
        record.modeOrderId = compact
            ? "logical-radial-(k,l,j); compact split z-adjacent"
            : "logical-radial-(k,l,j) assembled into WVM full-half order";
        record.schedulingId =
            "vertical-type1-internal-" + std::to_string(verticalWorkers) +
            ";horizontal-internal-1-outer-" +
            std::to_string(horizontalWorkers) + ";pointwise-" +
            std::string(pointwiseAdvectionPolicyName(pointwisePolicy)) + '-' +
            std::to_string(pointwiseWorkers);
        record.workers = std::max(
            {verticalWorkers, horizontalWorkers, pointwiseWorkers});
        record.internalWorkers = verticalWorkers;
        record.outerWorkers = horizontalWorkers;
        record.planningConfiguration = authoritative
            ? "authoritative WVM coefficient formulas and phase; exact production "
              "type-I family schedule; coefficient workers 2; FFTW MEASURE/UNALIGNED "
              "when selected; horizontal tile 16; no timed application allocation"
            : "synthetic development coefficient map; exact production type-I "
              "family schedule; FFTW MEASURE/UNALIGNED when selected; horizontal "
              "tile 16; no timed application allocation";
        const auto& stages = compact ? compactStages : fullStages;
        record.timings = {
            timing("component", "mode-keyed coefficient assembly and retained/full clearing",
                   "inverse", compact ? compactArenaBytes : fullArenaBytes,
                   stages.coefficientAssembly),
            timing("component", "15 inverse complex type-I channels",
                   "inverse", 2 * (compact ? compactArenaBytes : fullArenaBytes),
                   stages.verticalInverse),
            timing("retained-operator-total", "five horizontal inverse transforms",
                   "inverse", 5 * report.fullRealBytes,
                   stages.horizontalInverse),
            timing("component", "four streamed pointwise advection expressions",
                   "pointwise", byteCount(28 * volume, sizeof(double)),
                   stages.pointwise),
            timing("retained-operator-total", "four horizontal forward transforms and radial retention",
                   "forward", 4 * report.fullRealBytes,
                   stages.horizontalForward),
            timing("component", "four forward complex type-I channels and normalization",
                   "forward", 2 * (compact ? compactArenaBytes : fullArenaBytes),
                   stages.verticalForward),
            timing("component", authoritative
                       ? "four authoritative flux-target accumulations"
                       : "coefficient reset and four target accumulations",
                   "forward", 2 * coefficientBytes,
                   stages.coefficientProjection),
            timing("uninstrumented-total", totalStage, "complete",
                   0, compact ? compactTotals : fullTotals)};
        if (authoritative) {
            record.timings.insert(
                record.timings.begin(),
                timing("component", "phase evaluation and flux reset",
                       "phase", phaseBytes + coefficientBytes,
                       stages.phaseEvaluation));
        }
        record.correctness = {
            correctness(
                authoritative
                    ? (compact
                           ? "complete compact composition versus authoritative WVM oracle"
                           : "complete full-half composition versus authoritative WVM oracle")
                    : (compact
                           ? "complete compact composition versus full-half control"
                           : "full-half composition deterministic replay"),
                compact ? compactOutput.data() : fullOutput.data(),
                expectedOutput.data(), expectedOutput.size()),
            realCorrectness(
                compact
                    ? "compact shared U,V,W versus full-half control"
                    : "full-half shared U,V,W deterministic replay",
                compact ? compactShared.data() : fullShared.data(),
                expectedShared.data(), expectedShared.size()),
            realCorrectness(
                compact
                    ? "compact final derivative triple versus full-half control"
                    : "full-half derivative triple deterministic replay",
                compact ? compactDerivative.data() : fullDerivative.data(),
                expectedDerivative.data(), expectedDerivative.size()),
            realCorrectness(
                compact
                    ? "compact final pointwise target versus full-half control"
                    : "full-half pointwise target deterministic replay",
                compact ? compactTarget.data() : fullTarget.data(),
                expectedTarget.data(), expectedTarget.size()),
            scalarCorrectness(
                authoritative
                    ? "DC Fm=conjugate(Fp) and F0 remains real"
                    : "DC coefficient output remains real",
                authoritative
                    ? dcWaveVortexConstraintError(
                          compact ? compactOutput : fullOutput,
                          coefficientWorkload, modes)
                    : dcImaginaryError(
                          compact ? compactOutput : fullOutput,
                          coefficientWorkload, modes))};
        if (authoritative) {
            record.correctness.push_back({
                "fixture MATLAB versus compiled WVM nonlinear-flux cross-check",
                fixture->oracleMaximumScaleNormalizedError,
                tolerance,
                fixture->oracleMaximumScaleNormalizedError <= tolerance &&
                    fixture->oracleRelativeL2Error <= tolerance,
                fixture->oracleRelativeL2Error});
        }
        record.execution.forward.nativePlacement = "out-of-place";
        record.execution.forward.adapterPlacement = "out-of-place";
        record.execution.forward.destroysNativeInput = false;
        record.execution.forward.adapterPreservesCallerInput = true;
        record.execution.forward.nativeInputRepresentationId =
            "three retained mode-keyed coefficient arrays";
        record.execution.forward.nativeOutputRepresentationId =
            "three retained mode-keyed accumulated coefficient arrays";
        record.execution.forward.adapterInputRepresentationId =
            record.execution.forward.nativeInputRepresentationId;
        record.execution.forward.adapterOutputRepresentationId =
            record.execution.forward.nativeOutputRepresentationId;
        record.execution.forward.physicalExtents =
            "five inverse triples, four real targets, four forward scalars";
        record.execution.forward.stridesElements = compact
            ? "split z=1, field=Nz, mode=3*Nz; radial logical order"
            : "interleaved z=1, field=Nz, WVM half-row=3*Nz";
        record.execution.forward.aliasing = "caller input and output do not alias";
        record.execution.inverse = record.execution.forward;
        record.explicitPersistentBytes = compact
            ? compactArenaBytes : fullArenaBytes;
        record.scratchBytes = realBufferBytes;
        if (compact) {
            record.scratchBytes += compactInverse.scratchBytes() +
                compactForward.scratchBytes();
            record.allocationSeconds = compactInverse.allocationSeconds() +
                compactForward.allocationSeconds();
            record.planningSeconds += compactInverse.planningSeconds() +
                compactForward.planningSeconds();
            record.opaquePlanningBytes = compactInverse.planningBytes() +
                compactForward.planningBytes();
        } else {
            record.allocationSeconds = fullInverse.allocationSeconds() +
                fullForward.allocationSeconds();
            record.planningSeconds += fullInverse.planningSeconds() +
                fullForward.planningSeconds();
            record.opaquePlanningBytes = fullInverse.planningBytes() +
                fullForward.planningBytes();
        }
        record.explicitPersistentBytes += compact
            ? compactPointwise.persistentBytes()
            : fullPointwise.persistentBytes();
        record.explicitPersistentBytes += modeTableBytes;
        record.scratchBytes += phaseBytes;
        record.algorithmResidentBytes =
            record.explicitPersistentBytes + record.scratchBytes;
        record.estimatedProcessPeakBytes = record.algorithmResidentBytes +
            (authoritative ? 3 : 2) * coefficientBytes;
        record.benchmarkHarnessBytes =
            (authoritative ? 3 : 2) * coefficientBytes;
        record.otherSetupSeconds = setupSeconds / 2.0;
        record.opaqueProviderMemory = true;
        record.ledger = {
            {"coefficient fixture", StageState::setupOnly,
             authoritative
                 ? "authoritative WVM state and nonlinear-flux oracle; loaded and validated before timing"
                 : "deterministic symmetry-preserving development map; not the WVM coefficient-formula oracle"},
            {"coefficient assembly", StageState::executed,
             compact
                 ? "writes only retained compact split rows"
                 : "clears and assembles the WVM full half-spectrum"},
            {"vertical inverse", StageState::executed,
             "exact 15-channel REDFT00/RODFT00 production family schedule"},
            {"horizontal inverse", StageState::executed,
             compact
                 ? "fixed tile-16 partial-column-pruned retained inverse"
                 : "full WVM-order horizontal inverse"},
            {"pointwise advection", StageState::executed,
             "four streamed -(U*qx+V*qy+W*qz) expressions"},
            {"horizontal forward", StageState::executed,
             compact
                 ? "fixed tile-16 partial-column-pruned retained forward"
                 : "full WVM-order horizontal forward"},
            {"vertical forward", StageState::executed,
             "exact four-channel REDFT00/RODFT00 production family schedule and normalization"},
            {"coefficient accumulation", StageState::executed,
             "four target spectra accumulated into three mode-keyed outputs"},
            {"steady-state application allocation", StageState::elided,
             "all benchmark-owned storage and worker pools are persistent; "
             "FFTW-owned execution behavior remains opaque"}};
        if (authoritative) {
            record.ledger.insert(
                record.ledger.begin() + 1,
                {"phase evaluation and flux reset", StageState::executed,
                 "scalar sincos for every retained (k,l,j), plus zeroing Fp/Fm/F0"});
            record.ledger.push_back(
                {"exact WVM coefficient formulas and complete nonlinear flux",
                 StageState::executed,
                 "natural-dimensional-prescaled reconstruction, derivatives, "
                 "inertial/MDA exceptions, phase removal, and four-target accumulation"});
        } else {
            record.ledger.push_back(
                {"exact WVM coefficient formulas and complete nonlinear flux",
                 StageState::unsupported,
                 "requires a later WVM-exported or in-repository oracle; this "
                 "composed screen is performance-shape evidence only"});
        }
        return record;
    };

    report.providers.push_back(makeRecord(false));
    report.providers.push_back(makeRecord(true));
    const auto observedHighWater = processHighWaterBytes();
    for (auto& provider : report.providers)
        provider.observedProcessHighWaterBytes = observedHighWater;
    // Each provider reports its own complete explicit peak. The top-level
    // single-peak field remains unset because the matched providers have
    // different algorithm-resident storage.
    report.spectralPipelineEstimatedExplicitPeakBytes = 0;
    report.status = std::all_of(
        report.providers.begin(), report.providers.end(), passed)
        ? "passed" : "failed";
    return report;
}

} // namespace skbench
