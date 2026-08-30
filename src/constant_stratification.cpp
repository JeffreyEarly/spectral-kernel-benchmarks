#include "skbench/skbench.hpp"

#include <fftw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
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

} // namespace skbench
