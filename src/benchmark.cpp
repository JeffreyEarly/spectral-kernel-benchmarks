#include "skbench/skbench.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

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
    stream << std::put_time(&utc, compact ? "%Y%m%dT%H%M%SZ" : "%Y-%m-%dT%H:%M:%SZ");
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

std::vector<LedgerEntry> fftwLedger() {
    return {
        {"setup/planning", StageState::setupOnly, "FFTW guru64 MEASURE|UNALIGNED plans"},
        {"raw forward FFT", StageState::executed, "provider-native WVM-strided r2c"},
        {"horizontal retention", StageState::executed, "radial two-thirds mode-keyed gather"},
        {"representation conversion", StageState::elided, "FFTW writes the WVM frequency-major representation directly"},
        {"permutation/packing", StageState::elided, "no packing required for the FFTW primitive"},
        {"raw forward vertical MM", StageState::unsupported, "outside this FFT vertical slice"},
        {"modal work", StageState::unsupported, "outside this FFT vertical slice"},
        {"raw inverse vertical MM", StageState::unsupported, "outside this FFT vertical slice"},
        {"horizontal embedding", StageState::executed, "mode-keyed retained spectrum embedded into the Hermitian half-spectrum"},
        {"raw inverse FFT", StageState::executed, "provider-native WVM-strided c2r"},
        {"uninstrumented total", StageState::executed, "retained horizontal forward or inverse operator"}};
}

std::vector<LedgerEntry> vdspLedger() {
    return {
        {"setup/planning", StageState::setupOnly, "one radix-2 setup per persistent worker"},
        {"raw forward FFT", StageState::executed, "batched outer scheduling of native vDSP_fft2d_zripD calls"},
        {"horizontal retention", StageState::executed, "radial two-thirds mode-keyed gather"},
        {"representation conversion", StageState::fused, "split/interleaved conversion is fused with vDSP packing and WVM reordering"},
        {"permutation/packing", StageState::fused, "real packing and frequency-major WVM reordering are timed adapter components"},
        {"raw forward vertical MM", StageState::unsupported, "outside this FFT vertical slice"},
        {"modal work", StageState::unsupported, "outside this FFT vertical slice"},
        {"raw inverse vertical MM", StageState::unsupported, "outside this FFT vertical slice"},
        {"horizontal embedding", StageState::executed, "mode-keyed retained spectrum embedded before the inverse adapter"},
        {"raw inverse FFT", StageState::executed, "batched outer scheduling of native vDSP_fft2d_zripD calls"},
        {"uninstrumented total", StageState::executed, "retained horizontal forward or inverse operator"}};
}

ExecutionContract fftwExecutionContract(const Workload& workload) {
    const auto planes = workload.planes();
    const auto realExtents = "[planes=" + std::to_string(planes) + "][Ny=" + std::to_string(workload.ny) +
        "][Nx=" + std::to_string(workload.nx) + "]";
    const auto spectrumExtents = "[Ny=" + std::to_string(workload.ny) + "][NxHalf=" +
        std::to_string(workload.nxHalf()) + "][planes=" + std::to_string(planes) + "]";
    const auto realStrides = "x=1,y=" + std::to_string(workload.nx) + ",plane=" +
        std::to_string(workload.realPlaneElements());
    const auto spectrumStrides = "plane=1,kx=" + std::to_string(planes) + ",ky=" +
        std::to_string(planes * workload.nxHalf());

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
        "wvm-frequency-major-interleaved-half-spectrum",
        "wvm-x-fastest-real-grid",
        "wvm-frequency-major-interleaved-half-spectrum",
        "input=" + realExtents + "; output=" + spectrumExtents,
        "input{" + realStrides + "}; output{" + spectrumStrides + "}",
        0,
        1,
        "input and output do not overlap; FFTW_UNALIGNED accepts arbitrary scalar alignment",
        0,
        true};
    contract.inverse = {
        "out-of-place",
        "out-of-place",
        true,
        false,
        true,
        false,
        false,
        "wvm-frequency-major-interleaved-half-spectrum",
        "wvm-x-fastest-real-grid",
        "wvm-frequency-major-interleaved-half-spectrum",
        "wvm-x-fastest-real-grid",
        "input=" + spectrumExtents + "; output=" + realExtents,
        "input{" + spectrumStrides + "}; output{" + realStrides + "}",
        0,
        1,
        "input and output do not overlap; multidimensional FFTW c2r may destroy its input",
        0,
        true};
    return contract;
}

ExecutionContract vdspExecutionContract(const Workload& workload) {
    const auto half = workload.nx / 2;
    const auto planes = workload.planes();
    const auto nativeExtents = "two split arrays [planes=" + std::to_string(planes) + "][Ny=" +
        std::to_string(workload.ny) + "][Nx/2=" + std::to_string(half) + "]";
    const auto nativeStrides = "split-slot=1,row=" + std::to_string(half) + ",plane=" +
        std::to_string(half * workload.ny);

    DirectionExecutionContract nativeDirection{
        "in-place",
        "out-of-place",
        true,
        true,
        true,
        false,
        true,
        "vdsp-packed-split-complex",
        "vdsp-packed-split-complex",
        "wvm-x-fastest-real-grid",
        "wvm-frequency-major-interleaved-half-spectrum",
        nativeExtents,
        nativeStrides,
        0,
        alignof(double),
        "real and imaginary split arrays are disjoint; each transform overwrites its native input",
        0,
        true};
    ExecutionContract contract;
    contract.forward = nativeDirection;
    contract.inverse = nativeDirection;
    contract.inverse.adapterInputRepresentationId = "wvm-frequency-major-interleaved-half-spectrum";
    contract.inverse.adapterOutputRepresentationId = "wvm-x-fastest-real-grid";
    return contract;
}

void appendUnsupportedVdspRecord(BenchmarkReport& report, const Profile& selected, const VDSPProvider& provider) {
    ProviderRecord record;
    record.id = "accelerate-vdsp";
    record.version = "system";
    record.libraryIdentity = provider.libraryIdentity();
    record.algorithmId = "vdsp-radix2-2d-persistent-outer-batch";
    record.nativeRepresentationId = "vdsp-packed-split-complex";
    record.modeOrderId = "vdsp-packed-special-boundaries";
    record.schedulingId = "persistent-thread-pool";
    record.sourceIdentity = "Apple Accelerate system framework";
    record.workers = selected.defaultWorkers;
    record.execution = vdspExecutionContract(report.workload);
    record.otherSetupSeconds = provider.otherSetupSeconds();
    record.allocationSeconds = provider.allocationSeconds();
    record.planningSeconds = provider.planningSeconds();
    record.ledger = vdspLedger();
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
        {"exhaustive", "Initial larger reference shape; broader sweeps are added by later issues.", {512, 512, 129, 4, 1.0, 1.0, true}, totalWorkers, 3, 20}};
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
    VDSPProvider vdsp(workload, 1);
    if (!vdsp.supported()) {
        report.messages.push_back(vdsp.capability());
        return report;
    }

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

        vdsp.forwardAdapter(input.data(), actual.data());
        const auto vdspForwardError = maximumRelativeError(actual.data(), oracle.data(), oracle.size());
        passed = passed && vdspForwardError <= tolerance;
        report.messages.push_back("vdsp/" + std::string(fixtureName(fixture)) + "/forward=" + std::to_string(vdspForwardError));

        vdsp.inverseAdapter(oracle.data(), output.data());
        const auto vdspInverseError = maximumRelativeError(output.data(), input.data(), input.size(), 1.0 / static_cast<double>(workload.nx * workload.ny));
        passed = passed && vdspInverseError <= tolerance;
        report.messages.push_back("vdsp/" + std::string(fixtureName(fixture)) + "/inverse=" + std::to_string(vdspInverseError));

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
    const auto workers = options.workers == 0 ? selected.defaultWorkers : options.workers;
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

    auto input = makeFixture(workload, FixtureKind::random, options.seed);
    std::vector<Complex> referenceSpectrum(workload.spectrumElements());
    std::vector<Complex> workingSpectrum(workload.spectrumElements());
    std::vector<Complex> inverseSpectrum(workload.spectrumElements());
    std::vector<Complex> retainedSpectrum(modes.size() * workload.planes());
    std::vector<Complex> retainedWorking(modes.size() * workload.planes());
    std::vector<double> fftwOutput(workload.realElements());
    std::vector<double> output(workload.realElements());

    FFTWProvider fftw(workload, workers);
    fftw.forward(input.data(), referenceSpectrum.data());
    inverseSpectrum = referenceSpectrum;
    fftw.inverse(inverseSpectrum.data(), fftwOutput.data());
    gatherRetained(workload, modes, referenceSpectrum.data(), retainedSpectrum.data());

    ProviderRecord fftwRecord;
    fftwRecord.id = "fftw";
    fftwRecord.version = fftw.version();
    fftwRecord.libraryIdentity = fftw.libraryIdentity();
    fftwRecord.algorithmId = "wvm-production-guru64-measure-unaligned";
    fftwRecord.nativeRepresentationId = "wvm-frequency-major-interleaved-half-spectrum";
    fftwRecord.modeOrderId = "full-r2c-kx-nonnegative-ky-wrapped";
    fftwRecord.schedulingId = "fftw-internal-pthreads";
    fftwRecord.sourceIdentity = "https://fftw.org/pub/fftw/fftw-3.3.11.tar.gz";
    fftwRecord.sourceSha256 = "5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1";
    fftwRecord.configureFlags = "--host=aarch64-apple-darwin --enable-neon --enable-threads --disable-fortran --disable-openmp --enable-shared --disable-static";
    fftwRecord.compilerFlags = "-O3 -mcpu=native -mmacosx-version-min=13.3";
    fftwRecord.workers = workers;
    fftwRecord.execution = fftwExecutionContract(workload);
    fftwRecord.otherSetupSeconds = fftw.otherSetupSeconds();
    fftwRecord.allocationSeconds = fftw.allocationSeconds();
    fftwRecord.planningSeconds = fftw.planningSeconds();
    fftwRecord.opaquePlanningBytes = fftw.planningBytes();
    fftwRecord.ledger = fftwLedger();
    fftwRecord.correctness.push_back(metric("full inverse round trip", maximumRelativeError(fftwOutput.data(), input.data(), input.size(), 1.0 / static_cast<double>(workload.nx * workload.ny))));

    fftwRecord.timings.push_back(series("primitive", "raw FFT", "forward", StageState::executed,
        report.fullRealBytes + report.fullSpectrumBytes,
        measure(warmups, sampleCount, [&] { fftw.forward(input.data(), workingSpectrum.data()); })));
    fftwRecord.timings.push_back(series("primitive", "raw FFT", "inverse", StageState::executed,
        report.fullSpectrumBytes + report.fullRealBytes,
        measure(warmups, sampleCount,
            [&] { std::copy(referenceSpectrum.begin(), referenceSpectrum.end(), inverseSpectrum.begin()); },
            [&] { fftw.inverse(inverseSpectrum.data(), output.data()); })));
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
    report.providers.push_back(std::move(fftwRecord));

    VDSPProvider vdsp(workload, workers);
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
    vdspRecord.algorithmId = "vdsp-radix2-2d-persistent-outer-batch";
    vdspRecord.nativeRepresentationId = "vdsp-packed-split-complex";
    vdspRecord.modeOrderId = "vdsp-packed-special-boundaries";
    vdspRecord.schedulingId = "persistent-thread-pool";
    vdspRecord.sourceIdentity = "Apple Accelerate system framework";
    vdspRecord.configureFlags = "system framework";
    vdspRecord.compilerFlags = report.environment.compilerFlags;
    vdspRecord.workers = workers;
    vdspRecord.execution = vdspExecutionContract(workload);
    vdspRecord.explicitPersistentBytes = vdsp.explicitPersistentBytes();
    vdspRecord.otherSetupSeconds = vdsp.otherSetupSeconds();
    vdspRecord.allocationSeconds = vdsp.allocationSeconds();
    vdspRecord.planningSeconds = vdsp.planningSeconds();
    vdspRecord.ledger = vdspLedger();
    vdspRecord.correctness = {
        metric("full forward versus FFTW", vdspForwardError),
        metric("full inverse versus FFTW", vdspInverseReferenceError),
        metric("full inverse round trip", vdspRoundTripError),
        metric("retained forward versus FFTW", vdspRetainedError),
        metric("retained inverse versus FFTW", vdspRetainedInverseError)};

    vdspRecord.timings.push_back(series("primitive", "raw FFT", "forward", StageState::executed,
        vdsp.nativeBufferBytes() * 2,
        measure(warmups, sampleCount,
            [&] { vdsp.packForwardInput(input.data()); },
            [&] { vdsp.executeForwardNative(); })));
    vdspRecord.timings.push_back(series("primitive", "raw FFT", "inverse", StageState::executed,
        vdsp.nativeBufferBytes() * 2,
        measure(warmups, sampleCount,
            [&] { vdsp.packInverseInput(referenceSpectrum.data()); },
            [&] { vdsp.executeInverseNative(); })));

    vdspRecord.timings.push_back(series("adapter-component", "real-to-vDSP packing", "forward", StageState::executed,
        report.fullRealBytes + vdsp.nativeBufferBytes(),
        measure(warmups, sampleCount, [&] { vdsp.packForwardInput(input.data()); })));
    vdsp.packForwardInput(input.data());
    vdsp.executeForwardNative();
    vdspRecord.timings.push_back(series("adapter-component", "vDSP-to-WVM conversion and permutation", "forward", StageState::executed,
        vdsp.nativeBufferBytes() + report.fullSpectrumBytes,
        measure(warmups, sampleCount, [&] { vdsp.unpackForwardOutput(workingSpectrum.data()); })));
    vdspRecord.timings.push_back(series("adapter-component", "WVM-to-vDSP conversion and permutation", "inverse", StageState::executed,
        report.fullSpectrumBytes + vdsp.nativeBufferBytes(),
        measure(warmups, sampleCount, [&] { vdsp.packInverseInput(referenceSpectrum.data()); })));
    vdsp.packInverseInput(referenceSpectrum.data());
    vdsp.executeInverseNative();
    vdspRecord.timings.push_back(series("adapter-component", "vDSP-to-real unpacking", "inverse", StageState::executed,
        vdsp.nativeBufferBytes() + report.fullRealBytes,
        measure(warmups, sampleCount, [&] { vdsp.unpackInverseOutput(output.data()); })));

    vdspRecord.timings.push_back(series("adapter-total", "WVM-compatible full-spectrum adapter", "forward", StageState::executed,
        report.fullRealBytes + 2 * vdsp.nativeBufferBytes() + report.fullSpectrumBytes,
        measure(warmups, sampleCount, [&] { vdsp.forwardAdapter(input.data(), workingSpectrum.data()); })));
    vdspRecord.timings.push_back(series("adapter-total", "WVM-compatible full-spectrum adapter", "inverse", StageState::executed,
        report.fullSpectrumBytes + 2 * vdsp.nativeBufferBytes() + report.fullRealBytes,
        measure(warmups, sampleCount, [&] { vdsp.inverseAdapter(referenceSpectrum.data(), output.data()); })));
    vdspRecord.timings.push_back(series("operator-component", "horizontal retention", "forward", StageState::executed,
        report.fullSpectrumBytes + report.retainedSpectrumBytes,
        measure(warmups, sampleCount, [&] { gatherRetained(workload, modes, referenceSpectrum.data(), retainedWorking.data()); })));
    vdspRecord.timings.push_back(series("operator-component", "horizontal embedding", "inverse", StageState::executed,
        report.retainedSpectrumBytes + report.fullSpectrumBytes,
        measure(warmups, sampleCount, [&] { embedRetained(workload, modes, retainedSpectrum.data(), inverseSpectrum.data()); })));
    vdspRecord.timings.push_back(series("uninstrumented-total", "retained horizontal operator", "forward", StageState::executed,
        report.fullRealBytes + 2 * vdsp.nativeBufferBytes() + report.fullSpectrumBytes + report.retainedSpectrumBytes,
        measure(warmups, sampleCount, [&] {
            vdsp.forwardAdapter(input.data(), workingSpectrum.data());
            gatherRetained(workload, modes, workingSpectrum.data(), retainedWorking.data());
        })));
    vdspRecord.timings.push_back(series("uninstrumented-total", "retained horizontal operator", "inverse", StageState::executed,
        report.retainedSpectrumBytes + report.fullSpectrumBytes + 2 * vdsp.nativeBufferBytes() + report.fullRealBytes,
        measure(warmups, sampleCount, [&] {
            embedRetained(workload, modes, retainedSpectrum.data(), inverseSpectrum.data());
            vdsp.inverseAdapter(inverseSpectrum.data(), output.data());
        })));
    report.providers.push_back(std::move(vdspRecord));

    report.status = std::all_of(report.providers.begin(), report.providers.end(), correctnessPassed) ? "passed" : "failed";
    return report;
}

} // namespace skbench
