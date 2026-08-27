#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace skbench {

struct Complex {
    double real = 0.0;
    double imag = 0.0;
};

Complex conjugate(Complex value) noexcept;
double magnitude(Complex value) noexcept;

struct Workload {
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t nz = 0;
    std::size_t fields = 0;
    double lx = 1.0;
    double ly = 1.0;
    bool antialias = true;

    std::size_t planes() const;
    std::size_t nxHalf() const;
    std::size_t realPlaneElements() const;
    std::size_t halfRows() const;
    std::size_t realElements() const;
    std::size_t spectrumElements() const;
    std::size_t retainedVerticalModes() const;
};

struct RetainedMode {
    std::int64_t k = 0;
    std::int64_t l = 0;
    std::size_t storedKx = 0;
    std::size_t storedKy = 0;
    bool conjugatesStoredValue = false;
    double radialMode = 0.0;
};

std::vector<RetainedMode> retainedHorizontalModes(const Workload& workload);
std::size_t realIndex(const Workload& workload, std::size_t x, std::size_t y, std::size_t z, std::size_t field);
std::size_t wvmSpectrumIndex(const Workload& workload, std::size_t kx, std::size_t ky, std::size_t z, std::size_t field);
std::size_t planeMajorSpectrumIndex(const Workload& workload, std::size_t kx, std::size_t ky, std::size_t z, std::size_t field);
std::size_t retainedSpectrumIndex(const Workload& workload, std::size_t mode, std::size_t z, std::size_t field);
std::size_t modalSpectrumIndex(const Workload& workload, std::size_t mode, std::size_t j, std::size_t field);

void gatherRetained(const Workload& workload, const std::vector<RetainedMode>& modes, const Complex* fullSpectrum, Complex* retainedSpectrum);
void embedRetained(const Workload& workload, const std::vector<RetainedMode>& modes, const Complex* retainedSpectrum, Complex* fullSpectrum);
void wvmToPlaneMajor(const Workload& workload, const Complex* wvmSpectrum, Complex* planeMajorSpectrum);
void planeMajorToWvm(const Workload& workload, const Complex* planeMajorSpectrum, Complex* wvmSpectrum);
std::string modeOrderHash(const std::vector<RetainedMode>& modes);
std::string wvmSpectrumOrderHash(const Workload& workload);

struct VerticalOperators {
    std::string id;
    std::size_t nz = 0;
    std::size_t nj = 0;
    std::vector<double> forward;
    std::vector<double> inverse;
};

VerticalOperators orthonormalVerticalFixture(std::size_t nz, std::size_t nj);
void verticalForward(const Workload& workload, std::size_t horizontalModeCount, const VerticalOperators& operators, const Complex* physicalCoefficients, Complex* modalCoefficients);
void verticalInverse(const Workload& workload, std::size_t horizontalModeCount, const VerticalOperators& operators, const Complex* modalCoefficients, Complex* physicalCoefficients);

enum class FixtureKind {
    impulse,
    sinusoid,
    random,
    dc,
    nyquist
};

std::string_view fixtureName(FixtureKind fixture) noexcept;
std::vector<double> makeFixture(const Workload& workload, FixtureKind fixture, std::uint64_t seed);
void directR2C(const Workload& workload, const double* input, Complex* wvmSpectrum);
void directC2R(const Workload& workload, const Complex* wvmSpectrum, double* output);
double maximumRelativeError(const Complex* actual, const Complex* expected, std::size_t count);
double maximumRelativeError(const double* actual, const double* expected, std::size_t count, double actualScale = 1.0);

class FFTWProvider {
public:
    FFTWProvider(const Workload& workload, std::size_t workers);
    ~FFTWProvider();
    FFTWProvider(FFTWProvider&&) noexcept;
    FFTWProvider& operator=(FFTWProvider&&) noexcept;
    FFTWProvider(const FFTWProvider&) = delete;
    FFTWProvider& operator=(const FFTWProvider&) = delete;

    void forward(const double* input, Complex* wvmSpectrum);
    void inverse(Complex* wvmSpectrum, double* output);
    double otherSetupSeconds() const noexcept;
    double allocationSeconds() const noexcept;
    double planningSeconds() const noexcept;
    std::size_t planningBytes() const noexcept;
    std::string libraryIdentity() const;
    std::string version() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class VDSPProvider {
public:
    VDSPProvider(const Workload& workload, std::size_t workers);
    ~VDSPProvider();
    VDSPProvider(VDSPProvider&&) noexcept;
    VDSPProvider& operator=(VDSPProvider&&) noexcept;
    VDSPProvider(const VDSPProvider&) = delete;
    VDSPProvider& operator=(const VDSPProvider&) = delete;

    bool supported() const noexcept;
    std::string capability() const;
    void packForwardInput(const double* input);
    void executeForwardNative();
    void unpackForwardOutput(Complex* wvmSpectrum) const;
    void packInverseInput(const Complex* wvmSpectrum);
    void executeInverseNative();
    void unpackInverseOutput(double* output) const;
    void forwardAdapter(const double* input, Complex* wvmSpectrum);
    void inverseAdapter(const Complex* wvmSpectrum, double* output);
    double otherSetupSeconds() const noexcept;
    double allocationSeconds() const noexcept;
    double planningSeconds() const noexcept;
    std::size_t nativeBufferBytes() const noexcept;
    std::size_t explicitPersistentBytes() const noexcept;
    std::string libraryIdentity() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

enum class StageState {
    executed,
    fused,
    elided,
    setupOnly,
    unsupported
};

std::string_view stageStateName(StageState state) noexcept;

struct TimingSeries {
    std::string scope;
    std::string stage;
    std::string direction;
    StageState state = StageState::executed;
    std::uint64_t bytesMoved = 0;
    std::vector<double> seconds;
};

struct LedgerEntry {
    std::string stage;
    StageState state = StageState::unsupported;
    std::string detail;
};

struct CorrectnessMetric {
    std::string name;
    double maximumRelativeError = 0.0;
    double tolerance = 1.0e-12;
    bool passed = false;
};

struct DirectionExecutionContract {
    std::string nativePlacement;
    std::string adapterPlacement;
    bool destroysNativeInput = false;
    bool adapterPreservesCallerInput = false;
    bool requiresPreservationCopyForRepeatedExecution = false;
    bool preservationIncludedInPrimitiveTiming = false;
    bool preservationIncludedInAdapterTiming = false;
    std::string nativeInputRepresentationId;
    std::string nativeOutputRepresentationId;
    std::string adapterInputRepresentationId;
    std::string adapterOutputRepresentationId;
    std::string physicalExtents;
    std::string stridesElements;
    std::size_t paddingElements = 0;
    std::size_t minimumAlignmentBytes = 0;
    std::string aliasing;
    std::size_t reusableWorkBytes = 0;
    bool outputCanFeedOppositeDirection = false;
};

struct ExecutionContract {
    DirectionExecutionContract forward;
    DirectionExecutionContract inverse;
};

struct ProviderRecord {
    std::string id;
    std::string version;
    std::string libraryIdentity;
    std::string algorithmId;
    std::string nativeRepresentationId;
    std::string modeOrderId;
    std::string schedulingId;
    std::string sourceIdentity;
    std::string sourceSha256;
    std::string configureFlags;
    std::string compilerFlags;
    std::size_t workers = 1;
    ExecutionContract execution;
    std::size_t explicitPersistentBytes = 0;
    std::size_t scratchBytes = 0;
    std::size_t opaquePlanningBytes = 0;
    double otherSetupSeconds = 0.0;
    double allocationSeconds = 0.0;
    double planningSeconds = 0.0;
    std::vector<TimingSeries> timings;
    std::vector<LedgerEntry> ledger;
    std::vector<CorrectnessMetric> correctness;
};

struct EnvironmentRecord {
    std::string timestampUtc;
    std::string hostname;
    std::string operatingSystem;
    std::string machineModel;
    std::string cpuBrand;
    std::size_t totalCores = 0;
    std::size_t performanceCores = 0;
    std::size_t efficiencyCores = 0;
    std::uint64_t physicalMemoryBytes = 0;
    std::string compiler;
    std::string compilerVersion;
    std::string compilerFlags;
    std::string buildType;
    std::string gitCommit;
    bool gitDirty = false;
};

struct BenchmarkReport {
    std::string schema = "spectral-kernel-benchmark-v1";
    std::string status;
    std::string scalarTypeId = "float64";
    std::size_t scalarBits = 64;
    std::string runId;
    std::string profile;
    std::uint64_t seed = 0;
    std::size_t warmups = 0;
    std::size_t samples = 0;
    Workload workload;
    std::size_t retainedHorizontalModeCount = 0;
    std::string retainedModeOrderHash;
    std::string wvmFullSpectrumOrderHash;
    std::uint64_t fullRealBytes = 0;
    std::uint64_t fullSpectrumBytes = 0;
    std::uint64_t retainedSpectrumBytes = 0;
    EnvironmentRecord environment;
    std::vector<ProviderRecord> providers;
};

struct Profile {
    std::string name;
    std::string purpose;
    Workload workload;
    std::size_t defaultWorkers = 1;
    std::size_t warmups = 1;
    std::size_t samples = 3;
};

std::vector<Profile> profiles();
Profile profileNamed(std::string_view name);

struct RunOptions {
    std::string profile = "quick";
    std::size_t workers = 0;
    std::size_t warmups = 0;
    std::size_t samples = 0;
    std::uint64_t seed = 129;
    std::filesystem::path outputJson;
};

struct ValidationReport {
    bool passed = false;
    std::vector<std::string> messages;
};

BenchmarkReport runBenchmark(const RunOptions& options);
ValidationReport validateBenchmark(std::string_view profileName);
EnvironmentRecord environmentRecord();
double median(std::vector<double> values);
void writeJson(const BenchmarkReport& report, const std::filesystem::path& path);
void writeCsv(const BenchmarkReport& report, const std::filesystem::path& path);
int compareCsv(const std::filesystem::path& path, std::ostream& output);

} // namespace skbench
