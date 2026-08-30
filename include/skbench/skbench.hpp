#pragma once

#include <array>
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

struct VerticalModeGroup {
    std::uint64_t squaredModeKey = 0;
    std::size_t firstMode = 0;
    std::size_t modeCount = 0;

    bool operator==(const VerticalModeGroup&) const = default;
};

std::vector<RetainedMode> retainedHorizontalModes(const Workload& workload);
std::vector<VerticalModeGroup> squaredWavenumberGroups(const std::vector<RetainedMode>& modes);
std::string verticalModeGroupHash(const std::vector<VerticalModeGroup>& groups);
std::size_t realIndex(const Workload& workload, std::size_t x, std::size_t y, std::size_t z, std::size_t field);
std::size_t wvmSpectrumIndex(const Workload& workload, std::size_t kx, std::size_t ky, std::size_t z, std::size_t field);
std::size_t planeMajorSpectrumIndex(const Workload& workload, std::size_t kx, std::size_t ky, std::size_t z, std::size_t field);
std::size_t retainedSpectrumIndex(const Workload& workload, std::size_t mode, std::size_t z, std::size_t field);
std::size_t modalSpectrumIndex(const Workload& workload, std::size_t mode, std::size_t j, std::size_t field);
std::size_t wvmModalSpectrumIndex(const Workload& workload, std::size_t kx, std::size_t ky,
                                  std::size_t j, std::size_t field);

std::vector<double> syntheticModalWorkWeights(
    const Workload& workload, const std::vector<RetainedMode>& modes);
void applySyntheticModalWorkInterleaved(
    std::size_t count, const double* weights, const Complex* input, Complex* output) noexcept;
void applySyntheticModalWorkSplit(
    std::size_t count, const double* weights,
    const double* inputReal, const double* inputImaginary,
    double* outputReal, double* outputImaginary) noexcept;
void applySyntheticModalWorkWvm(
    const Workload& workload, const std::vector<RetainedMode>& modes,
    const double* weights, const Complex* input, Complex* output);

void gatherRetained(const Workload& workload, const std::vector<RetainedMode>& modes, const Complex* fullSpectrum, Complex* retainedSpectrum);
void embedRetained(const Workload& workload, const std::vector<RetainedMode>& modes, const Complex* retainedSpectrum, Complex* fullSpectrum);
void gatherRetainedModal(const Workload& workload, const std::vector<RetainedMode>& modes,
                         const Complex* fullModalSpectrum, Complex* retainedModalSpectrum);
void embedRetainedModal(const Workload& workload, const std::vector<RetainedMode>& modes,
                        const Complex* retainedModalSpectrum, Complex* fullModalSpectrum);
void interleavedToSplit(std::size_t count, const Complex* interleaved, double* real, double* imag);
void splitToInterleaved(std::size_t count, const double* real, const double* imag, Complex* interleaved);
void gatherRetainedSplit(const Workload& workload, const std::vector<RetainedMode>& modes,
                         const double* fullReal, const double* fullImag,
                         double* retainedReal, double* retainedImag);
void embedRetainedSplit(const Workload& workload, const std::vector<RetainedMode>& modes,
                        const double* retainedReal, const double* retainedImag,
                        double* fullReal, double* fullImag);
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

struct GroupedVerticalOperators {
    std::string id;
    std::size_t nz = 0;
    std::size_t nj = 0;
    std::vector<VerticalModeGroup> groups;
    std::vector<std::size_t> matrixSourceGroups;
    std::vector<double> forward;
    std::vector<double> inverse;
};

VerticalOperators orthonormalVerticalFixture(std::size_t nz, std::size_t nj);
GroupedVerticalOperators commonVerticalFixture(std::size_t horizontalModeCount, const VerticalOperators& operators);
GroupedVerticalOperators squaredWavenumberVerticalFixture(
    const Workload& workload, const std::vector<RetainedMode>& modes);
void verticalForward(const Workload& workload, std::size_t horizontalModeCount, const VerticalOperators& operators, const Complex* physicalCoefficients, Complex* modalCoefficients);
void verticalInverse(const Workload& workload, std::size_t horizontalModeCount, const VerticalOperators& operators, const Complex* modalCoefficients, Complex* physicalCoefficients);

enum class VerticalGemmLayout {
    complexInterleaved,
    split
};

std::string_view verticalGemmLayoutName(VerticalGemmLayout layout) noexcept;

enum class VerticalGemmBufferPolicy {
    bidirectional,
    forwardOnly,
    inverseOnly
};

enum class VerticalGemmSchedule {
    serial,
    outerStatic,
    outerDynamic
};

std::string_view verticalGemmScheduleName(VerticalGemmSchedule schedule) noexcept;
VerticalGemmSchedule verticalGemmScheduleNamed(std::string_view name);

struct VerticalGemmStrategy {
    VerticalGemmSchedule schedule = VerticalGemmSchedule::serial;
    std::size_t outerWorkers = 1;
};

class VerticalGemmProvider {
public:
    VerticalGemmProvider(const Workload& workload, std::size_t horizontalModeCount,
                         const VerticalOperators& operators, VerticalGemmLayout layout);
    VerticalGemmProvider(const Workload& workload, const GroupedVerticalOperators& operators,
                         VerticalGemmLayout layout);
    VerticalGemmProvider(const Workload& workload, const GroupedVerticalOperators& operators,
                         VerticalGemmLayout layout, VerticalGemmStrategy strategy);
    VerticalGemmProvider(const Workload& workload, const GroupedVerticalOperators& operators,
                         VerticalGemmLayout layout, VerticalGemmStrategy strategy,
                         VerticalGemmBufferPolicy bufferPolicy);
    ~VerticalGemmProvider();
    VerticalGemmProvider(VerticalGemmProvider&&) noexcept;
    VerticalGemmProvider& operator=(VerticalGemmProvider&&) noexcept;
    VerticalGemmProvider(const VerticalGemmProvider&) = delete;
    VerticalGemmProvider& operator=(const VerticalGemmProvider&) = delete;

    bool supported() const noexcept;
    std::string capability() const;
    VerticalGemmLayout layout() const noexcept;
    std::size_t columns() const noexcept;
    std::size_t physicalElements() const noexcept;
    std::size_t modalElements() const noexcept;
    std::size_t groupCount() const noexcept;
    std::size_t gemmCallsPerExecution() const noexcept;
    VerticalGemmStrategy strategy() const noexcept;
    std::size_t outerWorkers() const noexcept;
    std::size_t persistentBytes() const noexcept;
    std::size_t schedulerPersistentBytes() const noexcept;
    std::size_t matrixBytesPerDirection() const noexcept;
    std::size_t minimumAlignmentBytes() const noexcept;
    double allocationSeconds() const noexcept;
    double matrixPreparationSeconds() const noexcept;
    double schedulerSetupSeconds() const noexcept;
    bool hasOpaqueSchedulerMemory() const noexcept;
    std::string libraryIdentity() const;

    void loadPhysicalInput(const Complex* input);
    void loadModalInput(const Complex* input);
    void packPhysicalInputFromWvm(const std::vector<RetainedMode>& modes, const Complex* wvmSpectrum);
    void executeForward();
    void executeInverse();
    void executeForwardReal();
    void executeForwardImaginary();
    void executeInverseReal();
    void executeInverseImaginary();
    void executeSchedulerNoop();
    Complex* interleavedPhysicalInputData();
    Complex* interleavedModalInputData();
    const Complex* interleavedModalOutputData() const;
    const Complex* interleavedPhysicalOutputData() const;
    double* splitPhysicalInputRealData();
    double* splitPhysicalInputImaginaryData();
    double* splitModalInputRealData();
    double* splitModalInputImaginaryData();
    const double* splitModalOutputRealData() const;
    const double* splitModalOutputImaginaryData() const;
    const double* splitPhysicalOutputRealData() const;
    const double* splitPhysicalOutputImaginaryData() const;
    void copyForwardOutput(Complex* output) const;
    void copyInverseOutput(Complex* output) const;
    void embedPhysicalOutputToWvm(const std::vector<RetainedMode>& modes, Complex* wvmSpectrum) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class WvmDirectVerticalGemmProvider {
public:
    WvmDirectVerticalGemmProvider(const Workload& workload,
                                  const std::vector<RetainedMode>& modes,
                                  const GroupedVerticalOperators& operators,
                                  VerticalGemmStrategy strategy,
                                  VerticalGemmBufferPolicy bufferPolicy =
                                      VerticalGemmBufferPolicy::bidirectional);
    ~WvmDirectVerticalGemmProvider();
    WvmDirectVerticalGemmProvider(WvmDirectVerticalGemmProvider&&) noexcept;
    WvmDirectVerticalGemmProvider& operator=(WvmDirectVerticalGemmProvider&&) noexcept;
    WvmDirectVerticalGemmProvider(const WvmDirectVerticalGemmProvider&) = delete;
    WvmDirectVerticalGemmProvider& operator=(const WvmDirectVerticalGemmProvider&) = delete;

    bool supported() const noexcept;
    std::string capability() const;
    void initializeModalOutput(Complex* fullModalSpectrum) const;
    void initializeSpectrumOutput(Complex* fullSpectrum) const;
    void executeForward(const Complex* fullSpectrum, Complex* fullModalSpectrum);
    void executeInverse(const Complex* fullModalSpectrum, Complex* fullSpectrum);
    void executeSchedulerNoop();
    std::size_t modalSpectrumElements() const noexcept;
    std::size_t gemmCallsPerExecution() const noexcept;
    std::size_t outerWorkers() const noexcept;
    VerticalGemmStrategy strategy() const noexcept;
    std::size_t persistentBytes() const noexcept;
    std::size_t schedulerPersistentBytes() const noexcept;
    std::size_t matrixBytesPerDirection() const noexcept;
    double allocationSeconds() const noexcept;
    double matrixPreparationSeconds() const noexcept;
    double schedulerSetupSeconds() const noexcept;
    bool hasOpaqueSchedulerMemory() const noexcept;
    std::string libraryIdentity() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class PlaneMajorDirectVerticalGemmProvider {
public:
    PlaneMajorDirectVerticalGemmProvider(const Workload& workload,
                                         const std::vector<RetainedMode>& modes,
                                         const GroupedVerticalOperators& operators,
                                         VerticalGemmStrategy strategy);
    ~PlaneMajorDirectVerticalGemmProvider();
    PlaneMajorDirectVerticalGemmProvider(PlaneMajorDirectVerticalGemmProvider&&) noexcept;
    PlaneMajorDirectVerticalGemmProvider& operator=(PlaneMajorDirectVerticalGemmProvider&&) noexcept;
    PlaneMajorDirectVerticalGemmProvider(const PlaneMajorDirectVerticalGemmProvider&) = delete;
    PlaneMajorDirectVerticalGemmProvider& operator=(const PlaneMajorDirectVerticalGemmProvider&) = delete;

    bool supported() const noexcept;
    std::string capability() const;
    void initializeModalOutput(Complex* fullModalSpectrum) const;
    void initializeSpectrumOutput(Complex* fullSpectrum) const;
    void executeForward(const Complex* fullSpectrum, Complex* fullModalSpectrum);
    void executeInverse(const Complex* fullModalSpectrum, Complex* fullSpectrum);
    void executeSchedulerNoop();
    std::size_t modalSpectrumElements() const noexcept;
    std::size_t gemvCallsPerExecution() const noexcept;
    std::size_t outerWorkers() const noexcept;
    VerticalGemmStrategy strategy() const noexcept;
    std::size_t persistentBytes() const noexcept;
    std::size_t schedulerPersistentBytes() const noexcept;
    std::size_t matrixBytesPerDirection() const noexcept;
    double allocationSeconds() const noexcept;
    double matrixPreparationSeconds() const noexcept;
    double schedulerSetupSeconds() const noexcept;
    bool hasOpaqueSchedulerMemory() const noexcept;
    std::string libraryIdentity() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

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

enum class FFTWPlanningMode {
    estimate,
    measure,
    patient,
    exhaustive
};

std::string_view fftwPlanningModeName(FFTWPlanningMode mode) noexcept;
FFTWPlanningMode fftwPlanningModeNamed(std::string_view name);

enum class FFTWAlignmentStrategy {
    aligned,
    unaligned
};

std::string_view fftwAlignmentStrategyName(FFTWAlignmentStrategy strategy) noexcept;
FFTWAlignmentStrategy fftwAlignmentStrategyNamed(std::string_view name);

enum class FFTWWisdomStrategy {
    cold,
    generatedImport
};

std::string_view fftwWisdomStrategyName(FFTWWisdomStrategy strategy) noexcept;
FFTWWisdomStrategy fftwWisdomStrategyNamed(std::string_view name);

enum class FFTWDataLayout {
    interleaved,
    split
};

std::string_view fftwDataLayoutName(FFTWDataLayout layout) noexcept;
FFTWDataLayout fftwDataLayoutNamed(std::string_view name);

enum class FFTWSpectrumOrder {
    wvmFrequencyMajor,
    planeMajor
};

std::string_view fftwSpectrumOrderName(FFTWSpectrumOrder order) noexcept;
FFTWSpectrumOrder fftwSpectrumOrderNamed(std::string_view name);

struct FFTWStrategy {
    FFTWPlanningMode planningMode = FFTWPlanningMode::measure;
    FFTWAlignmentStrategy alignment = FFTWAlignmentStrategy::unaligned;
    FFTWWisdomStrategy wisdom = FFTWWisdomStrategy::cold;
    std::size_t internalWorkers = 1;
    std::size_t outerWorkers = 1;
    double planningTimeLimitSeconds = 0.0;
    FFTWDataLayout layout = FFTWDataLayout::interleaved;
    FFTWSpectrumOrder spectrumOrder = FFTWSpectrumOrder::wvmFrequencyMajor;
};

class FFTWProvider {
public:
    FFTWProvider(const Workload& workload, std::size_t workers);
    FFTWProvider(const Workload& workload, FFTWStrategy strategy);
    ~FFTWProvider();
    FFTWProvider(FFTWProvider&&) noexcept;
    FFTWProvider& operator=(FFTWProvider&&) noexcept;
    FFTWProvider(const FFTWProvider&) = delete;
    FFTWProvider& operator=(const FFTWProvider&) = delete;

    void forward(const double* input, Complex* wvmSpectrum);
    void inverse(Complex* wvmSpectrum, double* output);
    void gatherRetainedOuter(const std::vector<RetainedMode>& modes,
                             const Complex* wvmSpectrum, Complex* retainedSpectrum);
    void embedRetainedOuter(const std::vector<RetainedMode>& modes,
                            const Complex* retainedSpectrum, Complex* wvmSpectrum);
    void gatherRetainedToSplitOuter(const std::vector<RetainedMode>& modes,
                                    const Complex* spectrum,
                                    double* retainedReal, double* retainedImag,
                                    double scale = 1.0);
    void embedRetainedFromSplitOuter(const std::vector<RetainedMode>& modes,
                                    const double* retainedReal, const double* retainedImag,
                                    Complex* spectrum);
    void scaleRetainedSplitOuter(const std::vector<RetainedMode>& modes,
                                 double* retainedReal, double* retainedImag,
                                 double scale);
    void forwardSplit(const double* input, double* wvmSpectrumReal, double* wvmSpectrumImag);
    void inverseSplit(double* wvmSpectrumReal, double* wvmSpectrumImag, double* output);
    void gatherRetainedSplitOuter(const std::vector<RetainedMode>& modes,
                                  const double* spectrumReal, const double* spectrumImag,
                                  double* retainedReal, double* retainedImag);
    void embedRetainedSplitOuter(const std::vector<RetainedMode>& modes,
                                 const double* retainedReal, const double* retainedImag,
                                 double* spectrumReal, double* spectrumImag);
    void executeSchedulerNoop();
    bool splitInPlaceWvmOrderSupported() const noexcept;
    std::string splitInPlaceWvmOrderCapability() const;
    double otherSetupSeconds() const noexcept;
    double allocationSeconds() const noexcept;
    double planningSeconds() const noexcept;
    double wisdomGenerationSeconds() const noexcept;
    double wisdomImportSeconds() const noexcept;
    double planningTimeLimitSeconds() const noexcept;
    bool planningBudgetExhausted() const noexcept;
    std::size_t wisdomBytes() const noexcept;
    std::size_t planningBytes() const noexcept;
    std::size_t internalWorkers() const noexcept;
    std::size_t outerWorkers() const noexcept;
    std::size_t totalLogicalWorkers() const noexcept;
    std::size_t minimumAlignmentBytes() const noexcept;
    FFTWStrategy strategy() const noexcept;
    std::string libraryIdentity() const;
    std::string version() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class FFTWPrunedProvider {
public:
    FFTWPrunedProvider(const Workload& workload, const std::vector<RetainedMode>& modes,
                       FFTWPlanningMode planningMode, std::size_t internalWorkers,
                       std::size_t outerWorkers = 1);
    ~FFTWPrunedProvider();
    FFTWPrunedProvider(FFTWPrunedProvider&&) noexcept;
    FFTWPrunedProvider& operator=(FFTWPrunedProvider&&) noexcept;
    FFTWPrunedProvider(const FFTWPrunedProvider&) = delete;
    FFTWPrunedProvider& operator=(const FFTWPrunedProvider&) = delete;

    void executeForwardRows(const double* input);
    void executeForwardColumns();
    void gatherForward(Complex* retainedSpectrum);
    void gatherForwardSplit(double* retainedReal, double* retainedImag,
                            double scale = 1.0);
    void forward(const double* input, Complex* retainedSpectrum);
    void forwardSplit(const double* input, double* retainedReal,
                      double* retainedImag, double scale = 1.0);
    void embedInverse(const Complex* retainedSpectrum);
    void embedInverseSplit(const double* retainedReal, const double* retainedImag);
    void executeInverseColumns();
    void executeInverseRows(double* output);
    void inverse(const Complex* retainedSpectrum, double* output);
    void inverseSplit(const double* retainedReal, const double* retainedImag,
                      double* output);
    void executeSchedulerNoop();

    std::size_t activeKxCount() const noexcept;
    std::size_t fullKxCount() const noexcept;
    std::size_t rowTransformsPerDirection() const noexcept;
    std::size_t columnTransformsPerDirection() const noexcept;
    std::size_t omittedColumnTransformsPerDirection() const noexcept;
    std::size_t scratchBytes() const noexcept;
    std::size_t planningBytes() const noexcept;
    std::size_t minimumAlignmentBytes() const noexcept;
    std::size_t internalWorkers() const noexcept;
    std::size_t outerWorkers() const noexcept;
    std::size_t totalLogicalWorkers() const noexcept;
    std::size_t maximumShardScratchBytes() const noexcept;
    double otherSetupSeconds() const noexcept;
    double allocationSeconds() const noexcept;
    double planningSeconds() const noexcept;
    FFTWPlanningMode planningMode() const noexcept;
    bool completeHalfSpectrumOutputMaterialized() const noexcept;
    bool inPlaceRetainedOperatorSupported() const noexcept;
    std::string inPlaceRetainedOperatorCapability() const;
    std::string libraryIdentity() const;
    std::string version() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

enum class StreamingInversePreparationPolicy {
    fullZero,
    activeReset,
    compactPreserved,
    fullPreserved
};

std::string_view streamingInversePreparationPolicyName(
    StreamingInversePreparationPolicy policy) noexcept;
StreamingInversePreparationPolicy streamingInversePreparationPolicyNamed(
    std::string_view name);

class FFTWStreamingPrunedSplitProvider {
public:
    struct ConstFieldView {
        const double* real = nullptr;
        const double* imaginary = nullptr;
        std::size_t modeStride = 0;
    };

    struct FieldView {
        double* real = nullptr;
        double* imaginary = nullptr;
        std::size_t modeStride = 0;
    };

    FFTWStreamingPrunedSplitProvider(
        const Workload& workload, const std::vector<RetainedMode>& modes,
        FFTWPlanningMode planningMode, std::size_t internalWorkers,
        std::size_t outerWorkers, std::size_t tileWidth = 1,
        StreamingInversePreparationPolicy inversePreparationPolicy =
            StreamingInversePreparationPolicy::fullZero);
    ~FFTWStreamingPrunedSplitProvider();
    FFTWStreamingPrunedSplitProvider(FFTWStreamingPrunedSplitProvider&&) noexcept;
    FFTWStreamingPrunedSplitProvider& operator=(
        FFTWStreamingPrunedSplitProvider&&) noexcept;
    FFTWStreamingPrunedSplitProvider(
        const FFTWStreamingPrunedSplitProvider&) = delete;
    FFTWStreamingPrunedSplitProvider& operator=(
        const FFTWStreamingPrunedSplitProvider&) = delete;

    void forwardSplit(const double* input, double* retainedReal,
                      double* retainedImag, double scale = 1.0);
    void inverseSplit(const double* retainedReal, const double* retainedImag,
                      double* output);
    void forwardSplitFields(const double* input, const FieldView* fields,
                            std::size_t fieldCount, double scale = 1.0);
    void inverseSplitFields(const ConstFieldView* fields,
                            std::size_t fieldCount, double* output);

    void executeForwardRowsDiagnostic(const double* input);
    void executeForwardColumnsDiagnostic();
    void writeForwardSplitDiagnostic(double* retainedReal,
                                     double* retainedImag,
                                     double scale = 1.0);
    void embedInverseSplitDiagnostic(const double* retainedReal,
                                     const double* retainedImag);
    void writeForwardSplitFieldsDiagnostic(const FieldView* fields,
                                           std::size_t fieldCount,
                                           double scale = 1.0);
    void embedInverseSplitFieldsDiagnostic(const ConstFieldView* fields,
                                           std::size_t fieldCount);
    void loadInverseSplitFieldsDiagnostic(const ConstFieldView* fields,
                                          std::size_t fieldCount);
    void clearInverseInputDiagnostic();
    void scatterInverseInputDiagnostic();
    void executeInverseColumnsDiagnostic();
    void executeInverseRowsDiagnostic(double* output);
    void executeSchedulerNoop();

    std::size_t activeKxCount() const noexcept;
    std::size_t fullKxCount() const noexcept;
    std::size_t rowTransformsPerDirection() const noexcept;
    std::size_t columnTransformsPerDirection() const noexcept;
    std::size_t omittedColumnTransformsPerDirection() const noexcept;
    std::size_t scratchBytes() const noexcept;
    std::size_t workerScratchBytes() const noexcept;
    std::size_t fftScratchBytes() const noexcept;
    std::size_t inverseInputScratchBytes() const noexcept;
    std::size_t compactTileBytes() const noexcept;
    std::size_t tileWidth() const noexcept;
    std::size_t planningBytes() const noexcept;
    std::size_t minimumAlignmentBytes() const noexcept;
    std::size_t internalWorkers() const noexcept;
    std::size_t outerWorkers() const noexcept;
    std::size_t totalLogicalWorkers() const noexcept;
    double otherSetupSeconds() const noexcept;
    double allocationSeconds() const noexcept;
    double planningSeconds() const noexcept;
    FFTWPlanningMode planningMode() const noexcept;
    StreamingInversePreparationPolicy inversePreparationPolicy() const noexcept;
    std::uint64_t inverseTileLoadBytesPerExecution() const noexcept;
    std::uint64_t inverseClearBytesPerExecution() const noexcept;
    std::uint64_t inverseScatterBytesPerExecution() const noexcept;
    bool rowInversePreservesInput() const noexcept;
    bool completeHalfSpectrumMaterialized() const noexcept;
    std::string libraryIdentity() const;
    std::string version() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

enum class VDSPTransformStrategy {
    inPlace,
    inPlaceExplicitScratch,
    outOfPlace,
    outOfPlaceExplicitScratch
};

std::string_view vdspTransformStrategyName(VDSPTransformStrategy strategy) noexcept;
VDSPTransformStrategy vdspTransformStrategyNamed(std::string_view name);

enum class VDSPBatchStrategy {
    directPersistent,
    directGcd,
    separablePersistent,
    separableGcd
};

std::string_view vdspBatchStrategyName(VDSPBatchStrategy strategy) noexcept;
VDSPBatchStrategy vdspBatchStrategyNamed(std::string_view name);

class VDSPProvider {
public:
    VDSPProvider(const Workload& workload, std::size_t workers, VDSPTransformStrategy strategy,
                 VDSPBatchStrategy batchStrategy = VDSPBatchStrategy::directPersistent);
    ~VDSPProvider();
    VDSPProvider(VDSPProvider&&) noexcept;
    VDSPProvider& operator=(VDSPProvider&&) noexcept;
    VDSPProvider(const VDSPProvider&) = delete;
    VDSPProvider& operator=(const VDSPProvider&) = delete;

    bool supported() const noexcept;
    std::string capability() const;
    void packForwardInput(const double* input);
    void executeForwardNative();
    void executeForwardRowsNative();
    void executeForwardColumnsNative();
    void unpackForwardOutput(Complex* wvmSpectrum) const;
    void gatherRetainedNativeSplit(const std::vector<RetainedMode>& modes,
                                   double* retainedReal, double* retainedImag);
    void packInverseInput(const Complex* wvmSpectrum);
    void embedRetainedNativeSplit(const std::vector<RetainedMode>& modes,
                                  const double* retainedReal, const double* retainedImag);
    void executeInverseNative();
    void executeInverseColumnsNative();
    void executeInverseRowsNative();
    void executeSchedulerNoop();
    void unpackInverseOutput(double* output) const;
    void forwardAdapter(const double* input, Complex* wvmSpectrum);
    void inverseAdapter(const Complex* wvmSpectrum, double* output);
    void forwardRetainedNativeSplit(const double* input,
                                    const std::vector<RetainedMode>& modes,
                                    double* retainedReal, double* retainedImag);
    void inverseRetainedNativeSplit(const std::vector<RetainedMode>& modes,
                                    const double* retainedReal, const double* retainedImag,
                                    double* output);
    double otherSetupSeconds() const noexcept;
    double allocationSeconds() const noexcept;
    double planningSeconds() const noexcept;
    VDSPTransformStrategy strategy() const noexcept;
    VDSPBatchStrategy batchStrategy() const noexcept;
    bool separable() const noexcept;
    std::size_t workers() const noexcept;
    std::size_t nativeOperandBytes() const noexcept;
    std::size_t nativeBufferBytes() const noexcept;
    std::size_t explicitPersistentBytes() const noexcept;
    std::size_t scratchBytes() const noexcept;
    std::size_t minimumAlignmentBytes() const noexcept;
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
    double relativeL2Error = 0.0;
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
    std::string planningConfiguration;
    std::size_t workers = 1;
    std::size_t internalWorkers = 1;
    std::size_t outerWorkers = 1;
    std::size_t gemmCallsPerExecution = 0;
    ExecutionContract execution;
    std::size_t explicitPersistentBytes = 0;
    std::size_t scratchBytes = 0;
    std::uint64_t algorithmResidentBytes = 0;
    std::uint64_t benchmarkHarnessBytes = 0;
    std::uint64_t estimatedProcessPeakBytes = 0;
    std::uint64_t observedProcessHighWaterBytes = 0;
    bool opaqueProviderMemory = true;
    std::size_t opaquePlanningBytes = 0;
    double otherSetupSeconds = 0.0;
    double allocationSeconds = 0.0;
    double planningSeconds = 0.0;
    double wisdomGenerationSeconds = 0.0;
    double wisdomImportSeconds = 0.0;
    double planningTimeLimitSeconds = 0.0;
    bool planningBudgetExhausted = false;
    std::size_t wisdomBytes = 0;
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

struct FixtureProvenanceRecord {
    std::string status = "provider-independent-synthetic-development";
    std::string schema = "spectral-flux-fixture-v1";
    std::string waveVortexModelRepository = "JeffreyEarly/wave-vortex-model";
    std::string waveVortexModelCommit;
    std::string generatorIdentity;
    std::string fixtureHash;
    std::string normalization;
    std::string modeMapping;
    std::string derivativeConvention;
    bool authoritative = false;
};

struct SpectralFluxFixture {
    std::string fixtureId;
    std::string waveVortexModelRepository;
    std::string waveVortexModelCommit;
    std::string generatorIdentity;
    std::string fixtureHash;
    std::string normalization;
    std::string modeMapping;
    std::string derivativeConvention;
    Workload workload;
    double lz = 0.0;
    double latitude = 0.0;
    double pointwiseScale = 0.0;
    std::vector<RetainedMode> modes;
    std::vector<std::uint32_t> modeGroupIndices;
    std::vector<std::uint64_t> groupKeys;
    std::vector<std::uint32_t> inputFieldFamilies;
    std::vector<std::uint32_t> targetFieldFamilies;
    std::array<GroupedVerticalOperators, 2> operatorFamilies;
    std::uint64_t verticalOperatorSourceBytes = 0;
    std::vector<Complex> modalInputs;
    std::vector<Complex> expectedModalTargets;
};

SpectralFluxFixture loadPreparedSpectralFluxFixture(
    const std::filesystem::path& path);

struct ConstantStratificationFluxFixture {
    std::string fixtureId;
    std::string waveVortexModelRepository;
    std::string waveVortexModelCommit;
    std::string generatorIdentity;
    std::string fixtureHash;
    std::string normalization;
    std::string modeMapping;
    std::string coefficientContract;
    std::string compiledModuleSha256;
    Workload workload;
    double lz = 0.0;
    double n0 = 0.0;
    double rotationRate = 0.0;
    double latitude = 0.0;
    double gravity = 0.0;
    double elapsedTime = 0.0;
    double pointwiseScale = 0.0;
    double oracleMaximumScaleNormalizedError = 0.0;
    double oracleRelativeL2Error = 0.0;
    std::vector<RetainedMode> modes;
    std::vector<Complex> modalState;
    std::vector<Complex> expectedModalFlux;
};

ConstantStratificationFluxFixture
loadPreparedConstantStratificationFluxFixture(
    const std::filesystem::path& path);

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
    std::uint64_t modalSpectrumBytes = 0;
    std::uint64_t verticalMatrixFamilySourceBytes = 0;
    std::uint64_t verticalBenchmarkEstimatedExplicitPeakBytes = 0;
    std::uint64_t orderingPackingEstimatedExplicitPeakBytes = 0;
    std::uint64_t spectralPipelineEstimatedExplicitPeakBytes = 0;
    std::string verticalMatrixFamilyId = "orthonormal-dct2-truncated-v1";
    std::size_t verticalGroupCount = 0;
    std::size_t minimumVerticalGroupModes = 0;
    double medianVerticalGroupModes = 0.0;
    std::size_t maximumVerticalGroupModes = 0;
    std::size_t minimumVerticalGroupColumns = 0;
    double medianVerticalGroupColumns = 0.0;
    std::size_t maximumVerticalGroupColumns = 0;
    std::string verticalGroupOrderHash;
    FixtureProvenanceRecord fixtureProvenance;
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
    std::string kernel = "fft";
    std::string profile = "quick";
    std::string providers = "both";
    std::string fftwLayout = "interleaved";
    std::string fftwSpectrumOrder = "wvm";
    std::string fftwPlanning = "measure";
    std::string fftwAlignment = "unaligned";
    std::string fftwWisdom = "cold";
    std::string retainedRepresentation = "interleaved";
    std::size_t fftwInternalWorkers = 0;
    std::size_t fftwOuterWorkers = 1;
    double fftwPlanningTimeLimitSeconds = 0.0;
    std::string vdspStrategy = "in-place";
    std::string vdspBatchStrategy = "direct-persistent";
    std::string verticalGemmFamily = "common";
    std::string verticalGemmSchedule = "serial";
    std::size_t verticalGemmOuterWorkers = 1;
    std::string boundaryPolicy = "wvm-packed-split";
    std::size_t streamingTileWidth = 1;
    std::string streamingInversePolicy = "full-zero";
    std::string pointwisePolicy = "serial";
    std::size_t pointwiseWorkers = 0;
    std::string convolutionMap = "independent-products";
    std::string convolutionCandidate = "all";
    std::size_t convolutionProducts = 4;
    std::size_t convolutionCenteredM = 0;
    std::size_t workers = 0;
    std::size_t warmups = 0;
    std::size_t samples = 0;
    std::uint64_t seed = 129;
    std::filesystem::path spectralFluxFixture;
    std::filesystem::path constantStratificationFluxFixture;
    std::filesystem::path outputJson;
};

struct ValidationReport {
    bool passed = false;
    std::vector<std::string> messages;
};

BenchmarkReport runBenchmark(const RunOptions& options);
BenchmarkReport runPrunedHorizontalBenchmark(const RunOptions& options);
BenchmarkReport runVerticalGemmBenchmark(const RunOptions& options);
BenchmarkReport runOrderingPackingBenchmark(const RunOptions& options);
BenchmarkReport runSpectralBoundaryBenchmark(const RunOptions& options);
BenchmarkReport runSpectralPipelineBenchmark(const RunOptions& options);
BenchmarkReport runProductionLifetimeFluxBenchmark(const RunOptions& options);
BenchmarkReport runAuthoritativeProductionLifetimeFluxBenchmark(
    const RunOptions& options);
BenchmarkReport runConstantStratificationVerticalBenchmark(
    const RunOptions& options);
BenchmarkReport runConstantStratificationFluxBenchmark(
    const RunOptions& options);
BenchmarkReport runDealiasedConvolutionBenchmark(const RunOptions& options);
BenchmarkReport runVerticallyBatchedAdvectionBenchmark(const RunOptions& options);
ValidationReport validateBenchmark(std::string_view profileName);
EnvironmentRecord environmentRecord();
double median(std::vector<double> values);
double relativeL2Error(const Complex* actual, const Complex* expected, std::size_t count);
void writeJson(const BenchmarkReport& report, const std::filesystem::path& path);
void writeCsv(const BenchmarkReport& report, const std::filesystem::path& path);
int compareCsv(const std::filesystem::path& path, std::ostream& output);

} // namespace skbench
