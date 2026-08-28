#include "skbench/skbench.hpp"
#include "allocation_tracker.hpp"

#include <fftw3.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Value>
struct FreeDeleter {
    void operator()(Value* pointer) const noexcept { std::free(pointer); }
};

template <typename Value>
std::unique_ptr<Value, FreeDeleter<Value>> alignedBuffer(std::size_t count) {
    void* storage = nullptr;
    if (posix_memalign(&storage, 64, count * sizeof(Value)) != 0 || storage == nullptr) {
        throw std::bad_alloc();
    }
    return std::unique_ptr<Value, FreeDeleter<Value>>(static_cast<Value*>(storage));
}

void requireAllocationFreeExecution(const skbench::Workload& workload, skbench::FFTWStrategy strategy) {
    auto input = alignedBuffer<double>(workload.realElements());
    auto spectrum = alignedBuffer<skbench::Complex>(workload.spectrumElements());
    auto splitSpectrum = alignedBuffer<double>(2 * workload.spectrumElements());
    auto* spectrumReal = splitSpectrum.get();
    auto* spectrumImag = splitSpectrum.get() + workload.spectrumElements();
    auto output = alignedBuffer<double>(workload.realElements());
    for (std::size_t index = 0; index < workload.realElements(); ++index) {
        input.get()[index] = static_cast<double>(index % 31) / 31.0;
    }

    skbench::FFTWProvider provider(workload, strategy);
    for (std::size_t repetition = 0; repetition < 3; ++repetition) {
        if (strategy.layout == skbench::FFTWDataLayout::interleaved) {
            provider.forward(input.get(), spectrum.get());
            provider.inverse(spectrum.get(), output.get());
        } else {
            provider.forwardSplit(input.get(), spectrumReal, spectrumImag);
            provider.inverseSplit(spectrumReal, spectrumImag, output.get());
        }
        if (strategy.outerWorkers > 1) provider.executeSchedulerNoop();
    }

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) {
        if (strategy.layout == skbench::FFTWDataLayout::interleaved) {
            provider.forward(input.get(), spectrum.get());
            provider.inverse(spectrum.get(), output.get());
        } else {
            provider.forwardSplit(input.get(), spectrumReal, spectrumImag);
            provider.inverseSplit(spectrumReal, spectrumImag, output.get());
        }
        if (strategy.outerWorkers > 1) provider.executeSchedulerNoop();
    }
    require(skbench::test::endAllocationTracking() == 0, "FFTW steady-state execution allocated memory");
}

void requireAllocationFreeVerticalExecution(const skbench::Workload& workload,
                                            const skbench::GroupedVerticalOperators& operators,
                                            skbench::VerticalGemmLayout layout,
                                            skbench::VerticalGemmStrategy strategy = {}) {
    std::size_t horizontalModeCount = 0;
    for (const auto& group : operators.groups) horizontalModeCount += group.modeCount;
    const auto physicalCount = workload.nz * horizontalModeCount * workload.fields;
    const auto modalCount = operators.nj * horizontalModeCount * workload.fields;
    std::vector<skbench::Complex> physical(physicalCount);
    std::vector<skbench::Complex> modal(modalCount);
    for (std::size_t index = 0; index < physical.size(); ++index) {
        physical[index] = {static_cast<double>(index % 31) / 31.0,
                           -static_cast<double>(index % 29) / 29.0};
    }
    for (std::size_t index = 0; index < modal.size(); ++index) {
        modal[index] = {static_cast<double>(index % 23) / 23.0,
                        -static_cast<double>(index % 19) / 19.0};
    }

    skbench::VerticalGemmProvider provider(workload, operators, layout, strategy);
    require(provider.supported(), "Accelerate vertical GEMM provider is unavailable");
    provider.loadPhysicalInput(physical.data());
    provider.loadModalInput(modal.data());
    for (std::size_t repetition = 0; repetition < 3; ++repetition) {
        provider.executeForward();
        provider.executeInverse();
    }

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) {
        provider.executeForward();
        provider.executeInverse();
    }
    require(skbench::test::endAllocationTracking() == 0,
            "Accelerate vertical GEMM steady-state execution allocated memory");
}

void requireAllocationFreeOrderingPacking(
    const skbench::Workload& workload,
    const std::vector<skbench::RetainedMode>& modes,
    const skbench::GroupedVerticalOperators& operators,
    skbench::VerticalGemmLayout layout,
    skbench::VerticalGemmStrategy strategy) {
    const auto physicalCount = workload.nz * modes.size() * workload.fields;
    const auto modalCount = operators.nj * modes.size() * workload.fields;
    std::vector<skbench::Complex> physical(physicalCount);
    std::vector<skbench::Complex> modal(modalCount);
    std::vector<skbench::Complex> fullInput(workload.spectrumElements());
    std::vector<skbench::Complex> fullOutput(workload.spectrumElements());
    for (std::size_t index = 0; index < physical.size(); ++index) {
        physical[index] = {static_cast<double>(index % 31) / 31.0,
                           -static_cast<double>(index % 29) / 29.0};
    }
    for (std::size_t index = 0; index < modal.size(); ++index) {
        modal[index] = {static_cast<double>(index % 23) / 23.0,
                        -static_cast<double>(index % 19) / 19.0};
    }
    skbench::embedRetained(workload, modes, physical.data(), fullInput.data());

    skbench::VerticalGemmProvider provider(workload, operators, layout, strategy);
    require(provider.supported(), "Accelerate ordering/packing provider is unavailable");
    provider.loadModalInput(modal.data());
    for (std::size_t repetition = 0; repetition < 3; ++repetition) {
        provider.packPhysicalInputFromWvm(modes, fullInput.data());
        provider.executeForward();
        provider.executeInverse();
        provider.embedPhysicalOutputToWvm(modes, fullOutput.data());
    }

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) {
        provider.packPhysicalInputFromWvm(modes, fullInput.data());
        provider.executeForward();
        provider.executeInverse();
        provider.embedPhysicalOutputToWvm(modes, fullOutput.data());
    }
    require(skbench::test::endAllocationTracking() == 0,
            "ordering/packing steady-state execution allocated memory");
}

void requireExactSplitInPlaceWvmOrderUnsupported(const skbench::Workload& workload) {
    const auto storageCount = 2 * workload.spectrumElements();
    auto* storage = static_cast<double*>(fftw_malloc(storageCount * sizeof(double)));
    if (storage == nullptr) throw std::bad_alloc();

    const auto planes = workload.planes();
    const auto nxHalf = workload.nxHalf();
    const auto realPlane = workload.realPlaneElements();
    fftw_iodim64 forwardDimensions[2] = {
        {static_cast<ptrdiff_t>(workload.ny), static_cast<ptrdiff_t>(workload.nx), static_cast<ptrdiff_t>(planes * nxHalf)},
        {static_cast<ptrdiff_t>(workload.nx), 1, static_cast<ptrdiff_t>(planes)}};
    fftw_iodim64 forwardBatches[2] = {
        {static_cast<ptrdiff_t>(workload.nz), static_cast<ptrdiff_t>(realPlane), 1},
        {static_cast<ptrdiff_t>(workload.fields), static_cast<ptrdiff_t>(realPlane * workload.nz), static_cast<ptrdiff_t>(workload.nz)}};
    fftw_iodim64 inverseDimensions[2] = {
        {static_cast<ptrdiff_t>(workload.ny), static_cast<ptrdiff_t>(planes * nxHalf), static_cast<ptrdiff_t>(workload.nx)},
        {static_cast<ptrdiff_t>(workload.nx), static_cast<ptrdiff_t>(planes), 1}};
    fftw_iodim64 inverseBatches[2] = {
        {static_cast<ptrdiff_t>(workload.nz), 1, static_cast<ptrdiff_t>(realPlane)},
        {static_cast<ptrdiff_t>(workload.fields), static_cast<ptrdiff_t>(workload.nz), static_cast<ptrdiff_t>(realPlane * workload.nz)}};
    auto* splitReal = storage;
    auto* splitImag = storage + workload.spectrumElements();
    const auto flags = FFTW_ESTIMATE | FFTW_UNALIGNED;
    auto forward = fftw_plan_guru64_split_dft_r2c(
        2, forwardDimensions, 2, forwardBatches, storage, splitReal, splitImag, flags);
    auto inverse = fftw_plan_guru64_split_dft_c2r(
        2, inverseDimensions, 2, inverseBatches, splitReal, splitImag, storage, flags);
    const bool forwardUnsupported = forward == nullptr;
    const bool inverseUnsupported = inverse == nullptr;
    if (forward != nullptr) fftw_destroy_plan(forward);
    if (inverse != nullptr) fftw_destroy_plan(inverse);
    fftw_free(storage);
    require(forwardUnsupported, "exact WVM-order split forward in-place plan unexpectedly succeeded");
    require(inverseUnsupported, "exact WVM-order split inverse in-place plan unexpectedly succeeded");
}

} // namespace

int main() {
    try {
        const skbench::Workload workload{8, 8, 7, 2, 1.0, 1.0, true};
        require(workload.planes() == 14, "plane count");
        require(workload.nxHalf() == 5, "half width");
        require(workload.retainedVerticalModes() == 4, "retained vertical count");

        for (const auto strategy : {skbench::VDSPTransformStrategy::inPlace,
                                    skbench::VDSPTransformStrategy::inPlaceExplicitScratch,
                                    skbench::VDSPTransformStrategy::outOfPlace,
                                    skbench::VDSPTransformStrategy::outOfPlaceExplicitScratch}) {
            const auto name = skbench::vdspTransformStrategyName(strategy);
            require(skbench::vdspTransformStrategyNamed(name) == strategy, "vDSP strategy name round trip");
        }
        for (const auto strategy : {skbench::VDSPBatchStrategy::directPersistent,
                                    skbench::VDSPBatchStrategy::directGcd,
                                    skbench::VDSPBatchStrategy::separablePersistent,
                                    skbench::VDSPBatchStrategy::separableGcd}) {
            const auto name = skbench::vdspBatchStrategyName(strategy);
            require(skbench::vdspBatchStrategyNamed(name) == strategy, "vDSP batch strategy name round trip");
        }
        for (const auto mode : {skbench::FFTWPlanningMode::estimate,
                               skbench::FFTWPlanningMode::measure,
                               skbench::FFTWPlanningMode::patient,
                               skbench::FFTWPlanningMode::exhaustive}) {
            const auto name = skbench::fftwPlanningModeName(mode);
            require(skbench::fftwPlanningModeNamed(name) == mode, "FFTW planning name round trip");
        }
        for (const auto strategy : {skbench::FFTWAlignmentStrategy::aligned,
                                    skbench::FFTWAlignmentStrategy::unaligned}) {
            const auto name = skbench::fftwAlignmentStrategyName(strategy);
            require(skbench::fftwAlignmentStrategyNamed(name) == strategy, "FFTW alignment name round trip");
        }
        for (const auto strategy : {skbench::FFTWWisdomStrategy::cold,
                                    skbench::FFTWWisdomStrategy::generatedImport}) {
            const auto name = skbench::fftwWisdomStrategyName(strategy);
            require(skbench::fftwWisdomStrategyNamed(name) == strategy, "FFTW wisdom name round trip");
        }
        for (const auto layout : {skbench::FFTWDataLayout::interleaved,
                                  skbench::FFTWDataLayout::split}) {
            const auto name = skbench::fftwDataLayoutName(layout);
            require(skbench::fftwDataLayoutNamed(name) == layout, "FFTW data layout name round trip");
        }
        for (const auto layout : {skbench::VerticalGemmLayout::complexInterleaved,
                                  skbench::VerticalGemmLayout::split}) {
            require(!skbench::verticalGemmLayoutName(layout).empty(), "vertical GEMM layout name");
        }

        const auto profileList = skbench::profiles();
        require(skbench::profileNamed("wvm-historical-256-nz65-f4").workload.planes() == 260,
                "historical workload profile");
        require(skbench::profileNamed("wvm-current-512-nz257-f4").workload.planes() == 1028,
                "current workload profile");
        require(profileList.size() == 13, "unexpected profile count");

        skbench::RunOptions fftwOptions;
        fftwOptions.profile = "smoke";
        fftwOptions.providers = "fftw";
        fftwOptions.fftwLayout = "paired";
        fftwOptions.fftwPlanning = "measure";
        fftwOptions.fftwAlignment = "aligned";
        fftwOptions.fftwWisdom = "generated-import";
        fftwOptions.fftwInternalWorkers = 2;
        fftwOptions.fftwOuterWorkers = 2;
        fftwOptions.warmups = 1;
        fftwOptions.samples = 2;
        const auto fftwReport = skbench::runBenchmark(fftwOptions);
        require(fftwReport.status == "passed", "FFTW strategy benchmark failed");
        require(fftwReport.providers.size() == 2, "paired FFTW provider selection failed");
        const auto& fftwRecord = fftwReport.providers.front();
        require(fftwRecord.internalWorkers == 2 && fftwRecord.outerWorkers == 2 && fftwRecord.workers == 4,
                "FFTW worker topology");
        require(fftwRecord.wisdomGenerationSeconds > 0.0 && fftwRecord.wisdomImportSeconds > 0.0,
                "FFTW wisdom accounting");
        require(fftwRecord.wisdomBytes > 0, "FFTW wisdom bytes");
        require(fftwRecord.correctness.size() == 3, "FFTW strategy correctness metrics");
        require(fftwRecord.correctness[0].passed && fftwRecord.correctness[1].passed && fftwRecord.correctness[2].passed,
                "FFTW strategy correctness");
        bool foundScheduler = false;
        for (const auto& timing : fftwRecord.timings) {
            if (timing.stage == "batch scheduler empty dispatch") {
                foundScheduler = timing.state == skbench::StageState::executed && timing.seconds.size() == 2;
            }
        }
        require(foundScheduler, "FFTW outer scheduler timing");
        const auto& splitRecord = fftwReport.providers[1];
        require(splitRecord.id == "fftw-split", "split FFTW provider identity");
        require(splitRecord.nativeRepresentationId == "wvm-frequency-major-split-half-spectrum",
                "split FFTW native representation");
        require(splitRecord.execution.forward.nativePlacement == "out-of-place" &&
                splitRecord.execution.inverse.nativePlacement == "out-of-place",
                "split FFTW placement contract");
        require(!splitRecord.execution.forward.destroysNativeInput &&
                splitRecord.execution.inverse.destroysNativeInput,
                "split FFTW destructive-input contract");
        require(splitRecord.execution.forward.minimumAlignmentBytes == 64 &&
                splitRecord.execution.inverse.minimumAlignmentBytes == 64,
                "split FFTW alignment contract");
        require(splitRecord.execution.forward.paddingElements == 0 &&
                splitRecord.execution.inverse.paddingElements == 0,
                "split FFTW out-of-place padding contract");
        require(splitRecord.correctness.size() == 8, "split FFTW correctness metric count");
        bool foundForwardConversion = false;
        bool foundRetainedTotal = false;
        bool foundInPlaceCapability = false;
        for (const auto& timing : splitRecord.timings) {
            if (timing.stage == "split-to-interleaved conversion" && timing.direction == "forward") {
                foundForwardConversion = timing.state == skbench::StageState::executed && timing.seconds.size() == 2;
            }
            if (timing.stage == "persistent split retained horizontal operator" && timing.direction == "forward") {
                foundRetainedTotal = timing.state == skbench::StageState::executed && timing.seconds.size() == 2;
            }
            if (timing.stage == "exact WVM-order split in-place") {
                foundInPlaceCapability = timing.state == skbench::StageState::unsupported && timing.seconds.empty();
            }
        }
        require(foundForwardConversion, "split FFTW conversion timing");
        require(foundRetainedTotal, "split FFTW retained total timing");
        require(foundInPlaceCapability, "split FFTW in-place capability timing");

        const auto splitInput = skbench::makeFixture(workload, skbench::FixtureKind::random, 441);
        auto splitStrategy = skbench::FFTWStrategy{
            skbench::FFTWPlanningMode::estimate,
            skbench::FFTWAlignmentStrategy::unaligned,
            skbench::FFTWWisdomStrategy::cold,
            1,
            2,
            0.0,
            skbench::FFTWDataLayout::split};
        auto referenceStrategy = splitStrategy;
        referenceStrategy.layout = skbench::FFTWDataLayout::interleaved;
        skbench::FFTWProvider splitProvider(workload, splitStrategy);
        skbench::FFTWProvider referenceProvider(workload, referenceStrategy);
        std::vector<double> splitStorage(2 * workload.spectrumElements());
        auto* splitReal = splitStorage.data();
        auto* splitImag = splitStorage.data() + workload.spectrumElements();
        std::vector<skbench::Complex> splitInterleaved(workload.spectrumElements());
        std::vector<skbench::Complex> referenceInterleaved(workload.spectrumElements());
        splitProvider.forwardSplit(splitInput.data(), splitReal, splitImag);
        referenceProvider.forward(splitInput.data(), referenceInterleaved.data());
        skbench::splitToInterleaved(workload.spectrumElements(), splitReal, splitImag, splitInterleaved.data());
        require(skbench::maximumRelativeError(splitInterleaved.data(), referenceInterleaved.data(), splitInterleaved.size()) < 1.0e-12,
                "split FFTW forward equivalence");

        std::vector<double> convertedReal(workload.spectrumElements());
        std::vector<double> convertedImag(workload.spectrumElements());
        skbench::interleavedToSplit(workload.spectrumElements(), referenceInterleaved.data(), convertedReal.data(), convertedImag.data());
        require(skbench::maximumRelativeError(convertedReal.data(), splitReal, convertedReal.size()) < 1.0e-12,
                "interleaved-to-split real conversion");
        require(skbench::maximumRelativeError(convertedImag.data(), splitImag, convertedImag.size()) < 1.0e-12,
                "interleaved-to-split imaginary conversion");

        std::vector<double> splitInverse(workload.realElements());
        std::vector<double> referenceInverse(workload.realElements());
        splitProvider.inverseSplit(splitReal, splitImag, splitInverse.data());
        referenceProvider.inverse(referenceInterleaved.data(), referenceInverse.data());
        require(skbench::maximumRelativeError(splitInverse.data(), referenceInverse.data(), splitInverse.size()) < 1.0e-12,
                "split FFTW inverse equivalence");
        require(skbench::maximumRelativeError(splitInverse.data(), splitInput.data(), splitInverse.size(),
                                               1.0 / static_cast<double>(workload.nx * workload.ny)) < 1.0e-12,
                "split FFTW round trip");
        require(!splitProvider.splitInPlaceWvmOrderSupported(), "exact WVM-order split in-place unexpectedly supported");
        require(!splitProvider.splitInPlaceWvmOrderCapability().empty(), "split in-place capability explanation");
        bool rejectedWrongSplitSeparation = false;
        try {
            splitProvider.forwardSplit(splitInput.data(), splitReal, splitImag + 1);
        } catch (const std::invalid_argument&) {
            rejectedWrongSplitSeparation = true;
        }
        require(rejectedWrongSplitSeparation, "split FFTW accepted a new-array separation that differs from planning");
        requireExactSplitInPlaceWvmOrderUnsupported(workload);

        skbench::RunOptions exhaustiveOptions = fftwOptions;
        exhaustiveOptions.fftwPlanning = "exhaustive";
        exhaustiveOptions.fftwWisdom = "cold";
        exhaustiveOptions.fftwInternalWorkers = 1;
        exhaustiveOptions.fftwOuterWorkers = 1;
        exhaustiveOptions.fftwPlanningTimeLimitSeconds = 0.001;
        exhaustiveOptions.samples = 1;
        const auto exhaustiveReport = skbench::runBenchmark(exhaustiveOptions);
        require(exhaustiveReport.status == "passed", "time-bounded FFTW exhaustive benchmark failed");
        require(exhaustiveReport.providers.front().planningBudgetExhausted,
                "FFTW exhaustive planning budget was not recorded");

        if (skbench::test::allocationTrackingSupported()) {
            void* proof = nullptr;
            skbench::test::beginAllocationTracking();
            const auto proofResult = posix_memalign(&proof, 64, 64);
            const auto proofCount = skbench::test::endAllocationTracking();
            require(proofResult == 0 && proof != nullptr, "allocation tracker proof allocation failed");
            std::free(proof);
            require(proofCount > 0, "allocation tracker did not observe a proof allocation");

            requireAllocationFreeExecution(workload, {
                skbench::FFTWPlanningMode::estimate,
                skbench::FFTWAlignmentStrategy::aligned,
                skbench::FFTWWisdomStrategy::cold,
                2,
                1,
                0.0});
            requireAllocationFreeExecution(workload, {
                skbench::FFTWPlanningMode::estimate,
                skbench::FFTWAlignmentStrategy::aligned,
                skbench::FFTWWisdomStrategy::cold,
                1,
                2,
                0.0});
            requireAllocationFreeExecution(workload, {
                skbench::FFTWPlanningMode::estimate,
                skbench::FFTWAlignmentStrategy::aligned,
                skbench::FFTWWisdomStrategy::cold,
                2,
                2,
                0.0});
            requireAllocationFreeExecution(workload, {
                skbench::FFTWPlanningMode::estimate,
                skbench::FFTWAlignmentStrategy::aligned,
                skbench::FFTWWisdomStrategy::cold,
                2,
                2,
                0.0,
                skbench::FFTWDataLayout::split});
        }

        skbench::VDSPProvider inPlace(workload, 2, skbench::VDSPTransformStrategy::inPlace);
        skbench::VDSPProvider outOfPlaceScratch(workload, 2, skbench::VDSPTransformStrategy::outOfPlaceExplicitScratch);
        skbench::VDSPProvider separable(workload, 2, skbench::VDSPTransformStrategy::inPlace,
                                        skbench::VDSPBatchStrategy::separablePersistent);
        if (inPlace.supported() && outOfPlaceScratch.supported()) {
            const auto operandBytes = workload.realElements() * sizeof(double);
            require(inPlace.nativeOperandBytes() == operandBytes, "vDSP operand bytes");
            require(inPlace.nativeBufferBytes() == operandBytes, "in-place vDSP persistent bytes");
            require(inPlace.scratchBytes() == 0, "in-place vDSP scratch bytes");
            require(outOfPlaceScratch.nativeBufferBytes() == 2 * operandBytes, "out-of-place vDSP persistent bytes");
            require(outOfPlaceScratch.scratchBytes() == 2 * 2 * 8 * sizeof(double), "vDSP per-worker scratch bytes");
            require(outOfPlaceScratch.minimumAlignmentBytes() == 64, "vDSP buffer alignment");
        }
        if (inPlace.supported() && separable.supported()) {
            const auto operandBytes = workload.realElements() * sizeof(double);
            require(separable.nativeBufferBytes() == operandBytes, "separable vDSP persistent bytes");
            require(separable.scratchBytes() == 2 * 2 * workload.ny * sizeof(double),
                    "separable vDSP boundary scratch bytes");
            const auto input = skbench::makeFixture(workload, skbench::FixtureKind::random, 129);
            std::vector<skbench::Complex> directSpectrum(workload.spectrumElements());
            std::vector<skbench::Complex> separableSpectrum(workload.spectrumElements());
            std::vector<double> directOutput(workload.realElements());
            std::vector<double> separableOutput(workload.realElements());
            inPlace.forwardAdapter(input.data(), directSpectrum.data());
            separable.forwardAdapter(input.data(), separableSpectrum.data());
            require(skbench::maximumRelativeError(separableSpectrum.data(), directSpectrum.data(), directSpectrum.size()) < 1.0e-12,
                    "separable vDSP forward equivalence");
            inPlace.inverseAdapter(directSpectrum.data(), directOutput.data());
            separable.inverseAdapter(directSpectrum.data(), separableOutput.data());
            require(skbench::maximumRelativeError(separableOutput.data(), directOutput.data(), directOutput.size()) < 1.0e-12,
                    "separable vDSP inverse equivalence");
            separable.executeSchedulerNoop();
        }

        const auto modes = skbench::retainedHorizontalModes(workload);
        require(!modes.empty(), "retained modes are empty");
        require(modes.front().k == 0 && modes.front().l == 0, "DC is not first");
        require(skbench::modeOrderHash(modes) == skbench::modeOrderHash(modes), "mode hash is unstable");
        const auto k2Groups = skbench::squaredWavenumberGroups(modes);
        require(k2Groups.size() == 5, "smoke K-squared group count");
        require(k2Groups.front().squaredModeKey == 0 && k2Groups.front().firstMode == 0,
                "K-squared DC group");
        std::size_t groupedModeCount = 0;
        for (const auto& group : k2Groups) {
            require(group.firstMode == groupedModeCount && group.modeCount > 0,
                    "K-squared groups are not contiguous");
            groupedModeCount += group.modeCount;
        }
        require(groupedModeCount == modes.size(), "K-squared groups do not cover retained modes");
        require(skbench::verticalModeGroupHash(k2Groups) == "fnv1a64:69843c1f93cead00",
                "K-squared group hash");

        std::vector<skbench::Complex> wvm(workload.spectrumElements());
        for (std::size_t index = 0; index < wvm.size(); ++index) {
            wvm[index] = {static_cast<double>(index), -static_cast<double>(index)};
        }
        std::vector<skbench::Complex> planeMajor(wvm.size());
        std::vector<skbench::Complex> roundTrip(wvm.size());
        skbench::wvmToPlaneMajor(workload, wvm.data(), planeMajor.data());
        skbench::planeMajorToWvm(workload, planeMajor.data(), roundTrip.data());
        require(skbench::maximumRelativeError(roundTrip.data(), wvm.data(), wvm.size()) == 0.0, "layout round trip");

        std::vector<skbench::Complex> retained(modes.size() * workload.planes());
        std::vector<skbench::Complex> embedded(wvm.size());
        std::vector<skbench::Complex> retainedAgain(retained.size());
        skbench::gatherRetained(workload, modes, wvm.data(), retained.data());
        skbench::embedRetained(workload, modes, retained.data(), embedded.data());
        skbench::gatherRetained(workload, modes, embedded.data(), retainedAgain.data());
        require(skbench::maximumRelativeError(retainedAgain.data(), retained.data(), retained.size()) == 0.0, "retained gather/embed round trip");

        std::vector<double> wvmReal(wvm.size());
        std::vector<double> wvmImag(wvm.size());
        std::vector<double> retainedReal(retained.size());
        std::vector<double> retainedImag(retained.size());
        std::vector<double> embeddedReal(wvm.size());
        std::vector<double> embeddedImag(wvm.size());
        std::vector<skbench::Complex> splitRetainedAgain(retained.size());
        skbench::interleavedToSplit(wvm.size(), wvm.data(), wvmReal.data(), wvmImag.data());
        skbench::gatherRetainedSplit(workload, modes, wvmReal.data(), wvmImag.data(), retainedReal.data(), retainedImag.data());
        skbench::embedRetainedSplit(workload, modes, retainedReal.data(), retainedImag.data(), embeddedReal.data(), embeddedImag.data());
        skbench::gatherRetainedSplit(workload, modes, embeddedReal.data(), embeddedImag.data(), retainedReal.data(), retainedImag.data());
        skbench::splitToInterleaved(retained.size(), retainedReal.data(), retainedImag.data(), splitRetainedAgain.data());
        require(skbench::maximumRelativeError(splitRetainedAgain.data(), retained.data(), retained.size()) == 0.0,
                "split retained gather/embed round trip");

        const auto vertical = skbench::orthonormalVerticalFixture(workload.nz, workload.retainedVerticalModes());
        const auto commonVertical = skbench::commonVerticalFixture(modes.size(), vertical);
        const auto groupedVertical = skbench::squaredWavenumberVerticalFixture(workload, modes);
        require(commonVertical.groups.size() == 1 && commonVertical.groups.front().modeCount == modes.size(),
                "common vertical matrix family");
        require(groupedVertical.groups == k2Groups, "grouped vertical fixture groups");
        require(groupedVertical.forward.size() == k2Groups.size() * vertical.nz * vertical.nj,
                "grouped vertical forward matrix storage");
        require(groupedVertical.inverse.size() == groupedVertical.forward.size(),
                "grouped vertical inverse matrix storage");
        for (std::size_t groupIndex = 0; groupIndex < groupedVertical.groups.size(); ++groupIndex) {
            const auto offset = groupIndex * vertical.nz * vertical.nj;
            for (std::size_t first = 0; first < vertical.nj; ++first) {
                for (std::size_t second = 0; second < vertical.nj; ++second) {
                    double product = 0.0;
                    for (std::size_t z = 0; z < vertical.nz; ++z) {
                        product += groupedVertical.forward[offset + first * vertical.nz + z] *
                                   groupedVertical.forward[offset + second * vertical.nz + z];
                    }
                    const double expected = first == second ? 1.0 : 0.0;
                    require(std::abs(product - expected) < 1.0e-12,
                            "grouped vertical matrix rows are not orthonormal");
                }
            }
        }
        std::vector<skbench::Complex> modal(modes.size() * workload.fields * workload.retainedVerticalModes());
        std::vector<skbench::Complex> physicalAgain(retained.size());
        std::vector<skbench::Complex> modalAgain(modal.size());
        skbench::verticalForward(workload, modes.size(), vertical, retained.data(), modal.data());
        skbench::verticalInverse(workload, modes.size(), vertical, modal.data(), physicalAgain.data());
        skbench::verticalForward(workload, modes.size(), vertical, physicalAgain.data(), modalAgain.data());
        require(skbench::maximumRelativeError(modalAgain.data(), modal.data(), modal.size()) < 1.0e-12, "vertical modal round trip");

        for (const auto layout : {skbench::VerticalGemmLayout::complexInterleaved,
                                  skbench::VerticalGemmLayout::split}) {
            skbench::VerticalGemmProvider provider(workload, modes.size(), vertical, layout);
            require(provider.supported(), "vertical GEMM provider support");
            require(provider.columns() == modes.size() * workload.fields, "vertical GEMM K dimension");
            require(provider.physicalElements() == retained.size(), "vertical GEMM physical element count");
            require(provider.modalElements() == modal.size(), "vertical GEMM modal element count");
            require(provider.groupCount() == 1, "common vertical GEMM group count");
            require(provider.gemmCallsPerExecution() ==
                        (layout == skbench::VerticalGemmLayout::split ? 2 : 1),
                    "common vertical GEMM call count");
            require(provider.minimumAlignmentBytes() == 64, "vertical GEMM alignment");
            require(provider.persistentBytes() > 0, "vertical GEMM persistent-memory accounting");
            provider.loadPhysicalInput(retained.data());
            provider.loadModalInput(modal.data());
            provider.executeForward();
            provider.executeInverse();
            std::vector<skbench::Complex> gemmModal(modal.size());
            std::vector<skbench::Complex> gemmPhysical(physicalAgain.size());
            provider.copyForwardOutput(gemmModal.data());
            provider.copyInverseOutput(gemmPhysical.data());
            require(skbench::maximumRelativeError(gemmModal.data(), modal.data(), modal.size()) < 1.0e-12,
                    "vertical GEMM forward equivalence");
            require(skbench::maximumRelativeError(gemmPhysical.data(), physicalAgain.data(), physicalAgain.size()) < 1.0e-12,
                    "vertical GEMM inverse equivalence");

            std::vector<skbench::Complex> fullInput(workload.spectrumElements());
            std::vector<skbench::Complex> fullOutput(workload.spectrumElements());
            std::vector<skbench::Complex> gatheredOutput(physicalAgain.size());
            skbench::embedRetained(workload, modes, retained.data(), fullInput.data());
            provider.packPhysicalInputFromWvm(modes, fullInput.data());
            provider.executeForward();
            provider.executeInverse();
            provider.embedPhysicalOutputToWvm(modes, fullOutput.data());
            skbench::gatherRetained(
                workload, modes, fullOutput.data(), gatheredOutput.data());
            require(skbench::maximumRelativeError(
                        gatheredOutput.data(), gemmPhysical.data(), gatheredOutput.size()) < 1.0e-12,
                    "vertical GEMM direct WVM pack/embed equivalence");
        }

        for (const auto schedule : {skbench::VerticalGemmSchedule::outerStatic,
                                    skbench::VerticalGemmSchedule::outerDynamic}) {
            require(!skbench::verticalGemmScheduleName(schedule).empty(), "vertical GEMM schedule name");
            for (const auto layout : {skbench::VerticalGemmLayout::complexInterleaved,
                                      skbench::VerticalGemmLayout::split}) {
                skbench::VerticalGemmProvider serialProvider(
                    workload, groupedVertical, layout, {skbench::VerticalGemmSchedule::serial, 1});
                skbench::VerticalGemmProvider scheduledProvider(
                    workload, groupedVertical, layout, {schedule, 2});
                serialProvider.loadPhysicalInput(retained.data());
                serialProvider.loadModalInput(modal.data());
                scheduledProvider.loadPhysicalInput(retained.data());
                scheduledProvider.loadModalInput(modal.data());
                serialProvider.executeForward();
                serialProvider.executeInverse();
                scheduledProvider.executeForward();
                scheduledProvider.executeInverse();
                scheduledProvider.executeSchedulerNoop();
                std::vector<skbench::Complex> serialForward(modal.size());
                std::vector<skbench::Complex> serialInverse(physicalAgain.size());
                std::vector<skbench::Complex> scheduledForward(modal.size());
                std::vector<skbench::Complex> scheduledInverse(physicalAgain.size());
                serialProvider.copyForwardOutput(serialForward.data());
                serialProvider.copyInverseOutput(serialInverse.data());
                scheduledProvider.copyForwardOutput(scheduledForward.data());
                scheduledProvider.copyInverseOutput(scheduledInverse.data());
                require(skbench::maximumRelativeError(
                            scheduledForward.data(), serialForward.data(), serialForward.size()) == 0.0,
                        "scheduled vertical GEMM forward equivalence");
                require(skbench::maximumRelativeError(
                            scheduledInverse.data(), serialInverse.data(), serialInverse.size()) == 0.0,
                        "scheduled vertical GEMM inverse equivalence");
                require(scheduledProvider.outerWorkers() == 2, "scheduled vertical GEMM worker count");
                require(scheduledProvider.strategy().schedule == schedule, "scheduled vertical GEMM strategy");
                require(scheduledProvider.schedulerPersistentBytes() > 0, "scheduler memory accounting");
                require(scheduledProvider.hasOpaqueSchedulerMemory(), "scheduler opaque thread-stack accounting");
            }
        }

        skbench::RunOptions verticalOptions;
        verticalOptions.kernel = "vertical-gemm";
        verticalOptions.profile = "smoke";
        verticalOptions.warmups = 1;
        verticalOptions.samples = 2;
        const auto verticalReport = skbench::runBenchmark(verticalOptions);
        require(verticalReport.status == "passed", "vertical GEMM smoke benchmark failed");
        require(verticalReport.providers.size() == 2, "vertical GEMM candidate count");
        require(verticalReport.modalSpectrumBytes > 0, "vertical GEMM modal byte accounting");
        require(verticalReport.verticalGroupCount == 1, "common vertical GEMM report group count");
        for (const auto& provider : verticalReport.providers) {
            require(!provider.opaqueProviderMemory, "vertical GEMM memory should be explicit");
            require(provider.correctness.size() == 4, "vertical GEMM correctness metric count");
            for (const auto& correctness : provider.correctness) {
                require(correctness.passed && correctness.relativeL2Error <= 1.0e-12,
                        "vertical GEMM correctness");
            }
            bool foundForward = false;
            bool foundInverse = false;
            bool foundElidedPacking = false;
            for (const auto& timing : provider.timings) {
                if (timing.scope == "primitive" && timing.stage == "raw vertical GEMM" && timing.direction == "forward") {
                    foundForward = timing.state == skbench::StageState::executed && timing.seconds.size() == 2;
                }
                if (timing.scope == "primitive" && timing.stage == "raw vertical GEMM" && timing.direction == "inverse") {
                    foundInverse = timing.state == skbench::StageState::executed && timing.seconds.size() == 2;
                }
                if (timing.stage == "packing and representation conversion") {
                    foundElidedPacking = timing.state == skbench::StageState::elided && timing.seconds.empty();
                }
            }
            require(foundForward && foundInverse, "vertical GEMM primitive timing series");
            require(foundElidedPacking, "vertical GEMM excluded packing contract");
        }

        skbench::RunOptions groupedOptions = verticalOptions;
        groupedOptions.verticalGemmFamily = "k2-grouped";
        const auto groupedReport = skbench::runBenchmark(groupedOptions);
        require(groupedReport.status == "passed", "grouped vertical GEMM smoke benchmark failed");
        require(groupedReport.verticalMatrixFamilyId == groupedVertical.id,
                "grouped vertical GEMM family identity");
        require(groupedReport.verticalGroupCount == k2Groups.size(), "grouped vertical GEMM group count");
        require(groupedReport.minimumVerticalGroupModes == 1 &&
                groupedReport.medianVerticalGroupModes == 2.0 &&
                groupedReport.maximumVerticalGroupModes == 4,
                "grouped vertical GEMM mode distribution");
        require(groupedReport.minimumVerticalGroupColumns == 2 &&
                groupedReport.medianVerticalGroupColumns == 4.0 &&
                groupedReport.maximumVerticalGroupColumns == 8,
                "grouped vertical GEMM column distribution");
        require(groupedReport.verticalGroupOrderHash == skbench::verticalModeGroupHash(k2Groups),
                "grouped vertical GEMM order hash");
        require(groupedReport.verticalMatrixFamilySourceBytes ==
                    2 * groupedVertical.forward.size() * sizeof(double),
                "grouped vertical GEMM source-matrix bytes");
        const auto groupedProviderBytes = groupedReport.providers[0].explicitPersistentBytes +
            groupedReport.providers[1].explicitPersistentBytes;
        require(groupedReport.verticalBenchmarkEstimatedExplicitPeakBytes >
                    groupedReport.verticalMatrixFamilySourceBytes + groupedProviderBytes,
                "grouped vertical GEMM explicit peak-memory estimate");
        for (const auto& provider : groupedReport.providers) {
            require(provider.algorithmId.find("k2-group-serial") != std::string::npos,
                    "grouped vertical GEMM algorithm identity");
            require(provider.correctness.size() == 4, "grouped vertical GEMM correctness count");
            for (const auto& correctness : provider.correctness) {
                require(correctness.passed, "grouped vertical GEMM correctness");
            }
            bool foundFamilySetup = false;
            for (const auto& timing : provider.timings) {
                if (timing.scope == "setup-shared-component" &&
                    timing.stage == "logical matrix-family fixture generation") {
                    foundFamilySetup = timing.state == skbench::StageState::setupOnly &&
                        timing.seconds.size() == 1 &&
                        timing.bytesMoved == groupedReport.verticalMatrixFamilySourceBytes;
                }
            }
            require(foundFamilySetup, "grouped vertical GEMM shared setup accounting");
        }

        skbench::RunOptions orderingOptions = verticalOptions;
        orderingOptions.kernel = "ordering-packing";
        orderingOptions.verticalGemmFamily = "k2-grouped";
        const auto orderingReport = skbench::runBenchmark(orderingOptions);
        require(orderingReport.status == "passed", "ordering/packing smoke benchmark failed");
        require(orderingReport.providers.size() == 2, "ordering/packing candidate count");
        require(orderingReport.orderingPackingEstimatedExplicitPeakBytes >
                    orderingReport.fullSpectrumBytes + orderingReport.retainedSpectrumBytes,
                "ordering/packing explicit peak-memory estimate");
        for (const auto& provider : orderingReport.providers) {
            require(provider.correctness.size() == 6, "ordering/packing correctness count");
            bool foundPack = false;
            bool foundEmbed = false;
            bool foundCombinedForward = false;
            bool foundCombinedInverse = false;
            std::size_t reuseSeries = 0;
            for (const auto& timing : provider.timings) {
                foundPack = foundPack ||
                    (timing.scope == "adapter-component" && timing.direction == "forward" &&
                     timing.stage == "WVM retained gather and radial pack" &&
                     timing.state == skbench::StageState::executed && !timing.seconds.empty());
                foundEmbed = foundEmbed ||
                    (timing.scope == "adapter-component" && timing.direction == "inverse" &&
                     timing.stage == "WVM scatter and Hermitian embed" &&
                     timing.state == skbench::StageState::executed && !timing.seconds.empty());
                foundCombinedForward = foundCombinedForward ||
                    (timing.scope == "adapter-total" && timing.direction == "forward" &&
                     timing.state == skbench::StageState::executed);
                foundCombinedInverse = foundCombinedInverse ||
                    (timing.scope == "adapter-total" && timing.direction == "inverse" &&
                     timing.state == skbench::StageState::executed);
                if (timing.scope == "reuse-total") ++reuseSeries;
            }
            require(foundPack && foundEmbed, "ordering/packing movement timing series");
            require(foundCombinedForward && foundCombinedInverse,
                    "ordering/packing combined timing series");
            require(reuseSeries == 12, "ordering/packing reuse timing matrix");
            for (const auto& correctness : provider.correctness) {
                require(correctness.passed, "ordering/packing correctness");
            }
        }

        if (skbench::test::allocationTrackingSupported()) {
            requireAllocationFreeVerticalExecution(
                workload, commonVertical, skbench::VerticalGemmLayout::complexInterleaved);
            requireAllocationFreeVerticalExecution(
                workload, commonVertical, skbench::VerticalGemmLayout::split);
            requireAllocationFreeVerticalExecution(
                workload, groupedVertical, skbench::VerticalGemmLayout::complexInterleaved);
            requireAllocationFreeVerticalExecution(
                workload, groupedVertical, skbench::VerticalGemmLayout::split);
            for (const auto schedule : {skbench::VerticalGemmSchedule::outerStatic,
                                        skbench::VerticalGemmSchedule::outerDynamic}) {
                requireAllocationFreeVerticalExecution(
                    workload, groupedVertical, skbench::VerticalGemmLayout::complexInterleaved,
                    {schedule, 2});
                requireAllocationFreeVerticalExecution(
                    workload, groupedVertical, skbench::VerticalGemmLayout::split,
                    {schedule, 2});
            }
            for (const auto layout : {skbench::VerticalGemmLayout::complexInterleaved,
                                      skbench::VerticalGemmLayout::split}) {
                requireAllocationFreeOrderingPacking(
                    workload, modes, groupedVertical, layout,
                    {skbench::VerticalGemmSchedule::outerDynamic, 2});
            }
        }

        std::cout << "skbench unit tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "skbench_tests: " << error.what() << '\n';
        return 1;
    }
}
