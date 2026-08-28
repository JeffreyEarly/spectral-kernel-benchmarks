#include "skbench/skbench.hpp"
#include "allocation_tracker.hpp"

#include <fftw3.h>

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
    auto spectrumReal = alignedBuffer<double>(workload.spectrumElements());
    auto spectrumImag = alignedBuffer<double>(workload.spectrumElements());
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
            provider.forwardSplit(input.get(), spectrumReal.get(), spectrumImag.get());
            provider.inverseSplit(spectrumReal.get(), spectrumImag.get(), output.get());
        }
        if (strategy.outerWorkers > 1) provider.executeSchedulerNoop();
    }

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) {
        if (strategy.layout == skbench::FFTWDataLayout::interleaved) {
            provider.forward(input.get(), spectrum.get());
            provider.inverse(spectrum.get(), output.get());
        } else {
            provider.forwardSplit(input.get(), spectrumReal.get(), spectrumImag.get());
            provider.inverseSplit(spectrumReal.get(), spectrumImag.get(), output.get());
        }
        if (strategy.outerWorkers > 1) provider.executeSchedulerNoop();
    }
    require(skbench::test::endAllocationTracking() == 0, "FFTW steady-state execution allocated memory");
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
        std::vector<double> splitReal(workload.spectrumElements());
        std::vector<double> splitImag(workload.spectrumElements());
        std::vector<skbench::Complex> splitInterleaved(workload.spectrumElements());
        std::vector<skbench::Complex> referenceInterleaved(workload.spectrumElements());
        splitProvider.forwardSplit(splitInput.data(), splitReal.data(), splitImag.data());
        referenceProvider.forward(splitInput.data(), referenceInterleaved.data());
        skbench::splitToInterleaved(workload.spectrumElements(), splitReal.data(), splitImag.data(), splitInterleaved.data());
        require(skbench::maximumRelativeError(splitInterleaved.data(), referenceInterleaved.data(), splitInterleaved.size()) < 1.0e-12,
                "split FFTW forward equivalence");

        std::vector<double> convertedReal(workload.spectrumElements());
        std::vector<double> convertedImag(workload.spectrumElements());
        skbench::interleavedToSplit(workload.spectrumElements(), referenceInterleaved.data(), convertedReal.data(), convertedImag.data());
        require(skbench::maximumRelativeError(convertedReal.data(), splitReal.data(), convertedReal.size()) < 1.0e-12,
                "interleaved-to-split real conversion");
        require(skbench::maximumRelativeError(convertedImag.data(), splitImag.data(), convertedImag.size()) < 1.0e-12,
                "interleaved-to-split imaginary conversion");

        std::vector<double> splitInverse(workload.realElements());
        std::vector<double> referenceInverse(workload.realElements());
        splitProvider.inverseSplit(splitReal.data(), splitImag.data(), splitInverse.data());
        referenceProvider.inverse(referenceInterleaved.data(), referenceInverse.data());
        require(skbench::maximumRelativeError(splitInverse.data(), referenceInverse.data(), splitInverse.size()) < 1.0e-12,
                "split FFTW inverse equivalence");
        require(skbench::maximumRelativeError(splitInverse.data(), splitInput.data(), splitInverse.size(),
                                               1.0 / static_cast<double>(workload.nx * workload.ny)) < 1.0e-12,
                "split FFTW round trip");
        require(!splitProvider.splitInPlaceWvmOrderSupported(), "exact WVM-order split in-place unexpectedly supported");
        require(!splitProvider.splitInPlaceWvmOrderCapability().empty(), "split in-place capability explanation");
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
        std::vector<skbench::Complex> modal(modes.size() * workload.fields * workload.retainedVerticalModes());
        std::vector<skbench::Complex> physicalAgain(retained.size());
        std::vector<skbench::Complex> modalAgain(modal.size());
        skbench::verticalForward(workload, modes.size(), vertical, retained.data(), modal.data());
        skbench::verticalInverse(workload, modes.size(), vertical, modal.data(), physicalAgain.data());
        skbench::verticalForward(workload, modes.size(), vertical, physicalAgain.data(), modalAgain.data());
        require(skbench::maximumRelativeError(modalAgain.data(), modal.data(), modal.size()) < 1.0e-12, "vertical modal round trip");

        std::cout << "skbench unit tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "skbench_tests: " << error.what() << '\n';
        return 1;
    }
}
