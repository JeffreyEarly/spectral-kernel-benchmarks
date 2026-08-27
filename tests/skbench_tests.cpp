#include "skbench/skbench.hpp"
#include "allocation_tracker.hpp"

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
    auto output = alignedBuffer<double>(workload.realElements());
    for (std::size_t index = 0; index < workload.realElements(); ++index) {
        input.get()[index] = static_cast<double>(index % 31) / 31.0;
    }

    skbench::FFTWProvider provider(workload, strategy);
    for (std::size_t repetition = 0; repetition < 3; ++repetition) {
        provider.forward(input.get(), spectrum.get());
        provider.inverse(spectrum.get(), output.get());
        if (strategy.outerWorkers > 1) provider.executeSchedulerNoop();
    }

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) {
        provider.forward(input.get(), spectrum.get());
        provider.inverse(spectrum.get(), output.get());
        if (strategy.outerWorkers > 1) provider.executeSchedulerNoop();
    }
    require(skbench::test::endAllocationTracking() == 0, "FFTW steady-state execution allocated memory");
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

        const auto profileList = skbench::profiles();
        require(skbench::profileNamed("wvm-historical-256-nz65-f4").workload.planes() == 260,
                "historical workload profile");
        require(skbench::profileNamed("wvm-current-512-nz257-f4").workload.planes() == 1028,
                "current workload profile");
        require(profileList.size() == 13, "unexpected profile count");

        skbench::RunOptions fftwOptions;
        fftwOptions.profile = "smoke";
        fftwOptions.providers = "fftw";
        fftwOptions.fftwPlanning = "measure";
        fftwOptions.fftwAlignment = "aligned";
        fftwOptions.fftwWisdom = "generated-import";
        fftwOptions.fftwInternalWorkers = 2;
        fftwOptions.fftwOuterWorkers = 2;
        fftwOptions.warmups = 1;
        fftwOptions.samples = 2;
        const auto fftwReport = skbench::runBenchmark(fftwOptions);
        require(fftwReport.status == "passed", "FFTW strategy benchmark failed");
        require(fftwReport.providers.size() == 1, "FFTW-only provider selection failed");
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
