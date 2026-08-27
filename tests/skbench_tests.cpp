#include "skbench/skbench.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
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

        const auto profileList = skbench::profiles();
        require(skbench::profileNamed("wvm-historical-256-nz65-f4").workload.planes() == 260,
                "historical workload profile");
        require(skbench::profileNamed("wvm-current-512-nz257-f4").workload.planes() == 1028,
                "current workload profile");
        require(profileList.size() == 13, "unexpected profile count");

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
