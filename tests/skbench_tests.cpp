#include "skbench/skbench.hpp"
#include "allocation_tracker.hpp"

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SKBENCH_TEST_HAVE_FFTWPP
#define SKBENCH_TEST_HAVE_FFTWPP 0
#endif

#if SKBENCH_TEST_HAVE_FFTWPP
namespace skbench {
std::uint64_t probeDealiasedConvolutionSteadyStateAllocationsForTesting(
    std::size_t n, std::size_t products,
    void (*beginTracking)(), std::uint64_t (*endTracking)());
std::uint64_t probeWvmAdvectiveConvolutionSteadyStateAllocationsForTesting(
    std::size_t n, void (*beginTracking)(), std::uint64_t (*endTracking)());
}
#endif

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

bool accelerateAllocationAssertionsEnabled() {
    const auto* skip = std::getenv("SKBENCH_SKIP_ACCELERATE_ALLOCATION_ASSERTIONS");
    return skip == nullptr || std::string(skip) != "1";
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

void requireAllocationFreePrunedExecution(
    const skbench::Workload& workload, const std::vector<skbench::RetainedMode>& modes,
    std::size_t outerWorkers) {
    auto input = alignedBuffer<double>(workload.realElements());
    auto retained = alignedBuffer<skbench::Complex>(modes.size() * workload.planes());
    auto output = alignedBuffer<double>(workload.realElements());
    for (std::size_t index = 0; index < workload.realElements(); ++index) {
        input.get()[index] = static_cast<double>(index % 31) / 31.0;
    }

    skbench::FFTWPrunedProvider provider(
        workload, modes, skbench::FFTWPlanningMode::estimate, 1, outerWorkers);
    for (std::size_t repetition = 0; repetition < 3; ++repetition) {
        provider.forward(input.get(), retained.get());
        provider.inverse(retained.get(), output.get());
        if (outerWorkers > 1) provider.executeSchedulerNoop();
    }

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) {
        provider.forward(input.get(), retained.get());
        provider.inverse(retained.get(), output.get());
        if (outerWorkers > 1) provider.executeSchedulerNoop();
    }
    require(skbench::test::endAllocationTracking() == 0,
            "partially pruned FFTW steady-state execution allocated memory");
}

void requireAllocationFreeFusedRetainedSplitExecution(
    const skbench::Workload& workload,
    const std::vector<skbench::RetainedMode>& modes,
    std::size_t outerWorkers) {
    auto input = alignedBuffer<double>(workload.realElements());
    auto spectrum = alignedBuffer<skbench::Complex>(workload.spectrumElements());
    auto retainedReal = alignedBuffer<double>(modes.size() * workload.planes());
    auto retainedImag = alignedBuffer<double>(modes.size() * workload.planes());
    auto output = alignedBuffer<double>(workload.realElements());
    for (std::size_t index = 0; index < workload.realElements(); ++index) {
        input.get()[index] = static_cast<double>(index % 43) / 43.0;
    }

    skbench::FFTWProvider provider(workload, {
        skbench::FFTWPlanningMode::estimate,
        skbench::FFTWAlignmentStrategy::unaligned,
        skbench::FFTWWisdomStrategy::cold,
        1,
        outerWorkers,
        0.0,
        skbench::FFTWDataLayout::interleaved,
        skbench::FFTWSpectrumOrder::planeMajor});
    const auto execute = [&] {
        provider.forward(input.get(), spectrum.get());
        provider.gatherRetainedToSplitOuter(
            modes, spectrum.get(), retainedReal.get(), retainedImag.get());
        provider.scaleRetainedSplitOuter(
            modes, retainedReal.get(), retainedImag.get(), 0.5);
        provider.embedRetainedFromSplitOuter(
            modes, retainedReal.get(), retainedImag.get(), spectrum.get());
        provider.inverse(spectrum.get(), output.get());
        if (outerWorkers > 1) provider.executeSchedulerNoop();
    };
    for (std::size_t repetition = 0; repetition < 3; ++repetition) execute();

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) execute();
    require(skbench::test::endAllocationTracking() == 0,
            "fused retained-split FFTW execution allocated memory");
}

void requireAllocationFreePrunedSplitExecution(
    const skbench::Workload& workload,
    const std::vector<skbench::RetainedMode>& modes,
    std::size_t outerWorkers) {
    auto input = alignedBuffer<double>(workload.realElements());
    auto retainedReal = alignedBuffer<double>(modes.size() * workload.planes());
    auto retainedImag = alignedBuffer<double>(modes.size() * workload.planes());
    auto output = alignedBuffer<double>(workload.realElements());
    for (std::size_t index = 0; index < workload.realElements(); ++index) {
        input.get()[index] = static_cast<double>(index % 47) / 47.0;
    }

    skbench::FFTWPrunedProvider provider(
        workload, modes, skbench::FFTWPlanningMode::estimate, 1, outerWorkers);
    const auto execute = [&] {
        provider.forwardSplit(
            input.get(), retainedReal.get(), retainedImag.get(), 0.5);
        provider.inverseSplit(
            retainedReal.get(), retainedImag.get(), output.get());
        if (outerWorkers > 1) provider.executeSchedulerNoop();
    };
    for (std::size_t repetition = 0; repetition < 3; ++repetition) execute();

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) execute();
    require(skbench::test::endAllocationTracking() == 0,
            "pruned retained-split FFTW execution allocated memory");
}

void requireAllocationFreeStreamingPrunedSplitExecution(
    const skbench::Workload& workload,
    const std::vector<skbench::RetainedMode>& modes,
    std::size_t outerWorkers, std::size_t tileWidth = 1) {
    auto input = alignedBuffer<double>(workload.realElements());
    auto retainedReal = alignedBuffer<double>(modes.size() * workload.planes());
    auto retainedImag = alignedBuffer<double>(modes.size() * workload.planes());
    auto output = alignedBuffer<double>(workload.realElements());
    for (std::size_t index = 0; index < workload.realElements(); ++index) {
        input.get()[index] = static_cast<double>(index % 53) / 53.0;
    }

    skbench::FFTWStreamingPrunedSplitProvider provider(
        workload, modes, skbench::FFTWPlanningMode::estimate, 1,
        outerWorkers, tileWidth);
    const auto execute = [&] {
        provider.forwardSplit(
            input.get(), retainedReal.get(), retainedImag.get());
        provider.inverseSplit(
            retainedReal.get(), retainedImag.get(), output.get());
        if (outerWorkers > 1) provider.executeSchedulerNoop();
    };
    for (std::size_t repetition = 0; repetition < 3; ++repetition) execute();

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) execute();
    require(skbench::test::endAllocationTracking() == 0,
            "streaming pruned split FFTW execution allocated memory");
}

void requireAllocationFreeRetainedOuterExecution(
    const skbench::Workload& workload, const std::vector<skbench::RetainedMode>& modes,
    std::size_t outerWorkers,
    skbench::FFTWSpectrumOrder spectrumOrder = skbench::FFTWSpectrumOrder::wvmFrequencyMajor) {
    auto input = alignedBuffer<double>(workload.realElements());
    auto spectrum = alignedBuffer<skbench::Complex>(workload.spectrumElements());
    auto retained = alignedBuffer<skbench::Complex>(modes.size() * workload.planes());
    auto output = alignedBuffer<double>(workload.realElements());
    for (std::size_t index = 0; index < workload.realElements(); ++index) {
        input.get()[index] = static_cast<double>(index % 31) / 31.0;
    }

    skbench::FFTWProvider provider(workload, {
        skbench::FFTWPlanningMode::estimate,
        skbench::FFTWAlignmentStrategy::unaligned,
        skbench::FFTWWisdomStrategy::cold,
        1,
        outerWorkers,
        0.0,
        skbench::FFTWDataLayout::interleaved,
        spectrumOrder});
    for (std::size_t repetition = 0; repetition < 3; ++repetition) {
        provider.forward(input.get(), spectrum.get());
        provider.gatherRetainedOuter(modes, spectrum.get(), retained.get());
        provider.embedRetainedOuter(modes, retained.get(), spectrum.get());
        provider.inverse(spectrum.get(), output.get());
        if (outerWorkers > 1) provider.executeSchedulerNoop();
    }

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) {
        provider.forward(input.get(), spectrum.get());
        provider.gatherRetainedOuter(modes, spectrum.get(), retained.get());
        provider.embedRetainedOuter(modes, retained.get(), spectrum.get());
        provider.inverse(spectrum.get(), output.get());
        if (outerWorkers > 1) provider.executeSchedulerNoop();
    }
    require(skbench::test::endAllocationTracking() == 0,
            "outer-sharded full retained FFTW execution allocated memory");
}

void requireAllocationFreeSplitRetainedOuterExecution(
    const skbench::Workload& workload, const std::vector<skbench::RetainedMode>& modes,
    std::size_t outerWorkers, skbench::FFTWSpectrumOrder spectrumOrder) {
    auto input = alignedBuffer<double>(workload.realElements());
    auto split = alignedBuffer<double>(2 * workload.spectrumElements());
    auto retainedReal = alignedBuffer<double>(modes.size() * workload.planes());
    auto retainedImag = alignedBuffer<double>(modes.size() * workload.planes());
    auto output = alignedBuffer<double>(workload.realElements());
    auto* spectrumReal = split.get();
    auto* spectrumImag = split.get() + workload.spectrumElements();
    for (std::size_t index = 0; index < workload.realElements(); ++index) {
        input.get()[index] = static_cast<double>(index % 37) / 37.0;
    }

    skbench::FFTWProvider provider(workload, {
        skbench::FFTWPlanningMode::estimate,
        skbench::FFTWAlignmentStrategy::unaligned,
        skbench::FFTWWisdomStrategy::cold,
        1,
        outerWorkers,
        0.0,
        skbench::FFTWDataLayout::split,
        spectrumOrder});
    const auto execute = [&] {
        provider.forwardSplit(input.get(), spectrumReal, spectrumImag);
        provider.gatherRetainedSplitOuter(
            modes, spectrumReal, spectrumImag, retainedReal.get(), retainedImag.get());
        provider.embedRetainedSplitOuter(
            modes, retainedReal.get(), retainedImag.get(), spectrumReal, spectrumImag);
        provider.inverseSplit(spectrumReal, spectrumImag, output.get());
        if (outerWorkers > 1) provider.executeSchedulerNoop();
    };
    for (std::size_t repetition = 0; repetition < 3; ++repetition) execute();

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) execute();
    require(skbench::test::endAllocationTracking() == 0,
            "outer-sharded split retained FFTW execution allocated memory");
}

void requireAllocationFreeVdspRetainedExecution(
    const skbench::Workload& workload,
    const std::vector<skbench::RetainedMode>& modes,
    std::size_t workers) {
    auto input = alignedBuffer<double>(workload.realElements());
    auto retainedReal = alignedBuffer<double>(modes.size() * workload.planes());
    auto retainedImag = alignedBuffer<double>(modes.size() * workload.planes());
    auto output = alignedBuffer<double>(workload.realElements());
    for (std::size_t index = 0; index < workload.realElements(); ++index) {
        input.get()[index] = static_cast<double>(index % 41) / 41.0;
    }

    skbench::VDSPProvider provider(
        workload, workers, skbench::VDSPTransformStrategy::inPlace,
        skbench::VDSPBatchStrategy::directPersistent);
    if (!provider.supported()) return;
    const auto execute = [&] {
        provider.forwardRetainedNativeSplit(
            input.get(), modes, retainedReal.get(), retainedImag.get());
        provider.inverseRetainedNativeSplit(
            modes, retainedReal.get(), retainedImag.get(), output.get());
    };
    for (std::size_t repetition = 0; repetition < 3; ++repetition) execute();

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) execute();
    require(skbench::test::endAllocationTracking() == 0,
            "vDSP native retained steady-state execution allocated memory");
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

void requireAllocationFreeDirectOrdering(
    const skbench::Workload& workload,
    const std::vector<skbench::RetainedMode>& modes,
    const skbench::GroupedVerticalOperators& operators,
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

    skbench::WvmDirectVerticalGemmProvider provider(
        workload, modes, operators, strategy);
    require(provider.supported(), "Accelerate direct WVM-order provider is unavailable");
    std::vector<skbench::Complex> fullModalInput(provider.modalSpectrumElements());
    std::vector<skbench::Complex> fullModalOutput(provider.modalSpectrumElements());
    skbench::embedRetainedModal(
        workload, modes, modal.data(), fullModalInput.data());
    provider.initializeModalOutput(fullModalOutput.data());
    provider.initializeSpectrumOutput(fullOutput.data());
    for (std::size_t repetition = 0; repetition < 3; ++repetition) {
        provider.executeForward(fullInput.data(), fullModalOutput.data());
        provider.executeInverse(fullModalInput.data(), fullOutput.data());
        if (strategy.outerWorkers > 1) provider.executeSchedulerNoop();
    }

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) {
        provider.executeForward(fullInput.data(), fullModalOutput.data());
        provider.executeInverse(fullModalInput.data(), fullOutput.data());
        if (strategy.outerWorkers > 1) provider.executeSchedulerNoop();
    }
    require(skbench::test::endAllocationTracking() == 0,
            "direct WVM-order steady-state execution allocated memory");
}

void requireAllocationFreePlaneMajorOrdering(
    const skbench::Workload& workload,
    const std::vector<skbench::RetainedMode>& modes,
    const skbench::GroupedVerticalOperators& operators,
    skbench::VerticalGemmStrategy strategy) {
    const auto physicalCount = workload.nz * modes.size() * workload.fields;
    const auto modalCount = operators.nj * modes.size() * workload.fields;
    const auto fullModalCount = workload.halfRows() * operators.nj * workload.fields;
    std::vector<skbench::Complex> physical(physicalCount);
    std::vector<skbench::Complex> modal(modalCount);
    std::vector<skbench::Complex> wvmInput(workload.spectrumElements());
    std::vector<skbench::Complex> planeInput(workload.spectrumElements());
    std::vector<skbench::Complex> planeOutput(workload.spectrumElements());
    std::vector<skbench::Complex> planeModalInput(fullModalCount);
    std::vector<skbench::Complex> planeModalOutput(fullModalCount);
    for (std::size_t index = 0; index < physical.size(); ++index) {
        physical[index] = {static_cast<double>(index % 31) / 31.0,
                           -static_cast<double>(index % 29) / 29.0};
    }
    for (std::size_t index = 0; index < modal.size(); ++index) {
        modal[index] = {static_cast<double>(index % 23) / 23.0,
                        -static_cast<double>(index % 19) / 19.0};
    }
    skbench::embedRetained(workload, modes, physical.data(), wvmInput.data());
    skbench::wvmToPlaneMajor(workload, wvmInput.data(), planeInput.data());
    for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        const auto& mode = modes[modeIndex];
        const auto frequency = mode.storedKx + workload.nxHalf() * mode.storedKy;
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t j = 0; j < operators.nj; ++j) {
                auto value = modal[skbench::modalSpectrumIndex(
                    workload, modeIndex, j, field)];
                if (mode.conjugatesStoredValue) value = skbench::conjugate(value);
                planeModalInput[frequency + workload.halfRows() *
                    (j + operators.nj * field)] = value;
            }
        }
    }

    skbench::PlaneMajorDirectVerticalGemmProvider provider(
        workload, modes, operators, strategy);
    require(provider.supported(), "Accelerate plane-major view provider is unavailable");
    provider.initializeModalOutput(planeModalOutput.data());
    provider.initializeSpectrumOutput(planeOutput.data());
    for (std::size_t repetition = 0; repetition < 3; ++repetition) {
        provider.executeForward(planeInput.data(), planeModalOutput.data());
        provider.initializeSpectrumOutput(planeOutput.data());
        provider.executeInverse(planeModalInput.data(), planeOutput.data());
        if (strategy.outerWorkers > 1) provider.executeSchedulerNoop();
    }

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) {
        provider.executeForward(planeInput.data(), planeModalOutput.data());
        provider.initializeSpectrumOutput(planeOutput.data());
        provider.executeInverse(planeModalInput.data(), planeOutput.data());
        if (strategy.outerWorkers > 1) provider.executeSchedulerNoop();
    }
    require(skbench::test::endAllocationTracking() == 0,
            "plane-major retained-view steady-state execution allocated memory");
}

void requireAllocationFreeWvmPipeline(
    const skbench::Workload& workload,
    const std::vector<skbench::RetainedMode>& modes,
    const skbench::GroupedVerticalOperators& operators,
    skbench::VerticalGemmStrategy strategy) {
    auto input = alignedBuffer<double>(workload.realElements());
    auto spectrum = alignedBuffer<skbench::Complex>(workload.spectrumElements());
    auto output = alignedBuffer<double>(workload.realElements());
    const auto weights = skbench::syntheticModalWorkWeights(workload, modes);
    for (std::size_t index = 0; index < workload.realElements(); ++index) {
        input.get()[index] = static_cast<double>(index % 53) / 53.0;
    }

    skbench::FFTWProvider fftw(workload, {
        skbench::FFTWPlanningMode::estimate,
        skbench::FFTWAlignmentStrategy::unaligned,
        skbench::FFTWWisdomStrategy::cold,
        1,
        2,
        0.0,
        skbench::FFTWDataLayout::interleaved,
        skbench::FFTWSpectrumOrder::wvmFrequencyMajor});
    skbench::WvmDirectVerticalGemmProvider vertical(
        workload, modes, operators, strategy);
    require(vertical.supported(), "WVM pipeline vertical provider is unavailable");
    auto modalForward = alignedBuffer<skbench::Complex>(vertical.modalSpectrumElements());
    auto modalPostWork = alignedBuffer<skbench::Complex>(vertical.modalSpectrumElements());
    vertical.initializeModalOutput(modalForward.get());
    vertical.initializeModalOutput(modalPostWork.get());

    const auto execute = [&] {
        fftw.forward(input.get(), spectrum.get());
        vertical.executeForward(spectrum.get(), modalForward.get());
        skbench::applySyntheticModalWorkWvm(
            workload, modes, weights.data(), modalForward.get(), modalPostWork.get());
        vertical.initializeSpectrumOutput(spectrum.get());
        vertical.executeInverse(modalPostWork.get(), spectrum.get());
        fftw.inverse(spectrum.get(), output.get());
    };
    for (std::size_t repetition = 0; repetition < 3; ++repetition) execute();

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) execute();
    require(skbench::test::endAllocationTracking() == 0,
            "WVM synthetic pipeline steady-state execution allocated memory");
}

void requireAllocationFreeFusedSplitPipeline(
    const skbench::Workload& workload,
    const std::vector<skbench::RetainedMode>& modes,
    const skbench::GroupedVerticalOperators& operators,
    skbench::VerticalGemmStrategy strategy) {
    auto input = alignedBuffer<double>(workload.realElements());
    auto spectrum = alignedBuffer<skbench::Complex>(workload.spectrumElements());
    auto output = alignedBuffer<double>(workload.realElements());
    const auto weights = skbench::syntheticModalWorkWeights(workload, modes);
    const auto modalElements =
        workload.retainedVerticalModes() * modes.size() * workload.fields;
    for (std::size_t index = 0; index < workload.realElements(); ++index) {
        input.get()[index] = static_cast<double>(index % 59) / 59.0;
    }

    skbench::FFTWProvider fftw(workload, {
        skbench::FFTWPlanningMode::estimate,
        skbench::FFTWAlignmentStrategy::unaligned,
        skbench::FFTWWisdomStrategy::cold,
        1,
        2,
        0.0,
        skbench::FFTWDataLayout::interleaved,
        skbench::FFTWSpectrumOrder::planeMajor});
    skbench::VerticalGemmProvider vertical(
        workload, operators, skbench::VerticalGemmLayout::split, strategy);
    require(vertical.supported(), "fused-split pipeline vertical provider is unavailable");

    const auto execute = [&] {
        fftw.forward(input.get(), spectrum.get());
        fftw.gatherRetainedToSplitOuter(
            modes, spectrum.get(), vertical.splitPhysicalInputRealData(),
            vertical.splitPhysicalInputImaginaryData());
        vertical.executeForward();
        skbench::applySyntheticModalWorkSplit(
            modalElements, weights.data(),
            vertical.splitModalOutputRealData(),
            vertical.splitModalOutputImaginaryData(),
            vertical.splitModalInputRealData(),
            vertical.splitModalInputImaginaryData());
        vertical.executeInverse();
        fftw.embedRetainedFromSplitOuter(
            modes, vertical.splitPhysicalOutputRealData(),
            vertical.splitPhysicalOutputImaginaryData(), spectrum.get());
        fftw.inverse(spectrum.get(), output.get());
    };
    for (std::size_t repetition = 0; repetition < 3; ++repetition) execute();

    skbench::test::beginAllocationTracking();
    for (std::size_t repetition = 0; repetition < 32; ++repetition) execute();
    require(skbench::test::endAllocationTracking() == 0,
            "fused-split synthetic pipeline steady-state execution allocated memory");
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
        for (const auto order : {skbench::FFTWSpectrumOrder::wvmFrequencyMajor,
                                 skbench::FFTWSpectrumOrder::planeMajor}) {
            const auto name = skbench::fftwSpectrumOrderName(order);
            require(skbench::fftwSpectrumOrderNamed(name) == order,
                    "FFTW spectrum order name round trip");
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
        const auto largeProfile = skbench::profileNamed("wvm-large-1024-nz129-f4");
        const auto largeModes = skbench::retainedHorizontalModes(largeProfile.workload);
        require(largeModes.size() == 183037,
                "large four-field retained-mode count");
        require(skbench::squaredWavenumberGroups(largeModes).size() == 27779,
                "large four-field squared-wavenumber group count");
        require(profileList.size() == 14, "unexpected profile count");

        const auto prunedModes = skbench::retainedHorizontalModes(workload);
        skbench::FFTWPrunedProvider prunedProvider(
            workload, prunedModes, skbench::FFTWPlanningMode::estimate, 1, 2);
        require(prunedProvider.activeKxCount() < prunedProvider.fullKxCount(),
                "pruned FFTW candidate did not omit any kx columns");
        require(prunedProvider.columnTransformsPerDirection() +
                    prunedProvider.omittedColumnTransformsPerDirection() ==
                    workload.nxHalf() * workload.planes(),
                "pruned FFTW column accounting");
        require(prunedProvider.scratchBytes() ==
                    workload.spectrumElements() * sizeof(skbench::Complex),
                "pruned FFTW scratch accounting");
        require(prunedProvider.internalWorkers() == 1 &&
                    prunedProvider.outerWorkers() == 2 &&
                    prunedProvider.totalLogicalWorkers() == 2,
                "pruned FFTW outer worker topology");
        require(prunedProvider.maximumShardScratchBytes() * 2 >=
                    prunedProvider.scratchBytes(),
                "pruned FFTW maximum shard scratch accounting");
        require(!prunedProvider.completeHalfSpectrumOutputMaterialized(),
                "pruned FFTW unexpectedly claims a complete transformed output");
        require(!prunedProvider.inPlaceRetainedOperatorSupported() &&
                    !prunedProvider.inPlaceRetainedOperatorCapability().empty(),
                "pruned FFTW in-place capability contract");
        skbench::FFTWStreamingPrunedSplitProvider streamingPrunedProvider(
            workload, prunedModes, skbench::FFTWPlanningMode::estimate, 1, 2);
        skbench::FFTWStreamingPrunedSplitProvider tiledStreamingPrunedProvider(
            workload, prunedModes, skbench::FFTWPlanningMode::estimate, 1, 2, 8);
        require(streamingPrunedProvider.activeKxCount() ==
                    prunedProvider.activeKxCount(),
                "streaming pruned active-column count");
        require(streamingPrunedProvider.scratchBytes() ==
                    2 * workload.halfRows() * sizeof(skbench::Complex),
                "streaming pruned worker-local scratch accounting");
        require(streamingPrunedProvider.scratchBytes() <
                    prunedProvider.scratchBytes(),
                "streaming pruned scratch must be smaller than batch scratch");
        require(streamingPrunedProvider.workerScratchBytes() ==
                    workload.halfRows() * sizeof(skbench::Complex),
                "streaming pruned per-worker scratch accounting");
        require(!streamingPrunedProvider.completeHalfSpectrumMaterialized(),
                "streaming pruned provider unexpectedly materializes a full spectrum");
        require(tiledStreamingPrunedProvider.tileWidth() == 8,
                "tiled streaming pruned width");
        require(tiledStreamingPrunedProvider.fftScratchBytes() ==
                    workload.halfRows() * sizeof(skbench::Complex),
                "tiled streaming per-worker FFT scratch");
        require(tiledStreamingPrunedProvider.compactTileBytes() ==
                    2 * 8 * prunedModes.size() * sizeof(skbench::Complex),
                "tiled streaming compact scratch accounting");
        require(tiledStreamingPrunedProvider.scratchBytes() ==
                    2 * workload.halfRows() * sizeof(skbench::Complex) +
                    tiledStreamingPrunedProvider.compactTileBytes(),
                "tiled streaming total scratch accounting");
        require(!tiledStreamingPrunedProvider.completeHalfSpectrumMaterialized(),
                "tiled streaming unexpectedly materializes a full spectrum");

        std::vector<skbench::Complex> prunedOracleSpectrum(workload.spectrumElements());
        std::vector<skbench::Complex> prunedEmbeddedSpectrum(workload.spectrumElements());
        std::vector<skbench::Complex> prunedRetainedOracle(
            prunedModes.size() * workload.planes());
        std::vector<skbench::Complex> prunedRetainedActual(prunedRetainedOracle.size());
        std::vector<double> prunedInverseOracle(workload.realElements());
        std::vector<double> prunedInverseActual(workload.realElements());
        for (std::size_t fixtureIndex = 0; fixtureIndex < 5; ++fixtureIndex) {
            const auto fixture = static_cast<skbench::FixtureKind>(fixtureIndex);
            const auto fixtureInput = skbench::makeFixture(
                workload, fixture, 700 + fixtureIndex);
            const auto preservedInput = fixtureInput;
            skbench::directR2C(
                workload, fixtureInput.data(), prunedOracleSpectrum.data());
            skbench::gatherRetained(
                workload, prunedModes, prunedOracleSpectrum.data(),
                prunedRetainedOracle.data());
            prunedProvider.forward(fixtureInput.data(), prunedRetainedActual.data());
            require(skbench::maximumRelativeError(
                        prunedRetainedActual.data(), prunedRetainedOracle.data(),
                        prunedRetainedActual.size()) < 1.0e-12,
                    "pruned FFTW forward versus direct mathematical oracle");
            require(fixtureInput == preservedInput,
                    "pruned FFTW forward modified caller input");

            skbench::embedRetained(
                workload, prunedModes, prunedRetainedOracle.data(),
                prunedEmbeddedSpectrum.data());
            skbench::directC2R(
                workload, prunedEmbeddedSpectrum.data(), prunedInverseOracle.data());
            const auto preservedRetained = prunedRetainedOracle;
            prunedProvider.inverse(
                prunedRetainedOracle.data(), prunedInverseActual.data());
            require(skbench::maximumRelativeError(
                        prunedInverseActual.data(), prunedInverseOracle.data(),
                        prunedInverseActual.size()) < 1.0e-12,
                    "pruned FFTW inverse versus direct mathematical oracle");
            require(skbench::maximumRelativeError(
                        prunedRetainedOracle.data(), preservedRetained.data(),
                        prunedRetainedOracle.size()) == 0.0,
                    "pruned FFTW inverse modified caller retained input");
        }

        skbench::FFTWProvider fusedSplitProvider(workload, {
            skbench::FFTWPlanningMode::estimate,
            skbench::FFTWAlignmentStrategy::unaligned,
            skbench::FFTWWisdomStrategy::cold,
            1,
            2,
            0.0,
            skbench::FFTWDataLayout::interleaved,
            skbench::FFTWSpectrumOrder::planeMajor});
        std::vector<skbench::Complex> planeMajorSpectrum(workload.spectrumElements());
        std::vector<double> retainedOracleReal(prunedRetainedOracle.size());
        std::vector<double> retainedOracleImag(prunedRetainedOracle.size());
        std::vector<double> retainedSplitReal(prunedRetainedOracle.size());
        std::vector<double> retainedSplitImag(prunedRetainedOracle.size());
        std::vector<double> retainedNormalizedReal(prunedRetainedOracle.size());
        std::vector<double> retainedNormalizedImag(prunedRetainedOracle.size());
        for (std::size_t fixtureIndex = 0; fixtureIndex < 5; ++fixtureIndex) {
            const auto fixture = static_cast<skbench::FixtureKind>(fixtureIndex);
            const auto fixtureInput = skbench::makeFixture(
                workload, fixture, 900 + fixtureIndex);
            skbench::directR2C(
                workload, fixtureInput.data(), prunedOracleSpectrum.data());
            skbench::gatherRetained(
                workload, prunedModes, prunedOracleSpectrum.data(),
                prunedRetainedOracle.data());
            skbench::interleavedToSplit(
                prunedRetainedOracle.size(), prunedRetainedOracle.data(),
                retainedOracleReal.data(), retainedOracleImag.data());

            fusedSplitProvider.forward(
                fixtureInput.data(), planeMajorSpectrum.data());
            fusedSplitProvider.gatherRetainedToSplitOuter(
                prunedModes, planeMajorSpectrum.data(), retainedSplitReal.data(),
                retainedSplitImag.data());
            require(std::max(
                        skbench::maximumRelativeError(
                            retainedSplitReal.data(), retainedOracleReal.data(),
                            retainedSplitReal.size()),
                        skbench::maximumRelativeError(
                            retainedSplitImag.data(), retainedOracleImag.data(),
                            retainedSplitImag.size())) < 1.0e-12,
                    "fused retained-split forward versus mode-keyed oracle");

            constexpr double scale = 0.25;
            fusedSplitProvider.gatherRetainedToSplitOuter(
                prunedModes, planeMajorSpectrum.data(), retainedNormalizedReal.data(),
                retainedNormalizedImag.data(), scale);
            for (std::size_t index = 0; index < retainedSplitReal.size(); ++index) {
                retainedSplitReal[index] *= scale;
                retainedSplitImag[index] *= scale;
            }
            require(std::max(
                        skbench::maximumRelativeError(
                            retainedNormalizedReal.data(), retainedSplitReal.data(),
                            retainedSplitReal.size()),
                        skbench::maximumRelativeError(
                            retainedNormalizedImag.data(), retainedSplitImag.data(),
                            retainedSplitImag.size())) < 1.0e-12,
                    "fused retained-split normalization versus separate scale");

            fusedSplitProvider.embedRetainedFromSplitOuter(
                prunedModes, retainedOracleReal.data(), retainedOracleImag.data(),
                planeMajorSpectrum.data());
            fusedSplitProvider.inverse(
                planeMajorSpectrum.data(), prunedInverseActual.data());
            skbench::embedRetained(
                workload, prunedModes, prunedRetainedOracle.data(),
                prunedEmbeddedSpectrum.data());
            skbench::directC2R(
                workload, prunedEmbeddedSpectrum.data(), prunedInverseOracle.data());
            require(skbench::maximumRelativeError(
                        prunedInverseActual.data(), prunedInverseOracle.data(),
                        prunedInverseActual.size()) < 1.0e-12,
                    "fused retained-split inverse versus mode-keyed oracle");

            prunedProvider.forwardSplit(
                fixtureInput.data(), retainedSplitReal.data(),
                retainedSplitImag.data());
            require(std::max(
                        skbench::maximumRelativeError(
                            retainedSplitReal.data(), retainedOracleReal.data(),
                            retainedSplitReal.size()),
                        skbench::maximumRelativeError(
                            retainedSplitImag.data(), retainedOracleImag.data(),
                            retainedSplitImag.size())) < 1.0e-12,
                    "pruned retained-split forward versus mode-keyed oracle");
            prunedProvider.inverseSplit(
                retainedOracleReal.data(), retainedOracleImag.data(),
                prunedInverseActual.data());
            require(skbench::maximumRelativeError(
                        prunedInverseActual.data(), prunedInverseOracle.data(),
                        prunedInverseActual.size()) < 1.0e-12,
                    "pruned retained-split inverse versus mode-keyed oracle");

            streamingPrunedProvider.forwardSplit(
                fixtureInput.data(), retainedSplitReal.data(),
                retainedSplitImag.data());
            require(std::max(
                        skbench::maximumRelativeError(
                            retainedSplitReal.data(), retainedOracleReal.data(),
                            retainedSplitReal.size()),
                        skbench::maximumRelativeError(
                            retainedSplitImag.data(), retainedOracleImag.data(),
                            retainedSplitImag.size())) < 1.0e-12,
                    "streaming pruned direct-split forward versus mode-keyed oracle");
            streamingPrunedProvider.inverseSplit(
                retainedOracleReal.data(), retainedOracleImag.data(),
                prunedInverseActual.data());
            require(skbench::maximumRelativeError(
                        prunedInverseActual.data(), prunedInverseOracle.data(),
                        prunedInverseActual.size()) < 1.0e-12,
                    "streaming pruned split inverse versus mode-keyed oracle");

            tiledStreamingPrunedProvider.forwardSplit(
                fixtureInput.data(), retainedSplitReal.data(),
                retainedSplitImag.data());
            require(std::max(
                        skbench::maximumRelativeError(
                            retainedSplitReal.data(), retainedOracleReal.data(),
                            retainedSplitReal.size()),
                        skbench::maximumRelativeError(
                            retainedSplitImag.data(), retainedOracleImag.data(),
                            retainedSplitImag.size())) < 1.0e-12,
                    "tiled streaming forward versus mode-keyed oracle");
            tiledStreamingPrunedProvider.inverseSplit(
                retainedOracleReal.data(), retainedOracleImag.data(),
                prunedInverseActual.data());
            require(skbench::maximumRelativeError(
                        prunedInverseActual.data(), prunedInverseOracle.data(),
                        prunedInverseActual.size()) < 1.0e-12,
                    "tiled streaming inverse versus mode-keyed oracle");
        }

        skbench::RunOptions prunedOptions;
        prunedOptions.kernel = "pruned-horizontal";
        prunedOptions.profile = "smoke";
        prunedOptions.providers = "fftw";
        prunedOptions.fftwPlanning = "estimate";
        prunedOptions.fftwInternalWorkers = 1;
        prunedOptions.fftwOuterWorkers = 2;
        prunedOptions.warmups = 1;
        prunedOptions.samples = 2;
        const auto prunedReport = skbench::runBenchmark(prunedOptions);
        require(prunedReport.status == "passed",
                "pruned FFTW smoke benchmark failed");
        require(prunedReport.providers.size() == 2,
                "pruned FFTW benchmark candidate count");
        require(prunedReport.providers[0].id == "fftw-full-2d-retained-reference" &&
                    prunedReport.providers[1].id == "fftw-partial-column-pruned",
                "pruned FFTW provider identities");
        require(prunedReport.providers[1].scratchBytes ==
                    prunedReport.fullSpectrumBytes,
                "pruned FFTW report does not expose full-sized scratch");
        for (const auto& provider : prunedReport.providers) {
            require(provider.internalWorkers == 1 && provider.outerWorkers == 2 &&
                        provider.workers == 2,
                    "pruned benchmark matched outer worker topology");
            require(provider.schedulingId.find("persistent-outer") != std::string::npos,
                    "pruned benchmark scheduling identity");
        }
        for (const auto& provider : prunedReport.providers) {
            for (const auto& correctness : provider.correctness) {
                require(correctness.passed, "pruned FFTW benchmark correctness");
            }
        }

        auto prunedSplitOptions = prunedOptions;
        prunedSplitOptions.retainedRepresentation = "split";
        const auto prunedSplitReport = skbench::runBenchmark(prunedSplitOptions);
        require(prunedSplitReport.status == "passed" &&
                    prunedSplitReport.providers.size() == 2 &&
                    prunedSplitReport.providers[1].id ==
                        "fftw-partial-column-pruned-fused-split",
                "pruned retained-split smoke benchmark");

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
        require(fftwRecord.correctness.size() == 5, "FFTW strategy correctness metrics");
        for (const auto& metric : fftwRecord.correctness) {
            require(metric.passed, "FFTW strategy correctness");
        }
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
        require(splitRecord.correctness.size() == 9, "split FFTW correctness metric count");
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
            if (timing.stage == "multidimensional split in-place") {
                foundInPlaceCapability = timing.state == skbench::StageState::unsupported && timing.seconds.empty();
            }
        }
        require(foundForwardConversion, "split FFTW conversion timing");
        require(foundRetainedTotal, "split FFTW retained total timing");
        require(foundInPlaceCapability, "split FFTW in-place capability timing");

        auto planeMajorOptions = fftwOptions;
        planeMajorOptions.fftwSpectrumOrder = "plane-major";
        planeMajorOptions.fftwPlanning = "estimate";
        planeMajorOptions.fftwAlignment = "unaligned";
        planeMajorOptions.fftwWisdom = "cold";
        const auto planeMajorReport = skbench::runBenchmark(planeMajorOptions);
        require(planeMajorReport.status == "passed",
                "plane-major paired FFTW benchmark failed");
        require(planeMajorReport.providers.size() == 2,
                "plane-major paired FFTW provider selection failed");
        require(planeMajorReport.providers[0].nativeRepresentationId ==
                    "plane-major-interleaved-half-spectrum" &&
                planeMajorReport.providers[1].nativeRepresentationId ==
                    "plane-major-split-half-spectrum",
                "plane-major FFTW native representations");
        for (const auto& provider : planeMajorReport.providers) {
            require(provider.execution.forward.stridesElements.find("kx=1") !=
                        std::string::npos,
                    "plane-major FFTW physical stride contract");
            for (const auto& metric : provider.correctness) {
                require(metric.passed, "plane-major FFTW correctness");
            }
        }

        auto retainedViewOptions = planeMajorOptions;
        retainedViewOptions.fftwLayout = "interleaved";
        retainedViewOptions.retainedRepresentation = "view";
        const auto retainedViewReport = skbench::runBenchmark(retainedViewOptions);
        require(retainedViewReport.status == "passed" &&
                    retainedViewReport.providers.size() == 1 &&
                    retainedViewReport.providers[0].id ==
                        "fftw-plane-major-retained-view",
                "persistent retained-view smoke benchmark");
        require(!retainedViewReport.providers[0]
                     .execution.inverse.adapterPreservesCallerInput &&
                    !retainedViewReport.providers[0]
                         .execution.inverse.requiresPreservationCopyForRepeatedExecution,
                "persistent retained-view dead inverse-input contract");
        bool foundElidedViewRetention = false;
        for (const auto& timing : retainedViewReport.providers[0].timings) {
            if (timing.stage == "logical retained index view" &&
                timing.direction == "forward") {
                foundElidedViewRetention =
                    timing.state == skbench::StageState::elided &&
                    timing.seconds.empty();
            }
        }
        require(foundElidedViewRetention,
                "persistent retained-view elided retention timing");

        auto fusedRetainedSplitOptions = retainedViewOptions;
        fusedRetainedSplitOptions.retainedRepresentation = "split";
        const auto fusedRetainedSplitReport =
            skbench::runBenchmark(fusedRetainedSplitOptions);
        require(fusedRetainedSplitReport.status == "passed" &&
                    fusedRetainedSplitReport.providers.size() == 1 &&
                    fusedRetainedSplitReport.providers[0].id ==
                        "fftw-plane-major-fused-retained-split",
                "fused retained-split smoke benchmark");
        require(fusedRetainedSplitReport.providers[0].correctness.size() == 9,
                "fused retained-split correctness metric count");
        bool foundFusedConversion = false;
        bool foundNormalizationDiagnostic = false;
        for (const auto& timing : fusedRetainedSplitReport.providers[0].timings) {
            if (timing.stage ==
                    "fused horizontal retention and split conversion" &&
                timing.direction == "forward") {
                foundFusedConversion =
                    timing.state == skbench::StageState::executed &&
                    timing.seconds.size() == 2;
            }
            if (timing.scope == "diagnostic-total" &&
                timing.stage ==
                    "retained operator with fused horizontal normalization") {
                foundNormalizationDiagnostic =
                    timing.state == skbench::StageState::executed &&
                    timing.seconds.size() == 2;
            }
        }
        require(foundFusedConversion,
                "fused retained-split conversion timing");
        require(foundNormalizationDiagnostic,
                "fused retained-split normalization diagnostic timing");

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
            requireAllocationFreePrunedExecution(workload, prunedModes, 1);
            requireAllocationFreePrunedExecution(workload, prunedModes, 2);
            requireAllocationFreePrunedSplitExecution(workload, prunedModes, 2);
            requireAllocationFreeStreamingPrunedSplitExecution(
                workload, prunedModes, 2);
            requireAllocationFreeStreamingPrunedSplitExecution(
                workload, prunedModes, 2, 8);
            requireAllocationFreeRetainedOuterExecution(workload, prunedModes, 2);
            requireAllocationFreeRetainedOuterExecution(
                workload, prunedModes, 2, skbench::FFTWSpectrumOrder::planeMajor);
            requireAllocationFreeFusedRetainedSplitExecution(
                workload, prunedModes, 2);
            requireAllocationFreeSplitRetainedOuterExecution(
                workload, prunedModes, 2, skbench::FFTWSpectrumOrder::planeMajor);
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
        if (inPlace.supported()) {
            const auto input = skbench::makeFixture(
                workload, skbench::FixtureKind::random, 7331);
            std::vector<skbench::Complex> full(workload.spectrumElements());
            std::vector<skbench::Complex> expectedRetained(
                modes.size() * workload.planes());
            std::vector<skbench::Complex> actualRetained(expectedRetained.size());
            std::vector<skbench::Complex> projected(workload.spectrumElements());
            std::vector<double> retainedReal(expectedRetained.size());
            std::vector<double> retainedImag(expectedRetained.size());
            std::vector<double> expectedOutput(workload.realElements());
            std::vector<double> actualOutput(workload.realElements());

            inPlace.forwardAdapter(input.data(), full.data());
            skbench::gatherRetained(
                workload, modes, full.data(), expectedRetained.data());
            inPlace.forwardRetainedNativeSplit(
                input.data(), modes, retainedReal.data(), retainedImag.data());
            for (std::size_t plane = 0; plane < workload.planes(); ++plane) {
                const auto z = plane % workload.nz;
                const auto field = plane / workload.nz;
                for (std::size_t mode = 0; mode < modes.size(); ++mode) {
                    const auto native = mode + modes.size() * plane;
                    actualRetained[skbench::retainedSpectrumIndex(
                        workload, mode, z, field)] = {
                            retainedReal[native], retainedImag[native]};
                }
            }
            require(skbench::maximumRelativeError(
                        actualRetained.data(), expectedRetained.data(),
                        expectedRetained.size()) < 1.0e-12,
                    "vDSP native split retained forward equivalence");

            skbench::embedRetained(
                workload, modes, expectedRetained.data(), projected.data());
            inPlace.inverseAdapter(projected.data(), expectedOutput.data());
            inPlace.inverseRetainedNativeSplit(
                modes, retainedReal.data(), retainedImag.data(), actualOutput.data());
            require(skbench::maximumRelativeError(
                        actualOutput.data(), expectedOutput.data(), actualOutput.size()) <
                        1.0e-12,
                    "vDSP native split retained inverse equivalence");
        }
        if (skbench::test::allocationTrackingSupported()) {
            requireAllocationFreeVdspRetainedExecution(workload, modes, 2);
        }
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
        std::vector<skbench::Complex> fullModal(
            workload.halfRows() * workload.fields * workload.retainedVerticalModes());
        std::vector<skbench::Complex> modalLayoutRoundTrip(modal.size());
        skbench::embedRetainedModal(workload, modes, modal.data(), fullModal.data());
        skbench::gatherRetainedModal(
            workload, modes, fullModal.data(), modalLayoutRoundTrip.data());
        require(skbench::maximumRelativeError(
                    modalLayoutRoundTrip.data(), modal.data(), modal.size()) == 0.0,
                "frequency-major modal gather/embed round trip");
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

        skbench::VerticalGemmProvider packedGroupedProvider(
            workload, groupedVertical, skbench::VerticalGemmLayout::complexInterleaved,
            {skbench::VerticalGemmSchedule::serial, 1});
        std::vector<skbench::Complex> groupedFullInput(workload.spectrumElements());
        std::vector<skbench::Complex> groupedFullExpected(workload.spectrumElements());
        skbench::embedRetained(workload, modes, retained.data(), groupedFullInput.data());
        packedGroupedProvider.packPhysicalInputFromWvm(modes, groupedFullInput.data());
        packedGroupedProvider.loadModalInput(modal.data());
        packedGroupedProvider.executeForward();
        packedGroupedProvider.executeInverse();
        std::vector<skbench::Complex> groupedForwardExpected(modal.size());
        std::vector<skbench::Complex> groupedInverseExpected(retained.size());
        packedGroupedProvider.copyForwardOutput(groupedForwardExpected.data());
        packedGroupedProvider.copyInverseOutput(groupedInverseExpected.data());
        packedGroupedProvider.embedPhysicalOutputToWvm(modes, groupedFullExpected.data());
        for (const auto strategy : {
                 skbench::VerticalGemmStrategy{skbench::VerticalGemmSchedule::serial, 1},
                 skbench::VerticalGemmStrategy{skbench::VerticalGemmSchedule::outerStatic, 2},
                 skbench::VerticalGemmStrategy{skbench::VerticalGemmSchedule::outerDynamic, 2}}) {
            skbench::WvmDirectVerticalGemmProvider directProvider(
                workload, modes, groupedVertical, strategy);
            require(directProvider.supported(), "direct WVM-order provider support");
            require(directProvider.gemmCallsPerExecution() == modes.size(),
                    "direct WVM-order per-mode GEMM call count");
            require(directProvider.matrixBytesPerDirection() > 0,
                    "direct WVM-order matrix memory accounting");
            std::vector<skbench::Complex> directModalInput(directProvider.modalSpectrumElements());
            std::vector<skbench::Complex> directModalOutput(directProvider.modalSpectrumElements());
            std::vector<skbench::Complex> directFullOutput(workload.spectrumElements());
            std::vector<skbench::Complex> directForward(modal.size());
            std::vector<skbench::Complex> directInverse(retained.size());
            skbench::embedRetainedModal(
                workload, modes, modal.data(), directModalInput.data());
            directProvider.initializeModalOutput(directModalOutput.data());
            directProvider.initializeSpectrumOutput(directFullOutput.data());
            directProvider.executeForward(groupedFullInput.data(), directModalOutput.data());
            directProvider.executeInverse(directModalInput.data(), directFullOutput.data());
            skbench::gatherRetainedModal(
                workload, modes, directModalOutput.data(), directForward.data());
            skbench::gatherRetained(
                workload, modes, directFullOutput.data(), directInverse.data());
            require(skbench::maximumRelativeError(
                        directForward.data(), groupedForwardExpected.data(), directForward.size()) < 1.0e-12,
                    "direct WVM-order forward equivalence");
            require(skbench::maximumRelativeError(
                        directInverse.data(), groupedInverseExpected.data(), directInverse.size()) < 1.0e-12,
                    "direct WVM-order inverse equivalence");
            require(skbench::maximumRelativeError(
                        directFullOutput.data(), groupedFullExpected.data(), directFullOutput.size()) < 1.0e-12,
                    "direct WVM-order zero-padding and Hermitian-boundary equivalence");
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
        require(orderingReport.providers.size() == 3, "ordering/packing candidate count");
        require(orderingReport.orderingPackingEstimatedExplicitPeakBytes >
                    orderingReport.fullSpectrumBytes + orderingReport.retainedSpectrumBytes,
                "ordering/packing explicit peak-memory estimate");
        for (const auto& provider : orderingReport.providers) {
            require(provider.correctness.size() == 6, "ordering/packing correctness count");
            const bool direct = provider.id == "ordering-no-reorder-accelerate-zgemm";
            bool foundPack = false;
            bool foundEmbed = false;
            bool foundCombinedForward = false;
            bool foundCombinedInverse = false;
            std::size_t reuseSeries = 0;
            for (const auto& timing : provider.timings) {
                foundPack = foundPack ||
                    (timing.scope == "adapter-component" && timing.direction == "forward" &&
                     timing.stage == "WVM retained gather and radial pack" &&
                     timing.state == (direct ? skbench::StageState::elided
                                             : skbench::StageState::executed) &&
                     (direct ? timing.seconds.empty() : !timing.seconds.empty()));
                foundEmbed = foundEmbed ||
                    (timing.scope == "adapter-component" && timing.direction == "inverse" &&
                     timing.stage == "WVM scatter and Hermitian embed" &&
                     timing.state == (direct ? skbench::StageState::elided
                                             : skbench::StageState::executed) &&
                     (direct ? timing.seconds.empty() : !timing.seconds.empty()));
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
            require(reuseSeries == (direct ? 6 : 12), "ordering/packing reuse timing matrix");
            require(provider.gemmCallsPerExecution ==
                        (direct ? modes.size()
                                : groupedVertical.groups.size() *
                                    (provider.id == "ordering-pack-accelerate-split-dgemm" ? 2 : 1)),
                    "ordering/packing GEMM call count");
            for (const auto& correctness : provider.correctness) {
                require(correctness.passed, "ordering/packing correctness");
            }
        }

        for (const std::string boundaryPolicy : {
                 "wvm-direct", "wvm-packed-split",
                 "pruned-compact-interleaved", "plane-major-fused-split",
                 "plane-major-view"}) {
            skbench::RunOptions boundaryOptions;
            boundaryOptions.kernel = "spectral-boundary";
            boundaryOptions.boundaryPolicy = boundaryPolicy;
            boundaryOptions.profile = "smoke";
            boundaryOptions.verticalGemmFamily = "k2-grouped";
            boundaryOptions.verticalGemmSchedule = "serial";
            boundaryOptions.verticalGemmOuterWorkers = 1;
            boundaryOptions.fftwPlanning = "estimate";
            boundaryOptions.fftwInternalWorkers = 1;
            boundaryOptions.fftwOuterWorkers = 2;
            boundaryOptions.warmups = 1;
            boundaryOptions.samples = 2;
            const auto boundaryReport = skbench::runBenchmark(boundaryOptions);
            require(boundaryReport.status == "passed",
                    "spectral-boundary smoke benchmark failed");
            require(boundaryReport.providers.size() == 1,
                    "spectral-boundary isolated provider count");
            const auto& provider = boundaryReport.providers.front();
            require(provider.correctness.size() == 4,
                    "spectral-boundary correctness metric count");
            require(std::all_of(
                        provider.correctness.begin(), provider.correctness.end(),
                        [](const skbench::CorrectnessMetric& item) {
                            return item.passed;
                        }),
                    "spectral-boundary correctness");
            bool foundForwardTotal = false;
            bool foundInverseTotal = false;
            bool foundForwardVertical = false;
            bool foundInverseVertical = false;
            for (const auto& timing : provider.timings) {
                foundForwardTotal = foundForwardTotal ||
                    (timing.scope == "uninstrumented-total" &&
                     timing.direction == "forward" && timing.seconds.size() == 2);
                foundInverseTotal = foundInverseTotal ||
                    (timing.scope == "uninstrumented-total" &&
                     timing.direction == "inverse" && timing.seconds.size() == 2);
                foundForwardVertical = foundForwardVertical ||
                    (timing.scope == "primitive" &&
                     timing.stage == "raw vertical MM" &&
                     timing.direction == "forward" && timing.seconds.size() == 2);
                foundInverseVertical = foundInverseVertical ||
                    (timing.scope == "primitive" &&
                     timing.stage == "raw vertical MM" &&
                     timing.direction == "inverse" && timing.seconds.size() == 2);
            }
            require(foundForwardTotal && foundInverseTotal,
                    "spectral-boundary uninstrumented totals");
            require(foundForwardVertical && foundInverseVertical,
                    "spectral-boundary primitive vertical timings");
            require(boundaryReport.orderingPackingEstimatedExplicitPeakBytes >
                        boundaryReport.fullSpectrumBytes,
                    "spectral-boundary explicit peak estimate");
        }

        {
            const auto modalElements =
                workload.retainedVerticalModes() * modes.size() * workload.fields;
            const auto weights = skbench::syntheticModalWorkWeights(workload, modes);
            std::vector<skbench::Complex> modalInput(modalElements);
            std::vector<skbench::Complex> interleavedOutput(modalElements);
            std::vector<double> inputReal(modalElements);
            std::vector<double> inputImaginary(modalElements);
            std::vector<double> outputReal(modalElements);
            std::vector<double> outputImaginary(modalElements);
            std::vector<skbench::Complex> splitOutput(modalElements);
            for (std::size_t index = 0; index < modalElements; ++index) {
                modalInput[index] = {
                    static_cast<double>(index % 31) / 31.0,
                    -static_cast<double>(index % 29) / 29.0};
            }
            skbench::interleavedToSplit(
                modalElements, modalInput.data(), inputReal.data(), inputImaginary.data());
            skbench::applySyntheticModalWorkInterleaved(
                modalElements, weights.data(), modalInput.data(), interleavedOutput.data());
            skbench::applySyntheticModalWorkSplit(
                modalElements, weights.data(), inputReal.data(), inputImaginary.data(),
                outputReal.data(), outputImaginary.data());
            skbench::splitToInterleaved(
                modalElements, outputReal.data(), outputImaginary.data(), splitOutput.data());
            require(skbench::relativeL2Error(
                        splitOutput.data(), interleavedOutput.data(), modalElements) == 0.0,
                    "synthetic modal work split/interleaved equivalence");

            const auto fullModalElements = workload.halfRows() *
                workload.retainedVerticalModes() * workload.fields;
            std::vector<skbench::Complex> fullModalInput(fullModalElements);
            std::vector<skbench::Complex> fullModalOutput(fullModalElements);
            std::vector<skbench::Complex> gatheredModalOutput(modalElements);
            skbench::embedRetainedModal(
                workload, modes, modalInput.data(), fullModalInput.data());
            skbench::applySyntheticModalWorkWvm(
                workload, modes, weights.data(), fullModalInput.data(),
                fullModalOutput.data());
            skbench::gatherRetainedModal(
                workload, modes, fullModalOutput.data(), gatheredModalOutput.data());
            require(skbench::relativeL2Error(
                        gatheredModalOutput.data(), interleavedOutput.data(), modalElements) == 0.0,
                    "synthetic modal work WVM/compact equivalence");
        }

        for (const std::string pipelinePolicy : {
                 "wvm-direct", "plane-major-fused-split",
                 "streaming-pruned-compact-split"}) {
            skbench::RunOptions pipelineOptions;
            pipelineOptions.kernel = "spectral-pipeline";
            pipelineOptions.boundaryPolicy = pipelinePolicy;
            pipelineOptions.profile = "smoke";
            pipelineOptions.verticalGemmFamily = "k2-grouped";
            pipelineOptions.verticalGemmSchedule = "serial";
            pipelineOptions.verticalGemmOuterWorkers = 1;
            pipelineOptions.fftwPlanning = "estimate";
            pipelineOptions.fftwInternalWorkers = 1;
            pipelineOptions.fftwOuterWorkers = 2;
            pipelineOptions.warmups = 1;
            pipelineOptions.samples = 2;
            const auto pipelineReport = skbench::runBenchmark(pipelineOptions);
            require(pipelineReport.status == "passed",
                    "spectral-pipeline smoke benchmark failed");
            require(pipelineReport.providers.size() == 1,
                    "spectral-pipeline isolated provider count");
            const auto& provider = pipelineReport.providers.front();
            require(provider.correctness.size() == 5,
                    "spectral-pipeline correctness metric count");
            require(std::all_of(
                        provider.correctness.begin(), provider.correctness.end(),
                        [](const skbench::CorrectnessMetric& item) {
                            return item.passed;
                        }),
                    "spectral-pipeline correctness");
            bool foundRoundTrip = false;
            bool foundModalWork = false;
            bool foundRetainedForward = false;
            bool foundRetainedInverse = false;
            bool foundPrunedRows = false;
            bool foundPrunedColumns = false;
            bool modalLedgerExecuted = false;
            for (const auto& timing : provider.timings) {
                foundRoundTrip = foundRoundTrip ||
                    (timing.scope == "uninstrumented-total" &&
                     timing.stage == "synthetic antialiased spectral pipeline" &&
                     timing.direction == "round-trip" && timing.seconds.size() == 2);
                foundModalWork = foundModalWork ||
                    (timing.scope == "component" &&
                     timing.stage == "mode-keyed modal work" &&
                     timing.direction == "modal" && timing.seconds.size() == 2);
                foundRetainedForward = foundRetainedForward ||
                    (timing.scope == "retained-operator-total" &&
                     timing.direction == "forward" && timing.seconds.size() == 2);
                foundRetainedInverse = foundRetainedInverse ||
                    (timing.scope == "retained-operator-total" &&
                     timing.direction == "inverse" && timing.seconds.size() == 2);
                foundPrunedRows = foundPrunedRows ||
                    (timing.scope == "primitive-component" &&
                     timing.stage == "real row FFTs");
                foundPrunedColumns = foundPrunedColumns ||
                    (timing.scope == "primitive-component" &&
                     timing.stage == "selected-kx complex column FFTs");
            }
            for (const auto& entry : provider.ledger) {
                modalLedgerExecuted = modalLedgerExecuted ||
                    (entry.stage == "modal work" &&
                     entry.state == skbench::StageState::executed);
            }
            require(foundRoundTrip && foundModalWork,
                    "spectral-pipeline total and modal timing series");
            require(modalLedgerExecuted,
                    "spectral-pipeline modal ledger state");
            require(pipelineReport.spectralPipelineEstimatedExplicitPeakBytes >
                        pipelineReport.fullSpectrumBytes,
                    "spectral-pipeline explicit peak estimate");
            require(provider.algorithmResidentBytes > provider.explicitPersistentBytes,
                    "spectral-pipeline algorithm-resident memory");
            require(provider.benchmarkHarnessBytes > 0,
                    "spectral-pipeline benchmark-only memory");
            require(provider.estimatedProcessPeakBytes ==
                        pipelineReport.spectralPipelineEstimatedExplicitPeakBytes,
                    "spectral-pipeline provider peak estimate");
            require(provider.algorithmResidentBytes + provider.benchmarkHarnessBytes ==
                        provider.estimatedProcessPeakBytes,
                    "spectral-pipeline memory partition");
            require(provider.observedProcessHighWaterBytes > 0,
                    "spectral-pipeline observed process high-water memory");
            if (pipelinePolicy == "plane-major-fused-split") {
                require(foundRetainedForward && foundRetainedInverse,
                        "full fused-split retained horizontal totals");
            }
            if (pipelinePolicy == "streaming-pruned-compact-split") {
                require(provider.id ==
                            "pipeline-streaming-pruned-compact-split",
                        "streaming pipeline provider identity");
                require(foundRetainedForward && foundRetainedInverse &&
                            foundPrunedRows && foundPrunedColumns,
                        "streaming pipeline horizontal component ledger");
                require(provider.scratchBytes ==
                            2 * workload.halfRows() * sizeof(skbench::Complex),
                        "streaming pipeline worker-local scratch");
                require(provider.scratchBytes < pipelineReport.fullSpectrumBytes,
                        "streaming pipeline avoids batch-sized spectrum scratch");
            }
        }

        skbench::RunOptions tiledPipelineOptions;
        tiledPipelineOptions.kernel = "spectral-pipeline";
        tiledPipelineOptions.boundaryPolicy =
            "streaming-pruned-compact-split";
        tiledPipelineOptions.streamingTileWidth = 8;
        tiledPipelineOptions.profile = "smoke";
        tiledPipelineOptions.verticalGemmFamily = "k2-grouped";
        tiledPipelineOptions.verticalGemmSchedule = "serial";
        tiledPipelineOptions.verticalGemmOuterWorkers = 1;
        tiledPipelineOptions.fftwPlanning = "estimate";
        tiledPipelineOptions.fftwInternalWorkers = 1;
        tiledPipelineOptions.fftwOuterWorkers = 2;
        tiledPipelineOptions.warmups = 1;
        tiledPipelineOptions.samples = 2;
        const auto tiledPipelineReport =
            skbench::runBenchmark(tiledPipelineOptions);
        require(tiledPipelineReport.status == "passed",
                "tiled streaming pipeline smoke benchmark failed");
        const auto& tiledPipelineProvider =
            tiledPipelineReport.providers.front();
        require(tiledPipelineProvider.scratchBytes ==
                    2 * workload.halfRows() * sizeof(skbench::Complex) +
                    2 * 8 * modes.size() * sizeof(skbench::Complex),
                "tiled streaming pipeline scratch accounting");
        require(std::any_of(
                    tiledPipelineProvider.timings.begin(),
                    tiledPipelineProvider.timings.end(),
                    [](const skbench::TimingSeries& timing) {
                        return timing.scope == "adapter-component" &&
                            timing.stage ==
                                "plane-major compact staging and blocked split transpose" &&
                            timing.direction == "forward";
                    }),
                "tiled streaming forward adapter timing");
        require(std::any_of(
                    tiledPipelineProvider.timings.begin(),
                    tiledPipelineProvider.timings.end(),
                    [](const skbench::TimingSeries& timing) {
                        return timing.scope == "adapter-component" &&
                            timing.stage ==
                                "blocked split load, compact staging, embed, and zero fill" &&
                            timing.direction == "inverse";
                    }),
                "tiled streaming inverse adapter timing");

        if (skbench::test::allocationTrackingSupported() &&
            accelerateAllocationAssertionsEnabled()) {
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
            for (const auto schedule : {skbench::VerticalGemmSchedule::outerStatic,
                                        skbench::VerticalGemmSchedule::outerDynamic}) {
                requireAllocationFreeDirectOrdering(
                    workload, modes, groupedVertical, {schedule, 2});
                requireAllocationFreePlaneMajorOrdering(
                    workload, modes, groupedVertical, {schedule, 2});
            }
            requireAllocationFreeWvmPipeline(
                workload, modes, groupedVertical,
                {skbench::VerticalGemmSchedule::outerDynamic, 2});
            requireAllocationFreeFusedSplitPipeline(
                workload, modes, groupedVertical,
                {skbench::VerticalGemmSchedule::outerDynamic, 2});
#if SKBENCH_TEST_HAVE_FFTWPP
            require(
                skbench::probeDealiasedConvolutionSteadyStateAllocationsForTesting(
                    8, 4, skbench::test::beginAllocationTracking,
                    skbench::test::endAllocationTracking) == 0,
                "FFTW++ four-product steady-state execution allocated memory");
            require(
                skbench::probeDealiasedConvolutionSteadyStateAllocationsForTesting(
                    8, 12, skbench::test::beginAllocationTracking,
                    skbench::test::endAllocationTracking) == 0,
                "FFTW++ twelve-product steady-state execution allocated memory");
            require(
                skbench::probeWvmAdvectiveConvolutionSteadyStateAllocationsForTesting(
                    8, skbench::test::beginAllocationTracking,
                    skbench::test::endAllocationTracking) == 0,
                "WVM-like advective convolution steady-state execution allocated memory");
#endif
        }

        std::cout << "skbench unit tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "skbench_tests: " << error.what() << '\n';
        return 1;
    }
}
