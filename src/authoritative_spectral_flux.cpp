#include "skbench/skbench.hpp"
#include "pointwise_advection.hpp"

#include <fftw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <sys/resource.h>

namespace skbench {
namespace {

using Clock = std::chrono::steady_clock;
constexpr double tolerance = 1.0e-12;

std::uint64_t byteCount(std::size_t count, std::size_t elementSize) {
    if (count != 0 && elementSize >
            std::numeric_limits<std::uint64_t>::max() / count) {
        throw std::overflow_error("byte count overflow");
    }
    return static_cast<std::uint64_t>(count) * elementSize;
}

std::uint64_t highWaterBytes() noexcept {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) return 0;
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
#endif
}

template <typename Action>
std::vector<double> sample(std::size_t warmups, std::size_t samples,
                           Action action) {
    for (std::size_t index = 0; index < warmups; ++index) action();
    std::vector<double> result;
    result.reserve(samples);
    for (std::size_t index = 0; index < samples; ++index) {
        const auto start = Clock::now();
        action();
        result.push_back(std::chrono::duration<double>(Clock::now() - start).count());
    }
    return result;
}

TimingSeries timing(std::string scope, std::string stage,
                    std::string direction, StageState state,
                    std::uint64_t bytesMoved,
                    std::vector<double> seconds = {}) {
    return {std::move(scope), std::move(stage), std::move(direction), state,
            bytesMoved, std::move(seconds)};
}

CorrectnessMetric correctness(std::string name, const Complex* actual,
                              const Complex* expected, std::size_t count) {
    const auto maximum = maximumRelativeError(actual, expected, count);
    const auto l2 = relativeL2Error(actual, expected, count);
    return {std::move(name), maximum, tolerance,
            maximum <= tolerance && l2 <= tolerance, l2};
}

CorrectnessMetric scalarCorrectness(std::string name, double error) {
    return {std::move(name), error, tolerance, error <= tolerance, error};
}

bool allCorrect(const ProviderRecord& record) {
    return std::all_of(record.correctness.begin(), record.correctness.end(),
                       [](const CorrectnessMetric& item) { return item.passed; });
}

std::string runTimestamp(std::string value) {
    value.erase(std::remove_if(value.begin(), value.end(), [](char character) {
        return character == '-' || character == ':';
    }), value.end());
    return value;
}

struct FieldFamilyMap {
    std::array<std::vector<std::size_t>, 2> originalFields;
    std::vector<std::size_t> family;
    std::vector<std::size_t> local;
};

FieldFamilyMap fieldFamilyMap(const std::vector<std::uint32_t>& families) {
    FieldFamilyMap result;
    result.family.resize(families.size());
    result.local.resize(families.size());
    for (std::size_t field = 0; field < families.size(); ++field) {
        if (families[field] >= 2) {
            throw std::invalid_argument("Spectral-flux field family is out of range.");
        }
        const auto family = static_cast<std::size_t>(families[field]);
        result.family[field] = family;
        result.local[field] = result.originalFields[family].size();
        result.originalFields[family].push_back(field);
    }
    if (result.originalFields[0].empty() || result.originalFields[1].empty()) {
        throw std::invalid_argument("Both spectral-flux operator families must be used.");
    }
    return result;
}

Workload familyWorkload(const Workload& base, std::size_t fields) {
    auto result = base;
    result.fields = fields;
    return result;
}

std::vector<Complex> packModalFamily(
    const Workload& canonicalWorkload,
    const std::vector<Complex>& canonical,
    const std::vector<std::size_t>& originalFields,
    const Workload& packedWorkload, std::size_t modeCount) {
    const auto nj = canonicalWorkload.retainedVerticalModes();
    std::vector<Complex> result(modeCount * nj * originalFields.size());
    for (std::size_t mode = 0; mode < modeCount; ++mode) {
        for (std::size_t local = 0; local < originalFields.size(); ++local) {
            for (std::size_t j = 0; j < nj; ++j) {
                result[modalSpectrumIndex(packedWorkload, mode, j, local)] =
                    canonical[modalSpectrumIndex(
                        canonicalWorkload, mode, j, originalFields[local])];
            }
        }
    }
    return result;
}

void unpackModalFamilies(
    const Workload& canonicalWorkload, std::size_t modeCount,
    const FieldFamilyMap& map,
    const std::array<Workload, 2>& packedWorkloads,
    const std::array<std::vector<Complex>, 2>& packed,
    std::vector<Complex>& canonical) {
    const auto nj = canonicalWorkload.retainedVerticalModes();
    for (std::size_t original = 0; original < map.family.size(); ++original) {
        const auto family = map.family[original];
        const auto local = map.local[original];
        for (std::size_t mode = 0; mode < modeCount; ++mode) {
            for (std::size_t j = 0; j < nj; ++j) {
                canonical[modalSpectrumIndex(canonicalWorkload, mode, j, original)] =
                    packed[family][modalSpectrumIndex(
                        packedWorkloads[family], mode, j, local)];
            }
        }
    }
}

struct SharedContext {
    Workload outputWorkload;
    Workload inputWorkload;
    Workload tripleWorkload;
    Workload targetWorkload;
    std::vector<RetainedMode> modes;
    std::size_t modeCount = 0;
    std::size_t nj = 0;
    std::size_t realVolumeElements = 0;
    double pointwiseScale = 0.0;
    FieldFamilyMap inputMap;
    FieldFamilyMap targetMap;
    std::array<Workload, 2> inputFamilyWorkloads;
    std::array<Workload, 2> targetFamilyWorkloads;
};

SharedContext sharedContext(const SpectralFluxFixture& fixture) {
    SharedContext result;
    result.outputWorkload = fixture.workload;
    result.outputWorkload.fields = 4;
    result.inputWorkload = fixture.workload;
    result.inputWorkload.fields = 15;
    result.tripleWorkload = fixture.workload;
    result.tripleWorkload.fields = 3;
    result.targetWorkload = fixture.workload;
    result.targetWorkload.fields = 1;
    result.modes = fixture.modes;
    result.modeCount = fixture.modes.size();
    result.nj = fixture.workload.retainedVerticalModes();
    result.realVolumeElements = fixture.workload.nx * fixture.workload.ny *
        fixture.workload.nz;
    result.pointwiseScale = fixture.pointwiseScale;
    result.inputMap = fieldFamilyMap(fixture.inputFieldFamilies);
    result.targetMap = fieldFamilyMap(fixture.targetFieldFamilies);
    for (std::size_t family = 0; family < 2; ++family) {
        result.inputFamilyWorkloads[family] = familyWorkload(
            fixture.workload, result.inputMap.originalFields[family].size());
        result.targetFamilyWorkloads[family] = familyWorkload(
            fixture.workload, result.targetMap.originalFields[family].size());
    }
    return result;
}

std::size_t resolvedPointwiseWorkers(
    const RunOptions& options, PointwiseAdvectionPolicy policy) {
    if (options.pointwiseWorkers != 0) return options.pointwiseWorkers;
    return policy == PointwiseAdvectionPolicy::spatialStatic
        ? options.fftwOuterWorkers
        : std::size_t{1};
}

double dcImaginaryError(const SharedContext& context,
                        const std::vector<Complex>& actual) {
    double maximum = 0.0;
    for (std::size_t mode = 0; mode < context.modeCount; ++mode) {
        if (context.modes[mode].k != 0 || context.modes[mode].l != 0) continue;
        for (std::size_t target = 0; target < 4; ++target) {
            for (std::size_t j = 0; j < context.nj; ++j) {
                maximum = std::max(maximum, std::abs(actual[modalSpectrumIndex(
                    context.outputWorkload, mode, j, target)].imag));
            }
        }
    }
    return maximum;
}

void addTargetMetrics(ProviderRecord& record, const SharedContext& context,
                      const std::vector<Complex>& actual,
                      const std::vector<Complex>& expected) {
    for (std::size_t target = 0; target < 4; ++target) {
        std::vector<Complex> actualTarget(context.modeCount * context.nj);
        std::vector<Complex> expectedTarget(actualTarget.size());
        for (std::size_t mode = 0; mode < context.modeCount; ++mode) {
            for (std::size_t j = 0; j < context.nj; ++j) {
                const auto compact = j + context.nj * mode;
                actualTarget[compact] = actual[modalSpectrumIndex(
                    context.outputWorkload, mode, j, target)];
                expectedTarget[compact] = expected[modalSpectrumIndex(
                    context.outputWorkload, mode, j, target)];
            }
        }
        record.correctness.push_back(correctness(
            "target " + std::to_string(target) +
                " modal output versus authoritative WVM oracle",
            actualTarget.data(), expectedTarget.data(), actualTarget.size()));
    }
}

void requireOuterThreadContract(const VerticalGemmStrategy& strategy) {
    if (strategy.outerWorkers <= 1) return;
    const char* value = std::getenv("VECLIB_MAXIMUM_THREADS");
    if (value == nullptr || std::string_view(value) != "1") {
        throw std::invalid_argument(
            "Authoritative production-lifetime outer vertical scheduling requires "
            "VECLIB_MAXIMUM_THREADS=1.");
    }
}

DirectionExecutionContract executionContract(const SharedContext& context) {
    DirectionExecutionContract result;
    result.nativePlacement = "out-of-place";
    result.adapterPlacement = "out-of-place";
    result.adapterPreservesCallerInput = true;
    result.nativeInputRepresentationId =
        "15 ready retained truncated modal inputs partitioned by WVM operator family";
    result.nativeOutputRepresentationId =
        "four ready retained truncated modal targets partitioned by WVM operator family";
    result.adapterInputRepresentationId = "mode-keyed authoritative WVM fixture";
    result.adapterOutputRepresentationId = "mode-keyed authoritative WVM oracle";
    result.physicalExtents =
        "15 modal inputs -> shared U,V,W + one reusable derivative triple and target -> four modal targets";
    result.stridesElements =
        "provider-native family permutation is setup-only; j remains fastest";
    result.minimumAlignmentBytes = 64;
    result.aliasing =
        "caller inputs, provider state, seven live real volumes, and outputs do not overlap";
    result.outputCanFeedOppositeDirection = false;
    (void)context;
    return result;
}

struct FamilyOperators {
    std::array<GroupedVerticalOperators, 2> values;
};

FamilyOperators familyOperators(SpectralFluxFixture& fixture) {
    return {{std::move(fixture.operatorFamilies[0]),
             std::move(fixture.operatorFamilies[1])}};
}

template <typename Values>
void releaseStorage(Values& values) {
    Values{}.swap(values);
}

void releaseOperatorSource(GroupedVerticalOperators& operators) {
    releaseStorage(operators.forward);
    releaseStorage(operators.inverse);
    releaseStorage(operators.groups);
    releaseStorage(operators.matrixSourceGroups);
}

ProviderRecord runStreaming(
    const RunOptions& options, const SharedContext& context,
    SpectralFluxFixture& fixture, FamilyOperators& operators,
    const VerticalGemmStrategy& verticalStrategy,
    std::size_t warmups, std::size_t samples, std::size_t fftwInternalWorkers,
    double fixtureSeconds, bool fusedFamilyViews) {
    const auto setupStart = Clock::now();
    const auto pointwisePolicy =
        pointwiseAdvectionPolicyNamed(options.pointwisePolicy);
    const auto pointwiseWorkers =
        resolvedPointwiseWorkers(options, pointwisePolicy);
    const auto tileWidth = options.streamingTileWidth == 1
        ? std::size_t{16} : options.streamingTileWidth;
    if (tileWidth != 16) {
        throw std::invalid_argument(
            "The frozen authoritative streaming graph requires tile width 16.");
    }

    std::array<std::vector<Complex>, 2> inputModal;
    for (std::size_t family = 0; family < 2; ++family) {
        inputModal[family] = packModalFamily(
            context.inputWorkload, fixture.modalInputs,
            context.inputMap.originalFields[family],
            context.inputFamilyWorkloads[family], context.modeCount);
    }
    std::array<std::unique_ptr<VerticalGemmProvider>, 2> reconstruction;
    std::array<std::unique_ptr<VerticalGemmProvider>, 2> projection;
    std::uint64_t operatorPreparationSourceBytes = 0;
    for (std::size_t family = 0; family < 2; ++family) {
        operatorPreparationSourceBytes += byteCount(
            operators.values[family].forward.size() +
                operators.values[family].inverse.size(), sizeof(double));
        reconstruction[family] = std::make_unique<VerticalGemmProvider>(
            context.inputFamilyWorkloads[family], operators.values[family],
            VerticalGemmLayout::split, verticalStrategy,
            VerticalGemmBufferPolicy::inverseOnly);
        projection[family] = std::make_unique<VerticalGemmProvider>(
            context.targetFamilyWorkloads[family], operators.values[family],
            VerticalGemmLayout::split, verticalStrategy,
            VerticalGemmBufferPolicy::forwardOnly);
        reconstruction[family]->loadModalInput(inputModal[family].data());
        releaseOperatorSource(operators.values[family]);
    }
    FFTWStreamingPrunedSplitProvider inverse(
        context.tripleWorkload, context.modes,
        fftwPlanningModeNamed(options.fftwPlanning), fftwInternalWorkers,
        options.fftwOuterWorkers, tileWidth);
    FFTWStreamingPrunedSplitProvider forward(
        context.targetWorkload, context.modes,
        fftwPlanningModeNamed(options.fftwPlanning), fftwInternalWorkers,
        options.fftwOuterWorkers, tileWidth);
    PointwiseAdvectionExecutor pointwiseExecutor(
        pointwisePolicy, pointwiseWorkers, context.realVolumeElements,
        context.pointwiseScale);
    const auto tripleElements = context.modeCount * context.outputWorkload.nz * 3;
    const auto targetElements = context.modeCount * context.outputWorkload.nz;
    std::vector<double> inverseReal(fusedFamilyViews ? 0 : tripleElements);
    std::vector<double> inverseImaginary(fusedFamilyViews ? 0 : tripleElements);
    std::vector<double> forwardReal(fusedFamilyViews ? 0 : targetElements);
    std::vector<double> forwardImaginary(fusedFamilyViews ? 0 : targetElements);
    std::vector<double> shared(context.tripleWorkload.realElements());
    std::vector<double> derivative(context.tripleWorkload.realElements());
    std::vector<double> target(context.targetWorkload.realElements());

    auto extractTriple = [&](std::size_t firstOriginalField) {
        for (std::size_t mode = 0; mode < context.modeCount; ++mode) {
            for (std::size_t field = 0; field < 3; ++field) {
                const auto original = firstOriginalField + field;
                const auto family = context.inputMap.family[original];
                const auto local = context.inputMap.local[original];
                const auto& provider = *reconstruction[family];
                for (std::size_t z = 0; z < context.outputWorkload.nz; ++z) {
                    const auto source = retainedSpectrumIndex(
                        context.inputFamilyWorkloads[family], mode, z, local);
                    const auto destination = retainedSpectrumIndex(
                        context.tripleWorkload, mode, z, field);
                    inverseReal[destination] =
                        provider.splitPhysicalOutputRealData()[source];
                    inverseImaginary[destination] =
                        provider.splitPhysicalOutputImaginaryData()[source];
                }
            }
        }
    };
    auto scatterTarget = [&](std::size_t originalTarget) {
        const auto family = context.targetMap.family[originalTarget];
        const auto local = context.targetMap.local[originalTarget];
        auto& provider = *projection[family];
        for (std::size_t mode = 0; mode < context.modeCount; ++mode) {
            for (std::size_t z = 0; z < context.outputWorkload.nz; ++z) {
                const auto source = retainedSpectrumIndex(
                    context.targetWorkload, mode, z, 0);
                const auto destination = retainedSpectrumIndex(
                    context.targetFamilyWorkloads[family], mode, z, local);
                provider.splitPhysicalInputRealData()[destination] =
                    forwardReal[source];
                provider.splitPhysicalInputImaginaryData()[destination] =
                    forwardImaginary[source];
            }
        }
    };
    auto inverseFieldViews = [&](std::size_t firstOriginalField) {
        std::array<FFTWStreamingPrunedSplitProvider::ConstFieldView, 3> views;
        for (std::size_t field = 0; field < views.size(); ++field) {
            const auto original = firstOriginalField + field;
            const auto family = context.inputMap.family[original];
            const auto local = context.inputMap.local[original];
            const auto& provider = *reconstruction[family];
            const auto offset = context.outputWorkload.nz * local;
            views[field] = {
                provider.splitPhysicalOutputRealData() + offset,
                provider.splitPhysicalOutputImaginaryData() + offset,
                context.inputFamilyWorkloads[family].planes()};
        }
        return views;
    };
    auto forwardFieldView = [&](std::size_t originalTarget) {
        const auto family = context.targetMap.family[originalTarget];
        const auto local = context.targetMap.local[originalTarget];
        auto& provider = *projection[family];
        const auto offset = context.outputWorkload.nz * local;
        return FFTWStreamingPrunedSplitProvider::FieldView{
            provider.splitPhysicalInputRealData() + offset,
            provider.splitPhysicalInputImaginaryData() + offset,
            context.targetFamilyWorkloads[family].planes()};
    };
    auto executeVerticalInverse = [&] {
        reconstruction[0]->executeInverse();
        reconstruction[1]->executeInverse();
    };
    auto executeVerticalForward = [&] {
        projection[0]->executeForward();
        projection[1]->executeForward();
    };
    auto executeShared = [&] {
        if (fusedFamilyViews) {
            const auto views = inverseFieldViews(0);
            inverse.inverseSplitFields(
                views.data(), views.size(), shared.data());
        } else {
            extractTriple(0);
            inverse.inverseSplit(inverseReal.data(), inverseImaginary.data(),
                                 shared.data());
        }
    };
    auto executeDerivatives = [&] {
        for (std::size_t targetIndex = 0; targetIndex < 4; ++targetIndex) {
            if (fusedFamilyViews) {
                const auto views = inverseFieldViews(3 + 3 * targetIndex);
                inverse.inverseSplitFields(
                    views.data(), views.size(), derivative.data());
            } else {
                extractTriple(3 + 3 * targetIndex);
                inverse.inverseSplit(
                    inverseReal.data(), inverseImaginary.data(),
                    derivative.data());
            }
        }
    };
    auto executeForwardTargets = [&] {
        for (std::size_t targetIndex = 0; targetIndex < 4; ++targetIndex) {
            if (fusedFamilyViews) {
                auto view = forwardFieldView(targetIndex);
                forward.forwardSplitFields(target.data(), &view, 1);
            } else {
                forward.forwardSplit(target.data(), forwardReal.data(),
                                     forwardImaginary.data());
            }
        }
    };
    auto executeAll = [&] {
        executeVerticalInverse();
        executeShared();
        for (std::size_t targetIndex = 0; targetIndex < 4; ++targetIndex) {
            if (fusedFamilyViews) {
                const auto views = inverseFieldViews(3 + 3 * targetIndex);
                inverse.inverseSplitFields(
                    views.data(), views.size(), derivative.data());
            } else {
                extractTriple(3 + 3 * targetIndex);
                inverse.inverseSplit(
                    inverseReal.data(), inverseImaginary.data(),
                    derivative.data());
            }
            pointwiseExecutor.execute(
                shared.data(), derivative.data(), target.data());
            if (fusedFamilyViews) {
                auto view = forwardFieldView(targetIndex);
                forward.forwardSplitFields(target.data(), &view, 1);
            } else {
                forward.forwardSplit(target.data(), forwardReal.data(),
                                     forwardImaginary.data());
                scatterTarget(targetIndex);
            }
        }
        executeVerticalForward();
    };
    auto executeExtractAll = [&] {
        extractTriple(0);
        for (std::size_t targetIndex = 0; targetIndex < 4; ++targetIndex) {
            extractTriple(3 + 3 * targetIndex);
        }
    };
    auto executeScatterAll = [&] {
        for (std::size_t targetIndex = 0; targetIndex < 4; ++targetIndex) {
            scatterTarget(targetIndex);
        }
    };
    auto executeFusedInverseViewAdapter = [&] {
        auto views = inverseFieldViews(0);
        inverse.embedInverseSplitFieldsDiagnostic(
            views.data(), views.size());
        for (std::size_t targetIndex = 0; targetIndex < 4; ++targetIndex) {
            views = inverseFieldViews(3 + 3 * targetIndex);
            inverse.embedInverseSplitFieldsDiagnostic(
                views.data(), views.size());
        }
    };
    auto executeFusedForwardViewAdapter = [&] {
        for (std::size_t targetIndex = 0; targetIndex < 4; ++targetIndex) {
            auto view = forwardFieldView(targetIndex);
            forward.writeForwardSplitFieldsDiagnostic(&view, 1);
        }
    };
    auto executeAdapterSchedulerLowerBound = [&] {
        for (std::size_t iteration = 0; iteration < 5; ++iteration) {
            inverse.executeSchedulerNoop();
        }
        for (std::size_t iteration = 0; iteration < 4; ++iteration) {
            forward.executeSchedulerNoop();
        }
    };
    const auto setupSeconds =
        std::chrono::duration<double>(Clock::now() - setupStart).count();
    executeAll();

    std::array<std::vector<Complex>, 2> actualFamily;
    std::array<std::vector<Complex>, 2> preservedFamily;
    for (std::size_t family = 0; family < 2; ++family) {
        actualFamily[family].resize(
            context.modeCount * context.nj *
            context.targetFamilyWorkloads[family].fields);
        projection[family]->copyForwardOutput(actualFamily[family].data());
        preservedFamily[family].resize(inputModal[family].size());
        splitToInterleaved(
            inputModal[family].size(),
            reconstruction[family]->splitModalInputRealData(),
            reconstruction[family]->splitModalInputImaginaryData(),
            preservedFamily[family].data());
    }
    std::vector<Complex> actual(fixture.expectedModalTargets.size());
    std::vector<Complex> preserved(fixture.modalInputs.size());
    unpackModalFamilies(context.outputWorkload, context.modeCount,
                        context.targetMap, context.targetFamilyWorkloads,
                        actualFamily, actual);
    unpackModalFamilies(context.inputWorkload, context.modeCount,
                        context.inputMap, context.inputFamilyWorkloads,
                        preservedFamily, preserved);

    ProviderRecord record;
    record.id = fusedFamilyViews
        ? "pipeline-production-lifetime-streaming-pruned-tile16-fused-vertical-views-authoritative"
        : "pipeline-production-lifetime-streaming-pruned-tile16-authoritative";
    record.version = "FFTW 3.3.11 + Apple Accelerate";
    record.libraryIdentity = "pinned FFTW 3.3.11 and Apple Accelerate";
    record.algorithmId = fusedFamilyViews
        ? "partial-column-pruned-tile16+direct-split-family-views+streamed-3-shared-3-derivative+split-wvm-fg-k2-15to4-v3"
        : "partial-column-pruned-tile16+streamed-3-shared-3-derivative+split-wvm-fg-k2-15to4-v2";
    if (pointwisePolicy != PointwiseAdvectionPolicy::serial) {
        record.id += "-pointwise-" +
            std::string(pointwiseAdvectionPolicyName(pointwisePolicy));
        record.algorithmId += "+pointwise-" +
            std::string(pointwiseAdvectionPolicyName(pointwisePolicy)) +
            "-v1";
    }
    record.nativeRepresentationId =
        fusedFamilyViews
        ? "persistent radial compact split modal arrays; pruned FFT reads/writes wave-f/wave-g vertical buffers through direct strided field views"
        : "persistent-radial-compact-split-modal-and-retained-spectra; setup-only wave-f/wave-g field partition";
    record.modeOrderId = "logical-radial-(k,l,j,target); exact WVM F/G mapping";
    record.schedulingId =
        "horizontal-outer-" + std::to_string(options.fftwOuterWorkers) +
        ";vertical-" + std::string(verticalGemmScheduleName(
            verticalStrategy.schedule)) + '-' +
        std::to_string(verticalStrategy.outerWorkers) + "-per-operator-family";
    if (pointwisePolicy != PointwiseAdvectionPolicy::serial) {
        record.schedulingId += ";pointwise-" +
            std::string(pointwiseAdvectionPolicyName(pointwisePolicy)) + '-' +
            std::to_string(pointwiseWorkers);
    }
    record.sourceIdentity =
        "https://fftw.org/pub/fftw/fftw-3.3.11.tar.gz + Apple Accelerate system framework";
    record.sourceSha256 =
        "5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1";
    record.configureFlags =
        "FFTW --host=aarch64-apple-darwin --enable-neon --enable-threads; Accelerate system framework";
    record.internalWorkers = fftwInternalWorkers;
    record.outerWorkers = options.fftwOuterWorkers;
    record.workers = fftwInternalWorkers * options.fftwOuterWorkers +
        verticalStrategy.outerWorkers +
        (pointwisePolicy == PointwiseAdvectionPolicy::serial
             ? 0 : pointwiseWorkers);
    record.planningConfiguration =
        "authoritative spectral-flux-fixture-v1; Float64; radial two-thirds; "
        "Nj=floor(2*(Nz-1)/3); exact WVM wave-f/wave-g K2 operators; "
        "field-family permutation is setup-only; " +
        std::string(fusedFamilyViews
            ? "direct strided family views fuse horizontal/vertical representation movement"
            : "materialized canonical triples bridge horizontal/vertical representations");
    if (pointwisePolicy != PointwiseAdvectionPolicy::serial) {
        record.planningConfiguration += "; pointwise=" +
            std::string(pointwiseAdvectionPolicyName(pointwisePolicy)) +
            "; pointwise-workers=" + std::to_string(pointwiseWorkers);
    }
    record.execution.forward = executionContract(context);
    record.execution.inverse = record.execution.forward;
    record.execution.inverse.nativePlacement = "unsupported";
    record.execution.inverse.adapterPlacement = "unsupported";
    record.gemmCallsPerExecution = 0;
    for (std::size_t family = 0; family < 2; ++family) {
        record.gemmCallsPerExecution +=
            reconstruction[family]->gemmCallsPerExecution() +
            projection[family]->gemmCallsPerExecution();
        record.explicitPersistentBytes +=
            reconstruction[family]->persistentBytes() +
            projection[family]->persistentBytes();
        record.allocationSeconds +=
            reconstruction[family]->allocationSeconds() +
            projection[family]->allocationSeconds();
    }
    record.explicitPersistentBytes += pointwiseExecutor.persistentBytes();
    record.scratchBytes = inverse.scratchBytes() + forward.scratchBytes() +
        static_cast<std::size_t>(byteCount(
            inverseReal.size() + inverseImaginary.size() + forwardReal.size() +
                forwardImaginary.size() + shared.size() + derivative.size() +
                target.size(), sizeof(double)));
    record.algorithmResidentBytes = record.explicitPersistentBytes + record.scratchBytes;
    record.otherSetupSeconds = setupSeconds;
    record.allocationSeconds += inverse.allocationSeconds() + forward.allocationSeconds();
    record.planningSeconds = inverse.planningSeconds() + forward.planningSeconds();
    record.opaquePlanningBytes = inverse.planningBytes() + forward.planningBytes();
    record.correctness = {
        correctness("complete composition versus authoritative WVM oracle",
                    actual.data(), fixture.expectedModalTargets.data(), actual.size()),
        correctness("caller modal input preserved across exact F/G partition",
                    preserved.data(), fixture.modalInputs.data(), preserved.size()),
        scalarCorrectness("DC output remains real", dcImaginaryError(context, actual))};
    addTargetMetrics(record, context, actual, fixture.expectedModalTargets);

    const auto verticalInverseBytes = byteCount(
        fixture.modalInputs.size() + context.modeCount * context.outputWorkload.nz * 15,
        sizeof(Complex));
    const auto verticalForwardBytes = byteCount(
        context.modeCount * context.outputWorkload.nz * 4 +
            fixture.expectedModalTargets.size(), sizeof(Complex));
    const auto movementBytes = byteCount(
        context.modeCount * context.outputWorkload.nz * (15 + 4), sizeof(Complex));
    record.timings = {
        timing("setup-shared-component", "strict authoritative fixture load, validation, and F/G preparation",
               "shared", StageState::setupOnly,
               fixture.verticalOperatorSourceBytes +
                   byteCount(fixture.modalInputs.size() +
                                 fixture.expectedModalTargets.size(), sizeof(Complex)),
               {fixtureSeconds}),
        timing("primitive", "raw inverse vertical MM (15 modal inputs; exact WVM F/G families)",
               "inverse", StageState::executed, verticalInverseBytes,
               sample(warmups, samples, executeVerticalInverse)),
        timing("component", "shared U,V,W horizontal reconstruction", "inverse",
               StageState::executed,
               byteCount(3 * context.realVolumeElements, sizeof(double)),
               sample(warmups, samples, executeShared)),
        timing("component", "per-target derivative horizontal reconstruction", "inverse",
               StageState::executed,
               byteCount(12 * context.realVolumeElements, sizeof(double)),
               sample(warmups, samples, executeDerivatives)),
        timing("component", "four streamed pointwise advection expressions", "pointwise",
               StageState::executed,
               byteCount(28 * context.realVolumeElements, sizeof(double)),
               sample(warmups, samples, [&] {
                   for (std::size_t repeat = 0; repeat < 4; ++repeat)
                       pointwiseExecutor.execute(
                           shared.data(), derivative.data(), target.data());
               })),
        timing("retained-operator-total", "four streamed horizontal forward transforms and retention",
               "forward", StageState::executed,
               byteCount(4 * (context.realVolumeElements +
                                  context.modeCount * context.outputWorkload.nz),
                         sizeof(Complex)),
               sample(warmups, samples, executeForwardTargets)),
        timing("adapter-component",
               fusedFamilyViews
                   ? "native split family-view movement"
                   : "native split triple extraction and target scatter",
               "movement",
               fusedFamilyViews ? StageState::fused : StageState::executed,
               fusedFamilyViews ? 0 : movementBytes,
               fusedFamilyViews
                   ? std::vector<double>{}
                   : sample(warmups, samples, [&] {
                         executeExtractAll();
                         executeScatterAll();
                     })),
        timing("primitive", "raw forward vertical MM (4 modal targets; exact WVM F/G families)",
               "forward", StageState::executed, verticalForwardBytes,
               sample(warmups, samples, executeVerticalForward))};

    if (pointwisePolicy == PointwiseAdvectionPolicy::spatialStatic) {
        record.timings.push_back(timing(
            "scheduler-diagnostic", "pointwise empty persistent dispatch",
            "pointwise", StageState::executed, 0,
            sample(warmups, samples,
                   [&] { pointwiseExecutor.executeSchedulerNoop(); })));
    }

    if (fusedFamilyViews) {
        record.timings.push_back(timing(
            "adapter-diagnostic",
            "fused inverse family-view load and embedding",
            "inverse", StageState::executed,
            byteCount(context.modeCount * context.outputWorkload.nz * 15,
                      sizeof(Complex)),
            sample(warmups, samples, executeFusedInverseViewAdapter)));
        record.timings.push_back(timing(
            "adapter-diagnostic",
            "fused forward retention and family-view write",
            "forward", StageState::executed,
            byteCount(context.modeCount * context.outputWorkload.nz * 4,
                      sizeof(Complex)),
            sample(warmups, samples, executeFusedForwardViewAdapter)));
        record.timings.push_back(timing(
            "scheduler-diagnostic",
            "nine direct family-view executor dispatches",
            "movement", StageState::executed, 0,
            sample(warmups, samples, executeAdapterSchedulerLowerBound)));
    } else {
        record.timings.push_back(timing(
            "adapter-diagnostic", "native split triple extraction",
            "inverse", StageState::executed,
            byteCount(context.modeCount * context.outputWorkload.nz * 15,
                      sizeof(Complex)),
            sample(warmups, samples, executeExtractAll)));
        record.timings.push_back(timing(
            "adapter-diagnostic", "native split target scatter",
            "forward", StageState::executed,
            byteCount(context.modeCount * context.outputWorkload.nz * 4,
                      sizeof(Complex)),
            sample(warmups, samples, executeScatterAll)));
    }

    const auto correctnessBytes = byteCount(
        fixture.modalInputs.size() + fixture.expectedModalTargets.size() +
            actual.size() + preserved.size(), sizeof(Complex));
    releaseStorage(fixture.modalInputs);
    releaseStorage(fixture.expectedModalTargets);
    releaseStorage(actual);
    releaseStorage(preserved);
    releaseStorage(actualFamily[0]);
    releaseStorage(actualFamily[1]);
    releaseStorage(preservedFamily[0]);
    releaseStorage(preservedFamily[1]);
    releaseStorage(inputModal[0]);
    releaseStorage(inputModal[1]);
    record.timings.push_back(timing(
        "setup-component", "release prepared vertical operator source after provider preparation",
        "shared", StageState::setupOnly, operatorPreparationSourceBytes, {0.0}));
    record.timings.push_back(timing(
        "setup-component", "release fixture and correctness-only storage", "shared",
        StageState::setupOnly, correctnessBytes, {0.0}));
    record.timings.push_back(timing(
        "uninstrumented-total",
        "authoritative production-lifetime streamed four-target spectral-flux composition",
        "forward", StageState::executed,
        verticalInverseBytes + verticalForwardBytes + movementBytes,
        sample(warmups, samples, executeAll)));
    record.ledger = {
        {"fixture provenance", StageState::setupOnly,
         "strict SHA-256-verified authoritative WVM export; no synthetic fallback"},
        {"operator-family permutation", StageState::setupOnly,
         "logical fields partitioned into exact WVM wave-f/wave-g providers before timing"},
        {"shared U,V,W reconstruction", StageState::executed,
         "three shared real volumes reconstructed once"},
        {"derivative reconstruction", StageState::executed,
         "one reusable three-field derivative buffer per target"},
        {"pointwise advection", StageState::executed,
         pointwisePolicy == PointwiseAdvectionPolicy::serial
             ? "direct -(U*qx+V*qy+W*qz) with fixture normalization"
             : "direct -(U*qx+V*qy+W*qz) with fixture normalization; policy=" +
                   std::string(pointwiseAdvectionPolicyName(pointwisePolicy)) +
                   "; workers=" + std::to_string(pointwiseWorkers)},
        {"horizontal forward and retention", StageState::fused,
         "partial-column-pruned FFTW writes fixed tile-16 compact split output"},
        {"horizontal/vertical representation movement",
         fusedFamilyViews ? StageState::fused : StageState::executed,
         fusedFamilyViews
             ? "pruned inverse reads split F/G physical outputs directly and pruned forward writes split F/G physical inputs directly"
             : "materialized canonical split triples and target spectra bridge the FFT and vertical providers"},
        {"vertical projection", StageState::executed,
         "grouped split-real dgemm uses exact WVM F/G matrices"},
        {"steady-state allocation", StageState::elided,
         "all providers and reusable buffers are persistent"},
        {"complete WVM nonlinear flux", StageState::unsupported,
         "phase, coefficient assembly/accumulation, MATLAB dispatch, state, and time integration are excluded"}};
    record.execution.forward.reusableWorkBytes = record.scratchBytes;
    record.estimatedProcessPeakBytes = record.algorithmResidentBytes;
    record.observedProcessHighWaterBytes = highWaterBytes();
    record.opaqueProviderMemory = true;
    return record;
}

ProviderRecord runWvmDirect(
    const RunOptions& options, const SharedContext& context,
    SpectralFluxFixture& fixture, FamilyOperators& operators,
    const VerticalGemmStrategy& verticalStrategy,
    std::size_t warmups, std::size_t samples, std::size_t fftwInternalWorkers,
    double fixtureSeconds) {
    const auto setupStart = Clock::now();
    const auto pointwisePolicy =
        pointwiseAdvectionPolicyNamed(options.pointwisePolicy);
    const auto pointwiseWorkers =
        resolvedPointwiseWorkers(options, pointwisePolicy);
    std::array<std::vector<Complex>, 2> compactInput;
    std::array<std::vector<Complex>, 2> fullModalInput;
    std::array<std::vector<Complex>, 2> fullPhysicalInput;
    std::array<std::vector<Complex>, 2> fullPhysicalOutput;
    std::array<std::vector<Complex>, 2> fullModalOutput;
    std::array<std::unique_ptr<WvmDirectVerticalGemmProvider>, 2> reconstruction;
    std::array<std::unique_ptr<WvmDirectVerticalGemmProvider>, 2> projection;
    std::uint64_t operatorPreparationSourceBytes = 0;
    for (std::size_t family = 0; family < 2; ++family) {
        compactInput[family] = packModalFamily(
            context.inputWorkload, fixture.modalInputs,
            context.inputMap.originalFields[family],
            context.inputFamilyWorkloads[family], context.modeCount);
        fullModalInput[family].resize(
            context.inputFamilyWorkloads[family].halfRows() * context.nj *
            context.inputFamilyWorkloads[family].fields);
        embedRetainedModal(context.inputFamilyWorkloads[family], context.modes,
                           compactInput[family].data(), fullModalInput[family].data());
        fullPhysicalInput[family].resize(
            context.inputFamilyWorkloads[family].spectrumElements());
        fullPhysicalOutput[family].resize(
            context.targetFamilyWorkloads[family].spectrumElements());
        fullModalOutput[family].resize(
            context.targetFamilyWorkloads[family].halfRows() * context.nj *
            context.targetFamilyWorkloads[family].fields);
        operatorPreparationSourceBytes += byteCount(
            operators.values[family].forward.size() +
                operators.values[family].inverse.size(), sizeof(double));
        reconstruction[family] = std::make_unique<WvmDirectVerticalGemmProvider>(
            context.inputFamilyWorkloads[family], context.modes,
            operators.values[family], verticalStrategy,
            VerticalGemmBufferPolicy::inverseOnly);
        projection[family] = std::make_unique<WvmDirectVerticalGemmProvider>(
            context.targetFamilyWorkloads[family], context.modes,
            operators.values[family], verticalStrategy,
            VerticalGemmBufferPolicy::forwardOnly);
        releaseOperatorSource(operators.values[family]);
    }
    FFTWProvider inverse(
        context.tripleWorkload, FFTWStrategy{
            fftwPlanningModeNamed(options.fftwPlanning),
            fftwAlignmentStrategyNamed(options.fftwAlignment),
            fftwWisdomStrategyNamed(options.fftwWisdom), fftwInternalWorkers,
            options.fftwOuterWorkers, options.fftwPlanningTimeLimitSeconds,
            FFTWDataLayout::interleaved,
            FFTWSpectrumOrder::wvmFrequencyMajor});
    FFTWProvider forward(
        context.targetWorkload, FFTWStrategy{
            fftwPlanningModeNamed(options.fftwPlanning),
            fftwAlignmentStrategyNamed(options.fftwAlignment),
            fftwWisdomStrategyNamed(options.fftwWisdom), fftwInternalWorkers,
            options.fftwOuterWorkers, options.fftwPlanningTimeLimitSeconds,
            FFTWDataLayout::interleaved,
            FFTWSpectrumOrder::wvmFrequencyMajor});
    PointwiseAdvectionExecutor pointwiseExecutor(
        pointwisePolicy, pointwiseWorkers, context.realVolumeElements,
        context.pointwiseScale);
    std::vector<Complex> tripleSpectrum(context.tripleWorkload.spectrumElements());
    std::vector<Complex> targetSpectrum(context.targetWorkload.spectrumElements());
    std::vector<double> shared(context.tripleWorkload.realElements());
    std::vector<double> derivative(context.tripleWorkload.realElements());
    std::vector<double> target(context.targetWorkload.realElements());

    auto extractTriple = [&](std::size_t firstOriginalField) {
        for (std::size_t ky = 0; ky < context.outputWorkload.ny; ++ky) {
            for (std::size_t kx = 0; kx < context.outputWorkload.nxHalf(); ++kx) {
                for (std::size_t field = 0; field < 3; ++field) {
                    const auto original = firstOriginalField + field;
                    const auto family = context.inputMap.family[original];
                    const auto local = context.inputMap.local[original];
                    for (std::size_t z = 0; z < context.outputWorkload.nz; ++z) {
                        tripleSpectrum[wvmSpectrumIndex(
                            context.tripleWorkload, kx, ky, z, field)] =
                            fullPhysicalInput[family][wvmSpectrumIndex(
                                context.inputFamilyWorkloads[family], kx, ky,
                                z, local)];
                    }
                }
            }
        }
    };
    auto scatterTarget = [&](std::size_t originalTarget) {
        const auto family = context.targetMap.family[originalTarget];
        const auto local = context.targetMap.local[originalTarget];
        for (std::size_t ky = 0; ky < context.outputWorkload.ny; ++ky) {
            for (std::size_t kx = 0; kx < context.outputWorkload.nxHalf(); ++kx) {
                for (std::size_t z = 0; z < context.outputWorkload.nz; ++z) {
                    fullPhysicalOutput[family][wvmSpectrumIndex(
                        context.targetFamilyWorkloads[family], kx, ky, z,
                        local)] = targetSpectrum[wvmSpectrumIndex(
                            context.targetWorkload, kx, ky, z, 0)];
                }
            }
        }
    };
    auto executeVerticalInverse = [&] {
        for (std::size_t family = 0; family < 2; ++family) {
            reconstruction[family]->initializeSpectrumOutput(
                fullPhysicalInput[family].data());
            reconstruction[family]->executeInverse(
                fullModalInput[family].data(), fullPhysicalInput[family].data());
        }
    };
    auto executeVerticalForward = [&] {
        for (std::size_t family = 0; family < 2; ++family) {
            projection[family]->initializeModalOutput(fullModalOutput[family].data());
            projection[family]->executeForward(
                fullPhysicalOutput[family].data(), fullModalOutput[family].data());
        }
    };
    auto executeShared = [&] {
        extractTriple(0);
        inverse.inverse(tripleSpectrum.data(), shared.data());
    };
    auto executeDerivatives = [&] {
        for (std::size_t targetIndex = 0; targetIndex < 4; ++targetIndex) {
            extractTriple(3 + 3 * targetIndex);
            inverse.inverse(tripleSpectrum.data(), derivative.data());
        }
    };
    auto executeForwardTargets = [&] {
        for (std::size_t repeat = 0; repeat < 4; ++repeat)
            forward.forward(target.data(), targetSpectrum.data());
    };
    auto executeAll = [&] {
        executeVerticalInverse();
        extractTriple(0);
        inverse.inverse(tripleSpectrum.data(), shared.data());
        for (std::size_t targetIndex = 0; targetIndex < 4; ++targetIndex) {
            extractTriple(3 + 3 * targetIndex);
            inverse.inverse(tripleSpectrum.data(), derivative.data());
            pointwiseExecutor.execute(
                shared.data(), derivative.data(), target.data());
            forward.forward(target.data(), targetSpectrum.data());
            scatterTarget(targetIndex);
        }
        executeVerticalForward();
    };
    const auto setupSeconds =
        std::chrono::duration<double>(Clock::now() - setupStart).count();
    executeAll();

    std::array<std::vector<Complex>, 2> actualFamily;
    std::array<std::vector<Complex>, 2> preservedFamily;
    for (std::size_t family = 0; family < 2; ++family) {
        actualFamily[family].resize(
            context.modeCount * context.nj *
            context.targetFamilyWorkloads[family].fields);
        gatherRetainedModal(context.targetFamilyWorkloads[family], context.modes,
                            fullModalOutput[family].data(), actualFamily[family].data());
        preservedFamily[family].resize(compactInput[family].size());
        gatherRetainedModal(context.inputFamilyWorkloads[family], context.modes,
                            fullModalInput[family].data(), preservedFamily[family].data());
    }
    std::vector<Complex> actual(fixture.expectedModalTargets.size());
    std::vector<Complex> preserved(fixture.modalInputs.size());
    unpackModalFamilies(context.outputWorkload, context.modeCount,
                        context.targetMap, context.targetFamilyWorkloads,
                        actualFamily, actual);
    unpackModalFamilies(context.inputWorkload, context.modeCount,
                        context.inputMap, context.inputFamilyWorkloads,
                        preservedFamily, preserved);

    ProviderRecord record;
    record.id = "pipeline-production-lifetime-wvm-direct-authoritative";
    record.version = "FFTW 3.3.11 + Apple Accelerate";
    record.libraryIdentity = "pinned FFTW 3.3.11 and Apple Accelerate";
    record.algorithmId =
        "full-wvm-order-fftw+streamed-3-shared-3-derivative+direct-zgemm-wvm-fg-15to4-v2";
    if (pointwisePolicy != PointwiseAdvectionPolicy::serial) {
        record.id += "-pointwise-" +
            std::string(pointwiseAdvectionPolicyName(pointwisePolicy));
        record.algorithmId += "+pointwise-" +
            std::string(pointwiseAdvectionPolicyName(pointwisePolicy)) +
            "-v1";
    }
    record.nativeRepresentationId =
        "persistent-wvm-frequency-major-interleaved-full-spectrum-and-modal; setup-only wave-f/wave-g field partition";
    record.modeOrderId = "logical-radial-(k,l,j,target); exact WVM F/G mapping";
    record.schedulingId =
        "horizontal-outer-" + std::to_string(options.fftwOuterWorkers) +
        ";vertical-" + std::string(verticalGemmScheduleName(
            verticalStrategy.schedule)) + '-' +
        std::to_string(verticalStrategy.outerWorkers) + "-per-operator-family";
    if (pointwisePolicy != PointwiseAdvectionPolicy::serial) {
        record.schedulingId += ";pointwise-" +
            std::string(pointwiseAdvectionPolicyName(pointwisePolicy)) + '-' +
            std::to_string(pointwiseWorkers);
    }
    record.sourceIdentity =
        "https://fftw.org/pub/fftw/fftw-3.3.11.tar.gz + Apple Accelerate system framework";
    record.sourceSha256 =
        "5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1";
    record.configureFlags =
        "FFTW --host=aarch64-apple-darwin --enable-neon --enable-threads; Accelerate system framework";
    record.internalWorkers = fftwInternalWorkers;
    record.outerWorkers = options.fftwOuterWorkers;
    record.workers = fftwInternalWorkers * options.fftwOuterWorkers +
        verticalStrategy.outerWorkers +
        (pointwisePolicy == PointwiseAdvectionPolicy::serial
             ? 0 : pointwiseWorkers);
    record.planningConfiguration =
        "authoritative spectral-flux-fixture-v1; Float64; full WVM-order spectra; "
        "exact WVM wave-f/wave-g K2 operators; field-family permutation is setup-only";
    if (pointwisePolicy != PointwiseAdvectionPolicy::serial) {
        record.planningConfiguration += "; pointwise=" +
            std::string(pointwiseAdvectionPolicyName(pointwisePolicy)) +
            "; pointwise-workers=" + std::to_string(pointwiseWorkers);
    }
    record.execution.forward = executionContract(context);
    record.execution.inverse = record.execution.forward;
    record.execution.inverse.nativePlacement = "unsupported";
    record.execution.inverse.adapterPlacement = "unsupported";
    for (std::size_t family = 0; family < 2; ++family) {
        record.gemmCallsPerExecution +=
            reconstruction[family]->gemmCallsPerExecution() +
            projection[family]->gemmCallsPerExecution();
        record.explicitPersistentBytes +=
            reconstruction[family]->persistentBytes() +
            projection[family]->persistentBytes() +
            byteCount(fullModalInput[family].size() +
                          fullPhysicalInput[family].size() +
                          fullPhysicalOutput[family].size() +
                          fullModalOutput[family].size(), sizeof(Complex));
        record.allocationSeconds +=
            reconstruction[family]->allocationSeconds() +
            projection[family]->allocationSeconds();
    }
    record.explicitPersistentBytes += pointwiseExecutor.persistentBytes();
    record.scratchBytes = static_cast<std::size_t>(
        byteCount(tripleSpectrum.size() + targetSpectrum.size(), sizeof(Complex)) +
        byteCount(shared.size() + derivative.size() + target.size(), sizeof(double)));
    record.algorithmResidentBytes = record.explicitPersistentBytes + record.scratchBytes;
    record.otherSetupSeconds = setupSeconds;
    record.allocationSeconds += inverse.allocationSeconds() + forward.allocationSeconds();
    record.planningSeconds = inverse.planningSeconds() + forward.planningSeconds();
    record.opaquePlanningBytes = inverse.planningBytes() + forward.planningBytes();
    record.correctness = {
        correctness("complete composition versus authoritative WVM oracle",
                    actual.data(), fixture.expectedModalTargets.data(), actual.size()),
        correctness("caller modal input preserved across exact F/G partition",
                    preserved.data(), fixture.modalInputs.data(), preserved.size()),
        scalarCorrectness("DC output remains real", dcImaginaryError(context, actual))};
    addTargetMetrics(record, context, actual, fixture.expectedModalTargets);

    std::size_t fullInputElements = 0;
    std::size_t fullOutputElements = 0;
    for (std::size_t family = 0; family < 2; ++family) {
        fullInputElements += fullModalInput[family].size() +
            fullPhysicalInput[family].size();
        fullOutputElements += fullPhysicalOutput[family].size() +
            fullModalOutput[family].size();
    }
    const auto verticalInverseBytes = byteCount(fullInputElements, sizeof(Complex));
    const auto verticalForwardBytes = byteCount(fullOutputElements, sizeof(Complex));
    const auto movementBytes = byteCount(
        context.outputWorkload.nz * (15 + 4) *
            context.targetWorkload.halfRows(), sizeof(Complex));
    record.timings = {
        timing("setup-shared-component", "strict authoritative fixture load, validation, and F/G preparation",
               "shared", StageState::setupOnly,
               fixture.verticalOperatorSourceBytes +
                   byteCount(fixture.modalInputs.size() +
                                 fixture.expectedModalTargets.size(), sizeof(Complex)),
               {fixtureSeconds}),
        timing("primitive", "raw inverse vertical MM (15 modal inputs; exact WVM F/G families)",
               "inverse", StageState::executed, verticalInverseBytes,
               sample(warmups, samples, executeVerticalInverse)),
        timing("component", "shared U,V,W horizontal reconstruction", "inverse",
               StageState::executed,
               byteCount(3 * context.realVolumeElements, sizeof(double)),
               sample(warmups, samples, executeShared)),
        timing("component", "per-target derivative horizontal reconstruction", "inverse",
               StageState::executed,
               byteCount(12 * context.realVolumeElements, sizeof(double)),
               sample(warmups, samples, executeDerivatives)),
        timing("component", "four streamed pointwise advection expressions", "pointwise",
               StageState::executed,
               byteCount(28 * context.realVolumeElements, sizeof(double)),
               sample(warmups, samples, [&] {
                   for (std::size_t repeat = 0; repeat < 4; ++repeat)
                       pointwiseExecutor.execute(
                           shared.data(), derivative.data(), target.data());
               })),
        timing("primitive", "four streamed full horizontal forward FFTs", "forward",
               StageState::executed,
               byteCount(4 * (context.realVolumeElements +
                                  context.targetWorkload.spectrumElements()),
                         sizeof(Complex)),
               sample(warmups, samples, executeForwardTargets)),
        timing("adapter-component", "WVM-order triple extraction and target scatter",
               "movement", StageState::executed, movementBytes,
               sample(warmups, samples, [&] {
                   extractTriple(0);
                   for (std::size_t targetIndex = 0; targetIndex < 4; ++targetIndex) {
                       extractTriple(3 + 3 * targetIndex);
                       scatterTarget(targetIndex);
                   }
               })),
        timing("primitive", "raw forward vertical MM (4 modal targets; exact WVM F/G families)",
               "forward", StageState::executed, verticalForwardBytes,
               sample(warmups, samples, executeVerticalForward))};

    if (pointwisePolicy == PointwiseAdvectionPolicy::spatialStatic) {
        record.timings.push_back(timing(
            "scheduler-diagnostic", "pointwise empty persistent dispatch",
            "pointwise", StageState::executed, 0,
            sample(warmups, samples,
                   [&] { pointwiseExecutor.executeSchedulerNoop(); })));
    }

    const auto correctnessBytes = byteCount(
        fixture.modalInputs.size() + fixture.expectedModalTargets.size() +
            actual.size() + preserved.size(), sizeof(Complex));
    releaseStorage(fixture.modalInputs);
    releaseStorage(fixture.expectedModalTargets);
    releaseStorage(actual);
    releaseStorage(preserved);
    for (std::size_t family = 0; family < 2; ++family) {
        releaseStorage(compactInput[family]);
        releaseStorage(actualFamily[family]);
        releaseStorage(preservedFamily[family]);
    }
    record.timings.push_back(timing(
        "setup-component", "release prepared vertical operator source after provider preparation",
        "shared", StageState::setupOnly, operatorPreparationSourceBytes, {0.0}));
    record.timings.push_back(timing(
        "setup-component", "release fixture and correctness-only storage", "shared",
        StageState::setupOnly, correctnessBytes, {0.0}));
    record.timings.push_back(timing(
        "uninstrumented-total",
        "authoritative production-lifetime streamed four-target spectral-flux composition",
        "forward", StageState::executed,
        verticalInverseBytes + verticalForwardBytes + movementBytes,
        sample(warmups, samples, executeAll)));
    record.ledger = {
        {"fixture provenance", StageState::setupOnly,
         "strict SHA-256-verified authoritative WVM export; no synthetic fallback"},
        {"operator-family permutation", StageState::setupOnly,
         "logical fields partitioned into exact WVM wave-f/wave-g providers before timing"},
        {"shared U,V,W reconstruction", StageState::executed,
         "three shared real volumes reconstructed once"},
        {"derivative reconstruction", StageState::executed,
         "one reusable three-field derivative buffer per target"},
        {"pointwise advection", StageState::executed,
         pointwisePolicy == PointwiseAdvectionPolicy::serial
             ? "direct -(U*qx+V*qy+W*qz) with fixture normalization"
             : "direct -(U*qx+V*qy+W*qz) with fixture normalization; policy=" +
                   std::string(pointwiseAdvectionPolicyName(pointwisePolicy)) +
                   "; workers=" + std::to_string(pointwiseWorkers)},
        {"horizontal forward and retention", StageState::executed,
         "full WVM-order FFTW output feeds direct vertical zgemm"},
        {"vertical projection", StageState::executed,
         "direct complex zgemm uses exact WVM F/G matrices"},
        {"steady-state allocation", StageState::elided,
         "all providers and reusable buffers are persistent"},
        {"complete WVM nonlinear flux", StageState::unsupported,
         "phase, coefficient assembly/accumulation, MATLAB dispatch, state, and time integration are excluded"}};
    record.execution.forward.reusableWorkBytes = record.scratchBytes;
    record.estimatedProcessPeakBytes = record.algorithmResidentBytes;
    record.observedProcessHighWaterBytes = highWaterBytes();
    record.opaqueProviderMemory = true;
    return record;
}

} // namespace

BenchmarkReport runAuthoritativeProductionLifetimeFluxBenchmark(
    const RunOptions& options) {
    if (options.spectralFluxFixture.empty()) {
        throw std::invalid_argument(
            "Authoritative production-lifetime benchmark requires a prepared fixture.");
    }
    if (options.boundaryPolicy != "wvm-direct" &&
        options.boundaryPolicy != "streaming-pruned-compact-split" &&
        options.boundaryPolicy !=
            "streaming-pruned-compact-split-fused-vertical-views") {
        throw std::invalid_argument(
            "Authoritative production-lifetime boundary policy must be wvm-direct "
            "streaming-pruned-compact-split, or "
            "streaming-pruned-compact-split-fused-vertical-views.");
    }
    if (options.verticalGemmFamily != "k2-grouped") {
        throw std::invalid_argument(
            "Authoritative production-lifetime benchmark requires k2-grouped vertical GEMM.");
    }
    if (options.workers != 0) {
        throw std::invalid_argument(
            "Authoritative production-lifetime benchmark uses independent worker controls.");
    }
    const auto pointwisePolicy =
        pointwiseAdvectionPolicyNamed(options.pointwisePolicy);
    const auto pointwiseWorkers =
        resolvedPointwiseWorkers(options, pointwisePolicy);
    if (pointwiseWorkers == 0) {
        throw std::invalid_argument(
            "Authoritative pointwise worker count must be positive.");
    }
    if (pointwisePolicy != PointwiseAdvectionPolicy::spatialStatic &&
        pointwiseWorkers != 1) {
        throw std::invalid_argument(
            "Serial pointwise policies require exactly one worker.");
    }
    const VerticalGemmStrategy verticalStrategy{
        verticalGemmScheduleNamed(options.verticalGemmSchedule),
        options.verticalGemmOuterWorkers};
    if (verticalStrategy.schedule == VerticalGemmSchedule::serial &&
        verticalStrategy.outerWorkers != 1) {
        throw std::invalid_argument("Serial vertical scheduling requires one worker.");
    }
    requireOuterThreadContract(verticalStrategy);

    const auto fixtureStart = Clock::now();
    auto fixture = loadPreparedSpectralFluxFixture(options.spectralFluxFixture);
    const auto selected = profileNamed(options.profile);
    if (selected.workload.nx != fixture.workload.nx ||
        selected.workload.ny != fixture.workload.ny ||
        selected.workload.nz != fixture.workload.nz) {
        throw std::invalid_argument(
            "Prepared spectral-flux fixture dimensions do not match the selected profile.");
    }
    const auto context = sharedContext(fixture);
    auto operators = familyOperators(fixture);
    const auto fixtureSeconds =
        std::chrono::duration<double>(Clock::now() - fixtureStart).count();
    const auto warmups = options.warmups == 0 ? std::size_t{1} : options.warmups;
    const auto samples = options.samples == 0 ? std::size_t{3} : options.samples;
    const auto fftwInternalWorkers = options.fftwInternalWorkers == 0
        ? std::size_t{1} : options.fftwInternalWorkers;

    BenchmarkReport report;
    report.environment = environmentRecord();
    const auto issueTag = pointwisePolicy != PointwiseAdvectionPolicy::serial
        ? "issue22-authoritative-"
        : options.boundaryPolicy ==
              "streaming-pruned-compact-split-fused-vertical-views"
            ? "issue21-authoritative-"
            : "issue19-authoritative-";
    report.runId = runTimestamp(report.environment.timestampUtc) + '-' +
        issueTag + options.boundaryPolicy;
    if (pointwisePolicy != PointwiseAdvectionPolicy::serial) {
        report.runId += "-pointwise-" +
            std::string(pointwiseAdvectionPolicyName(pointwisePolicy)) + '-' +
            std::to_string(pointwiseWorkers);
    }
    report.runId += '-' + report.environment.hostname;
    report.profile = options.profile;
    report.seed = 0;
    report.warmups = warmups;
    report.samples = samples;
    report.workload = context.outputWorkload;
    report.retainedHorizontalModeCount = context.modeCount;
    report.retainedModeOrderHash = modeOrderHash(context.modes);
    report.wvmFullSpectrumOrderHash = wvmSpectrumOrderHash(context.outputWorkload);
    report.fullRealBytes = byteCount(7 * context.realVolumeElements, sizeof(double));
    report.fullSpectrumBytes = byteCount(
        context.inputWorkload.spectrumElements() +
            context.outputWorkload.spectrumElements(), sizeof(Complex));
    report.retainedSpectrumBytes = byteCount(
        context.modeCount * context.outputWorkload.nz * 19, sizeof(Complex));
    report.modalSpectrumBytes = byteCount(
        fixture.modalInputs.size() + fixture.expectedModalTargets.size(),
        sizeof(Complex));
    report.verticalMatrixFamilySourceBytes = byteCount(
        fixture.verticalOperatorSourceBytes, 1);
    report.verticalMatrixFamilyId = "wvm-wave-f+wave-g-floating-k2-exact";
    const auto& groups = operators.values[0].groups;
    report.verticalGroupCount = groups.size();
    report.verticalGroupOrderHash = verticalModeGroupHash(groups);
    std::vector<double> groupModes;
    std::vector<double> groupColumns;
    for (const auto& group : groups) {
        groupModes.push_back(static_cast<double>(group.modeCount));
        groupColumns.push_back(static_cast<double>(group.modeCount * 15));
    }
    report.minimumVerticalGroupModes = static_cast<std::size_t>(
        *std::min_element(groupModes.begin(), groupModes.end()));
    report.medianVerticalGroupModes = median(groupModes);
    report.maximumVerticalGroupModes = static_cast<std::size_t>(
        *std::max_element(groupModes.begin(), groupModes.end()));
    report.minimumVerticalGroupColumns = static_cast<std::size_t>(
        *std::min_element(groupColumns.begin(), groupColumns.end()));
    report.medianVerticalGroupColumns = median(groupColumns);
    report.maximumVerticalGroupColumns = static_cast<std::size_t>(
        *std::max_element(groupColumns.begin(), groupColumns.end()));
    report.fixtureProvenance = {
        "authoritative-wvm-export", "spectral-flux-fixture-v1",
        fixture.waveVortexModelRepository, fixture.waveVortexModelCommit,
        fixture.generatorIdentity, fixture.fixtureHash, fixture.normalization,
        fixture.modeMapping, fixture.derivativeConvention, true};

    const auto streaming = options.boundaryPolicy != "wvm-direct";
    const auto fusedFamilyViews = options.boundaryPolicy ==
        "streaming-pruned-compact-split-fused-vertical-views";
    auto record = streaming
        ? runStreaming(options, context, fixture, operators, verticalStrategy,
                       warmups, samples, fftwInternalWorkers, fixtureSeconds,
                       fusedFamilyViews)
        : runWvmDirect(options, context, fixture, operators, verticalStrategy,
                       warmups, samples, fftwInternalWorkers, fixtureSeconds);
    record.compilerFlags = report.environment.compilerFlags;
    report.spectralPipelineEstimatedExplicitPeakBytes =
        record.estimatedProcessPeakBytes;
    report.providers.push_back(std::move(record));
    report.status = allCorrect(report.providers.front()) ? "passed" : "failed";
    fftw_forget_wisdom();
    return report;
}

} // namespace skbench
