#include "skbench/skbench.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace skbench {
namespace {

std::string escapeJson(std::string_view value) {
    std::ostringstream output;
    for (const char rawCharacter : value) {
        const auto character = static_cast<unsigned char>(rawCharacter);
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(character) << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

void quote(std::ostream& stream, std::string_view value) {
    stream << '"' << escapeJson(value) << '"';
}

void number(std::ostream& stream, double value) {
    if (std::isfinite(value)) stream << std::setprecision(17) << value;
    else stream << "null";
}

void samples(std::ostream& stream, const std::vector<double>& values) {
    stream << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) stream << ',';
        number(stream, values[index]);
    }
    stream << ']';
}

void executionDirection(std::ostream& stream, const DirectionExecutionContract& contract) {
    stream << "{\"nativePlacement\":"; quote(stream, contract.nativePlacement);
    stream << ",\"adapterPlacement\":"; quote(stream, contract.adapterPlacement);
    stream << ",\"destroysNativeInput\":" << (contract.destroysNativeInput ? "true" : "false");
    stream << ",\"adapterPreservesCallerInput\":" << (contract.adapterPreservesCallerInput ? "true" : "false");
    stream << ",\"requiresPreservationCopyForRepeatedExecution\":" <<
        (contract.requiresPreservationCopyForRepeatedExecution ? "true" : "false");
    stream << ",\"preservationIncludedInPrimitiveTiming\":" <<
        (contract.preservationIncludedInPrimitiveTiming ? "true" : "false");
    stream << ",\"preservationIncludedInAdapterTiming\":" <<
        (contract.preservationIncludedInAdapterTiming ? "true" : "false");
    stream << ",\"nativeInputRepresentationId\":"; quote(stream, contract.nativeInputRepresentationId);
    stream << ",\"nativeOutputRepresentationId\":"; quote(stream, contract.nativeOutputRepresentationId);
    stream << ",\"adapterInputRepresentationId\":"; quote(stream, contract.adapterInputRepresentationId);
    stream << ",\"adapterOutputRepresentationId\":"; quote(stream, contract.adapterOutputRepresentationId);
    stream << ",\"physicalExtents\":"; quote(stream, contract.physicalExtents);
    stream << ",\"stridesElements\":"; quote(stream, contract.stridesElements);
    stream << ",\"paddingElements\":" << contract.paddingElements;
    stream << ",\"minimumAlignmentBytes\":" << contract.minimumAlignmentBytes;
    stream << ",\"aliasing\":"; quote(stream, contract.aliasing);
    stream << ",\"reusableWorkBytes\":" << contract.reusableWorkBytes;
    stream << ",\"outputCanFeedOppositeDirection\":" << (contract.outputCanFeedOppositeDirection ? "true" : "false") << '}';
}

std::vector<std::string> splitCsv(std::string_view line) {
    std::vector<std::string> fields;
    std::string current;
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char character = line[index];
        if (character == '"') {
            if (quoted && index + 1 < line.size() && line[index + 1] == '"') {
                current.push_back('"');
                ++index;
            } else {
                quoted = !quoted;
            }
        } else if (character == ',' && !quoted) {
            fields.push_back(current);
            current.clear();
        } else {
            current.push_back(character);
        }
    }
    fields.push_back(current);
    return fields;
}

void csvField(std::ostream& stream, std::string_view value) {
    const bool requiresQuotes = value.find_first_of(",\"\n\r") != std::string_view::npos;
    if (!requiresQuotes) {
        stream << value;
        return;
    }
    stream << '"';
    for (const char character : value) {
        if (character == '"') stream << '"';
        stream << character;
    }
    stream << '"';
}

} // namespace

void writeJson(const BenchmarkReport& report, const std::filesystem::path& path) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("Unable to open JSON result: " + path.string());

    const auto& workload = report.workload;
    const auto verticalFraction = workload.nz > 1
        ? static_cast<double>(workload.retainedVerticalModes()) / static_cast<double>(workload.nz - 1)
        : 0.0;
    stream << "{\n";
    stream << "  \"schema\":"; quote(stream, report.schema); stream << ",\n";
    stream << "  \"status\":"; quote(stream, report.status); stream << ",\n";
    stream << "  \"numericType\":{\"id\":"; quote(stream, report.scalarTypeId);
    stream << ",\"scalarBits\":" << report.scalarBits << "},\n";
    stream << "  \"run\":{\"id\":"; quote(stream, report.runId);
    stream << ",\"profile\":"; quote(stream, report.profile);
    stream << ",\"seed\":" << report.seed << ",\"warmups\":" << report.warmups << ",\"samples\":" << report.samples << "},\n";
    stream << "  \"provenance\":{\"benchmarkRepository\":\"JeffreyEarly/spectral-kernel-benchmarks\",\"waveVortexModelRepository\":\"JeffreyEarly/wave-vortex-model\",\"waveVortexModelIssue\":129,\"auditedProductionBaselineCommit\":\"be0f78995c49a2bfe4c43d75827856e3812ac278\",\"historicalHarnessCommit\":\"b63fef8f6a9e400c6ec560205906578b763d8298\",\"historicalDecisionCommit\":\"34797085d1b642b9d9c822f30a6b5c18e139c2bf\"},\n";
    stream << "  \"logicalOperators\":{\"horizontalForward\":\"T_h^+=P_h F_xy\",\"horizontalInverse\":\"T_h^-=F_xy^-1 E_h\",\"verticalForward\":\"T_z^+=V_f\",\"verticalInverse\":\"T_z^-=V_i\",\"verticalFixtureId\":";
    quote(stream, report.verticalMatrixFamilyId); stream << ",\"resultKey\":\"(k,l,j,field)\"},\n";
    stream << "  \"workload\":{\n";
    stream << "    \"Nx\":" << workload.nx << ",\"Ny\":" << workload.ny << ",\"Nz\":" << workload.nz << ",\"fields\":" << workload.fields << ",\n";
    stream << "    \"H\":" << workload.halfRows() << ",\"Nkl\":" << report.retainedHorizontalModeCount << ",\"Nj\":" << workload.retainedVerticalModes() << ",\"planes\":" << workload.planes() << ",\n";
    stream << "    \"antialias\":{\"enabled\":" << (workload.antialias ? "true" : "false") << ",\"horizontalPolicy\":\"radial-two-thirds\",\"horizontalCutoffFraction\":0.66666666666666663,\"verticalPolicy\":\"floor(2*(Nz-1)/3)\",\"verticalRetainedFraction\":";
    number(stream, verticalFraction); stream << "},\n";
    stream << "    \"grouping\":{\"realPlaneOrder\":\"x-fastest,y,z,field\",\"fullSpectrumOrder\":\"z-fastest,field,kx,ky\",\"retainedOrder\":\"z-fastest,field,radial-mode\",\"verticalColumnOrder\":\"field-fastest-within-radial-mode\",\"verticalMatrixFamilyId\":";
    quote(stream, report.verticalMatrixFamilyId);
    stream << ",\"verticalGroupCount\":" << report.verticalGroupCount;
    stream << ",\"verticalGroupModes\":{\"minimum\":" << report.minimumVerticalGroupModes << ",\"median\":";
    number(stream, report.medianVerticalGroupModes); stream << ",\"maximum\":" << report.maximumVerticalGroupModes << "}";
    stream << ",\"verticalGroupColumns\":{\"minimum\":" << report.minimumVerticalGroupColumns << ",\"median\":";
    number(stream, report.medianVerticalGroupColumns); stream << ",\"maximum\":" << report.maximumVerticalGroupColumns << "}";
    stream << ",\"verticalGroupOrderHash\":"; quote(stream, report.verticalGroupOrderHash); stream << "},\n";
    stream << "    \"stridesElements\":{\"real\":{\"x\":1,\"y\":" << workload.nx << ",\"z\":" << workload.realPlaneElements() << ",\"field\":" << workload.realPlaneElements() * workload.nz << "},";
    stream << "\"fullSpectrum\":{\"z\":1,\"field\":" << workload.nz << ",\"kx\":" << workload.planes() << ",\"ky\":" << workload.planes() * workload.nxHalf() << "},";
    stream << "\"retainedSpectrum\":{\"z\":1,\"field\":" << workload.nz << ",\"mode\":" << workload.planes() << "},";
    stream << "\"modalSpectrum\":{\"j\":1,\"field\":" << workload.retainedVerticalModes() << ",\"mode\":" << workload.retainedVerticalModes() * workload.fields << "}},\n";
    stream << "    \"bytes\":{\"real\":" << report.fullRealBytes << ",\"fullSpectrum\":" << report.fullSpectrumBytes << ",\"retainedSpectrum\":" << report.retainedSpectrumBytes << ",\"modalSpectrum\":" << report.modalSpectrumBytes << ",\"verticalMatrixFamilySource\":" << report.verticalMatrixFamilySourceBytes << ",\"verticalBenchmarkEstimatedExplicitPeak\":" << report.verticalBenchmarkEstimatedExplicitPeakBytes << ",\"orderingPackingEstimatedExplicitPeak\":" << report.orderingPackingEstimatedExplicitPeakBytes << "},\n";
    stream << "    \"permutationHashes\":{\"retainedModeOrder\":"; quote(stream, report.retainedModeOrderHash);
    stream << ",\"wvmFullSpectrumOrder\":"; quote(stream, report.wvmFullSpectrumOrderHash); stream << "}\n";
    stream << "  },\n";
    const auto& environment = report.environment;
    stream << "  \"environment\":{\"timestampUtc\":"; quote(stream, environment.timestampUtc);
    stream << ",\"hostname\":"; quote(stream, environment.hostname);
    stream << ",\"operatingSystem\":"; quote(stream, environment.operatingSystem);
    stream << ",\"machineModel\":"; quote(stream, environment.machineModel);
    stream << ",\"cpuBrand\":"; quote(stream, environment.cpuBrand);
    stream << ",\"totalCores\":" << environment.totalCores << ",\"performanceCores\":" << environment.performanceCores << ",\"efficiencyCores\":" << environment.efficiencyCores;
    stream << ",\"physicalMemoryBytes\":" << environment.physicalMemoryBytes;
    stream << ",\"compiler\":"; quote(stream, environment.compiler);
    stream << ",\"compilerVersion\":"; quote(stream, environment.compilerVersion);
    stream << ",\"compilerFlags\":"; quote(stream, environment.compilerFlags);
    stream << ",\"buildType\":"; quote(stream, environment.buildType);
    stream << ",\"gitCommit\":"; quote(stream, environment.gitCommit);
    stream << ",\"gitDirty\":" << (environment.gitDirty ? "true" : "false") << "},\n";
    stream << "  \"providers\":[\n";
    for (std::size_t providerIndex = 0; providerIndex < report.providers.size(); ++providerIndex) {
        if (providerIndex != 0) stream << ",\n";
        const auto& provider = report.providers[providerIndex];
        stream << "    {\"id\":"; quote(stream, provider.id);
        stream << ",\"version\":"; quote(stream, provider.version);
        stream << ",\"libraryIdentity\":"; quote(stream, provider.libraryIdentity);
        stream << ",\"algorithmId\":"; quote(stream, provider.algorithmId);
        stream << ",\"nativeRepresentationId\":"; quote(stream, provider.nativeRepresentationId);
        stream << ",\"modeOrderId\":"; quote(stream, provider.modeOrderId);
        stream << ",\"schedulingId\":"; quote(stream, provider.schedulingId);
        stream << ",\"providerBuild\":{\"sourceIdentity\":"; quote(stream, provider.sourceIdentity);
        stream << ",\"sourceSha256\":"; quote(stream, provider.sourceSha256);
        stream << ",\"configureFlags\":"; quote(stream, provider.configureFlags);
        stream << ",\"compilerFlags\":"; quote(stream, provider.compilerFlags); stream << '}';
        stream << ",\"workers\":" << provider.workers;
        stream << ",\"scheduling\":{\"internalWorkers\":" << provider.internalWorkers
               << ",\"outerWorkers\":" << provider.outerWorkers
               << ",\"totalLogicalWorkers\":" << provider.workers << '}';
        stream << ",\"executionContract\":{\"forward\":";
        executionDirection(stream, provider.execution.forward);
        stream << ",\"inverse\":";
        executionDirection(stream, provider.execution.inverse);
        stream << '}';
        stream << ",\"setup\":{\"totalSeconds\":";
        number(stream, provider.otherSetupSeconds + provider.allocationSeconds + provider.planningSeconds +
                       provider.wisdomGenerationSeconds + provider.wisdomImportSeconds);
        stream << ",\"otherSeconds\":"; number(stream, provider.otherSetupSeconds);
        stream << ",\"allocationSeconds\":"; number(stream, provider.allocationSeconds);
        stream << ",\"wisdomGenerationSeconds\":"; number(stream, provider.wisdomGenerationSeconds);
        stream << ",\"wisdomImportSeconds\":"; number(stream, provider.wisdomImportSeconds);
        stream << "}";
        stream << ",\"planning\":{\"seconds\":"; number(stream, provider.planningSeconds);
        stream << ",\"configuration\":"; quote(stream, provider.planningConfiguration);
        stream << ",\"temporaryBytes\":" << provider.opaquePlanningBytes;
        stream << ",\"wisdomBytes\":" << provider.wisdomBytes;
        stream << ",\"timeLimitSeconds\":"; number(stream, provider.planningTimeLimitSeconds);
        stream << ",\"budgetExhausted\":" << (provider.planningBudgetExhausted ? "true" : "false") << "}";
        stream << ",\"memory\":{\"persistentBytes\":" << provider.explicitPersistentBytes << ",\"scratchBytes\":" << provider.scratchBytes << ",\"opaqueProviderMemory\":" << (provider.opaqueProviderMemory ? "true" : "false") << "}";
        stream << ",\"componentLedger\":[";
        for (std::size_t ledgerIndex = 0; ledgerIndex < provider.ledger.size(); ++ledgerIndex) {
            if (ledgerIndex != 0) stream << ',';
            const auto& entry = provider.ledger[ledgerIndex];
            stream << "{\"stage\":"; quote(stream, entry.stage);
            stream << ",\"state\":"; quote(stream, stageStateName(entry.state));
            stream << ",\"detail\":"; quote(stream, entry.detail); stream << '}';
        }
        stream << "],\"timings\":[";
        for (std::size_t timingIndex = 0; timingIndex < provider.timings.size(); ++timingIndex) {
            if (timingIndex != 0) stream << ',';
            const auto& timing = provider.timings[timingIndex];
            stream << "{\"scope\":"; quote(stream, timing.scope);
            stream << ",\"stage\":"; quote(stream, timing.stage);
            stream << ",\"direction\":"; quote(stream, timing.direction);
            stream << ",\"state\":"; quote(stream, stageStateName(timing.state));
            stream << ",\"bytesMoved\":" << timing.bytesMoved << ",\"medianSeconds\":";
            number(stream, median(timing.seconds));
            stream << ",\"samplesSeconds\":"; samples(stream, timing.seconds); stream << '}';
        }
        stream << "],\"correctness\":[";
        for (std::size_t metricIndex = 0; metricIndex < provider.correctness.size(); ++metricIndex) {
            if (metricIndex != 0) stream << ',';
            const auto& correctness = provider.correctness[metricIndex];
            stream << "{\"name\":"; quote(stream, correctness.name);
            stream << ",\"maximumRelativeError\":"; number(stream, correctness.maximumRelativeError);
            stream << ",\"relativeL2Error\":"; number(stream, correctness.relativeL2Error);
            stream << ",\"tolerance\":"; number(stream, correctness.tolerance);
            stream << ",\"passed\":" << (correctness.passed ? "true" : "false") << '}';
        }
        stream << "]}";
    }
    stream << "\n  ]\n}\n";
}

void writeCsv(const BenchmarkReport& report, const std::filesystem::path& path) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("Unable to open CSV result: " + path.string());
    stream << "run_id,provider,scope,stage,direction,state,bytes_moved,sample_index,seconds\n";
    for (const auto& provider : report.providers) {
        for (const auto& timing : provider.timings) {
            if (timing.seconds.empty()) {
                csvField(stream, report.runId); stream << ',';
                csvField(stream, provider.id); stream << ',';
                csvField(stream, timing.scope); stream << ',';
                csvField(stream, timing.stage); stream << ',';
                csvField(stream, timing.direction); stream << ',';
                csvField(stream, stageStateName(timing.state));
                stream << ',' << timing.bytesMoved << ",,\n";
                continue;
            }
            for (std::size_t sampleIndex = 0; sampleIndex < timing.seconds.size(); ++sampleIndex) {
                csvField(stream, report.runId); stream << ',';
                csvField(stream, provider.id); stream << ',';
                csvField(stream, timing.scope); stream << ',';
                csvField(stream, timing.stage); stream << ',';
                csvField(stream, timing.direction); stream << ',';
                csvField(stream, stageStateName(timing.state));
                stream << ',' << timing.bytesMoved << ',' << sampleIndex << ',' << std::setprecision(17) << timing.seconds[sampleIndex] << '\n';
            }
        }
    }
}

int compareCsv(const std::filesystem::path& path, std::ostream& output) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Unable to open CSV result: " + path.string());
    std::string line;
    if (!std::getline(stream, line)) throw std::runtime_error("CSV result is empty.");
    using Key = std::tuple<std::string, std::string, std::string, std::string>;
    std::map<Key, std::vector<double>> groups;
    while (std::getline(stream, line)) {
        const auto fields = splitCsv(line);
        if (fields.size() != 9 || fields[8].empty()) continue;
        groups[{fields[1], fields[2], fields[3], fields[4]}].push_back(std::stod(fields[8]));
    }
    output << "provider\tscope\tstage\tdirection\tmedian_ms\tsamples\n";
    for (const auto& [key, values] : groups) {
        output << std::get<0>(key) << '\t' << std::get<1>(key) << '\t' << std::get<2>(key) << '\t'
               << std::get<3>(key) << '\t' << std::fixed << std::setprecision(6) << 1000.0 * median(values)
               << '\t' << values.size() << '\n';
    }
    return groups.empty() ? 1 : 0;
}

} // namespace skbench
