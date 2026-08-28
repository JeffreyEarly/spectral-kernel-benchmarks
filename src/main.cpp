#include "skbench/skbench.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void usage(std::ostream& output) {
    output << "Usage:\n"
           << "  skbench list\n"
           << "  skbench validate [--profile NAME]\n"
           << "  skbench run [--profile NAME] [--providers both|fftw] [--workers N]\n"
           << "              [--fftw-planning estimate|measure|patient|exhaustive]\n"
           << "              [--fftw-layout interleaved|split|paired]\n"
           << "              [--fftw-alignment aligned|unaligned] [--fftw-wisdom cold|generated-import]\n"
           << "              [--fftw-internal-workers N] [--fftw-outer-workers N]\n"
           << "              [--fftw-planning-time-limit SECONDS]\n"
           << "              [--vdsp-strategy NAME] [--vdsp-batch-strategy NAME]\n"
           << "              [--warmups N] [--samples N] [--seed N] [--output PATH]\n"
           << "  skbench compare --input SAMPLES.csv\n";
}
std::string requireValue(int argc, char** argv, int& index) {
    if (index + 1 >= argc) throw std::invalid_argument("Missing value after " + std::string(argv[index]));
    return argv[++index];
}

void list() {
    std::cout << "schema: spectral-kernel-benchmark-v1\nprofiles:\n";
    for (const auto& profile : skbench::profiles()) {
        std::cout << "  " << profile.name << ": " << profile.workload.nx << 'x' << profile.workload.ny
                  << ", Nz=" << profile.workload.nz << ", fields=" << profile.workload.fields
                  << ", workers=" << profile.defaultWorkers << "\n    " << profile.purpose << '\n';
    }
    std::cout << "providers:\n"
              << "  fftw: pinned 3.3.11 NEON/pthreads, guru64 WVM strides\n"
              << "  accelerate-vdsp: double packed split-complex radix-2, direct or separable outer batching\n"
              << "FFTW planning modes:\n"
              << "  estimate\n"
              << "  measure\n"
              << "  patient\n"
              << "  exhaustive\n"
              << "FFTW alignment strategies:\n"
              << "  aligned\n"
              << "  unaligned\n"
              << "FFTW wisdom strategies:\n"
              << "  cold\n"
              << "  generated-import\n"
              << "FFTW data layouts:\n"
              << "  interleaved\n"
              << "  split\n"
              << "  paired\n"
              << "vDSP strategies:\n"
              << "  in-place\n"
              << "  in-place-explicit-scratch\n"
              << "  out-of-place\n"
              << "  out-of-place-explicit-scratch\n"
              << "vDSP batch strategies:\n"
              << "  direct-persistent\n"
              << "  direct-gcd\n"
              << "  separable-persistent\n"
              << "  separable-gcd\n"
              << "representations:\n"
              << "  logical retained modes keyed by (k,l,j,field)\n"
              << "  WVM frequency-major interleaved Hermitian half-spectrum\n"
              << "  plane-major interleaved Hermitian half-spectrum\n"
              << "  vDSP packed split-complex\n";
}

void printSummary(const skbench::BenchmarkReport& report) {
    std::cout << "run " << report.runId << " status=" << report.status << " profile=" << report.profile
              << " workload=" << report.workload.nx << 'x' << report.workload.ny
              << " Nz=" << report.workload.nz << " fields=" << report.workload.fields
              << " Nkl=" << report.retainedHorizontalModeCount << '\n';
    std::cout << "provider\tscope\tstage\tdirection\tmedian_ms\n";
    for (const auto& provider : report.providers) {
        for (const auto& timing : provider.timings) {
            if (timing.seconds.empty()) continue;
            std::cout << provider.id << '\t' << timing.scope << '\t' << timing.stage << '\t'
                      << timing.direction << '\t' << std::fixed << std::setprecision(6)
                      << 1000.0 * skbench::median(timing.seconds) << '\n';
        }
        for (const auto& metric : provider.correctness) {
            std::cout << provider.id << " correctness " << metric.name << '=' << std::scientific
                      << metric.maximumRelativeError << " passed=" << (metric.passed ? "true" : "false") << '\n';
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage(std::cerr);
            return 2;
        }
        const std::string_view command = argv[1];
        if (command == "list") {
            if (argc != 2) throw std::invalid_argument("list accepts no arguments.");
            list();
            return 0;
        }
        if (command == "validate") {
            std::string profile = "smoke";
            for (int index = 2; index < argc; ++index) {
                const std::string_view key = argv[index];
                if (key == "--profile") profile = requireValue(argc, argv, index);
                else throw std::invalid_argument("Unknown validate option: " + std::string(key));
            }
            const auto report = skbench::validateBenchmark(profile);
            for (const auto& message : report.messages) std::cout << message << '\n';
            return report.passed ? 0 : 1;
        }
        if (command == "run") {
            skbench::RunOptions options;
            for (int index = 2; index < argc; ++index) {
                const std::string_view key = argv[index];
                if (key == "--profile") options.profile = requireValue(argc, argv, index);
                else if (key == "--providers") options.providers = requireValue(argc, argv, index);
                else if (key == "--fftw-layout") options.fftwLayout = requireValue(argc, argv, index);
                else if (key == "--fftw-planning") options.fftwPlanning = requireValue(argc, argv, index);
                else if (key == "--fftw-alignment") options.fftwAlignment = requireValue(argc, argv, index);
                else if (key == "--fftw-wisdom") options.fftwWisdom = requireValue(argc, argv, index);
                else if (key == "--fftw-internal-workers") options.fftwInternalWorkers = std::stoull(requireValue(argc, argv, index));
                else if (key == "--fftw-outer-workers") options.fftwOuterWorkers = std::stoull(requireValue(argc, argv, index));
                else if (key == "--fftw-planning-time-limit") options.fftwPlanningTimeLimitSeconds = std::stod(requireValue(argc, argv, index));
                else if (key == "--vdsp-strategy") options.vdspStrategy = requireValue(argc, argv, index);
                else if (key == "--vdsp-batch-strategy") options.vdspBatchStrategy = requireValue(argc, argv, index);
                else if (key == "--workers") options.workers = std::stoull(requireValue(argc, argv, index));
                else if (key == "--warmups") options.warmups = std::stoull(requireValue(argc, argv, index));
                else if (key == "--samples") options.samples = std::stoull(requireValue(argc, argv, index));
                else if (key == "--seed") options.seed = std::stoull(requireValue(argc, argv, index));
                else if (key == "--output") options.outputJson = requireValue(argc, argv, index);
                else throw std::invalid_argument("Unknown run option: " + std::string(key));
            }
            auto report = skbench::runBenchmark(options);
            auto jsonPath = options.outputJson;
            if (jsonPath.empty()) jsonPath = std::filesystem::path("results/local") / (report.runId + ".json");
            auto csvPath = jsonPath;
            csvPath.replace_extension(".csv");
            skbench::writeJson(report, jsonPath);
            skbench::writeCsv(report, csvPath);
            printSummary(report);
            std::cout << "json=" << std::filesystem::absolute(jsonPath).string() << '\n';
            std::cout << "csv=" << std::filesystem::absolute(csvPath).string() << '\n';
            return report.status == "passed" ? 0 : 1;
        }
        if (command == "compare") {
            std::filesystem::path input;
            for (int index = 2; index < argc; ++index) {
                const std::string_view key = argv[index];
                if (key == "--input") input = requireValue(argc, argv, index);
                else if (input.empty() && !key.starts_with("--")) input = key;
                else throw std::invalid_argument("Unknown compare option: " + std::string(key));
            }
            if (input.empty()) throw std::invalid_argument("compare requires --input SAMPLES.csv.");
            return skbench::compareCsv(input, std::cout);
        }
        usage(std::cerr);
        throw std::invalid_argument("Unknown command: " + std::string(command));
    } catch (const std::exception& error) {
        std::cerr << "skbench: " << error.what() << '\n';
        return 2;
    }
}
