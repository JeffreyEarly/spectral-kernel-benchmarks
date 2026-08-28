#include "skbench/skbench.hpp"

#include <fftw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if SKBENCH_HAVE_FFTWPP
#include "Complex.h"
#include "convolve.h"
#endif

#ifndef SKBENCH_FFTWPP_COMMIT
#define SKBENCH_FFTWPP_COMMIT "unavailable"
#endif

#if SKBENCH_HAVE_FFTWPP
namespace {
void issue17MultiplyOffset(std::size_t offset, Complex** values, std::size_t count) {
    double* realValues[12]{};
    for (std::size_t array = 0; array < 4; ++array)
        realValues[array] = reinterpret_cast<double*>(values[array]);
    for (std::size_t index = 0; index < count; ++index) {
        double input[4] = {realValues[0][index], realValues[1][index],
                           realValues[2][index], realValues[3][index]};
        for (std::size_t product = 0; product < 4; ++product) {
            const auto logicalProduct = offset + product;
            const auto first = logicalProduct / 4;
            const auto second = logicalProduct % 4;
            realValues[product][index] = input[first] * input[second];
        }
    }
}
void issue17Multiply0(Complex** values, std::size_t count, fftwpp::Indices*, std::size_t) {
    issue17MultiplyOffset(0, values, count);
}
void issue17Multiply4(Complex** values, std::size_t count, fftwpp::Indices*, std::size_t) {
    issue17MultiplyOffset(4, values, count);
}
void issue17Multiply8(Complex** values, std::size_t count, fftwpp::Indices*, std::size_t) {
    issue17MultiplyOffset(8, values, count);
}
}
#endif

namespace skbench {
namespace {

using Clock = std::chrono::steady_clock;

#if SKBENCH_HAVE_FFTWPP
constexpr double tolerance = 1.0e-12;
struct StoredMode {
    int k = 0;
    int l = 0;
    std::size_t rectangleIndex = 0;
    std::size_t fullIndex = 0;
};

std::vector<StoredMode> storedDisk(std::size_t n) {
    const auto radius = static_cast<int>(n / 3);
    const auto length = static_cast<std::size_t>(2 * radius + 1);
    const auto half = static_cast<std::size_t>(radius + 1);
    std::vector<StoredMode> modes;
    for (int l = 0; l <= radius; ++l) {
        for (int k = -radius; k <= radius; ++k) {
            if (k * k + l * l > radius * radius) continue;
            const auto centeredK = static_cast<std::size_t>(k + radius);
            const auto fullK = k < 0 ? n - static_cast<std::size_t>(-k)
                                     : static_cast<std::size_t>(k);
            modes.push_back({k, l, centeredK * half + static_cast<std::size_t>(l),
                             fullK * (n / 2 + 1) + static_cast<std::size_t>(l)});
        }
    }
    (void)length;
    return modes;
}

std::vector<Complex> compactFixture(const std::vector<StoredMode>& modes,
                                    std::size_t products, std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::vector<Complex> values(4 * modes.size());
    for (std::size_t field = 0; field < 4; ++field) {
        for (std::size_t mode = 0; mode < modes.size(); ++mode) {
            const auto& key = modes[mode];
            if (key.l == 0 && key.k < 0) continue;
            Complex value{distribution(generator), distribution(generator)};
            if (key.l == 0 && key.k == 0) value.imag = 0.0;
            values[field * modes.size() + mode] = value;
            if (key.l == 0 && key.k > 0) {
                const auto match = std::find_if(modes.begin(), modes.end(), [&](const StoredMode& other) {
                    return other.k == -key.k && other.l == 0;
                });
                values[field * modes.size() + static_cast<std::size_t>(match - modes.begin())] =
                    {value.real, -value.imag};
            }
        }
    }
    (void)products;
    return values;
}

template <class Prepare, class Action>
std::vector<double> timed(std::size_t warmups, std::size_t samples,
                          Prepare prepare, Action action) {
    for (std::size_t i = 0; i < warmups; ++i) { prepare(); action(); }
    std::vector<double> result;
    result.reserve(samples);
    for (std::size_t i = 0; i < samples; ++i) {
        prepare();
        const auto start = Clock::now();
        action();
        result.push_back(std::chrono::duration<double>(Clock::now() - start).count());
    }
    return result;
}

TimingSeries timing(std::string scope, std::string stage, StageState state,
                    std::uint64_t moved, std::vector<double> samples = {}) {
    return {std::move(scope), std::move(stage), "forward", state, moved, std::move(samples)};
}

class ExplicitConvolution {
public:
    ExplicitConvolution(std::size_t n, std::size_t products,
                        const std::vector<StoredMode>& modes)
        : n_(n), products_(products), count_(std::max<std::size_t>(4, products)),
          halfCount_(n * (n / 2 + 1)), realCount_(n * n),
          modes_(modes), output_(products * modes.size()) {
        for (std::size_t i = 0; i < count_; ++i) {
            spectra_.push_back(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * halfCount_)));
            physical_.push_back(static_cast<double*>(fftw_malloc(sizeof(double) * realCount_)));
            if (!spectra_.back() || !physical_.back()) throw std::bad_alloc();
        }
        const auto flags = FFTW_MEASURE | FFTW_UNALIGNED;
        inverse_ = fftw_plan_dft_c2r_2d(static_cast<int>(n_), static_cast<int>(n_),
                                        spectra_[0], physical_[0], flags);
        forward_ = fftw_plan_dft_r2c_2d(static_cast<int>(n_), static_cast<int>(n_),
                                        physical_[0], spectra_[0], flags);
        if (!inverse_ || !forward_) throw std::runtime_error("Unable to create explicit FFTW plans.");
    }
    ~ExplicitConvolution() {
        if (inverse_) fftw_destroy_plan(inverse_);
        if (forward_) fftw_destroy_plan(forward_);
        for (auto* value : spectra_) fftw_free(value);
        for (auto* value : physical_) fftw_free(value);
    }
    void load(const std::vector<Complex>& compact) {
        for (std::size_t field = 0; field < 4; ++field) {
            std::memset(spectra_[field], 0, sizeof(fftw_complex) * halfCount_);
            for (std::size_t mode = 0; mode < modes_.size(); ++mode) {
                spectra_[field][modes_[mode].fullIndex][0] = compact[field * modes_.size() + mode].real;
                spectra_[field][modes_[mode].fullIndex][1] = compact[field * modes_.size() + mode].imag;
            }
        }
    }
    void inverse() {
        for (std::size_t field = 0; field < 4; ++field)
            fftw_execute_dft_c2r(inverse_, spectra_[field], physical_[field]);
    }
    void multiply() {
        for (std::size_t point = 0; point < realCount_; ++point) {
            double input[4] = {physical_[0][point], physical_[1][point],
                               physical_[2][point], physical_[3][point]};
            for (std::size_t product = 0; product < products_; ++product)
                physical_[product][point] = input[product / 4] * input[product % 4];
        }
    }
    void forward() {
        for (std::size_t product = 0; product < products_; ++product)
            fftw_execute_dft_r2c(forward_, physical_[product], spectra_[product]);
    }
    void gather() {
        const double scale = 1.0 / static_cast<double>(realCount_);
        for (std::size_t product = 0; product < products_; ++product)
            for (std::size_t mode = 0; mode < modes_.size(); ++mode)
                output_[product * modes_.size() + mode] = {
                    scale * spectra_[product][modes_[mode].fullIndex][0],
                    scale * spectra_[product][modes_[mode].fullIndex][1]};
    }
    void execute(const std::vector<Complex>& compact) { load(compact); inverse(); multiply(); forward(); gather(); }
    const std::vector<Complex>& output() const { return output_; }
    std::uint64_t residentBytes() const {
        return count_ * (halfCount_ * sizeof(fftw_complex) + realCount_ * sizeof(double)) +
               output_.size() * sizeof(Complex);
    }
private:
    std::size_t n_, products_, count_, halfCount_, realCount_;
    const std::vector<StoredMode>& modes_;
    std::vector<fftw_complex*> spectra_;
    std::vector<double*> physical_;
    std::vector<Complex> output_;
    fftw_plan inverse_ = nullptr;
    fftw_plan forward_ = nullptr;
};

#if SKBENCH_HAVE_FFTWPP
class ImplicitConvolution {
    struct Kernel;
public:
    ImplicitConvolution(std::size_t n, std::size_t products,
                        const std::vector<StoredMode>& modes)
        : n_(n), products_(products), modes_(modes), radius_(n / 3),
          length_(2 * radius_ + 1), half_(radius_ + 1),
          count_(4), output_(products * modes.size()) {
        fftwpp::fftw::maxthreads = 1;
        fftwpp::fftw::effort = FFTW_MEASURE | FFTW_UNALIGNED;
        fftwpp::multiplier* multipliers[] = {issue17Multiply0, issue17Multiply4, issue17Multiply8};
        for (std::size_t group = 0; group < (products_ + 3) / 4; ++group)
            kernels_.push_back(std::make_unique<Kernel>(n_, length_, half_, multipliers[group]));
    }
    void load(Kernel& kernel, const std::vector<Complex>& compact) {
        std::fill(kernel.arrays[0], kernel.arrays[0] + count_ * length_ * half_, ::Complex(0.0, 0.0));
        for (std::size_t field = 0; field < 4; ++field)
            for (std::size_t mode = 0; mode < modes_.size(); ++mode) {
                const auto value = compact[field * modes_.size() + mode];
                kernel.arrays[field][modes_[mode].rectangleIndex] = ::Complex(value.real, value.imag);
            }
    }
    void gatherGroup(const Kernel& kernel, std::size_t firstProduct) {
        const auto stop = std::min(products_, firstProduct + 4);
        for (std::size_t product = firstProduct; product < stop; ++product)
            for (std::size_t mode = 0; mode < modes_.size(); ++mode) {
                const auto value = kernel.arrays[product - firstProduct][modes_[mode].rectangleIndex];
                output_[product * modes_.size() + mode] = {value.real(), value.imag()};
            }
    }
    void executeNativeSequence(const std::vector<Complex>& compact, bool gatherOutput) {
        for (std::size_t group = 0; group < kernels_.size(); ++group) {
            auto& kernel = *kernels_[group];
            load(kernel, compact);
            kernel.convolution->convolve(kernel.arrays);
            if (gatherOutput) gatherGroup(kernel, group * 4);
        }
    }
    void execute(const std::vector<Complex>& compact) { executeNativeSequence(compact, true); }
    const std::vector<Complex>& output() const { return output_; }
    std::uint64_t residentBytes() const {
        std::uint64_t bytes = output_.size() * sizeof(Complex);
        for (const auto& kernel : kernels_) {
            const auto caller = count_ * length_ * half_;
            const auto internal = count_ * kernel->fftx->outputSize() + kernel->fftx->workSizeW() +
                                  4 * kernel->fftx->workSizeV();
            bytes += (caller + internal) * sizeof(::Complex);
        }
        return bytes;
    }
    bool nativeInPlace() const {
        return std::all_of(kernels_.begin(), kernels_.end(), [](const auto& kernel) {
            return kernel->fftx->inplace && kernel->ffty->inplace;
        });
    }
private:
    struct Kernel {
        Kernel(std::size_t n, std::size_t length, std::size_t half,
               fftwpp::multiplier* multiplier) {
            arrays = utils::ComplexAlign(4, length * half);
            if (!arrays) throw std::bad_alloc();
            appx = std::make_unique<fftwpp::Application>(4, 4, fftwpp::multNone, 1);
            fftx = std::make_unique<fftwpp::fftPadCentered>(length, n, *appx, half);
            appy = std::make_unique<fftwpp::Application>(4, 4, multiplier, *appx);
            ffty = std::make_unique<fftwpp::fftPadHermitian>(length, n, *appy);
            convolution = std::make_unique<fftwpp::Convolution2>(fftx.get(), ffty.get());
        }
        ~Kernel() {
            convolution.reset();
            ffty.reset();
            fftx.reset();
            appy.reset();
            appx.reset();
            if (arrays) { utils::deleteAlign(arrays[0]); delete[] arrays; }
        }
        ::Complex** arrays = nullptr;
        std::unique_ptr<fftwpp::Application> appx, appy;
        std::unique_ptr<fftwpp::fftPadCentered> fftx;
        std::unique_ptr<fftwpp::fftPadHermitian> ffty;
        std::unique_ptr<fftwpp::Convolution2> convolution;
    };
    std::size_t n_, products_;
    const std::vector<StoredMode>& modes_;
    std::size_t radius_, length_, half_, count_;
    std::vector<Complex> output_;
    std::vector<std::unique_ptr<Kernel>> kernels_;
};
#endif

CorrectnessMetric correctness(const std::vector<Complex>& actual,
                              const std::vector<Complex>& expected) {
    const auto maximum = maximumRelativeError(actual.data(), expected.data(), actual.size());
    const auto l2 = relativeL2Error(actual.data(), expected.data(), actual.size());
    return {"mode-keyed retained convolution versus explicit FFTW oracle", maximum,
            tolerance, maximum <= tolerance && l2 <= tolerance, l2};
}

CorrectnessMetric correctnessGroup(const std::vector<Complex>& actual,
                                   const std::vector<Complex>& expected,
                                   std::size_t modes, std::size_t firstProduct) {
    const auto count = std::min<std::size_t>(4, actual.size() / modes - firstProduct) * modes;
    const auto* actualStart = actual.data() + firstProduct * modes;
    const auto* expectedStart = expected.data() + firstProduct * modes;
    const auto maximum = maximumRelativeError(actualStart, expectedStart, count);
    const auto l2 = relativeL2Error(actualStart, expectedStart, count);
    return {"product group " + std::to_string(firstProduct / 4 + 1) + " versus explicit oracle",
            maximum, tolerance, maximum <= tolerance && l2 <= tolerance, l2};
}

ExecutionContract convolutionContract(std::size_t n, std::size_t products,
                                      bool nativeInPlace, bool implicit) {
    DirectionExecutionContract forward;
    forward.nativePlacement = nativeInPlace ? "in-place" : "out-of-place";
    forward.adapterPlacement = "out-of-place";
    forward.destroysNativeInput = true;
    forward.adapterPreservesCallerInput = true;
    forward.requiresPreservationCopyForRepeatedExecution = true;
    forward.preservationIncludedInAdapterTiming = true;
    forward.nativeInputRepresentationId = implicit ? "centered-rectangular-hermitian" : "full-fftw-half-spectrum";
    forward.nativeOutputRepresentationId = forward.nativeInputRepresentationId;
    forward.adapterInputRepresentationId = "mode-keyed-radial-hermitian-four-field";
    forward.adapterOutputRepresentationId = "mode-keyed-radial-hermitian-products";
    forward.physicalExtents = "Nx=Ny=" + std::to_string(n) + "; fields=4; products=" + std::to_string(products);
    forward.aliasing = "native execution overwrites provider buffers; the timed complete adapter copies from and preserves the caller-owned compact input";
    DirectionExecutionContract inverse;
    inverse.nativePlacement = "unsupported";
    inverse.adapterPlacement = "unsupported";
    inverse.nativeInputRepresentationId = "not-applicable";
    inverse.nativeOutputRepresentationId = "not-applicable";
    return {std::move(forward), std::move(inverse)};
}
#endif

} // namespace

BenchmarkReport runDealiasedConvolutionBenchmark(const RunOptions& options) {
#if !SKBENCH_HAVE_FFTWPP
    (void)options;
    throw std::runtime_error("dealiased-convolution requires configuring with SKBENCH_ENABLE_FFTWPP=ON");
#else
    if (options.convolutionProducts != 4 && options.convolutionProducts != 12)
        throw std::invalid_argument("convolution-products must be 4 or 12.");
    const auto selected = profileNamed(options.profile);
    if (selected.workload.nx != selected.workload.ny)
        throw std::invalid_argument("dealiased-convolution currently requires a square horizontal grid.");
    const auto n = selected.workload.nx;
    const auto products = options.convolutionProducts;
    const auto warmups = options.warmups ? options.warmups : 1;
    const auto samples = options.samples ? options.samples : 5;
    const auto modes = storedDisk(n);
    const auto input = compactFixture(modes, products, options.seed);

    const auto setupStart = Clock::now();
    ExplicitConvolution explicitPath(n, products, modes);
    const auto explicitSetup = std::chrono::duration<double>(Clock::now() - setupStart).count();
    const auto implicitSetupStart = Clock::now();
    ImplicitConvolution implicitPath(n, products, modes);
    const auto implicitSetup = std::chrono::duration<double>(Clock::now() - implicitSetupStart).count();

    explicitPath.execute(input);
    implicitPath.execute(input);
    const auto implicitCorrectness = correctness(implicitPath.output(), explicitPath.output());

    BenchmarkReport report;
    report.status = implicitCorrectness.passed ? "passed" : "failed";
    report.environment = environmentRecord();
    auto id = report.environment.timestampUtc;
    id.erase(std::remove_if(id.begin(), id.end(), [](char c) { return c == '-' || c == ':'; }), id.end());
    report.runId = id + "-issue17-n" + std::to_string(n) + "-p" + std::to_string(products);
    report.profile = options.profile;
    report.seed = options.seed;
    report.warmups = warmups;
    report.samples = samples;
    report.workload = {n, n, 1, 4, 1.0, 1.0, true};
    report.retainedHorizontalModeCount = modes.size();
    report.retainedModeOrderHash = "centered-k-then-nonnegative-l-radial-disk-v1";
    report.fullRealBytes = n * n * 4 * sizeof(double);
    report.fullSpectrumBytes = n * (n / 2 + 1) * 4 * sizeof(Complex);
    report.retainedSpectrumBytes = modes.size() * 4 * sizeof(Complex);

    ProviderRecord baseline;
    baseline.id = "fftw-explicit-dealiased-convolution";
    baseline.version = "3.3.11";
    baseline.libraryIdentity = "FFTW 3.3.11 pthread build";
    baseline.algorithmId = "explicit-full-grid-four-field-quadratic-p" + std::to_string(products);
    baseline.nativeRepresentationId = "full-fftw-half-spectrum";
    baseline.modeOrderId = report.retainedModeOrderHash;
    baseline.schedulingId = "single-thread";
    baseline.planningConfiguration = "FFTW_MEASURE|FFTW_UNALIGNED";
    baseline.execution = convolutionContract(n, products, false, false);
    baseline.explicitPersistentBytes = explicitPath.residentBytes();
    baseline.algorithmResidentBytes = explicitPath.residentBytes();
    baseline.opaqueProviderMemory = true;
    baseline.planningSeconds = explicitSetup;
    baseline.correctness = {{"independent oracle baseline", 0.0, tolerance, true, 0.0}};
    baseline.timings.push_back(timing("primitive-component", "inverse FFT batch", StageState::executed, report.fullSpectrumBytes + report.fullRealBytes,
        timed(warmups, samples, [&] { explicitPath.load(input); }, [&] { explicitPath.inverse(); })));
    baseline.timings.push_back(timing("primitive-component", "pointwise quadratic products", StageState::executed, (4 + products) * n * n * sizeof(double),
        timed(warmups, samples, [&] { explicitPath.load(input); explicitPath.inverse(); }, [&] { explicitPath.multiply(); })));
    baseline.timings.push_back(timing("primitive-component", "forward FFT batch", StageState::executed, products * n * n * sizeof(double) + products * n * (n / 2 + 1) * sizeof(Complex),
        timed(warmups, samples, [&] { explicitPath.load(input); explicitPath.inverse(); explicitPath.multiply(); }, [&] { explicitPath.forward(); })));
    baseline.timings.push_back(timing("uninstrumented-total", "dealiased four-field convolution", StageState::executed, explicitPath.residentBytes(),
        timed(warmups, samples, [] {}, [&] { explicitPath.execute(input); })));
    baseline.ledger = {
        {"setup/planning", StageState::setupOnly, "two reusable FFTW_MEASURE plans"},
        {"compact-to-full embedding", StageState::executed, "included in the authoritative total"},
        {"inverse FFTs", StageState::executed, "four full-grid c2r transforms"},
        {"quadratic products", StageState::executed, std::to_string(products) + " deterministic pointwise products"},
        {"forward FFTs", StageState::executed, std::to_string(products) + " full-grid r2c transforms"},
        {"radial retention", StageState::executed, "included in the authoritative total"},
        {"nonlinear WVM flux", StageState::unsupported, "explicitly outside issue #17"}};

    ProviderRecord candidate;
    candidate.id = "fftwpp-hybrid-hermitian-convolution";
    candidate.version = "3.04";
    candidate.libraryIdentity = "FFTW++ public master pinned at " SKBENCH_FFTWPP_COMMIT;
    candidate.algorithmId = "hybrid-centered-hermitian-four-field-quadratic-p" + std::to_string(products);
    candidate.nativeRepresentationId = "centered-rectangular-hermitian";
    candidate.modeOrderId = report.retainedModeOrderHash;
    candidate.schedulingId = "single-thread";
    candidate.sourceIdentity = "https://github.com/dealias/fftwpp/commit/" SKBENCH_FFTWPP_COMMIT;
    candidate.configureFlags = "FFTWPP_SINGLE_THREAD=1; optimizer-selected residue/hybrid parameters";
    candidate.planningConfiguration = "FFTW_MEASURE|FFTW_UNALIGNED";
    candidate.execution = convolutionContract(n, products, implicitPath.nativeInPlace(), true);
    candidate.explicitPersistentBytes = implicitPath.residentBytes();
    candidate.algorithmResidentBytes = implicitPath.residentBytes();
    candidate.opaqueProviderMemory = false;
    candidate.planningSeconds = implicitSetup;
    candidate.correctness = {implicitCorrectness};
    for (std::size_t firstProduct = 0; firstProduct < products; firstProduct += 4)
        candidate.correctness.push_back(correctnessGroup(
            implicitPath.output(), explicitPath.output(), modes.size(), firstProduct));
    candidate.timings.push_back(timing("fused-primitive", "implicit hybrid transform-multiply-transform", StageState::fused, implicitPath.residentBytes(),
        timed(warmups, samples, [] {}, [&] { implicitPath.executeNativeSequence(input, false); })));
    candidate.timings.push_back(timing("uninstrumented-total", "dealiased four-field convolution", StageState::executed, implicitPath.residentBytes(),
        timed(warmups, samples, [] {}, [&] { implicitPath.execute(input); })));
    candidate.ledger = {
        {"setup/planning", StageState::setupOnly, "FFTW++ optimizer and reusable FFTW plans"},
        {"compact-to-centered embedding", StageState::executed, "included in the authoritative total"},
        {"inverse FFTs", StageState::fused, "inseparable FFTW++ implicit/hybrid stage"},
        {"quadratic products", StageState::fused, std::to_string(products) + " outputs from " + std::to_string((products + 3) / 4) + " four-output custom-multiplier call(s); FFTW++ B>A is excluded after a reproducible optimizer crash"},
        {"forward FFTs", StageState::fused, "inseparable FFTW++ implicit/hybrid stage"},
        {"radial retention", StageState::executed, "included in the authoritative total"},
        {"nonlinear WVM flux", StageState::unsupported, "explicitly outside issue #17"}};

    report.providers = {std::move(baseline), std::move(candidate)};
    return report;
#endif
}

std::uint64_t probeDealiasedConvolutionSteadyStateAllocationsForTesting(
    std::size_t n, std::size_t products,
    void (*beginTracking)(), std::uint64_t (*endTracking)()) {
#if !SKBENCH_HAVE_FFTWPP
    (void)n;
    (void)products;
    (void)beginTracking;
    (void)endTracking;
    throw std::runtime_error("FFTW++ allocation probe is unavailable in this build.");
#else
    if (products != 4 && products != 12)
        throw std::invalid_argument("allocation probe products must be 4 or 12.");
    const auto modes = storedDisk(n);
    const auto input = compactFixture(modes, products, 129);
    ExplicitConvolution explicitPath(n, products, modes);
    ImplicitConvolution implicitPath(n, products, modes);
    explicitPath.execute(input);
    implicitPath.execute(input);
    beginTracking();
    explicitPath.execute(input);
    implicitPath.execute(input);
    return endTracking();
#endif
}

} // namespace skbench
