#include "skbench/skbench.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace skbench {
namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

std::size_t checkedProduct(std::size_t first, std::size_t second) {
    if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) {
        throw std::overflow_error("workload size overflow");
    }
    return first * second;
}

void validateWorkload(const Workload& workload) {
    if (workload.nx < 2 || workload.ny < 2 || workload.nz < 1 || workload.fields < 1) {
        throw std::invalid_argument("Nx and Ny must be at least two; Nz and fields must be positive.");
    }
    if (!(workload.lx > 0.0) || !(workload.ly > 0.0)) {
        throw std::invalid_argument("Horizontal domain lengths must be positive.");
    }
}

std::vector<std::int64_t> dftModes(std::size_t count) {
    std::vector<std::int64_t> modes;
    modes.reserve(count);
    const auto positiveCount = (count + 1) / 2;
    for (std::size_t value = 0; value < positiveCount; ++value) {
        modes.push_back(static_cast<std::int64_t>(value));
    }
    for (std::size_t value = count / 2; value > 0; --value) {
        modes.push_back(-static_cast<std::int64_t>(value));
    }
    return modes;
}

std::size_t dftIndex(std::int64_t mode, std::size_t count) {
    const auto wrapped = mode >= 0 ? mode : static_cast<std::int64_t>(count) + mode;
    return static_cast<std::size_t>(wrapped);
}

bool selfConjugate(std::int64_t mode, std::size_t count) {
    return mode == 0 || (count % 2 == 0 && mode == -static_cast<std::int64_t>(count / 2));
}

bool primaryMode(std::int64_t k, std::int64_t l, std::size_t nx, std::size_t ny) {
    const bool kSelf = selfConjugate(k, nx);
    const bool lSelf = selfConjugate(l, ny);
    return l > 0 || (lSelf && (k > 0 || kSelf));
}

bool nyquistMode(std::int64_t k, std::int64_t l, std::size_t nx, std::size_t ny) {
    return (nx % 2 == 0 && k == -static_cast<std::int64_t>(nx / 2)) ||
           (ny % 2 == 0 && l == -static_cast<std::int64_t>(ny / 2));
}

double radialMagnitude(double first, double second) {
    volatile double firstSquared = first * first;
    volatile double secondSquared = second * second;
    return std::sqrt(firstSquared + secondSquared);
}

Complex add(Complex first, Complex second) {
    return {first.real + second.real, first.imag + second.imag};
}

Complex multiply(Complex first, Complex second) {
    return {first.real * second.real - first.imag * second.imag,
            first.real * second.imag + first.imag * second.real};
}

Complex scale(Complex value, double factor) {
    return {factor * value.real, factor * value.imag};
}

void hashByte(std::uint64_t& hash, std::uint8_t value) {
    hash ^= value;
    hash *= UINT64_C(1099511628211);
}

template <typename Integer>
void hashInteger(std::uint64_t& hash, Integer value) {
    using Unsigned = std::make_unsigned_t<Integer>;
    const auto bits = static_cast<Unsigned>(value);
    for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
        hashByte(hash, static_cast<std::uint8_t>((bits >> (8 * byte)) & static_cast<Unsigned>(0xff)));
    }
}

std::string finishHash(std::uint64_t hash) {
    std::ostringstream stream;
    stream << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return stream.str();
}

} // namespace

Complex conjugate(Complex value) noexcept { return {value.real, -value.imag}; }

double magnitude(Complex value) noexcept { return std::hypot(value.real, value.imag); }

std::size_t Workload::planes() const { return checkedProduct(nz, fields); }

std::size_t Workload::nxHalf() const { return nx / 2 + 1; }

std::size_t Workload::realPlaneElements() const { return checkedProduct(nx, ny); }

std::size_t Workload::halfRows() const { return checkedProduct(nxHalf(), ny); }

std::size_t Workload::realElements() const { return checkedProduct(realPlaneElements(), planes()); }

std::size_t Workload::spectrumElements() const { return checkedProduct(halfRows(), planes()); }

std::size_t Workload::retainedVerticalModes() const {
    if (nz < 2) return 0;
    return 2 * (nz - 1) / 3;
}

std::vector<RetainedMode> retainedHorizontalModes(const Workload& workload) {
    validateWorkload(workload);
    const auto kModes = dftModes(workload.nx);
    const auto lModes = dftModes(workload.ny);
    const double maximumK = 2.0 * pi * (static_cast<double>(workload.nx / 2) / workload.lx);
    std::vector<RetainedMode> modes;
    modes.reserve(workload.halfRows());

    for (const auto l : lModes) {
        for (const auto k : kModes) {
            if (!primaryMode(k, l, workload.nx, workload.ny) || nyquistMode(k, l, workload.nx, workload.ny)) continue;
            const double physicalK = 2.0 * pi * static_cast<double>(k) / workload.lx;
            const double physicalL = 2.0 * pi * static_cast<double>(l) / workload.ly;
            const double radial = radialMagnitude(physicalK, physicalL);
            if (workload.antialias && radial > 2.0 * maximumK / 3.0) continue;

            RetainedMode mode;
            mode.k = k;
            mode.l = l;
            mode.radialMode = radial;
            if (k >= 0) {
                mode.storedKx = static_cast<std::size_t>(k);
                mode.storedKy = dftIndex(l, workload.ny);
            } else {
                mode.storedKx = static_cast<std::size_t>(-k);
                mode.storedKy = dftIndex(-l, workload.ny);
                mode.conjugatesStoredValue = true;
            }
            modes.push_back(mode);
        }
    }

    std::stable_sort(modes.begin(), modes.end(), [](const RetainedMode& first, const RetainedMode& second) {
        if (first.radialMode != second.radialMode) return first.radialMode < second.radialMode;
        if (first.k != second.k) return first.k < second.k;
        return first.l < second.l;
    });
    return modes;
}

std::size_t realIndex(const Workload& workload, std::size_t x, std::size_t y, std::size_t z, std::size_t field) {
    return x + workload.nx * (y + workload.ny * (z + workload.nz * field));
}

std::size_t wvmSpectrumIndex(const Workload& workload, std::size_t kx, std::size_t ky, std::size_t z, std::size_t field) {
    return z + workload.nz * field + workload.planes() * (kx + workload.nxHalf() * ky);
}

std::size_t planeMajorSpectrumIndex(const Workload& workload, std::size_t kx, std::size_t ky, std::size_t z, std::size_t field) {
    return kx + workload.nxHalf() * (ky + workload.ny * (z + workload.nz * field));
}

std::size_t retainedSpectrumIndex(const Workload& workload, std::size_t mode, std::size_t z, std::size_t field) {
    return z + workload.nz * field + workload.planes() * mode;
}

std::size_t modalSpectrumIndex(const Workload& workload, std::size_t mode, std::size_t j, std::size_t field) {
    const auto nj = workload.retainedVerticalModes();
    return j + nj * field + nj * workload.fields * mode;
}

void gatherRetained(const Workload& workload, const std::vector<RetainedMode>& modes, const Complex* fullSpectrum, Complex* retainedSpectrum) {
    for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        const auto& mode = modes[modeIndex];
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t z = 0; z < workload.nz; ++z) {
                auto value = fullSpectrum[wvmSpectrumIndex(workload, mode.storedKx, mode.storedKy, z, field)];
                if (mode.conjugatesStoredValue) value = conjugate(value);
                retainedSpectrum[retainedSpectrumIndex(workload, modeIndex, z, field)] = value;
            }
        }
    }
}

void embedRetained(const Workload& workload, const std::vector<RetainedMode>& modes, const Complex* retainedSpectrum, Complex* fullSpectrum) {
    std::fill_n(fullSpectrum, workload.spectrumElements(), Complex{});
    for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        const auto& mode = modes[modeIndex];
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t z = 0; z < workload.nz; ++z) {
                const auto compact = retainedSpectrum[retainedSpectrumIndex(workload, modeIndex, z, field)];
                const auto stored = mode.conjugatesStoredValue ? conjugate(compact) : compact;
                fullSpectrum[wvmSpectrumIndex(workload, mode.storedKx, mode.storedKy, z, field)] = stored;
                if (mode.storedKx == 0 && mode.storedKy != 0 && 2 * mode.storedKy != workload.ny) {
                    const auto conjugateKy = (workload.ny - mode.storedKy) % workload.ny;
                    fullSpectrum[wvmSpectrumIndex(workload, 0, conjugateKy, z, field)] = conjugate(stored);
                }
            }
        }
    }
}

void interleavedToSplit(std::size_t count, const Complex* interleaved, double* real, double* imag) {
    for (std::size_t index = 0; index < count; ++index) {
        real[index] = interleaved[index].real;
        imag[index] = interleaved[index].imag;
    }
}

void splitToInterleaved(std::size_t count, const double* real, const double* imag, Complex* interleaved) {
    for (std::size_t index = 0; index < count; ++index) {
        interleaved[index] = {real[index], imag[index]};
    }
}

void gatherRetainedSplit(const Workload& workload, const std::vector<RetainedMode>& modes,
                         const double* fullReal, const double* fullImag,
                         double* retainedReal, double* retainedImag) {
    for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        const auto& mode = modes[modeIndex];
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t z = 0; z < workload.nz; ++z) {
                const auto fullIndex = wvmSpectrumIndex(workload, mode.storedKx, mode.storedKy, z, field);
                const auto retainedIndex = retainedSpectrumIndex(workload, modeIndex, z, field);
                retainedReal[retainedIndex] = fullReal[fullIndex];
                retainedImag[retainedIndex] = mode.conjugatesStoredValue ? -fullImag[fullIndex] : fullImag[fullIndex];
            }
        }
    }
}

void embedRetainedSplit(const Workload& workload, const std::vector<RetainedMode>& modes,
                        const double* retainedReal, const double* retainedImag,
                        double* fullReal, double* fullImag) {
    std::fill_n(fullReal, workload.spectrumElements(), 0.0);
    std::fill_n(fullImag, workload.spectrumElements(), 0.0);
    for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        const auto& mode = modes[modeIndex];
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t z = 0; z < workload.nz; ++z) {
                const auto retainedIndex = retainedSpectrumIndex(workload, modeIndex, z, field);
                const auto storedIndex = wvmSpectrumIndex(workload, mode.storedKx, mode.storedKy, z, field);
                const auto storedImag = mode.conjugatesStoredValue ? -retainedImag[retainedIndex] : retainedImag[retainedIndex];
                fullReal[storedIndex] = retainedReal[retainedIndex];
                fullImag[storedIndex] = storedImag;
                if (mode.storedKx == 0 && mode.storedKy != 0 && 2 * mode.storedKy != workload.ny) {
                    const auto conjugateKy = (workload.ny - mode.storedKy) % workload.ny;
                    const auto conjugateIndex = wvmSpectrumIndex(workload, 0, conjugateKy, z, field);
                    fullReal[conjugateIndex] = retainedReal[retainedIndex];
                    fullImag[conjugateIndex] = -storedImag;
                }
            }
        }
    }
}

void wvmToPlaneMajor(const Workload& workload, const Complex* wvmSpectrum, Complex* planeMajorSpectrum) {
    for (std::size_t field = 0; field < workload.fields; ++field) {
        for (std::size_t z = 0; z < workload.nz; ++z) {
            for (std::size_t ky = 0; ky < workload.ny; ++ky) {
                for (std::size_t kx = 0; kx < workload.nxHalf(); ++kx) {
                    planeMajorSpectrum[planeMajorSpectrumIndex(workload, kx, ky, z, field)] =
                        wvmSpectrum[wvmSpectrumIndex(workload, kx, ky, z, field)];
                }
            }
        }
    }
}

void planeMajorToWvm(const Workload& workload, const Complex* planeMajorSpectrum, Complex* wvmSpectrum) {
    for (std::size_t field = 0; field < workload.fields; ++field) {
        for (std::size_t z = 0; z < workload.nz; ++z) {
            for (std::size_t ky = 0; ky < workload.ny; ++ky) {
                for (std::size_t kx = 0; kx < workload.nxHalf(); ++kx) {
                    wvmSpectrum[wvmSpectrumIndex(workload, kx, ky, z, field)] =
                        planeMajorSpectrum[planeMajorSpectrumIndex(workload, kx, ky, z, field)];
                }
            }
        }
    }
}

std::string modeOrderHash(const std::vector<RetainedMode>& modes) {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const auto& mode : modes) {
        hashInteger(hash, mode.k);
        hashInteger(hash, mode.l);
    }
    return finishHash(hash);
}

std::string wvmSpectrumOrderHash(const Workload& workload) {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (std::size_t ky = 0; ky < workload.ny; ++ky) {
        const auto signedKy = ky <= workload.ny / 2 ? static_cast<std::int64_t>(ky) : static_cast<std::int64_t>(ky) - static_cast<std::int64_t>(workload.ny);
        for (std::size_t kx = 0; kx < workload.nxHalf(); ++kx) {
            for (std::size_t field = 0; field < workload.fields; ++field) {
                for (std::size_t z = 0; z < workload.nz; ++z) {
                    hashInteger(hash, static_cast<std::int64_t>(kx));
                    hashInteger(hash, signedKy);
                    hashInteger(hash, z);
                    hashInteger(hash, field);
                }
            }
        }
    }
    return finishHash(hash);
}

VerticalOperators orthonormalVerticalFixture(std::size_t nz, std::size_t nj) {
    if (nz == 0 || nj == 0 || nj > nz) throw std::invalid_argument("Vertical fixture requires 1 <= Nj <= Nz.");
    VerticalOperators operators;
    operators.id = "orthonormal-dct2-truncated-v1";
    operators.nz = nz;
    operators.nj = nj;
    operators.forward.resize(nj * nz);
    operators.inverse.resize(nz * nj);
    for (std::size_t j = 0; j < nj; ++j) {
        const double normalization = j == 0 ? std::sqrt(1.0 / static_cast<double>(nz)) : std::sqrt(2.0 / static_cast<double>(nz));
        for (std::size_t z = 0; z < nz; ++z) {
            const double value = normalization * std::cos(pi * (static_cast<double>(z) + 0.5) * static_cast<double>(j) / static_cast<double>(nz));
            operators.forward[j * nz + z] = value;
            operators.inverse[z * nj + j] = value;
        }
    }
    return operators;
}

void verticalForward(const Workload& workload, std::size_t horizontalModeCount, const VerticalOperators& operators, const Complex* physicalCoefficients, Complex* modalCoefficients) {
    if (operators.nz != workload.nz || operators.nj != workload.retainedVerticalModes()) {
        throw std::invalid_argument("Vertical forward operator dimensions do not match the workload.");
    }
    for (std::size_t mode = 0; mode < horizontalModeCount; ++mode) {
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t j = 0; j < operators.nj; ++j) {
                Complex sum;
                for (std::size_t z = 0; z < operators.nz; ++z) {
                    const auto value = physicalCoefficients[retainedSpectrumIndex(workload, mode, z, field)];
                    const double factor = operators.forward[j * operators.nz + z];
                    sum.real += factor * value.real;
                    sum.imag += factor * value.imag;
                }
                modalCoefficients[modalSpectrumIndex(workload, mode, j, field)] = sum;
            }
        }
    }
}

void verticalInverse(const Workload& workload, std::size_t horizontalModeCount, const VerticalOperators& operators, const Complex* modalCoefficients, Complex* physicalCoefficients) {
    if (operators.nz != workload.nz || operators.nj != workload.retainedVerticalModes()) {
        throw std::invalid_argument("Vertical inverse operator dimensions do not match the workload.");
    }
    for (std::size_t mode = 0; mode < horizontalModeCount; ++mode) {
        for (std::size_t field = 0; field < workload.fields; ++field) {
            for (std::size_t z = 0; z < operators.nz; ++z) {
                Complex sum;
                for (std::size_t j = 0; j < operators.nj; ++j) {
                    const auto value = modalCoefficients[modalSpectrumIndex(workload, mode, j, field)];
                    const double factor = operators.inverse[z * operators.nj + j];
                    sum.real += factor * value.real;
                    sum.imag += factor * value.imag;
                }
                physicalCoefficients[retainedSpectrumIndex(workload, mode, z, field)] = sum;
            }
        }
    }
}

std::string_view fixtureName(FixtureKind fixture) noexcept {
    switch (fixture) {
        case FixtureKind::impulse: return "impulse";
        case FixtureKind::sinusoid: return "sinusoid";
        case FixtureKind::random: return "random";
        case FixtureKind::dc: return "dc";
        case FixtureKind::nyquist: return "nyquist";
    }
    return "unknown";
}

std::vector<double> makeFixture(const Workload& workload, FixtureKind fixture, std::uint64_t seed) {
    validateWorkload(workload);
    std::vector<double> values(workload.realElements(), 0.0);
    std::mt19937_64 generator(seed);
    std::normal_distribution<double> distribution;

    for (std::size_t field = 0; field < workload.fields; ++field) {
        for (std::size_t z = 0; z < workload.nz; ++z) {
            const auto plane = z + workload.nz * field;
            const double amplitude = 1.0 + 0.125 * static_cast<double>(plane);
            if (fixture == FixtureKind::impulse) {
                const auto x = (1 + plane) % workload.nx;
                const auto y = (2 + 3 * plane) % workload.ny;
                values[realIndex(workload, x, y, z, field)] = amplitude;
                continue;
            }
            for (std::size_t y = 0; y < workload.ny; ++y) {
                for (std::size_t x = 0; x < workload.nx; ++x) {
                    double value = 0.0;
                    switch (fixture) {
                        case FixtureKind::sinusoid: {
                            const double phase = 0.17 * static_cast<double>(plane);
                            value = amplitude * std::cos(2.0 * pi * (static_cast<double>(x) / static_cast<double>(workload.nx) + 2.0 * static_cast<double>(y) / static_cast<double>(workload.ny)) + phase);
                            break;
                        }
                        case FixtureKind::random:
                            value = distribution(generator);
                            break;
                        case FixtureKind::dc:
                            value = amplitude;
                            break;
                        case FixtureKind::nyquist:
                            value = amplitude * (((x + y) % 2 == 0) ? 1.0 : -1.0);
                            break;
                        case FixtureKind::impulse:
                            break;
                    }
                    values[realIndex(workload, x, y, z, field)] = value;
                }
            }
        }
    }
    return values;
}

void directR2C(const Workload& workload, const double* input, Complex* wvmSpectrum) {
    validateWorkload(workload);
    for (std::size_t field = 0; field < workload.fields; ++field) {
        for (std::size_t z = 0; z < workload.nz; ++z) {
            for (std::size_t ky = 0; ky < workload.ny; ++ky) {
                for (std::size_t kx = 0; kx < workload.nxHalf(); ++kx) {
                    Complex sum;
                    for (std::size_t y = 0; y < workload.ny; ++y) {
                        for (std::size_t x = 0; x < workload.nx; ++x) {
                            const double angle = -2.0 * pi *
                                (static_cast<double>(kx * x) / static_cast<double>(workload.nx) +
                                 static_cast<double>(ky * y) / static_cast<double>(workload.ny));
                            const Complex phase{std::cos(angle), std::sin(angle)};
                            sum = add(sum, scale(phase, input[realIndex(workload, x, y, z, field)]));
                        }
                    }
                    wvmSpectrum[wvmSpectrumIndex(workload, kx, ky, z, field)] = sum;
                }
            }
        }
    }
}

void directC2R(const Workload& workload, const Complex* wvmSpectrum, double* output) {
    validateWorkload(workload);
    for (std::size_t field = 0; field < workload.fields; ++field) {
        for (std::size_t z = 0; z < workload.nz; ++z) {
            for (std::size_t y = 0; y < workload.ny; ++y) {
                for (std::size_t x = 0; x < workload.nx; ++x) {
                    Complex sum;
                    for (std::size_t ky = 0; ky < workload.ny; ++ky) {
                        for (std::size_t kx = 0; kx < workload.nx; ++kx) {
                            std::size_t storedKx = kx;
                            std::size_t storedKy = ky;
                            bool shouldConjugate = false;
                            if (kx > workload.nx / 2) {
                                storedKx = workload.nx - kx;
                                storedKy = (workload.ny - ky) % workload.ny;
                                shouldConjugate = true;
                            }
                            auto coefficient = wvmSpectrum[wvmSpectrumIndex(workload, storedKx, storedKy, z, field)];
                            if (shouldConjugate) coefficient = conjugate(coefficient);
                            const double angle = 2.0 * pi *
                                (static_cast<double>(kx * x) / static_cast<double>(workload.nx) +
                                 static_cast<double>(ky * y) / static_cast<double>(workload.ny));
                            sum = add(sum, multiply(coefficient, {std::cos(angle), std::sin(angle)}));
                        }
                    }
                    output[realIndex(workload, x, y, z, field)] = sum.real;
                }
            }
        }
    }
}

double maximumRelativeError(const Complex* actual, const Complex* expected, std::size_t count) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        numerator = std::max(numerator, magnitude({actual[index].real - expected[index].real, actual[index].imag - expected[index].imag}));
        denominator = std::max(denominator, magnitude(expected[index]));
    }
    return numerator / std::max(denominator, 1.0);
}

double maximumRelativeError(const double* actual, const double* expected, std::size_t count, double actualScale) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        numerator = std::max(numerator, std::abs(actualScale * actual[index] - expected[index]));
        denominator = std::max(denominator, std::abs(expected[index]));
    }
    return numerator / std::max(denominator, 1.0);
}

double relativeL2Error(const Complex* actual, const Complex* expected, std::size_t count) {
    long double squaredError = 0.0;
    long double squaredReference = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const long double realError = static_cast<long double>(actual[index].real) - expected[index].real;
        const long double imaginaryError = static_cast<long double>(actual[index].imag) - expected[index].imag;
        squaredError += realError * realError + imaginaryError * imaginaryError;
        const long double expectedReal = expected[index].real;
        const long double expectedImaginary = expected[index].imag;
        squaredReference += expectedReal * expectedReal + expectedImaginary * expectedImaginary;
    }
    if (squaredReference == 0.0) return std::sqrt(static_cast<double>(squaredError));
    return std::sqrt(static_cast<double>(squaredError / squaredReference));
}

std::string_view stageStateName(StageState state) noexcept {
    switch (state) {
        case StageState::executed: return "executed";
        case StageState::fused: return "fused";
        case StageState::elided: return "elided";
        case StageState::setupOnly: return "setup-only";
        case StageState::unsupported: return "unsupported";
    }
    return "unsupported";
}

} // namespace skbench
