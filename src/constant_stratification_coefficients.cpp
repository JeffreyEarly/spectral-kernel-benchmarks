#include "constant_stratification_coefficients.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace skbench {
namespace {

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double matlabModalNormalizationGravity = 9.81;

Complex add(Complex first, Complex second) noexcept {
    return {first.real + second.real, first.imag + second.imag};
}

Complex subtract(Complex first, Complex second) noexcept {
    return {first.real - second.real, first.imag - second.imag};
}

Complex multiply(Complex first, Complex second) noexcept {
    return {
        first.real * second.real - first.imag * second.imag,
        first.real * second.imag + first.imag * second.real};
}

Complex multiply(Complex value, double scale) noexcept {
    return {value.real * scale, value.imag * scale};
}

Complex phase(double angle) noexcept {
    return {std::cos(angle), std::sin(angle)};
}

std::size_t modeIndex(const ConstantStratificationModeTable& table,
                      std::size_t mode, std::size_t j) noexcept {
    return j + table.nj * mode;
}

std::size_t coefficientIndex(const ConstantStratificationModeTable& table,
                             std::size_t mode, std::size_t j,
                             std::size_t field) noexcept {
    return j + table.nj * field + table.nj * 3 * mode;
}

struct EvolvedCoefficients {
    Complex ap;
    Complex am;
    Complex a0;
};

EvolvedCoefficients evolvedCoefficients(
    const ConstantStratificationModeTable& table,
    const std::vector<Complex>& state, const std::vector<Complex>& phases,
    std::size_t mode, std::size_t j) noexcept {
    const auto index = modeIndex(table, mode, j);
    return {
        multiply(state[coefficientIndex(table, mode, j, 0)], phases[index]),
        multiply(state[coefficientIndex(table, mode, j, 1)],
                 conjugate(phases[index])),
        state[coefficientIndex(table, mode, j, 2)]};
}

Complex fieldValue(const ConstantStratificationModeTable& table,
                   std::size_t index,
                   const EvolvedCoefficients& coefficients,
                   std::size_t target) noexcept {
    if (target == 0)
        return add(add(multiply(table.uApField[index], coefficients.ap),
                       multiply(conjugate(table.uApField[index]), coefficients.am)),
                   multiply(table.uA0Field[index], coefficients.a0));
    if (target == 1)
        return add(add(multiply(table.vApField[index], coefficients.ap),
                       multiply(conjugate(table.vApField[index]), coefficients.am)),
                   multiply(table.vA0Field[index], coefficients.a0));
    if (target == 2)
        return add(multiply(table.wApField[index], coefficients.ap),
                   multiply(table.wApField[index], coefficients.am));
    return add(add(multiply(coefficients.ap, table.nApField[index]),
                   multiply(coefficients.am, -table.nApField[index])),
               multiply(coefficients.a0, table.nA0Field[index]));
}

struct WaveCoefficientPair {
    Complex ap;
    Complex am;
};

WaveCoefficientPair waveCoefficientPair(Complex wave, Complex buoyancy) noexcept {
    return {add(wave, buoyancy), subtract(wave, buoyancy)};
}

WaveCoefficientPair inertialCoefficientPair(
    Complex value, double scale, std::size_t target) noexcept {
    Complex ap{};
    if (target == 0) ap = multiply(value, scale);
    if (target == 1) ap = multiply(multiply(value, Complex{0.0, -1.0}), scale);
    return {ap, conjugate(ap)};
}

template <typename Value>
std::uint64_t bytes(const std::vector<Value>& values) noexcept {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (values.capacity() > maximum / sizeof(Value)) return maximum;
    return static_cast<std::uint64_t>(values.capacity()) * sizeof(Value);
}

} // namespace

ConstantStratificationModeTable makeConstantStratificationModeTable(
    const ConstantStratificationFluxFixture& fixture) {
    if (fixture.modes.empty() || fixture.workload.retainedVerticalModes() == 0)
        throw std::invalid_argument(
            "Constant-stratification mode table requires retained modes.");
    ConstantStratificationModeTable table;
    table.nj = fixture.workload.retainedVerticalModes();
    const auto modeCount = fixture.modes.size();
    const auto coefficientCount = modeCount * table.nj;
    table.horizontal.reserve(modeCount);
    table.verticalWavenumber.resize(table.nj);
    table.omega.resize(coefficientCount);
    table.inertialScale.resize(table.nj);
    table.apmWProjectionPrefactor.resize(table.nj);
    table.uApField.resize(coefficientCount);
    table.vApField.resize(coefficientCount);
    table.wApField.resize(coefficientCount);
    table.nApField.resize(coefficientCount);
    table.uA0Field.resize(coefficientCount);
    table.vA0Field.resize(coefficientCount);
    table.nA0Field.resize(coefficientCount);
    table.a0FromVorticity.resize(coefficientCount);
    table.a0FromBuoyancy.resize(coefficientCount);
    table.apmDProjection.resize(coefficientCount);
    table.apmNProjection.resize(coefficientCount);
    table.apmDScaled.resize(coefficientCount);

    for (const auto& mode : fixture.modes) {
        const double k = 2.0 * pi * static_cast<double>(mode.k) /
            fixture.workload.lx;
        const double l = 2.0 * pi * static_cast<double>(mode.l) /
            fixture.workload.ly;
        const double kh = std::sqrt(k * k + l * l);
        table.horizontal.push_back({
            k, l, kh, kh == 0.0 ? 0.0 : k / kh,
            kh == 0.0 ? 0.0 : l / kh});
    }

    const double n02 = fixture.n0 * fixture.n0;
    const double coriolis = 2.0 * fixture.rotationRate *
        std::sin(fixture.latitude * pi / 180.0);
    const double f2 = coriolis * coriolis;
    if (n02 <= f2)
        throw std::invalid_argument(
            "Nonhydrostatic constant stratification requires N0 squared greater than f squared.");

    std::vector<double> h0(table.nj);
    std::vector<double> fg(table.nj);
    std::vector<double> gg(table.nj);
    std::vector<double> gWaveScale(table.nj);
    for (std::size_t j = 0; j < table.nj; ++j) {
        const double vertical = static_cast<double>(j) * pi / fixture.lz;
        const double signNorm = j % 2 == 0 ? 1.0 : -1.0;
        table.verticalWavenumber[j] = vertical;
        h0[j] = j == 0 ? fixture.lz : n02 /
            (fixture.gravity * vertical * vertical);
        fg[j] = j == 0 ? 2.0 : signNorm * h0[j] * vertical *
            std::sqrt(2.0 * matlabModalNormalizationGravity /
                      (fixture.lz * n02));
        gg[j] = j == 0 ? 1.0 : signNorm *
            std::sqrt(2.0 * matlabModalNormalizationGravity /
                      (fixture.lz * n02));
        const double gw = j == 0 ? gg[j] : signNorm *
            std::sqrt(2.0 * matlabModalNormalizationGravity /
                      (fixture.lz * (n02 - f2)));
        gWaveScale[j] = gg[j] / (gg[j] / gw);
    }

    for (std::size_t mode = 0; mode < modeCount; ++mode) {
        const auto& horizontal = table.horizontal[mode];
        const double kh2 = horizontal.kh * horizontal.kh;
        for (std::size_t j = 0; j < table.nj; ++j) {
            const auto index = modeIndex(table, mode, j);
            const double vertical = table.verticalWavenumber[j];
            const double signNorm = j % 2 == 0 ? 1.0 : -1.0;
            const bool isWave = horizontal.kh > 0.0 && j > 0;
            const bool isInertial = horizontal.kh == 0.0;
            const bool isGeostrophic = horizontal.kh > 0.0;
            const bool isMda = horizontal.kh == 0.0 && j > 0;
            double hpm = 1.0;
            if (j > 0)
                hpm = (n02 - f2) /
                    (fixture.gravity * (vertical * vertical + kh2));
            double fw = fg[j];
            if (j > 0)
                fw = signNorm * hpm * vertical *
                    std::sqrt(2.0 * matlabModalNormalizationGravity /
                              (fixture.lz * (n02 - f2)));
            const double gw = j == 0 ? gg[j] : signNorm *
                std::sqrt(2.0 * matlabModalNormalizationGravity /
                          (fixture.lz * (n02 - f2)));
            const double omega = std::sqrt(
                fixture.gravity * hpm * kh2 + f2);
            const double fwg = fg[j] / fw;
            const double gwg = gg[j] / gw;
            const double fWaveScale = fg[j] / fwg;
            table.omega[index] = omega;
            if (mode == 0) table.inertialScale[j] = 0.5 * fwg / fg[j];
            const double prefactor = signNorm * std::sqrt(
                fixture.gravity * fixture.lz / (2.0 * (n02 - f2)));
            table.apmDScaled[index] = (vertical / 2.0) * prefactor;
            if (mode == 0) table.apmWProjectionPrefactor[j] = prefactor;

            if (isWave) {
                const Complex uAp{
                    horizontal.cosAlpha,
                    -(coriolis / omega) * horizontal.sinAlpha};
                const Complex vAp{
                    horizontal.sinAlpha,
                    (coriolis / omega) * horizontal.cosAlpha};
                const Complex wAp{0.0, -horizontal.kh * hpm};
                const double nAp = -horizontal.kh * hpm / omega;
                const Complex apmD{0.0, -1.0 / (2.0 * horizontal.kh * hpm)};
                const double apmN = -omega / (2.0 * horizontal.kh * hpm);
                table.uApField[index] = multiply(uAp, fWaveScale);
                table.vApField[index] = multiply(vAp, fWaveScale);
                table.wApField[index] = multiply(wAp, gWaveScale[j]);
                table.nApField[index] = nAp * gWaveScale[j];
                const double deltaScale = h0[j] * gwg / fg[j];
                table.apmDProjection[index] = multiply(apmD, deltaScale);
                table.apmNProjection[index] = apmN * gwg / gg[j];
            } else if (isInertial) {
                table.uApField[index] = {fWaveScale, 0.0};
                table.vApField[index] = {0.0, fWaveScale};
            }

            if (isGeostrophic) {
                const double lr2Inverse = j == 0 ? 0.0 :
                    f2 / (fixture.gravity * h0[j]);
                const double denominator = kh2 + lr2Inverse;
                const Complex uA0{0.0, horizontal.l / denominator};
                const Complex vA0{0.0, -horizontal.k / denominator};
                const double nA0 = j == 0 ? 0.0 :
                    -(coriolis / fixture.gravity) / denominator;
                table.uA0Field[index] = multiply(uA0, fg[j]);
                table.vA0Field[index] = multiply(vA0, fg[j]);
                table.nA0Field[index] = nA0 * gg[j];
                table.a0FromVorticity[index] = 1.0 / fg[j];
                table.a0FromBuoyancy[index] = j == 0 ? 0.0 :
                    (-coriolis / h0[j]) / gg[j];
            } else if (isMda) {
                table.nA0Field[index] = gg[j];
                table.a0FromBuoyancy[index] = 1.0 / gg[j];
                table.a0FromVorticity[index] =
                    (f2 / (2.0 * h0[j])) / fg[j];
            }
        }
    }
    return table;
}

std::uint64_t constantStratificationModeTableBytes(
    const ConstantStratificationModeTable& table) noexcept {
    std::uint64_t result = bytes(table.horizontal) +
        bytes(table.verticalWavenumber) + bytes(table.omega) +
        bytes(table.inertialScale) + bytes(table.apmWProjectionPrefactor) +
        bytes(table.uApField) + bytes(table.vApField) +
        bytes(table.wApField) + bytes(table.nApField) +
        bytes(table.uA0Field) + bytes(table.vA0Field) +
        bytes(table.nA0Field) + bytes(table.a0FromVorticity) +
        bytes(table.a0FromBuoyancy) + bytes(table.apmDProjection) +
        bytes(table.apmNProjection) + bytes(table.apmDScaled);
    return result;
}

void evaluateConstantStratificationPhases(
    const ConstantStratificationModeTable& table, double elapsedTime,
    std::vector<Complex>& phases) {
    phases.resize(table.omega.size());
    for (std::size_t index = 0; index < table.omega.size(); ++index)
        phases[index] = phase(table.omega[index] * elapsedTime);
}

std::array<Complex, 3> constantStratificationVelocitySpectrum(
    const ConstantStratificationModeTable& table,
    const std::vector<Complex>& state, const std::vector<Complex>& phases,
    std::size_t mode, std::size_t j) {
    const auto index = modeIndex(table, mode, j);
    const auto coefficients = evolvedCoefficients(table, state, phases, mode, j);
    auto u = multiply(fieldValue(table, index, coefficients, 0), 0.5);
    auto v = multiply(fieldValue(table, index, coefficients, 1), 0.5);
    auto w = j == 0 ? Complex{} :
        multiply(fieldValue(table, index, coefficients, 2), 0.5);
    return {u, v, w};
}

std::array<Complex, 3> constantStratificationDerivativeSpectrum(
    const ConstantStratificationModeTable& table,
    const std::vector<Complex>& state, const std::vector<Complex>& phases,
    std::size_t mode, std::size_t j, std::size_t target,
    std::size_t nz) {
    const auto index = modeIndex(table, mode, j);
    const auto coefficients = evolvedCoefficients(table, state, phases, mode, j);
    const auto value = fieldValue(table, index, coefficients, target);
    const auto& horizontal = table.horizontal[mode];
    const bool cosine = target < 2;
    const bool endpoint = j == 0 || j + 1 == nz;
    auto x = multiply(value, Complex{0.0, horizontal.k});
    auto y = multiply(value, Complex{0.0, horizontal.l});
    auto z = multiply(value, cosine ? -table.verticalWavenumber[j] :
                      table.verticalWavenumber[j]);
    if (cosine) {
        x = multiply(x, 0.5);
        y = multiply(y, 0.5);
        z = endpoint ? Complex{} : multiply(z, 0.5);
    } else {
        x = endpoint ? Complex{} : multiply(x, 0.5);
        y = endpoint ? Complex{} : multiply(y, 0.5);
        z = multiply(z, 0.5);
    }
    return {x, y, z};
}

void accumulateConstantStratificationFluxTarget(
    const ConstantStratificationModeTable& table,
    const std::vector<Complex>& phases, std::vector<Complex>& flux,
    std::size_t mode, std::size_t j, std::size_t target,
    Complex transformedValue, double horizontalScale) {
    const auto index = modeIndex(table, mode, j);
    const auto value = multiply(transformedValue, horizontalScale);
    const auto& horizontal = table.horizontal[mode];
    Complex a0Contribution{};
    if (target == 0)
        a0Contribution = multiply(
            multiply(value, Complex{0.0, -horizontal.l}),
            table.a0FromVorticity[index]);
    if (target == 1)
        a0Contribution = multiply(
            multiply(value, Complex{0.0, horizontal.k}),
            table.a0FromVorticity[index]);
    if (target == 3)
        a0Contribution = multiply(value, table.a0FromBuoyancy[index]);
    const auto buoyancyValue = subtract(
        target == 3 ? value : Complex{},
        multiply(a0Contribution, table.nA0Field[index]));
    const auto buoyancyContribution = multiply(
        buoyancyValue, table.apmNProjection[index]);
    Complex waveContribution{};
    if (target == 0)
        waveContribution = multiply(
            value, horizontal.cosAlpha * table.apmDScaled[index]);
    if (target == 1)
        waveContribution = multiply(
            value, horizontal.sinAlpha * table.apmDScaled[index]);
    if (target == 2)
        waveContribution = multiply(
            value, Complex{0.0, (horizontal.kh / 2.0) *
                table.apmWProjectionPrefactor[j]});
    auto contribution = waveCoefficientPair(
        waveContribution, buoyancyContribution);
    if (mode == 0)
        contribution = inertialCoefficientPair(
            value, table.inertialScale[j], target);
    const auto fpIndex = coefficientIndex(table, mode, j, 0);
    const auto fmIndex = coefficientIndex(table, mode, j, 1);
    const auto f0Index = coefficientIndex(table, mode, j, 2);
    flux[fpIndex] = add(
        flux[fpIndex], multiply(contribution.ap, conjugate(phases[index])));
    flux[fmIndex] = add(
        flux[fmIndex], multiply(contribution.am, phases[index]));
    flux[f0Index] = add(flux[f0Index], a0Contribution);
}

} // namespace skbench
