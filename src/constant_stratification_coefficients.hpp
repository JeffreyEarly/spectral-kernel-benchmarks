#pragma once

#include "skbench/skbench.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skbench {

struct ConstantStratificationHorizontalMode {
    double k = 0.0;
    double l = 0.0;
    double kh = 0.0;
    double cosAlpha = 0.0;
    double sinAlpha = 0.0;
};

struct ConstantStratificationModeTable {
    std::size_t nj = 0;
    std::vector<ConstantStratificationHorizontalMode> horizontal;
    std::vector<double> verticalWavenumber;
    std::vector<double> omega;
    std::vector<double> inertialScale;
    std::vector<double> apmWProjectionPrefactor;
    std::vector<Complex> uApField;
    std::vector<Complex> vApField;
    std::vector<Complex> wApField;
    std::vector<double> nApField;
    std::vector<Complex> uA0Field;
    std::vector<Complex> vA0Field;
    std::vector<double> nA0Field;
    std::vector<double> a0FromVorticity;
    std::vector<double> a0FromBuoyancy;
    std::vector<Complex> apmDProjection;
    std::vector<double> apmNProjection;
    std::vector<double> apmDScaled;
};

ConstantStratificationModeTable makeConstantStratificationModeTable(
    const ConstantStratificationFluxFixture& fixture);

std::uint64_t constantStratificationModeTableBytes(
    const ConstantStratificationModeTable& table) noexcept;

void evaluateConstantStratificationPhases(
    const ConstantStratificationModeTable& table, double elapsedTime,
    std::vector<Complex>& phases);

std::array<Complex, 3> constantStratificationVelocitySpectrum(
    const ConstantStratificationModeTable& table,
    const std::vector<Complex>& state, const std::vector<Complex>& phases,
    std::size_t mode, std::size_t j);

std::array<Complex, 3> constantStratificationDerivativeSpectrum(
    const ConstantStratificationModeTable& table,
    const std::vector<Complex>& state, const std::vector<Complex>& phases,
    std::size_t mode, std::size_t j, std::size_t target,
    std::size_t nz);

void accumulateConstantStratificationFluxTarget(
    const ConstantStratificationModeTable& table,
    const std::vector<Complex>& phases, std::vector<Complex>& flux,
    std::size_t mode, std::size_t j, std::size_t target,
    Complex transformedValue, double horizontalScale);

} // namespace skbench
