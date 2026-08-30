#include "skbench/skbench.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace skbench {
namespace {

constexpr std::array<char, 8> preparedMagic{
    'S', 'K', 'C', 'F', 'P', '0', '0', '1'};
constexpr std::uint32_t preparedVersion = 1;
constexpr std::uint32_t endianMarker = UINT32_C(0x01020304);
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr std::string_view auditedWvmCommit =
    "6ad254fb9756ac918bb72e036020d004879df1f2";
constexpr std::string_view normalizationId =
    "raw horizontal FFT; inverse type-I factors placed in coefficient "
    "assembly; no explicit pointwise scale; forward type-I divided by "
    "Nz-1; modal projection includes 1/(Nx*Ny)";
constexpr std::string_view modeMappingId =
    "logical (k,l,j,coefficient); WVM radial magnitude then k then l; "
    "prepared payload reordered to skbench radial mode order; j fastest";
constexpr std::string_view coefficientContractId =
    "WVM constant-stratification natural-dimensional-prescaled "
    "nonhydrostatic nonlinear flux with phase and inertial/MDA exceptions";

std::size_t checkedProduct(std::size_t left, std::size_t right,
                           const char* label) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
        throw std::overflow_error(std::string(label) + " size overflows size_t.");
    return left * right;
}

class Reader {
public:
    explicit Reader(const std::filesystem::path& path)
        : path_(path), stream_(path_, std::ios::binary | std::ios::ate) {
        if (!stream_)
            throw std::runtime_error(
                "Unable to open prepared constant-stratification fixture: " +
                path_.string());
        const auto end = stream_.tellg();
        if (end < 0)
            throw std::runtime_error(
                "Unable to size prepared constant-stratification fixture: " +
                path_.string());
        size_ = static_cast<std::size_t>(end);
        stream_.seekg(0);
    }

    template <typename Value>
    Value value(const char* label) {
        static_assert(std::is_trivially_copyable_v<Value>);
        Value result{};
        read(&result, sizeof(Value), label);
        return result;
    }

    template <typename Value>
    std::vector<Value> values(std::size_t count, const char* label) {
        static_assert(std::is_trivially_copyable_v<Value>);
        const auto bytes = checkedProduct(count, sizeof(Value), label);
        std::vector<Value> result(count);
        read(result.data(), bytes, label);
        return result;
    }

    std::string string(const char* label) {
        const auto length = value<std::uint64_t>(label);
        if (length > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()))
            throw std::runtime_error(std::string(label) + " length exceeds size_t.");
        std::string result(static_cast<std::size_t>(length), '\0');
        read(result.data(), result.size(), label);
        if (result.find('\0') != std::string::npos)
            throw std::runtime_error(std::string(label) + " contains NUL bytes.");
        return result;
    }

    void requireFinished() const {
        if (offset_ != size_)
            throw std::runtime_error(
                "Prepared constant-stratification fixture contains trailing bytes.");
    }

private:
    void read(void* destination, std::size_t count, const char* label) {
        if (count > size_ - offset_)
            throw std::runtime_error(
                std::string("Prepared constant-stratification fixture is truncated at ") +
                label + '.');
        auto* output = static_cast<char*>(destination);
        std::size_t remaining = count;
        constexpr auto maximumRead = static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max());
        while (remaining != 0) {
            const auto chunk = std::min(remaining, maximumRead);
            stream_.read(output, static_cast<std::streamsize>(chunk));
            if (!stream_)
                throw std::runtime_error(
                    std::string("Unable to read prepared constant-stratification fixture at ") +
                    label + '.');
            output += chunk;
            remaining -= chunk;
            offset_ += chunk;
        }
    }

    std::filesystem::path path_;
    std::ifstream stream_;
    std::size_t size_ = 0;
    std::size_t offset_ = 0;
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::size_t dimension(std::uint64_t value, const char* label) {
    if (value == 0 || value > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error(std::string("Invalid fixture dimension ") + label + '.');
    return static_cast<std::size_t>(value);
}

bool sha256(std::string_view value) {
    return value.size() == 64 && std::all_of(
        value.begin(), value.end(), [](char character) {
            return (character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f');
        });
}

void requireFinite(const std::vector<Complex>& values, const char* label) {
    if (!std::all_of(values.begin(), values.end(), [](Complex value) {
            return std::isfinite(value.real) && std::isfinite(value.imag);
        }))
        throw std::runtime_error(std::string(label) + " contains a non-finite value.");
}

} // namespace

ConstantStratificationFluxFixture
loadPreparedConstantStratificationFluxFixture(
    const std::filesystem::path& path) {
    static_assert(sizeof(Complex) == 2 * sizeof(double));
    if constexpr (std::endian::native != std::endian::little)
        throw std::runtime_error(
            "Prepared constant-stratification fixtures require a little-endian host.");

    Reader reader(path);
    const auto magic = reader.values<char>(preparedMagic.size(), "magic");
    require(std::equal(magic.begin(), magic.end(), preparedMagic.begin()),
            "Prepared constant-stratification fixture has the wrong magic.");
    require(reader.value<std::uint32_t>("version") == preparedVersion,
            "Prepared constant-stratification fixture has an unsupported version.");
    require(reader.value<std::uint32_t>("endian marker") == endianMarker,
            "Prepared constant-stratification fixture has the wrong byte order.");
    require(reader.value<std::uint32_t>("authoritative flag") == 1,
            "Prepared constant-stratification fixture is not authoritative.");
    require(reader.value<std::uint32_t>("reserved header word") == 0,
            "Prepared constant-stratification fixture reserved word is nonzero.");

    ConstantStratificationFluxFixture fixture;
    fixture.workload.nx = dimension(reader.value<std::uint64_t>("Nx"), "Nx");
    fixture.workload.ny = dimension(reader.value<std::uint64_t>("Ny"), "Ny");
    fixture.workload.nz = dimension(reader.value<std::uint64_t>("Nz"), "Nz");
    const auto nkl = dimension(reader.value<std::uint64_t>("Nkl"), "Nkl");
    const auto nj = dimension(reader.value<std::uint64_t>("Nj"), "Nj");
    const auto coefficientCount = dimension(
        reader.value<std::uint64_t>("coefficient count"), "coefficient count");
    require(coefficientCount == 3,
            "Constant-stratification fixture must contain Ap, Am, and A0.");
    fixture.workload.fields = 4;
    fixture.workload.antialias = true;
    fixture.workload.lx = reader.value<double>("Lx");
    fixture.workload.ly = reader.value<double>("Ly");
    fixture.lz = reader.value<double>("Lz");
    fixture.n0 = reader.value<double>("N0");
    fixture.rotationRate = reader.value<double>("rotation rate");
    fixture.latitude = reader.value<double>("latitude");
    fixture.gravity = reader.value<double>("gravity");
    fixture.elapsedTime = reader.value<double>("elapsed time");
    fixture.pointwiseScale = reader.value<double>("pointwise scale");
    fixture.oracleMaximumScaleNormalizedError =
        reader.value<double>("oracle maximum scale-normalized error");
    fixture.oracleRelativeL2Error =
        reader.value<double>("oracle relative L2 error");

    require(fixture.workload.nx == fixture.workload.ny &&
                fixture.workload.nx % 2 == 0,
            "Constant-stratification fixture requires an even square grid.");
    require(fixture.workload.lx == fixture.workload.ly,
            "Constant-stratification fixture requires a square domain.");
    require(nj == fixture.workload.retainedVerticalModes(),
            "Constant-stratification fixture Nj violates two-thirds retention.");
    require(std::isfinite(fixture.workload.lx) && fixture.workload.lx > 0.0 &&
                std::isfinite(fixture.lz) && fixture.lz > 0.0 &&
                std::isfinite(fixture.n0) && fixture.n0 > 0.0 &&
                std::isfinite(fixture.rotationRate) &&
                std::isfinite(fixture.latitude) &&
                std::isfinite(fixture.gravity) && fixture.gravity > 0.0 &&
                std::isfinite(fixture.elapsedTime) &&
                std::isfinite(fixture.pointwiseScale) &&
                fixture.pointwiseScale > 0.0,
            "Constant-stratification fixture has invalid physical metadata.");
    require(fixture.pointwiseScale == 1.0,
            "Constant-stratification fixture pointwise scale is inconsistent.");
    const auto coriolis = 2.0 * fixture.rotationRate *
        std::sin(fixture.latitude * pi / 180.0);
    require(fixture.n0 * fixture.n0 > coriolis * coriolis,
            "Constant-stratification fixture violates N0 squared greater than f squared.");
    require(fixture.oracleMaximumScaleNormalizedError <= 1.0e-12 &&
                fixture.oracleRelativeL2Error <= 1.0e-12,
            "Constant-stratification fixture WVM backend cross-check failed.");

    fixture.fixtureId = reader.string("fixture id");
    fixture.waveVortexModelRepository = reader.string("WVM repository");
    fixture.waveVortexModelCommit = reader.string("WVM commit");
    fixture.generatorIdentity = reader.string("generator identity");
    fixture.fixtureHash = reader.string("fixture hash");
    fixture.normalization = reader.string("normalization");
    fixture.modeMapping = reader.string("mode mapping");
    fixture.coefficientContract = reader.string("coefficient contract");
    fixture.compiledModuleSha256 = reader.string("compiled module SHA-256");
    require(!fixture.fixtureId.empty(),
            "Constant-stratification fixture id is empty.");
    require(fixture.waveVortexModelRepository ==
                "JeffreyEarly/wave-vortex-model",
            "Constant-stratification fixture identifies the wrong WVM repository.");
    require(fixture.waveVortexModelCommit == auditedWvmCommit,
            "Constant-stratification fixture identifies the wrong WVM commit.");
    require(fixture.fixtureHash.starts_with("sha256:") &&
                fixture.fixtureHash.size() == 71,
            "Constant-stratification fixture identity is not SHA-256.");
    require(fixture.normalization == normalizationId,
            "Constant-stratification fixture normalization is inconsistent.");
    require(fixture.modeMapping == modeMappingId,
            "Constant-stratification fixture mode mapping is inconsistent.");
    require(fixture.coefficientContract == coefficientContractId,
            "Constant-stratification fixture coefficient contract is inconsistent.");
    require(sha256(fixture.compiledModuleSha256),
            "Constant-stratification fixture module identity is not SHA-256.");

    const auto rawKeys = reader.values<std::int32_t>(
        checkedProduct(2, nkl, "mode keys"), "mode keys");
    fixture.modes = retainedHorizontalModes(fixture.workload);
    require(fixture.modes.size() == nkl,
            "Constant-stratification fixture retained mode count is inconsistent.");
    for (std::size_t mode = 0; mode < nkl; ++mode) {
        require(rawKeys[2 * mode] == fixture.modes[mode].k &&
                    rawKeys[2 * mode + 1] == fixture.modes[mode].l,
                "Constant-stratification fixture mode keys are not canonical.");
    }

    const auto coefficientElements = checkedProduct(
        checkedProduct(nkl, nj, "modal rows"), coefficientCount,
        "modal coefficients");
    fixture.modalState = reader.values<Complex>(
        coefficientElements, "modal state");
    fixture.expectedModalFlux = reader.values<Complex>(
        coefficientElements, "expected modal flux");
    reader.requireFinished();
    requireFinite(fixture.modalState, "Constant-stratification modal state");
    requireFinite(fixture.expectedModalFlux,
                  "Constant-stratification expected modal flux");
    for (std::size_t j = 0; j < nj; ++j) {
        const auto ap = fixture.modalState[j];
        const auto am = fixture.modalState[j + nj];
        const auto a0 = fixture.modalState[j + 2 * nj];
        require(am.real == ap.real && am.imag == -ap.imag,
                "Constant-stratification fixture DC Am is not conjugate(Ap).");
        require(a0.imag == 0.0,
                "Constant-stratification fixture DC A0 is not real.");
    }
    return fixture;
}

} // namespace skbench
