#include "skbench/skbench.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace skbench {
namespace {

constexpr std::array<char, 8> preparedMagic{
    'S', 'K', 'F', 'X', 'P', '0', '0', '1'};
constexpr std::uint32_t preparedVersion = 1;
constexpr std::uint32_t endianMarker = UINT32_C(0x01020304);

std::size_t checkedProduct(std::size_t left, std::size_t right,
                           const char* label) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string(label) + " size overflows size_t.");
    }
    return left * right;
}

class Reader {
public:
    explicit Reader(const std::filesystem::path& path) : path_(path) {
        std::ifstream stream(path_, std::ios::binary | std::ios::ate);
        if (!stream) {
            throw std::runtime_error(
                "Unable to open prepared spectral-flux fixture: " + path_.string());
        }
        const auto end = stream.tellg();
        if (end < 0) {
            throw std::runtime_error(
                "Unable to size prepared spectral-flux fixture: " + path_.string());
        }
        bytes_.resize(static_cast<std::size_t>(end));
        stream.seekg(0);
        stream.read(reinterpret_cast<char*>(bytes_.data()),
                    static_cast<std::streamsize>(bytes_.size()));
        if (!stream) {
            throw std::runtime_error(
                "Unable to read prepared spectral-flux fixture: " + path_.string());
        }
    }

    template <typename Value>
    Value value(const char* label) {
        static_assert(std::is_trivially_copyable_v<Value>);
        require(sizeof(Value), label);
        Value result{};
        std::memcpy(&result, bytes_.data() + offset_, sizeof(Value));
        offset_ += sizeof(Value);
        return result;
    }

    template <typename Value>
    std::vector<Value> values(std::size_t count, const char* label) {
        static_assert(std::is_trivially_copyable_v<Value>);
        const auto byteCount = checkedProduct(count, sizeof(Value), label);
        require(byteCount, label);
        std::vector<Value> result(count);
        std::memcpy(result.data(), bytes_.data() + offset_, byteCount);
        offset_ += byteCount;
        return result;
    }

    std::string string(const char* label) {
        const auto length = value<std::uint64_t>(label);
        if (length > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error(std::string(label) + " length exceeds size_t.");
        }
        require(static_cast<std::size_t>(length), label);
        const auto* first = reinterpret_cast<const char*>(bytes_.data() + offset_);
        std::string result(first, first + static_cast<std::ptrdiff_t>(length));
        offset_ += static_cast<std::size_t>(length);
        if (result.find('\0') != std::string::npos) {
            throw std::runtime_error(std::string(label) + " contains NUL bytes.");
        }
        return result;
    }

    void requireFinished() const {
        if (offset_ != bytes_.size()) {
            throw std::runtime_error(
                "Prepared spectral-flux fixture contains trailing bytes.");
        }
    }

private:
    void require(std::size_t count, const char* label) const {
        if (count > bytes_.size() - offset_) {
            throw std::runtime_error(
                std::string("Prepared spectral-flux fixture is truncated at ") + label + '.');
        }
    }

    std::filesystem::path path_;
    std::vector<std::byte> bytes_;
    std::size_t offset_ = 0;
};

std::size_t checkedDimension(std::uint64_t value, const char* label) {
    if (value == 0 || value > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string("Invalid fixture dimension ") + label + '.');
    }
    return static_cast<std::size_t>(value);
}

void requireText(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void requireFinite(const std::vector<double>& values, const char* label) {
    if (!std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
        })) {
        throw std::runtime_error(std::string(label) + " contains a non-finite value.");
    }
}

void requireFinite(const std::vector<Complex>& values, const char* label) {
    if (!std::all_of(values.begin(), values.end(), [](Complex value) {
            return std::isfinite(value.real) && std::isfinite(value.imag);
        })) {
        throw std::runtime_error(std::string(label) + " contains a non-finite value.");
    }
}

} // namespace

SpectralFluxFixture loadPreparedSpectralFluxFixture(
    const std::filesystem::path& path) {
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error(
            "Prepared spectral-flux fixtures currently require a little-endian host.");
    }

    Reader reader(path);
    const auto magic = reader.values<char>(preparedMagic.size(), "magic");
    requireText(std::equal(magic.begin(), magic.end(), preparedMagic.begin()),
                "Prepared spectral-flux fixture has the wrong magic.");
    requireText(reader.value<std::uint32_t>("version") == preparedVersion,
                "Prepared spectral-flux fixture has an unsupported version.");
    requireText(reader.value<std::uint32_t>("endian marker") == endianMarker,
                "Prepared spectral-flux fixture has the wrong byte order.");
    requireText(reader.value<std::uint32_t>("authoritative flag") == 1,
                "Prepared spectral-flux fixture is not authoritative.");
    requireText(reader.value<std::uint32_t>("reserved header word") == 0,
                "Prepared spectral-flux fixture reserved header word is nonzero.");

    SpectralFluxFixture fixture;
    fixture.workload.nx = checkedDimension(reader.value<std::uint64_t>("Nx"), "Nx");
    fixture.workload.ny = checkedDimension(reader.value<std::uint64_t>("Ny"), "Ny");
    fixture.workload.nz = checkedDimension(reader.value<std::uint64_t>("Nz"), "Nz");
    const auto nkl = checkedDimension(reader.value<std::uint64_t>("Nkl"), "Nkl");
    const auto nj = checkedDimension(reader.value<std::uint64_t>("Nj"), "Nj");
    const auto groupCount = checkedDimension(
        reader.value<std::uint64_t>("group count"), "group count");
    const auto familyCount = checkedDimension(
        reader.value<std::uint64_t>("family count"), "family count");
    const auto inputCount = checkedDimension(
        reader.value<std::uint64_t>("input count"), "input count");
    const auto targetCount = checkedDimension(
        reader.value<std::uint64_t>("target count"), "target count");
    requireText(familyCount == 2, "Spectral-flux fixture must contain wave-f and wave-g.");
    requireText(inputCount == 15, "Spectral-flux fixture must contain 15 input fields.");
    requireText(targetCount == 4, "Spectral-flux fixture must contain four targets.");
    fixture.workload.fields = targetCount;
    fixture.workload.antialias = true;
    fixture.workload.lx = reader.value<double>("Lx");
    fixture.workload.ly = reader.value<double>("Ly");
    fixture.lz = reader.value<double>("Lz");
    fixture.latitude = reader.value<double>("latitude");
    fixture.pointwiseScale = reader.value<double>("pointwise scale");
    requireText(fixture.workload.nx == fixture.workload.ny,
                "Authoritative pilot requires a square horizontal grid.");
    requireText(fixture.workload.nx % 2 == 0 && fixture.workload.ny % 2 == 0,
                "Authoritative pilot requires even horizontal dimensions.");
    requireText(fixture.workload.lx == fixture.workload.ly,
                "Authoritative pilot requires a square horizontal domain.");
    requireText(std::isfinite(fixture.workload.lx) && fixture.workload.lx > 0.0 &&
                    std::isfinite(fixture.lz) && fixture.lz > 0.0 &&
                    std::isfinite(fixture.latitude) &&
                    std::isfinite(fixture.pointwiseScale) &&
                    fixture.pointwiseScale > 0.0,
                "Spectral-flux fixture contains invalid physical metadata.");
    const auto horizontalElements = checkedProduct(
        fixture.workload.nx, fixture.workload.ny, "horizontal grid");
    const auto expectedScale = 1.0 / static_cast<double>(
        checkedProduct(horizontalElements, horizontalElements,
                       "pointwise normalization"));
    requireText(fixture.pointwiseScale == expectedScale,
                "Spectral-flux fixture pointwise normalization is inconsistent.");
    requireText(nj == fixture.workload.retainedVerticalModes(),
                "Spectral-flux fixture Nj violates floor(2*(Nz-1)/3).");

    fixture.fixtureId = reader.string("fixture id");
    fixture.waveVortexModelRepository = reader.string("WVM repository");
    fixture.waveVortexModelCommit = reader.string("WVM commit");
    fixture.generatorIdentity = reader.string("generator identity");
    fixture.fixtureHash = reader.string("fixture hash");
    fixture.normalization = reader.string("normalization");
    fixture.modeMapping = reader.string("mode mapping");
    fixture.derivativeConvention = reader.string("derivative convention");
    requireText(!fixture.fixtureId.empty(), "Spectral-flux fixture id is empty.");
    requireText(fixture.waveVortexModelRepository == "JeffreyEarly/wave-vortex-model",
                "Spectral-flux fixture identifies the wrong WVM repository.");
    requireText(fixture.waveVortexModelCommit.size() == 40,
                "Spectral-flux fixture WVM commit is not a full object id.");
    requireText(fixture.fixtureHash.starts_with("sha256:") &&
                    fixture.fixtureHash.size() == 71,
                "Spectral-flux fixture identity is not SHA-256.");

    const auto rawModeKeys = reader.values<std::int32_t>(
        checkedProduct(2, nkl, "mode keys"), "mode keys");
    const auto verticalKeys = reader.values<std::int32_t>(nj, "vertical mode keys");
    fixture.modeGroupIndices = reader.values<std::uint32_t>(
        nkl, "mode group indices");
    fixture.groupKeys = reader.values<std::uint64_t>(groupCount, "group keys");
    fixture.inputFieldFamilies = reader.values<std::uint32_t>(
        inputCount, "input field families");
    fixture.targetFieldFamilies = reader.values<std::uint32_t>(
        targetCount, "target field families");

    const auto matrixElements = checkedProduct(
        checkedProduct(fixture.workload.nz, nj, "operator matrix"),
        checkedProduct(familyCount, groupCount, "operator family"),
        "operator payload");
    fixture.inverseOperators = reader.values<double>(
        matrixElements, "inverse operators");
    fixture.forwardOperators = reader.values<double>(
        matrixElements, "forward operators");
    const auto inputElements = checkedProduct(
        checkedProduct(nj, inputCount, "modal input fields"), nkl,
        "modal inputs");
    const auto targetElements = checkedProduct(
        checkedProduct(nj, targetCount, "modal target fields"), nkl,
        "modal targets");
    fixture.modalInputs = reader.values<Complex>(inputElements, "modal inputs");
    fixture.expectedModalTargets = reader.values<Complex>(
        targetElements, "expected modal targets");
    reader.requireFinished();

    for (std::size_t j = 0; j < nj; ++j) {
        requireText(verticalKeys[j] == static_cast<std::int32_t>(j),
                    "Spectral-flux fixture vertical mode keys are not j=0..Nj-1.");
    }
    const auto expectedModes = retainedHorizontalModes(fixture.workload);
    requireText(expectedModes.size() == nkl,
                "Spectral-flux fixture retained horizontal mode count is inconsistent.");
    using ModeKey = std::pair<std::int64_t, std::int64_t>;
    std::map<ModeKey, std::size_t> sourceModeIndices;
    for (std::size_t mode = 0; mode < nkl; ++mode) {
        const ModeKey key{rawModeKeys[2 * mode], rawModeKeys[2 * mode + 1]};
        requireText(sourceModeIndices.emplace(key, mode).second,
                    "Spectral-flux fixture horizontal mode keys are not unique.");
    }
    std::vector<std::size_t> sourceForExpectedMode(nkl);
    for (std::size_t mode = 0; mode < nkl; ++mode) {
        const ModeKey key{expectedModes[mode].k, expectedModes[mode].l};
        const auto source = sourceModeIndices.find(key);
        requireText(source != sourceModeIndices.end(),
                    "Spectral-flux fixture is missing a retained horizontal mode key.");
        sourceForExpectedMode[mode] = source->second;
    }
    requireText(sourceModeIndices.size() == expectedModes.size(),
                "Spectral-flux fixture contains an unexpected horizontal mode key.");
    fixture.modes = expectedModes;
    std::vector<bool> usedGroups(groupCount, false);
    std::uint32_t previousGroup = 0;
    for (std::size_t sourceMode = 0; sourceMode < nkl; ++sourceMode) {
        const auto group = fixture.modeGroupIndices[sourceMode];
        requireText(group < groupCount,
                    "Spectral-flux fixture mode group index is out of range.");
        requireText(sourceMode == 0 || group >= previousGroup,
                    "Spectral-flux fixture mode groups are not contiguous.");
        previousGroup = group;
        usedGroups[group] = true;
        const auto k = static_cast<std::int64_t>(rawModeKeys[2 * sourceMode]);
        const auto l = static_cast<std::int64_t>(rawModeKeys[2 * sourceMode + 1]);
        const auto key = static_cast<std::uint64_t>(k * k + l * l);
        requireText(fixture.groupKeys[group] == key,
                    "Spectral-flux fixture group diagnostic key is inconsistent.");
    }
    requireText(std::all_of(usedGroups.begin(), usedGroups.end(),
                            [](bool used) { return used; }),
                "Spectral-flux fixture does not use every declared mode group.");
    requireText(std::is_sorted(fixture.groupKeys.begin(), fixture.groupKeys.end()),
                "Spectral-flux fixture group diagnostic keys are not nondecreasing.");
    const auto inputBlock = checkedProduct(nj, inputCount, "modal input mode block");
    const auto targetBlock = checkedProduct(nj, targetCount, "modal target mode block");
    std::vector<Complex> reorderedInputs(fixture.modalInputs.size());
    std::vector<Complex> reorderedTargets(fixture.expectedModalTargets.size());
    std::vector<std::uint32_t> reorderedGroups(nkl);
    for (std::size_t mode = 0; mode < nkl; ++mode) {
        const auto source = sourceForExpectedMode[mode];
        std::copy_n(fixture.modalInputs.begin() +
                        static_cast<std::ptrdiff_t>(source * inputBlock),
                    inputBlock,
                    reorderedInputs.begin() +
                        static_cast<std::ptrdiff_t>(mode * inputBlock));
        std::copy_n(fixture.expectedModalTargets.begin() +
                        static_cast<std::ptrdiff_t>(source * targetBlock),
                    targetBlock,
                    reorderedTargets.begin() +
                        static_cast<std::ptrdiff_t>(mode * targetBlock));
        reorderedGroups[mode] = fixture.modeGroupIndices[source];
    }
    fixture.modalInputs.swap(reorderedInputs);
    fixture.expectedModalTargets.swap(reorderedTargets);
    fixture.modeGroupIndices.swap(reorderedGroups);
    const std::vector<std::uint32_t> expectedInputFamilies{
        0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0};
    const std::vector<std::uint32_t> expectedTargetFamilies{0, 0, 1, 1};
    requireText(fixture.inputFieldFamilies == expectedInputFamilies,
                "Spectral-flux fixture input field-to-operator map is inconsistent.");
    requireText(fixture.targetFieldFamilies == expectedTargetFamilies,
                "Spectral-flux fixture target-to-operator map is inconsistent.");
    requireFinite(fixture.inverseOperators, "Inverse operator payload");
    requireFinite(fixture.forwardOperators, "Forward operator payload");
    requireFinite(fixture.modalInputs, "Modal input payload");
    requireFinite(fixture.expectedModalTargets, "Modal target payload");
    for (std::size_t field = 0; field < inputCount; ++field) {
        for (std::size_t j = 0; j < nj; ++j) {
            requireText(fixture.modalInputs[j + nj * field].imag == 0.0,
                        "Spectral-flux fixture DC modal inputs must be real.");
        }
    }
    return fixture;
}

GroupedVerticalOperators spectralFluxOperatorFamily(
    const SpectralFluxFixture& fixture, std::size_t family) {
    if (family >= 2) {
        throw std::invalid_argument("Spectral-flux operator family must be zero or one.");
    }
    std::vector<VerticalModeGroup> groups;
    std::vector<std::size_t> sourceGroups;
    for (std::size_t mode = 0; mode < fixture.modes.size(); ++mode) {
        const auto sourceGroup = static_cast<std::size_t>(
            fixture.modeGroupIndices[mode]);
        if (groups.empty() || sourceGroups.back() != sourceGroup) {
            groups.push_back({fixture.groupKeys[sourceGroup], mode, 1});
            sourceGroups.push_back(sourceGroup);
        } else {
            ++groups.back().modeCount;
        }
    }
    const auto nz = fixture.workload.nz;
    const auto nj = fixture.workload.retainedVerticalModes();
    const auto perGroup = checkedProduct(nz, nj, "operator matrix");
    GroupedVerticalOperators result;
    result.id = family == 0 ? "wvm-wave-f-floating-k2" :
                              "wvm-wave-g-floating-k2";
    result.nz = nz;
    result.nj = nj;
    result.groups = groups;
    result.inverse.resize(checkedProduct(groups.size(), perGroup,
                                         "inverse operator family"));
    result.forward.resize(result.inverse.size());
    for (std::size_t group = 0; group < groups.size(); ++group) {
        const auto sourceBase = perGroup * (family + 2 * sourceGroups[group]);
        const auto destinationBase = perGroup * group;
        for (std::size_t z = 0; z < nz; ++z) {
            for (std::size_t j = 0; j < nj; ++j) {
                result.inverse[destinationBase + z * nj + j] =
                    fixture.inverseOperators[sourceBase + z + nz * j];
                result.forward[destinationBase + j * nz + z] =
                    fixture.forwardOperators[sourceBase + j + nj * z];
            }
        }
    }
    return result;
}

} // namespace skbench
