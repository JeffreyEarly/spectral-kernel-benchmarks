#include "skbench/skbench.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <complex>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#if SKBENCH_HAVE_ACCELERATE
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

namespace skbench {
namespace {

using Clock = std::chrono::steady_clock;
#if SKBENCH_HAVE_ACCELERATE
using BlasComplex = __LAPACK_double_complex;
#else
using BlasComplex = std::complex<double>;
#endif
static_assert(sizeof(BlasComplex) == 2 * sizeof(double));

double elapsedSeconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::size_t checkedProduct(std::size_t left, std::size_t right, const char* label) {
    if (left != 0 && right > static_cast<std::size_t>(-1) / left) {
        throw std::overflow_error(std::string(label) + " size overflows size_t.");
    }
    return left * right;
}

int checkedBlasDimension(std::size_t value, const char* label) {
    if (value > static_cast<std::size_t>(INT_MAX)) {
        throw std::invalid_argument(std::string(label) + " exceeds the Accelerate BLAS integer range.");
    }
    return static_cast<int>(value);
}

template <typename Value>
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    ~AlignedBuffer() { std::free(data_); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    void allocate(std::size_t count) {
        if (data_ != nullptr) throw std::logic_error("Aligned buffer is already allocated.");
        if (count == 0) return;
        void* storage = nullptr;
        if (posix_memalign(&storage, 64, checkedProduct(count, sizeof(Value), "aligned buffer")) != 0 || storage == nullptr) {
            throw std::bad_alloc();
        }
        data_ = static_cast<Value*>(storage);
        count_ = count;
    }

    Value* data() noexcept { return data_; }
    const Value* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(Value); }

private:
    Value* data_ = nullptr;
    std::size_t count_ = 0;
};

} // namespace

std::string_view verticalGemmLayoutName(VerticalGemmLayout layout) noexcept {
    switch (layout) {
        case VerticalGemmLayout::complexInterleaved: return "complex-interleaved";
        case VerticalGemmLayout::split: return "split";
    }
    return "unknown";
}

struct VerticalGemmProvider::Impl {
    Workload workload;
    std::size_t horizontalModeCount = 0;
    std::size_t columnCount = 0;
    std::size_t physicalCount = 0;
    std::size_t modalCount = 0;
    VerticalGemmLayout layout = VerticalGemmLayout::complexInterleaved;
    bool available = SKBENCH_HAVE_ACCELERATE != 0;
    std::string capabilityText;
    double allocationTime = 0.0;
    double preparationTime = 0.0;

    AlignedBuffer<BlasComplex> complexForwardMatrix;
    AlignedBuffer<BlasComplex> complexInverseMatrix;
    AlignedBuffer<BlasComplex> complexPhysicalInput;
    AlignedBuffer<BlasComplex> complexModalInput;
    AlignedBuffer<BlasComplex> complexModalOutput;
    AlignedBuffer<BlasComplex> complexPhysicalOutput;

    AlignedBuffer<double> realForwardMatrix;
    AlignedBuffer<double> realInverseMatrix;
    AlignedBuffer<double> physicalInputReal;
    AlignedBuffer<double> physicalInputImaginary;
    AlignedBuffer<double> modalInputReal;
    AlignedBuffer<double> modalInputImaginary;
    AlignedBuffer<double> modalOutputReal;
    AlignedBuffer<double> modalOutputImaginary;
    AlignedBuffer<double> physicalOutputReal;
    AlignedBuffer<double> physicalOutputImaginary;

    int nz = 0;
    int nj = 0;
    int columns = 0;

    Impl(const Workload& inputWorkload, std::size_t inputHorizontalModeCount,
         const VerticalOperators& operators, VerticalGemmLayout inputLayout)
        : workload(inputWorkload), horizontalModeCount(inputHorizontalModeCount), layout(inputLayout) {
        if (operators.nz != workload.nz || operators.nj != workload.retainedVerticalModes()) {
            throw std::invalid_argument("Vertical GEMM operator dimensions do not match the workload.");
        }
        if (horizontalModeCount == 0 || workload.fields == 0) {
            throw std::invalid_argument("Vertical GEMM requires at least one horizontal mode and field.");
        }
        columnCount = checkedProduct(horizontalModeCount, workload.fields, "vertical GEMM column");
        physicalCount = checkedProduct(workload.nz, columnCount, "vertical GEMM physical operand");
        modalCount = checkedProduct(operators.nj, columnCount, "vertical GEMM modal operand");
        nz = checkedBlasDimension(workload.nz, "Nz");
        nj = checkedBlasDimension(operators.nj, "Nj");
        columns = checkedBlasDimension(columnCount, "K");

        if (!available) {
            capabilityText = "unsupported: Accelerate BLAS is available only on Apple platforms";
            return;
        }

        const auto allocationStart = Clock::now();
        if (layout == VerticalGemmLayout::complexInterleaved) {
            complexForwardMatrix.allocate(checkedProduct(operators.nj, operators.nz, "forward matrix"));
            complexInverseMatrix.allocate(checkedProduct(operators.nz, operators.nj, "inverse matrix"));
            complexPhysicalInput.allocate(physicalCount);
            complexModalInput.allocate(modalCount);
            complexModalOutput.allocate(modalCount);
            complexPhysicalOutput.allocate(physicalCount);
        } else {
            realForwardMatrix.allocate(checkedProduct(operators.nj, operators.nz, "forward matrix"));
            realInverseMatrix.allocate(checkedProduct(operators.nz, operators.nj, "inverse matrix"));
            physicalInputReal.allocate(physicalCount);
            physicalInputImaginary.allocate(physicalCount);
            modalInputReal.allocate(modalCount);
            modalInputImaginary.allocate(modalCount);
            modalOutputReal.allocate(modalCount);
            modalOutputImaginary.allocate(modalCount);
            physicalOutputReal.allocate(physicalCount);
            physicalOutputImaginary.allocate(physicalCount);
        }
        allocationTime = elapsedSeconds(allocationStart);

        const auto preparationStart = Clock::now();
        for (std::size_t z = 0; z < operators.nz; ++z) {
            for (std::size_t j = 0; j < operators.nj; ++j) {
                const double forwardValue = operators.forward[j * operators.nz + z];
                const double inverseValue = operators.inverse[z * operators.nj + j];
                const auto forwardIndex = j + operators.nj * z;
                const auto inverseIndex = z + operators.nz * j;
                if (layout == VerticalGemmLayout::complexInterleaved) {
                    complexForwardMatrix.data()[forwardIndex] = {forwardValue, 0.0};
                    complexInverseMatrix.data()[inverseIndex] = {inverseValue, 0.0};
                } else {
                    realForwardMatrix.data()[forwardIndex] = forwardValue;
                    realInverseMatrix.data()[inverseIndex] = inverseValue;
                }
            }
        }
        preparationTime = elapsedSeconds(preparationStart);
        capabilityText = "supported";
    }

    void requireAvailable() const {
        if (!available) throw std::runtime_error(capabilityText);
    }

    void requireSplit() const {
        requireAvailable();
        if (layout != VerticalGemmLayout::split) {
            throw std::logic_error("Split GEMM component requested from the complex GEMM provider.");
        }
    }

    std::size_t persistentBytes() const noexcept {
        return complexForwardMatrix.bytes() + complexInverseMatrix.bytes() +
            complexPhysicalInput.bytes() + complexModalInput.bytes() + complexModalOutput.bytes() +
            complexPhysicalOutput.bytes() + realForwardMatrix.bytes() + realInverseMatrix.bytes() +
            physicalInputReal.bytes() + physicalInputImaginary.bytes() + modalInputReal.bytes() +
            modalInputImaginary.bytes() + modalOutputReal.bytes() + modalOutputImaginary.bytes() +
            physicalOutputReal.bytes() + physicalOutputImaginary.bytes();
    }
};

VerticalGemmProvider::VerticalGemmProvider(const Workload& workload, std::size_t horizontalModeCount,
                                           const VerticalOperators& operators, VerticalGemmLayout layout)
    : impl_(std::make_unique<Impl>(workload, horizontalModeCount, operators, layout)) {}

VerticalGemmProvider::~VerticalGemmProvider() = default;
VerticalGemmProvider::VerticalGemmProvider(VerticalGemmProvider&&) noexcept = default;
VerticalGemmProvider& VerticalGemmProvider::operator=(VerticalGemmProvider&&) noexcept = default;

bool VerticalGemmProvider::supported() const noexcept { return impl_->available; }
std::string VerticalGemmProvider::capability() const { return impl_->capabilityText; }
VerticalGemmLayout VerticalGemmProvider::layout() const noexcept { return impl_->layout; }
std::size_t VerticalGemmProvider::columns() const noexcept { return impl_->columnCount; }
std::size_t VerticalGemmProvider::physicalElements() const noexcept { return impl_->physicalCount; }
std::size_t VerticalGemmProvider::modalElements() const noexcept { return impl_->modalCount; }
std::size_t VerticalGemmProvider::persistentBytes() const noexcept { return impl_->persistentBytes(); }
std::size_t VerticalGemmProvider::matrixBytesPerDirection() const noexcept {
    const auto scalarBytes = impl_->layout == VerticalGemmLayout::complexInterleaved ? sizeof(BlasComplex) : sizeof(double);
    return impl_->workload.nz * impl_->workload.retainedVerticalModes() * scalarBytes;
}
std::size_t VerticalGemmProvider::minimumAlignmentBytes() const noexcept { return 64; }
double VerticalGemmProvider::allocationSeconds() const noexcept { return impl_->allocationTime; }
double VerticalGemmProvider::matrixPreparationSeconds() const noexcept { return impl_->preparationTime; }
std::string VerticalGemmProvider::libraryIdentity() const {
#if SKBENCH_HAVE_ACCELERATE
    return "/System/Library/Frameworks/Accelerate.framework";
#else
    return "unavailable";
#endif
}

void VerticalGemmProvider::loadPhysicalInput(const Complex* input) {
    impl_->requireAvailable();
    if (impl_->layout == VerticalGemmLayout::complexInterleaved) {
        for (std::size_t index = 0; index < impl_->physicalCount; ++index) {
            impl_->complexPhysicalInput.data()[index] = {input[index].real, input[index].imag};
        }
        return;
    }
    for (std::size_t index = 0; index < impl_->physicalCount; ++index) {
        impl_->physicalInputReal.data()[index] = input[index].real;
        impl_->physicalInputImaginary.data()[index] = input[index].imag;
    }
}

void VerticalGemmProvider::loadModalInput(const Complex* input) {
    impl_->requireAvailable();
    if (impl_->layout == VerticalGemmLayout::complexInterleaved) {
        for (std::size_t index = 0; index < impl_->modalCount; ++index) {
            impl_->complexModalInput.data()[index] = {input[index].real, input[index].imag};
        }
        return;
    }
    for (std::size_t index = 0; index < impl_->modalCount; ++index) {
        impl_->modalInputReal.data()[index] = input[index].real;
        impl_->modalInputImaginary.data()[index] = input[index].imag;
    }
}

void VerticalGemmProvider::executeForward() {
    impl_->requireAvailable();
#if SKBENCH_HAVE_ACCELERATE
    if (impl_->layout == VerticalGemmLayout::complexInterleaved) {
        const BlasComplex alpha{1.0, 0.0};
        const BlasComplex beta{0.0, 0.0};
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    impl_->nj, impl_->columns, impl_->nz,
                    &alpha, impl_->complexForwardMatrix.data(), impl_->nj,
                    impl_->complexPhysicalInput.data(), impl_->nz,
                    &beta, impl_->complexModalOutput.data(), impl_->nj);
    } else {
        executeForwardReal();
        executeForwardImaginary();
    }
#endif
}

void VerticalGemmProvider::executeInverse() {
    impl_->requireAvailable();
#if SKBENCH_HAVE_ACCELERATE
    if (impl_->layout == VerticalGemmLayout::complexInterleaved) {
        const BlasComplex alpha{1.0, 0.0};
        const BlasComplex beta{0.0, 0.0};
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    impl_->nz, impl_->columns, impl_->nj,
                    &alpha, impl_->complexInverseMatrix.data(), impl_->nz,
                    impl_->complexModalInput.data(), impl_->nj,
                    &beta, impl_->complexPhysicalOutput.data(), impl_->nz);
    } else {
        executeInverseReal();
        executeInverseImaginary();
    }
#endif
}

void VerticalGemmProvider::executeForwardReal() {
    impl_->requireSplit();
#if SKBENCH_HAVE_ACCELERATE
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                impl_->nj, impl_->columns, impl_->nz, 1.0,
                impl_->realForwardMatrix.data(), impl_->nj,
                impl_->physicalInputReal.data(), impl_->nz, 0.0,
                impl_->modalOutputReal.data(), impl_->nj);
#endif
}

void VerticalGemmProvider::executeForwardImaginary() {
    impl_->requireSplit();
#if SKBENCH_HAVE_ACCELERATE
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                impl_->nj, impl_->columns, impl_->nz, 1.0,
                impl_->realForwardMatrix.data(), impl_->nj,
                impl_->physicalInputImaginary.data(), impl_->nz, 0.0,
                impl_->modalOutputImaginary.data(), impl_->nj);
#endif
}

void VerticalGemmProvider::executeInverseReal() {
    impl_->requireSplit();
#if SKBENCH_HAVE_ACCELERATE
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                impl_->nz, impl_->columns, impl_->nj, 1.0,
                impl_->realInverseMatrix.data(), impl_->nz,
                impl_->modalInputReal.data(), impl_->nj, 0.0,
                impl_->physicalOutputReal.data(), impl_->nz);
#endif
}

void VerticalGemmProvider::executeInverseImaginary() {
    impl_->requireSplit();
#if SKBENCH_HAVE_ACCELERATE
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                impl_->nz, impl_->columns, impl_->nj, 1.0,
                impl_->realInverseMatrix.data(), impl_->nz,
                impl_->modalInputImaginary.data(), impl_->nj, 0.0,
                impl_->physicalOutputImaginary.data(), impl_->nz);
#endif
}

void VerticalGemmProvider::copyForwardOutput(Complex* output) const {
    impl_->requireAvailable();
    if (impl_->layout == VerticalGemmLayout::complexInterleaved) {
        for (std::size_t index = 0; index < impl_->modalCount; ++index) {
            output[index] = {impl_->complexModalOutput.data()[index].real(),
                             impl_->complexModalOutput.data()[index].imag()};
        }
        return;
    }
    for (std::size_t index = 0; index < impl_->modalCount; ++index) {
        output[index] = {impl_->modalOutputReal.data()[index], impl_->modalOutputImaginary.data()[index]};
    }
}

void VerticalGemmProvider::copyInverseOutput(Complex* output) const {
    impl_->requireAvailable();
    if (impl_->layout == VerticalGemmLayout::complexInterleaved) {
        for (std::size_t index = 0; index < impl_->physicalCount; ++index) {
            output[index] = {impl_->complexPhysicalOutput.data()[index].real(),
                             impl_->complexPhysicalOutput.data()[index].imag()};
        }
        return;
    }
    for (std::size_t index = 0; index < impl_->physicalCount; ++index) {
        output[index] = {impl_->physicalOutputReal.data()[index], impl_->physicalOutputImaginary.data()[index]};
    }
}

} // namespace skbench
