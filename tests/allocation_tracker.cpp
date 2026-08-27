#include "allocation_tracker.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstdint>

namespace {

std::atomic<bool> tracking = false;
std::atomic<std::uint64_t> allocationCount = 0;

void recordAllocation() noexcept {
    if (tracking.load(std::memory_order_relaxed)) {
        allocationCount.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace

namespace skbench::test {

bool allocationTrackingSupported() noexcept {
#if defined(__APPLE__)
    return true;
#else
    return false;
#endif
}

void beginAllocationTracking() noexcept {
    allocationCount.store(0, std::memory_order_relaxed);
    tracking.store(true, std::memory_order_seq_cst);
}

std::uint64_t endAllocationTracking() noexcept {
    tracking.store(false, std::memory_order_seq_cst);
    return allocationCount.load(std::memory_order_relaxed);
}

} // namespace skbench::test

#if defined(__APPLE__)

extern "C" void* skbenchTrackedMalloc(std::size_t size) {
    recordAllocation();
    return std::malloc(size);
}

extern "C" void* skbenchTrackedCalloc(std::size_t count, std::size_t size) {
    recordAllocation();
    return std::calloc(count, size);
}

extern "C" void* skbenchTrackedRealloc(void* pointer, std::size_t size) {
    recordAllocation();
    return std::realloc(pointer, size);
}

extern "C" void* skbenchTrackedValloc(std::size_t size) {
    recordAllocation();
    return valloc(size);
}

extern "C" void* skbenchTrackedAlignedAlloc(std::size_t alignment, std::size_t size) {
    recordAllocation();
    return aligned_alloc(alignment, size);
}

extern "C" int skbenchTrackedPosixMemalign(void** pointer, std::size_t alignment, std::size_t size) {
    recordAllocation();
    return posix_memalign(pointer, alignment, size);
}

#define SKBENCH_INTERPOSE(replacement, replacee) \
    __attribute__((used)) static struct { \
        const void* replacementFunction; \
        const void* replacedFunction; \
    } skbenchInterpose##replacee __attribute__((section("__DATA,__interpose"))) = { \
        reinterpret_cast<const void*>(&replacement), reinterpret_cast<const void*>(&replacee) \
    }

SKBENCH_INTERPOSE(skbenchTrackedMalloc, malloc);
SKBENCH_INTERPOSE(skbenchTrackedCalloc, calloc);
SKBENCH_INTERPOSE(skbenchTrackedRealloc, realloc);
SKBENCH_INTERPOSE(skbenchTrackedValloc, valloc);
SKBENCH_INTERPOSE(skbenchTrackedAlignedAlloc, aligned_alloc);
SKBENCH_INTERPOSE(skbenchTrackedPosixMemalign, posix_memalign);

#endif
