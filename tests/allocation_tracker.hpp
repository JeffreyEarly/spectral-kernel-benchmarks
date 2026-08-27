#pragma once

#include <cstdint>

namespace skbench::test {

bool allocationTrackingSupported() noexcept;
void beginAllocationTracking() noexcept;
std::uint64_t endAllocationTracking() noexcept;

} // namespace skbench::test
