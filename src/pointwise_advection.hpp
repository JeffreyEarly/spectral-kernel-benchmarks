#pragma once

#include <cstddef>
#include <memory>
#include <string_view>

namespace skbench {

enum class PointwiseAdvectionPolicy {
    serial,
    vectorSerial,
    spatialStatic,
};

PointwiseAdvectionPolicy pointwiseAdvectionPolicyNamed(std::string_view name);
const char* pointwiseAdvectionPolicyName(PointwiseAdvectionPolicy policy);

class PointwiseAdvectionExecutor {
public:
    PointwiseAdvectionExecutor(PointwiseAdvectionPolicy policy,
                               std::size_t workers,
                               std::size_t volumeElements,
                               double scale);
    ~PointwiseAdvectionExecutor();

    PointwiseAdvectionExecutor(const PointwiseAdvectionExecutor&) = delete;
    PointwiseAdvectionExecutor& operator=(const PointwiseAdvectionExecutor&) = delete;
    PointwiseAdvectionExecutor(PointwiseAdvectionExecutor&&) noexcept;
    PointwiseAdvectionExecutor& operator=(PointwiseAdvectionExecutor&&) noexcept;

    void execute(const double* shared, const double* derivative,
                 double* target);
    void executeSchedulerNoop();

    PointwiseAdvectionPolicy policy() const noexcept;
    std::size_t workers() const noexcept;
    std::size_t persistentBytes() const noexcept;
    double setupSeconds() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace skbench
