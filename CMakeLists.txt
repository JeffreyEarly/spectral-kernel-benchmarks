cmake_minimum_required(VERSION 3.24)

project(spectral-kernel-benchmarks VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
find_package(Threads REQUIRED)

option(SKBENCH_BUILD_TESTS "Build the focused correctness test executable." ON)
set(SKBENCH_FFTW_ROOT "" CACHE PATH "Root of a pinned FFTW 3.3.11 installation. Leave empty to build the pinned provider.")
set(SKBENCH_FFTW_BUILD_JOBS "12" CACHE STRING "Parallel jobs used when building the pinned FFTW provider.")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SkbenchFFTW)

execute_process(
    COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE SKBENCH_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT SKBENCH_GIT_COMMIT)
    set(SKBENCH_GIT_COMMIT "unknown")
endif()
execute_process(
    COMMAND git status --porcelain --untracked-files=normal
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE SKBENCH_GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(SKBENCH_GIT_STATUS)
    set(SKBENCH_GIT_DIRTY 1)
else()
    set(SKBENCH_GIT_DIRTY 0)
endif()

add_library(skbench_core
    src/benchmark.cpp
    src/fftw_provider.cpp
    src/pruned_fftw_provider.cpp
    src/json.cpp
    src/model.cpp
    src/vertical_gemm.cpp
    src/vdsp_provider.cpp)
target_compile_features(skbench_core PUBLIC cxx_std_20)
target_include_directories(skbench_core
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(skbench_core PRIVATE skbench_fftw Threads::Threads)
target_compile_definitions(skbench_core PRIVATE
    SKBENCH_FFTW_VERSION="3.3.11"
    SKBENCH_GIT_COMMIT="${SKBENCH_GIT_COMMIT}"
    SKBENCH_GIT_DIRTY=${SKBENCH_GIT_DIRTY}
    SKBENCH_BUILD_FLAGS="$<$<CONFIG:Release>:-O3\ -mcpu=native>")

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(skbench_core PRIVATE "${ACCELERATE_FRAMEWORK}")
    target_compile_definitions(skbench_core PRIVATE SKBENCH_HAVE_ACCELERATE=1)
else()
    target_compile_definitions(skbench_core PRIVATE SKBENCH_HAVE_ACCELERATE=0)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(skbench_core PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
        $<$<CONFIG:Release>:-O3;-mcpu=native>)
endif()

add_executable(skbench src/main.cpp)
target_compile_features(skbench PRIVATE cxx_std_20)
target_link_libraries(skbench PRIVATE skbench_core)

if(SKBENCH_BUILD_TESTS)
    add_library(skbench_allocation_tracker SHARED tests/allocation_tracker.cpp)
    target_compile_features(skbench_allocation_tracker PRIVATE cxx_std_20)
    target_include_directories(skbench_allocation_tracker PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/tests")

    add_executable(skbench_tests tests/skbench_tests.cpp)
    target_compile_features(skbench_tests PRIVATE cxx_std_20)
    target_link_libraries(skbench_tests PRIVATE skbench_core skbench_allocation_tracker skbench_fftw)
    add_test(NAME skbench-unit COMMAND skbench_tests)
    add_test(NAME skbench-cli-list COMMAND skbench list)
    add_test(NAME skbench-cli-validate COMMAND skbench validate --profile smoke)
    add_test(NAME skbench-cli-pruned-horizontal COMMAND skbench run --kernel pruned-horizontal --providers fftw --fftw-planning estimate --fftw-internal-workers 1 --fftw-outer-workers 2 --profile smoke --warmups 1 --samples 2)
    add_test(NAME skbench-cli-pruned-horizontal-split COMMAND skbench run --kernel pruned-horizontal --providers fftw --retained-representation split --fftw-planning estimate --fftw-internal-workers 1 --fftw-outer-workers 2 --profile smoke --warmups 1 --samples 2)
    add_test(NAME skbench-cli-retained-view COMMAND skbench run --kernel fft --providers fftw --retained-representation view --fftw-spectrum-order plane-major --fftw-planning estimate --fftw-internal-workers 1 --fftw-outer-workers 2 --profile smoke --warmups 1 --samples 2)
    add_test(NAME skbench-cli-fused-retained-split COMMAND skbench run --kernel fft --providers fftw --retained-representation split --fftw-spectrum-order plane-major --fftw-planning estimate --fftw-internal-workers 1 --fftw-outer-workers 2 --profile smoke --warmups 1 --samples 2)
    add_test(NAME skbench-cli-vertical-gemm COMMAND skbench run --kernel vertical-gemm --profile smoke --warmups 1 --samples 2)
    add_test(NAME skbench-cli-vertical-gemm-k2-grouped COMMAND skbench run --kernel vertical-gemm --vertical-gemm-family k2-grouped --profile smoke --warmups 1 --samples 2)
    add_test(NAME skbench-cli-vertical-gemm-k2-outer-static COMMAND skbench run --kernel vertical-gemm --vertical-gemm-family k2-grouped --vertical-gemm-schedule outer-static --vertical-gemm-outer-workers 2 --profile smoke --warmups 1 --samples 2)
    add_test(NAME skbench-cli-vertical-gemm-k2-outer-dynamic COMMAND skbench run --kernel vertical-gemm --vertical-gemm-family k2-grouped --vertical-gemm-schedule outer-dynamic --vertical-gemm-outer-workers 2 --profile smoke --warmups 1 --samples 2)
    add_test(NAME skbench-cli-ordering-packing COMMAND skbench run --kernel ordering-packing --vertical-gemm-family k2-grouped --vertical-gemm-schedule outer-dynamic --vertical-gemm-outer-workers 2 --profile smoke --warmups 1 --samples 2)
    add_test(NAME skbench-cli-spectral-boundary COMMAND skbench run --kernel spectral-boundary --boundary-policy plane-major-fused-split --vertical-gemm-family k2-grouped --vertical-gemm-schedule outer-dynamic --vertical-gemm-outer-workers 2 --fftw-planning estimate --fftw-internal-workers 1 --fftw-outer-workers 2 --profile smoke --warmups 1 --samples 2)
    add_test(NAME skbench-cli-spectral-pipeline COMMAND skbench run --kernel spectral-pipeline --boundary-policy plane-major-fused-split --vertical-gemm-family k2-grouped --vertical-gemm-schedule outer-dynamic --vertical-gemm-outer-workers 2 --fftw-planning estimate --fftw-internal-workers 1 --fftw-outer-workers 2 --profile smoke --warmups 1 --samples 2)
    add_test(NAME skbench-cli-streaming-pruned-pipeline COMMAND skbench run --kernel spectral-pipeline --boundary-policy streaming-pruned-compact-split --vertical-gemm-family k2-grouped --vertical-gemm-schedule outer-dynamic --vertical-gemm-outer-workers 2 --fftw-planning estimate --fftw-internal-workers 1 --fftw-outer-workers 2 --profile smoke --warmups 1 --samples 2)
    add_test(NAME skbench-cli-streaming-pruned-tiled-pipeline COMMAND skbench run --kernel spectral-pipeline --boundary-policy streaming-pruned-compact-split --streaming-tile-width 8 --vertical-gemm-family k2-grouped --vertical-gemm-schedule outer-dynamic --vertical-gemm-outer-workers 2 --fftw-planning estimate --fftw-internal-workers 1 --fftw-outer-workers 2 --profile smoke --warmups 1 --samples 2)
    set_tests_properties(skbench-cli-validate PROPERTIES TIMEOUT 120)
    set_tests_properties(skbench-cli-pruned-horizontal PROPERTIES TIMEOUT 120)
    set_tests_properties(skbench-cli-pruned-horizontal-split PROPERTIES TIMEOUT 120)
    set_tests_properties(skbench-cli-retained-view PROPERTIES TIMEOUT 120)
    set_tests_properties(skbench-cli-fused-retained-split PROPERTIES TIMEOUT 120)
    set_tests_properties(skbench-cli-vertical-gemm PROPERTIES TIMEOUT 120)
    set_tests_properties(skbench-cli-vertical-gemm-k2-grouped PROPERTIES TIMEOUT 120)
    set_tests_properties(skbench-cli-vertical-gemm-k2-outer-static PROPERTIES TIMEOUT 120 ENVIRONMENT "VECLIB_MAXIMUM_THREADS=1")
    set_tests_properties(skbench-cli-vertical-gemm-k2-outer-dynamic PROPERTIES TIMEOUT 120 ENVIRONMENT "VECLIB_MAXIMUM_THREADS=1")
    set_tests_properties(skbench-cli-ordering-packing PROPERTIES TIMEOUT 120 ENVIRONMENT "VECLIB_MAXIMUM_THREADS=1")
    set_tests_properties(skbench-cli-spectral-boundary PROPERTIES TIMEOUT 120 ENVIRONMENT "VECLIB_MAXIMUM_THREADS=1")
    set_tests_properties(skbench-cli-spectral-pipeline PROPERTIES TIMEOUT 120 ENVIRONMENT "VECLIB_MAXIMUM_THREADS=1")
    set_tests_properties(skbench-cli-streaming-pruned-pipeline PROPERTIES TIMEOUT 120 ENVIRONMENT "VECLIB_MAXIMUM_THREADS=1")
    set_tests_properties(skbench-cli-streaming-pruned-tiled-pipeline PROPERTIES TIMEOUT 120 ENVIRONMENT "VECLIB_MAXIMUM_THREADS=1")
endif()
