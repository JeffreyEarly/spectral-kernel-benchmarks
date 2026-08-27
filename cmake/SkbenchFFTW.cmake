include(ExternalProject)

set(_skbench_fftw_source_url "https://fftw.org/pub/fftw/fftw-3.3.11.tar.gz")
set(_skbench_fftw_source_sha256 "5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1")

if(SKBENCH_FFTW_ROOT)
    find_path(_skbench_fftw_include fftw3.h
        PATHS "${SKBENCH_FFTW_ROOT}/include"
        NO_DEFAULT_PATH REQUIRED)
    find_library(_skbench_fftw_base
        NAMES fftw3 libfftw3.3.dylib
        PATHS "${SKBENCH_FFTW_ROOT}/lib"
        NO_DEFAULT_PATH REQUIRED)
    find_library(_skbench_fftw_threads
        NAMES fftw3_threads libfftw3_threads.3.dylib
        PATHS "${SKBENCH_FFTW_ROOT}/lib"
        NO_DEFAULT_PATH REQUIRED)

    add_library(skbench_fftw_base UNKNOWN IMPORTED)
    set_target_properties(skbench_fftw_base PROPERTIES
        IMPORTED_LOCATION "${_skbench_fftw_base}"
        INTERFACE_INCLUDE_DIRECTORIES "${_skbench_fftw_include}")
    add_library(skbench_fftw_threads UNKNOWN IMPORTED)
    set_target_properties(skbench_fftw_threads PROPERTIES
        IMPORTED_LOCATION "${_skbench_fftw_threads}")

    add_library(skbench_fftw INTERFACE)
    target_link_libraries(skbench_fftw INTERFACE skbench_fftw_threads skbench_fftw_base)
else()
    if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        message(FATAL_ERROR "The pinned automatic FFTW build currently supports Apple Silicon only. Set SKBENCH_FFTW_ROOT for another platform.")
    endif()

    execute_process(
        COMMAND /usr/bin/xcrun --show-sdk-path
        OUTPUT_VARIABLE _skbench_macos_sdk
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
        COMMAND /usr/bin/xcrun --find clang
        OUTPUT_VARIABLE _skbench_clang
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)

    set(_skbench_fftw_prefix "${CMAKE_BINARY_DIR}/_deps/fftw")
    set(_skbench_fftw_install "${_skbench_fftw_prefix}/install")
    set(_skbench_fftw_cflags "-O3 -mcpu=native -mmacosx-version-min=13.3 -isysroot ${_skbench_macos_sdk}")
    set(_skbench_fftw_ldflags "-mmacosx-version-min=13.3 -isysroot ${_skbench_macos_sdk} -Wl,-headerpad_max_install_names")
    file(MAKE_DIRECTORY "${_skbench_fftw_install}/include" "${_skbench_fftw_install}/lib")

    ExternalProject_Add(skbench_fftw_external
        URL "${_skbench_fftw_source_url}"
        URL_HASH "SHA256=${_skbench_fftw_source_sha256}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PREFIX "${_skbench_fftw_prefix}"
        CONFIGURE_COMMAND
            "${CMAKE_COMMAND}" -E env
            "SDKROOT=${_skbench_macos_sdk}"
            "MACOSX_DEPLOYMENT_TARGET=13.3"
            "CC=${_skbench_clang}"
            "CFLAGS=${_skbench_fftw_cflags}"
            "LDFLAGS=${_skbench_fftw_ldflags}"
            <SOURCE_DIR>/configure
            "--prefix=${_skbench_fftw_install}"
            --host=aarch64-apple-darwin
            --enable-neon
            --enable-threads
            --disable-fortran
            --disable-openmp
            --enable-shared
            --disable-static
        BUILD_COMMAND
            "${CMAKE_COMMAND}" -E env
            "SDKROOT=${_skbench_macos_sdk}"
            "MACOSX_DEPLOYMENT_TARGET=13.3"
            /usr/bin/make "-j${SKBENCH_FFTW_BUILD_JOBS}"
        INSTALL_COMMAND
            "${CMAKE_COMMAND}" -E env
            "SDKROOT=${_skbench_macos_sdk}"
            "MACOSX_DEPLOYMENT_TARGET=13.3"
            /usr/bin/make install
        TEST_COMMAND
            "${CMAKE_COMMAND}" -E env
            "SDKROOT=${_skbench_macos_sdk}"
            "MACOSX_DEPLOYMENT_TARGET=13.3"
            /usr/bin/make check
        TEST_BEFORE_INSTALL TRUE
        BUILD_BYPRODUCTS
            "${_skbench_fftw_install}/lib/libfftw3.3.dylib"
            "${_skbench_fftw_install}/lib/libfftw3_threads.3.dylib")

    add_library(skbench_fftw_base SHARED IMPORTED)
    set_target_properties(skbench_fftw_base PROPERTIES
        IMPORTED_LOCATION "${_skbench_fftw_install}/lib/libfftw3.3.dylib"
        INTERFACE_INCLUDE_DIRECTORIES "${_skbench_fftw_install}/include")
    add_library(skbench_fftw_threads SHARED IMPORTED)
    set_target_properties(skbench_fftw_threads PROPERTIES
        IMPORTED_LOCATION "${_skbench_fftw_install}/lib/libfftw3_threads.3.dylib")

    add_library(skbench_fftw INTERFACE)
    target_link_libraries(skbench_fftw INTERFACE skbench_fftw_threads skbench_fftw_base)
    add_dependencies(skbench_fftw skbench_fftw_external)
endif()
