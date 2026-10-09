cmake_minimum_required(VERSION 3.24)
project(tie VERSION 0.1.0 LANGUAGES CXX)
enable_language(OBJCXX)

if(NOT (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64"))
  message(FATAL_ERROR "tie targets Apple Silicon macOS (arm64)")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)
FetchContent_Declare(doctest URL https://github.com/doctest/doctest/archive/refs/tags/v2.4.12.tar.gz)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
set(DOCTEST_NO_INSTALL ON)
set(JSON_Install OFF)
FetchContent_MakeAvailable(doctest json)

set(TIE_WARNINGS -Wall -Wextra -Wpedantic -Wshadow)
# apple-m1 is the baseline of every Apple Silicon Mac and enables the int8 dot-product instructions.
set(TIE_ARCH -mcpu=apple-m1)

# Metal shaders are compiled at runtime from source embedded in the binary.
# Order matters: headers first, then kernels.
set(TIE_METAL_SOURCES
  ${CMAKE_SOURCE_DIR}/src/kernels/metal/common.h
  ${CMAKE_SOURCE_DIR}/src/kernels/metal/ggml_args.h
  ${CMAKE_SOURCE_DIR}/src/kernels/metal/elementwise.metal
  ${CMAKE_SOURCE_DIR}/src/kernels/metal/rms_norm.metal
  ${CMAKE_SOURCE_DIR}/src/kernels/metal/rope.metal
)
string(REPLACE ";" "|" TIE_METAL_SOURCES_ARG "${TIE_METAL_SOURCES}")
set(TIE_METAL_EMBED ${CMAKE_BINARY_DIR}/generated/metal_source.cpp)
add_custom_command(
  OUTPUT ${TIE_METAL_EMBED}
  COMMAND ${CMAKE_COMMAND} -DOUTPUT=${TIE_METAL_EMBED} -DSOURCES=${TIE_METAL_SOURCES_ARG}
          -P ${CMAKE_SOURCE_DIR}/cmake/embed_metal.cmake
  DEPENDS ${TIE_METAL_SOURCES} ${CMAKE_SOURCE_DIR}/cmake/embed_metal.cmake
  COMMENT "Embedding Metal shader source"
  VERBATIM)

add_library(tie_core STATIC
  src/backend/cpu/cpu_backend.cpp
  src/backend/metal/metal_backend.mm
  ${TIE_METAL_EMBED}
  src/core/buffer.cpp
  src/core/dtype.cpp
  src/core/tensor.cpp
  src/core/thread_pool.cpp
  src/kernels/cpu/q8_0.cpp
  src/kernels/cpu/vec.cpp
  src/model/config.cpp
  src/model/gguf.cpp
  src/model/loader.cpp
  src/model/mapped_file.cpp
  src/model/safetensors.cpp
  src/tokenizer/chat_template.cpp
  src/tokenizer/pretokenize.cpp
  src/tokenizer/tokenizer.cpp
  src/tokenizer/unicode.cpp
  src/tokenizer/unicode_tables.cpp
)
target_include_directories(tie_core PUBLIC src)
target_compile_options(tie_core PRIVATE ${TIE_WARNINGS} ${TIE_ARCH})
set_source_files_properties(src/backend/metal/metal_backend.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
# The generated file is one long string literal; -Woverlength-strings does not apply to it.
set_source_files_properties(${TIE_METAL_EMBED} PROPERTIES COMPILE_OPTIONS "-w")
target_link_libraries(tie_core PUBLIC nlohmann_json::nlohmann_json "-framework CoreFoundation" "-framework Metal" "-framework Foundation")

enable_testing()
add_executable(tie_tests
  tests/test_main.cpp
  tests/backend/test_cpu_ops.cpp
  tests/backend/test_matmul.cpp
  tests/backend/test_metal_backend.cpp
  tests/backend/test_metal_ops.cpp
  tests/core/test_dtype.cpp
  tests/core/test_tensor.cpp
  tests/core/test_thread_pool.cpp
  tests/model/test_config.cpp
  tests/model/test_gguf.cpp
  tests/model/test_loader.cpp
  tests/model/test_safetensors.cpp
  tests/support/npy.cpp
  tests/support/test_data.cpp
  tests/support/test_npy.cpp
  tests/tokenizer/test_chat_template.cpp
  tests/tokenizer/test_pretokenize.cpp
  tests/tokenizer/test_tokenizer.cpp
)
target_include_directories(tie_tests PRIVATE tests)
target_compile_options(tie_tests PRIVATE ${TIE_WARNINGS} ${TIE_ARCH})
target_compile_definitions(tie_tests PRIVATE TIE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(tie_tests PRIVATE tie_core doctest::doctest)
add_test(NAME tie_tests COMMAND tie_tests)
