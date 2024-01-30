cmake_minimum_required(VERSION 3.20)
project(sparse-solver-eval C CXX)

if (HYPRE_WITH_CUDA)
  enable_language(CUDA)
  set(HYPRE_CUDA_SM ${CUDA_ARCH} CACHE STRING "Target CUDA architecture." FORCE)
  find_package(CUDA REQUIRED)
  find_package(CUDAToolkit REQUIRED) 
endif()

find_package(MPI REQUIRED)

# Enable the address sanitizer for debug builds
# This checks for unsafe memory usage (buffer overflow etc)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Enabling address sanitizer")
  add_compile_options(-fno-omit-frame-pointer -fsanitize=address)
  add_link_options( -fno-omit-frame-pointer -fsanitize=address)
endif()

# add_subdirectory(libs/AMGX)
add_subdirectory(libs/hypre/src)
add_subdirectory(libs/fast_matrix_market)
add_subdirectory(libs/nlohman-json)
add_subdirectory(libs/spdlog)
add_subdirectory(libs/CLI11)

add_subdirectory(src)