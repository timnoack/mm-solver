#pragma once

#include "HYPRE_config.h"
#ifdef HYPRE_USING_CUDA

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

inline int CUDA_CALL(cudaError_t result) {
  if (result != cudaSuccess) {
    spdlog::error("CUDA Runtime Error: {}", cudaGetErrorString(result));
    std::exit(1);
  }
  return result;
}

#endif