add_executable(matrix-solver main.cpp solver.cpp solver-hypre.cpp)

target_compile_features(matrix-solver PUBLIC cxx_std_17)
target_link_libraries(matrix-solver stdc++fs)

target_link_libraries(matrix-solver MPI::MPI_CXX nlohmann_json::nlohmann_json fast_matrix_market spdlog::spdlog CLI11::CLI11)

target_link_libraries(matrix-solver HYPRE)
#target_link_libraries(matrix-solver amgxsh)

#target_include_directories(matrix-solver PUBLIC ${MPI_INCLUDE_PATH})
# target_include_directories(matrix "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# # Link against AMGX
# GET_FILENAME_COMPONENT(CMAKE_C_COMPILER_NAME "${CMAKE_C_COMPILER}" NAME)
# IF(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)
#   set(libs_all CUDA::cusparse CUDA::cusolver)
#   set(dyn_libs amgxsh CUDA::cudart_static CUDA::cublas)
# ELSE(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)
#   set(libs_all CUDA::cusparse CUDA::cusolver rt dl)
#   set(dyn_libs amgxsh rt dl CUDA::cudart_static CUDA::cublas)
# ENDIF(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)
# target_link_libraries(matrix-solver ${dyn_libs})