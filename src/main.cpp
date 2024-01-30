#include <mpi.h>

#include <fstream>
#include <iostream>

#include "CLI/CLI.hpp"
#include "fast_matrix_market/fast_matrix_market.hpp"
#include "nlohmann/json.hpp"
#include "solver.hpp"
#include "spdlog/spdlog.h"

using namespace std;
using namespace std::filesystem;
using namespace nlohmann;
using namespace solver;

json config;

// Paths within the config file are relative to the config file's directory
path configDir;

json loadConfig(const path& configPath) {
  configDir = configPath.parent_path();
  ifstream configStream(configPath);
  json config = json::parse(configStream, nullptr, true, true);
  configStream.close();
  return config;
}

Solver::COOMatrix loadMatrix() {
  path matrixPath = configDir / config["solution"]["matrix"].get<string>();

  Solver::COOMatrix mat;

  // Only rank 0 will load the matrix
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    spdlog::info("Loading matrix from {}", matrixPath.string());

    ifstream matrixStream(matrixPath);
    if (!matrixStream.is_open()) {
      spdlog::error("Failed to open matrix file {}", matrixPath.string());
      exit(1);
    }

    fast_matrix_market::read_matrix_market_triplet(
        matrixStream, mat.nrows, mat.ncols, mat.rows, mat.cols, mat.vals);
    matrixStream.close();

    spdlog::info("Loaded matrix with {} rows and {} columns", mat.nrows,
                 mat.ncols);
  }

  // Broadcast or receive the matrix to/from all ranks
  MPI_Bcast(&mat.nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&mat.ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int nnz = mat.rows.size();
  MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  mat.rows.resize(nnz);
  mat.cols.resize(nnz);
  mat.vals.resize(nnz);
  MPI_Bcast(mat.rows.data(), mat.rows.size(), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(mat.cols.data(), mat.cols.size(), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(mat.vals.data(), mat.vals.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return mat;
}

Solver::COOVector loadVector(const path& vectorPath) {
  Solver::COOVector vec;

  // Only rank 0 will load the vector
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    spdlog::info("Loading vector from {}", vectorPath.string());

    ifstream vectorStream(vectorPath);
    if (!vectorStream.is_open()) {
      spdlog::error("Failed to open vector file {}", vectorPath.string());
      exit(1);
    }

    fast_matrix_market::read_matrix_market_doublet(vectorStream, vec.size,
                                                   vec.rows, vec.vals);
    vectorStream.close();

    spdlog::info("Loaded vector with {} rows", vec.size);
  }

  // Broadcast or receive the vector to/from all ranks
  MPI_Bcast(&vec.size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  vec.rows.resize(vec.size);
  vec.vals.resize(vec.size);
  MPI_Bcast(vec.rows.data(), vec.rows.size(), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(vec.vals.data(), vec.vals.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return vec;
}

int main(int argc, char** argv) {
  CLI::App app{"Sparse Solver Evaluation"};
  path configPath;
  std::string framework;
  app.add_option("-c,--config", configPath, "Path to config file")->required();
  app.add_option("-f,--framework", framework,
                 "Solver framework to use (hypre or amgx)")
      ->required();
  CLI11_PARSE(app, argc, argv);

  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Rank 0 will handle all logging
  // All other ranks with log errors only
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  spdlog::set_level(rank == 0 ? spdlog::level::trace : spdlog::level::err);
  spdlog::info("Initialized MPI with {} ranks", size);

  config = loadConfig(configPath);
  spdlog::info("Loaded config from {}", configPath.string());
  spdlog::info("Using framework {}", framework);

  // Load the matrix
  Solver::COOMatrix mat = loadMatrix();

  // Load x and b if they are specified in the config
  // By default, x is set to 0 and b is set to A * [1, 1, ...]
  Solver::COOVector b;
  if (config["solution"].contains("b")) {
    if (!config["solution"]["b"].is_string()) {
      spdlog::error("b must be a string");
      exit(1);
    }
    b = loadVector(configDir / config["solution"]["b"].get<string>());
  }
  Solver::COOVector x;
  if (config["solution"].contains("x0")) {
    if (!config["solution"]["x0"].is_string()) {
      spdlog::error("x0 must be a string");
      exit(1);
    }
    x = loadVector(configDir / config["solution"]["x0"].get<string>());
  }

  // Create the solver and solve the system
  auto solver = Solver::create(framework, config["solvers"]["x"]);
  solver->solve(mat, x, b);

  MPI_Finalize();

  return 0;
}