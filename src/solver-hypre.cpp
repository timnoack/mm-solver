#include "solver-hypre.hpp"

#include <complex.h>

#include <fstream>
#include <ios>
#include <unordered_map>

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "cudaHelper.hpp"
#include "nlohmann/json.hpp"
#include "spdlog/spdlog.h"

using namespace std;
using namespace solver;

HypreSolver::~HypreSolver() = default;

void HypreSolver::initPartitioning(const COOMatrix &mat) {
  spdlog::info("Initializing partitioning");
  spdlog::info("Number of processes: {}", num_procs_);
  spdlog::info("My rank: {}", rank_);

  partitioning_.resize(num_procs_ + 1);
  for (int i = 0; i < num_procs_ + 1; i++) {
    partitioning_[i] = mat.nrows * i / num_procs_;
  }
}

void HypreSolver::setMatrixValues(HYPRE_IJMatrix &ij, const COOMatrix &coo) {
  // For each row, store the column indices and values in a map
  std::unordered_map<int, std::vector<HYPRE_BigInt>> cols_map;
  std::unordered_map<int, std::vector<HYPRE_Real>> vals_map;
  int nnz = 0;
  spdlog::debug("Translating matrix from coordinate format to maps");
  for (int k = 0; k < coo.rows.size(); ++k) {
    if (coo.rows[k] >= partitioning_[rank_] &&
        coo.rows[k] < partitioning_[rank_ + 1]) {
      double val = coo.vals[k];
      int col = coo.cols[k];
      int row = coo.rows[k];
      cols_map[row].push_back(col);
      vals_map[row].push_back(val);
      localRows_.insert(coo.rows[k]);
      nnz++;
    }
  }

  // Set matrix values
  spdlog::debug("Translating matrix from maps to CSR format");
  HYPRE_BigInt nrows = cols_map.size();
  std::vector<HYPRE_BigInt> ncols;
  std::vector<HYPRE_BigInt> rows;
  ncols.reserve(nrows);
  rows.reserve(nrows);

  std::vector<HYPRE_Real> values;
  std::vector<HYPRE_BigInt> cols;
  values.reserve(nnz);
  cols.reserve(nnz);

  for (auto &pair : cols_map) {
    rows.push_back(pair.first);
    ncols.push_back(pair.second.size());
    for (int i = 0; i < ncols.back(); ++i) {
      cols.push_back(pair.second[i]);
      values.push_back(vals_map[pair.first][i]);
    }
  }

#ifdef HYPRE_USING_CUDA
  // When using GPU, Hypre expects the arrays to be allocated on the device
  spdlog::debug("Copying matrix to GPU");
  // Allocate memory on the device
  HYPRE_BigInt *d_rows;
  HYPRE_BigInt *d_ncols;
  HYPRE_BigInt *d_cols;
  HYPRE_Real *d_values;
  CUDA_CALL(cudaMalloc((void **)&d_rows, nrows * sizeof(HYPRE_BigInt)));
  CUDA_CALL(cudaMalloc((void **)&d_ncols, nrows * sizeof(HYPRE_BigInt)));
  CUDA_CALL(cudaMalloc((void **)&d_cols, nnz * sizeof(HYPRE_BigInt)));
  CUDA_CALL(cudaMalloc((void **)&d_values, nnz * sizeof(HYPRE_Real)));

  // Copy the arrays to the device
  CUDA_CALL(cudaMemcpy(d_rows, rows.data(), nrows * sizeof(HYPRE_BigInt),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_ncols, ncols.data(), nrows * sizeof(HYPRE_BigInt),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_cols, cols.data(), nnz * sizeof(HYPRE_BigInt),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_values, values.data(), nnz * sizeof(HYPRE_Real),
                       cudaMemcpyHostToDevice));

  // Set the matrix values
  spdlog::debug("Setting matrix values on GPU");
  HYPRE_IJMatrixSetValues(ij, nrows, d_ncols, d_rows, d_cols, d_values);

  CUDA_CALL(cudaFree(d_rows));
  CUDA_CALL(cudaFree(d_ncols));
  CUDA_CALL(cudaFree(d_cols));
  CUDA_CALL(cudaFree(d_values));
#else
  // When using CPU, Hypre expects the arrays to be allocated on the host
  // So we can directly pass the pointers to the arrays
  spdlog::debug("Setting matrix values on CPU");
  HYPRE_IJMatrixSetValues(ij, nrows, ncols.data(), rows.data(), cols.data(),
                          values.data());
#endif
}

void HypreSolver::setVectorValues(HYPRE_IJVector &ij, const COOVector &coo) {
  // For each row, store the column indices and values in a map
  std::vector<HYPRE_BigInt> rows;
  std::vector<HYPRE_Real> vals;
  rows.reserve(coo.rows.size());
  vals.reserve(coo.rows.size());

  if (coo.rows.empty()) {
    // If no rows are given, set all values to zero
    for (auto row : localRows_) {
      rows.push_back(row);
      vals.push_back(0.0);
    }
  } else {
    for (int k = 0; k < coo.rows.size(); ++k) {
      if (coo.rows[k] >= partitioning_[rank_] &&
          coo.rows[k] < partitioning_[rank_ + 1]) {
        rows.push_back(coo.rows[k]);
        vals.push_back(coo.vals[k]);
      }
    }
  }

#ifdef HYPRE_USING_CUDA
  // When using GPU, Hypre expects the arrays to be allocated on the device
  spdlog::debug("Copying vector to GPU");
  // Allocate memory on the device
  HYPRE_BigInt *d_rows;
  HYPRE_Real *d_vals;
  CUDA_CALL(cudaMalloc((void **)&d_rows, rows.size() * sizeof(HYPRE_BigInt)));
  CUDA_CALL(cudaMalloc((void **)&d_vals, vals.size() * sizeof(HYPRE_Real)));

  // Copy the arrays to the device
  CUDA_CALL(cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(HYPRE_BigInt),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_vals, vals.data(), vals.size() * sizeof(HYPRE_Real),
                       cudaMemcpyHostToDevice));

  // Set the vector values
  spdlog::debug("Setting vector values on GPU");
  HYPRE_IJVectorSetValues(ij, rows.size(), d_rows, d_vals);

  CUDA_CALL(cudaFree(d_rows));
  CUDA_CALL(cudaFree(d_vals));
#else
  // When using CPU, Hypre expects the arrays to be allocated on the host
  // So we can directly pass the pointers to the arrays
  spdlog::debug("Setting vector values on CPU");
  HYPRE_IJVectorSetValues(ij, rows.size(), rows.data(), vals.data());
#endif
}

void HypreSolver::initMatrix(const COOMatrix &mat) {
  spdlog::info("Initializing ij Matrix");

  int ilower = partitioning_[rank_];
  int iupper = partitioning_[rank_ + 1] - 1;

  HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &A_);
  HYPRE_IJMatrixSetObjectType(
      A_, HYPRE_PARCSR);  // The only type supported by Hypre
  HYPRE_IJMatrixInitialize(A_);

  // Set matrix values
  setMatrixValues(A_, mat);

  spdlog::info("Assembling parcsr matrix");
  HYPRE_IJMatrixAssemble(A_);
  HYPRE_IJMatrixGetObject(A_, (void **)&par_A_);
}

void HypreSolver::initVectorx(const COOVector &x) {
  spdlog::info("Initializing ij vector x");
  int jlower = partitioning_[rank_];
  int jupper = partitioning_[rank_ + 1] - 1;
  int nvalues = jupper - jlower + 1;

  HYPRE_IJVectorCreate(comm, jlower, jupper, &x_);
  HYPRE_IJVectorSetObjectType(x_, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(x_);

  // Set vector values
  if (x.rows.size() > 0) {
    spdlog::info("Using given initial guess");
  } else {
    spdlog::info("Using zero initial guess");
  }
  setVectorValues(x_, x);

  HYPRE_IJVectorAssemble(x_);
  HYPRE_IJVectorGetObject(x_, (void **)&par_x_);
}

void HypreSolver::initVectorb(const COOVector &b) {
  spdlog::info("Initializing ij vector b");
  int jlower = partitioning_[rank_];
  int jupper = partitioning_[rank_ + 1] - 1;

  HYPRE_IJVectorCreate(comm, jlower, jupper, &b_);
  HYPRE_IJVectorSetObjectType(b_, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(b_);

  /* set vector values */
  if (b.rows.size() > 0) {
    spdlog::info("Using given RHS");
    setVectorValues(b_, b);
    HYPRE_IJVectorAssemble(b_);
    HYPRE_IJVectorGetObject(b_, (void **)&par_b_);
  } else {
    spdlog::info("Settings RHS to A * [1, 1, ...]");
    // Calculate b = A * [1, 1, ...] using matrix-vector multiplication with
    // hypre
    HYPRE_IJVectorGetObject(b_, (void **)&par_b_);

    // Create vector of ones
    HYPRE_IJVector ones;
    HYPRE_ParVector par_ones;
    HYPRE_IJVectorCreate(comm, jlower, jupper, &ones);
    HYPRE_IJVectorSetObjectType(ones, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ones);
    for (int i = 0; i < jupper - jlower + 1; ++i) {
      int nrows = 1;
      HYPRE_BigInt big_index = i + jlower;
      double value = 1.0;
      HYPRE_IJVectorSetValues(ones, nrows, &big_index, &value);
    }
    HYPRE_IJVectorAssemble(ones);
    HYPRE_IJVectorGetObject(ones, (void **)&par_ones);

    // Calculate b = A * [1, 1, ...]
    HYPRE_ParCSRMatrixMatvec(1.0, par_A_, par_ones, 0.0, par_b_);
  }
}

void HypreSolver::initSolver() {
  spdlog::info("Initializing solver");

  std::string solverType = config["type"];
  if (solverType == "PBiCGStab") {
    spdlog::info("Using PBiCGStab solver");
    /* Create Solver */
    HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver_);
    HYPRE_ParCSRBiCGSTABSetPrintLevel(solver_, 3);

    std::string preconditionerType =
        config["preconditioner"].value("type", "none");
    if (preconditionerType == "none") {
      spdlog::info("Using no preconditioner");
    } else if (preconditionerType == "ILU") {
      spdlog::info("Using ILU preconditioner");
      HYPRE_ILUCreate(&preconditioner_);
      HYPRE_ILUSetType(preconditioner_, 0);
      HYPRE_ILUSetMaxIter(preconditioner_, 1);
      HYPRE_ILUSetTol(preconditioner_, 0.0);
      HYPRE_ILUSetLocalReordering(preconditioner_, 1); /* 0: none, 1: RCM */
      // HYPRE_ILUSetPrintLevel(preconditioner_, 3);
      HYPRE_BiCGSTABSetPrecond(solver_, (HYPRE_PtrToSolverFcn)HYPRE_ILUSolve,
                               (HYPRE_PtrToSolverFcn)HYPRE_ILUSetup,
                               preconditioner_);
    } else {
      spdlog::error("Unknown preconditioner type {}", preconditionerType);
      exit(1);
    }

    double absTol = config.value("tolerance", 0.0);
    double relTol = config.value("relativeTolerance", 0.0);
    spdlog::info("Setting absolute tolerance to {}", absTol);
    HYPRE_ParCSRBiCGSTABSetAbsoluteTol(solver_, absTol);
    spdlog::info("Setting relative tolerance to {}", relTol);
    HYPRE_ParCSRBiCGSTABSetTol(solver_, relTol);

    int maxIter = config.value("maxIter", 1000000);
    spdlog::info("Setting max iterations to {}", maxIter);
    HYPRE_ParCSRBiCGSTABSetMaxIter(solver_, maxIter);

    HYPRE_ParCSRBiCGSTABSetup(solver_, par_A_, par_b_, par_x_);
  }
}

void HypreSolver::initHypre() {
  spdlog::info("Initializing Hypre");

#ifdef HYPRE_USING_CUDA
  spdlog::info("Using GPU");
  int device_count;
  CUDA_CALL(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    spdlog::error("No CUDA devices found");
    exit(1);
  }

  if (device_count != num_procs_) {
    spdlog::error(
        "Number of CUDA devices ({}) does not match number of processes ({})",
        device_count, num_procs_);
    exit(1);
  }

  spdlog::info("Number of CUDA devices: {}", device_count);

  CUDA_CALL(cudaSetDevice(rank_));
#else
  spdlog::info("Using CPU");
#endif

  HYPRE_Initialize();
#ifdef HYPRE_USING_CUDA
  HYPRE_DeviceInitialize();
#endif
}

void HypreSolver::solve(const COOMatrix &mat, const COOVector &x,
                        const COOVector &b) {
  spdlog::info("Solving with Hypre");

  initHypre();
  initPartitioning(mat);
  initMatrix(mat);
  initVectorb(b);
  initVectorx(x);
  initSolver();

  // HYPRE_ParVectorPrint(par_x_, "xinitial");
  // HYPRE_ParVectorPrint(par_b_, "b");

  spdlog::info("Starting solver");
  // Measure time
  double startTime = MPI_Wtime();
  HYPRE_ParCSRBiCGSTABSolve(solver_, par_A_, par_b_, par_x_);
  double endTime = MPI_Wtime();
  spdlog::info("Solver took {} seconds", endTime - startTime);
  spdlog::info("Solver finished");

  int numIterations;
  HYPRE_BiCGSTABGetNumIterations(solver_, &numIterations);
  spdlog::info("Number of iterations: {}", numIterations);

  double relNorm, absNorm;
  HYPRE_BiCGSTABGetFinalRelativeResidualNorm(solver_, &relNorm);
  spdlog::info("Final relative residual norm: {}", relNorm);

  // HYPRE_ParVectorPrint(par_x_, "xfinal");

  HYPRE_BiCGSTABDestroy(solver_);
  HYPRE_IJMatrixDestroy(A_);
  HYPRE_IJVectorDestroy(x_);
  HYPRE_IJVectorDestroy(b_);
}

HypreSolver::HypreSolver(const nlohmann::json &config) : Solver(config) {
  MPI_Comm_rank(comm, &rank_);
  MPI_Comm_size(comm, &num_procs_);
}