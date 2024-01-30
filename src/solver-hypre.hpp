#pragma once

#include <mpi.h>

#include <set>
#include <vector>

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "nlohmann/json_fwd.hpp"
#include "solver.hpp"

namespace solver {
class HypreSolver : public Solver {
 private:
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank_;
  int num_procs_;

  HYPRE_IJVector x_;
  HYPRE_IJVector b_;

  HYPRE_ParVector par_x_;
  HYPRE_ParVector par_b_;

  HYPRE_IJMatrix A_;
  HYPRE_ParCSRMatrix par_A_;

  HYPRE_Solver solver_, preconditioner_;

  std::vector<int> partitioning_;
  std::set<int> localRows_;  // row indices of rows on this process

  void initHypre();
  void sortMatrix(const COOMatrix& mat);
  void initPartitioning(const COOMatrix& mat);
  void initMatrix(const COOMatrix& mat);
  void initVectorb(const COOVector& b);
  void initVectorx(const COOVector& x);
  void initSolver();

  void setMatrixValues(HYPRE_IJMatrix& ij, const COOMatrix& coo);
  void setVectorValues(HYPRE_IJVector& ij, const COOVector* coo = nullptr,
                       double value = 0.0);

 public:
  HypreSolver(const nlohmann::json& config);
  virtual ~HypreSolver();

  void solve(const COOMatrix& mat, const COOVector& x,
             const COOVector& b) override;
};
}  // namespace solver