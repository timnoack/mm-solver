#pragma once

#include <memory>
#include <vector>

#include "nlohmann/json_fwd.hpp"

namespace solver {
class Solver {
 protected:
  const nlohmann::json& config;

 public:
  Solver() = delete;
  Solver(const nlohmann::json& config) : config(config) {}
  virtual ~Solver();

  struct COOMatrix {
    int nrows;
    int ncols;
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;
  };

  struct COOVector {
    int size;
    std::vector<int> rows;
    std::vector<double> vals;
  };

  virtual void solve(const COOMatrix& mat, const COOVector& x,
                     const COOVector& b) = 0;

  static std::unique_ptr<Solver> create(std::string framework,
                                        const nlohmann::json& config);
};
}  // namespace solver