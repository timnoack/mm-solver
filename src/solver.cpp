#include "solver.hpp"

#include "solver-hypre.hpp"

using namespace solver;

Solver::~Solver() = default;

std::unique_ptr<Solver> Solver::create(std::string framework,
                                       const nlohmann::json& config) {
  if (framework == "hypre") {
    return std::make_unique<HypreSolver>(config);
  } else {
    throw std::runtime_error("Unknown solver framework " + framework);
  }
}