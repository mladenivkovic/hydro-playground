/**
 * @file RiemannBase.h
 * @brief Base class for Riemann solvers.
 */

#pragma once

#include "Config.h"
#include "Gas.h"
#include "Logging.h"


class RiemannBase {

protected:
  //! The left state
  PrimitiveState& _left;

  //! The right state
  PrimitiveState& _right;

  //! Star state pressure
  Float _pstar;

  //! Star state velocity
  Float _vstar;

  //! In which dimension/direction to solve the problem.
  size_t _dim;

  //! Do we have vacuum generating conditions?
  bool hasVacuum();

  //! Get the vacuum solution
  PrimitiveState solveVacuum();

  //! Sample the solved Riemann problem.
  ConservedFlux sampleSolution();

public:
  RiemannBase(PrimitiveState& l, PrimitiveState& r, const size_t dimension):
    _left(l),
    _right(r),
    _dim(dimension) {};
  ~RiemannBase() = default;

  //! Call the actual solver.
  virtual ConservedFlux solve() {
    error("Shouldn't be called.");
    return {};
  }
};

