/**
 * @file RiemannBase.h
 * @brief Base class for Riemann solvers.
 */

#pragma once

#include "Config.h"
#include "Gas.h"
#include "Logging.h"


namespace riemann {


  class RiemannBase {

  protected:
    //! The left state
    idealGas::PrimitiveState& _left;

    //! The right state
    idealGas::PrimitiveState& _right;

    //! Star state pressure
    Float _pstar;

    //! Star state velocity
    Float _vstar;

    //! In which dimension/direction to solve the problem.
    size_t _dim;

    //! Do we have vacuum generating conditions?
    bool hasVacuum();

    //! Get the vacuum solution
    idealGas::PrimitiveState solveVacuum();


  public:
    RiemannBase(idealGas::PrimitiveState& l, idealGas::PrimitiveState& r, const size_t dimension):
      _left(l),
      _right(r),
      _dim(dimension) {};
    ~RiemannBase() = default;

    //! Call the actual solver.
    virtual idealGas::PrimitiveState solve() {
      error("Shouldn't be called.");
      return {};
    }
  };

} // namespace riemann
