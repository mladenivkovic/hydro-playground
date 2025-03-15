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
      idealGas::PrimitiveState& left;

      //! The right state
      idealGas::PrimitiveState& right;

      //! Star state pressure
      Float pstar;

      //! Star state velocity
      Float vstar;

      //! Do we have vacuum generating conditions?
      bool hasVacuum(size_t dimension);


  public:

    RiemannBase(idealGas::PrimitiveState& l, idealGas::PrimitiveState& r) : left(l), right(r){};
    ~RiemannBase() = default;

    //! Call the actual solver.
    virtual idealGas::PrimitiveState solve(){
      error("Shouldn't be called.");
      return {};
    }

  };

} // namespace riemann

