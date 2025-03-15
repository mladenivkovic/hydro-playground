/**
 * @file RiemannBase.h
 * @brief Base class for Riemann solvers.
 */

#pragma once

#include "RiemannBase.h"

namespace riemann {


  class RiemannExact : public RiemannBase {

    protected:


  public:

    RiemannExact(idealGas::PrimitiveState& l, idealGas::PrimitiveState& r) : RiemannBase(l, r){};
    ~RiemannExact() = default;

    //! Call the actual solver.
    idealGas::PrimitiveState solve() override{
      message("Solving Exact.");
      return {};
    }

  };

} // namespace riemann

