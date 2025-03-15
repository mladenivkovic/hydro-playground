/**
 * @file RiemannHLLC.h
 * @brief The HLLC Riemann solver.
 */

#pragma once

#include "RiemannBase.h"

namespace riemann {


  /**
   * The HLLC Riemann solver.
   */
  class RiemannHLLC : public RiemannBase {

    protected:


  public:

    RiemannHLLC(idealGas::PrimitiveState& l, idealGas::PrimitiveState& r) : RiemannBase(l, r){};
    ~RiemannHLLC() = default;

    //! Call the actual solver.
    idealGas::PrimitiveState solve() override{
      message("Solving HLLC.");
      return {};
    }

  };

} // namespace riemann

