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

    RiemannHLLC(idealGas::PrimitiveState& l, idealGas::PrimitiveState& r, const size_t dimension) : RiemannBase(l, r, dimension){};
    ~RiemannHLLC() = default;

    //! Call the actual solver.
    idealGas::PrimitiveState solve() override;

  };

} // namespace riemann

