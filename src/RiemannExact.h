/**
 * @file RiemannExact.h
 * @brief The exact Riemann solver.
 */

#pragma once

#include "RiemannBase.h"

namespace riemann {

  /**
   * The Exact riemann solver.
   */
  class RiemannExact: public RiemannBase {

  protected:
  public:
    RiemannExact(idealGas::PrimitiveState& l, idealGas::PrimitiveState& r, const size_t dimension):
      RiemannBase(l, r, dimension) {};
    ~RiemannExact() = default;

    //! Call the actual solver.
    idealGas::PrimitiveState solve() override {
      message("Solving Exact.");
      return {};
    }
  };

} // namespace riemann
