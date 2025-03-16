/**
 * @file RiemannExact.h
 * @brief The exact Riemann solver.
 */

#pragma once

#include "Gas.h"
#include "RiemannBase.h"

namespace riemann {

  /**
   * The Exact riemann solver.
   */
  class RiemannExact: public RiemannBase {

  private:
    //! Compute the star state pressure and velocity iteratively.
    void computeStarStates();

    //!
    Float fp(
      const Float                     pguess,
      const idealGas::PrimitiveState& state,
      const Float                     A,
      const Float                     B,
      const Float                     cs
    );

    //!
    Float dfpdp(
      const Float                     pguess,
      const idealGas::PrimitiveState& state,
      const Float                     A,
      const Float                     B,
      const Float                     cs
    );


  public:
    RiemannExact(idealGas::PrimitiveState& l, idealGas::PrimitiveState& r, const size_t dimension):
      RiemannBase(l, r, dimension) {};
    ~RiemannExact() = default;

    //! Call the actual solver.
    idealGas::ConservedFlux solve() override;
  };

} // namespace riemann
