/**
 * @file RiemannExact.h
 * @brief The exact Riemann solver.
 */

#pragma once

#include "Gas.h"
#include "RiemannBase.h"

/**
 * The Exact riemann solver.
 */
class RiemannExact: public RiemannBase {

private:
  //! Compute the star state pressure and velocity iteratively.
  void computeStarStates();

  //! f(p)
  Float fp(
    const Float                     pguess,
    const PrimitiveState& state,
    const Float                     A,
    const Float                     B,
    const Float                     cs
  );

  //! df(p)/dp
  Float dfpdp(
    const Float                     pguess,
    const PrimitiveState& state,
    const Float                     A,
    const Float                     B,
    const Float                     cs
  );


public:
  RiemannExact(PrimitiveState& l, PrimitiveState& r, const size_t dimension):
    RiemannBase(l, r, dimension) {};
  ~RiemannExact() = default;

  //! Call the actual solver.
  ConservedFlux solve() override;
};

