/**
 * @file RiemannExact.h
 * @brief The exact Riemann solver.
 */

#pragma once

#include "Gas.h"
#include "RiemannBase.h"
#include "Utils.h"

/**
 * The Exact riemann solver.
 */
class RiemannExact: public RiemannBase {

private:
  //! Compute the star state pressure and velocity iteratively.
  template<Device>
  void computeStarStates();

  //! f(p)
  __host__ __device__ Float fp(
    const Float pguess, const PrimitiveState& state, const Float A, const Float B, const Float cs
  );

  //! df(p)/dp
  __host__ __device__ Float dfpdp(
    const Float pguess, const PrimitiveState& state, const Float A, const Float B, const Float cs
  );


public:
  __host__ __device__
  RiemannExact(PrimitiveState& l, PrimitiveState& r, const size_t dimension):
    RiemannBase(l, r, dimension) {};
  ~RiemannExact() = default;

  //! Call the actual solver.
  ConservedFlux solve() override;
  __device__ ConservedFlux solveOnGpu() override;
};



/**
 * The left/right part of the pressure function.
 * Equation 60-63 in Theory document.
 *
 * @param pguess Star state pressure guess
 * @param state The left or right state for which to compute f_p
 * @param A   A_L or A_R (Eq. 62)
 * @param B   B_L or B_R (Eq. 63)
 * @param cs  soundspeed of state
 */
 inline Float RiemannExact::fp(
  const Float pguess, const PrimitiveState& state, const Float A, const Float B, const Float cs
) {

  Float p = state.getP();

  if (pguess > p) {
    // we have a shock situation
    return (pguess - p) * std::sqrt(A / (pguess + B));
  }
  // we have a rarefaction situation
  return cst::TWOOVERGM1 * cs * (std::pow(pguess / p, cst::BETA) - 1.);
}


/**
 * The derivative of the left/right part of the pressure function.
 * Equation 64 in Theory document.
 *
 * @param pguess Star state pressure guess
 * @param state The left or right state for which to compute f_p
 * @param A   A_L or A_R (Eq. 65)
 * @param B   B_L or B_R (Eq. 66)
 * @param cs  soundspeed of state
 */
inline Float RiemannExact::dfpdp(
  const Float pguess, const PrimitiveState& state, const Float A, const Float B, const Float cs
) {

  Float p   = state.getP();
  Float rho = state.getRho();

  if (pguess > p) {
    // we have a shock situation
    return std::sqrt(A / (pguess + B)) * (1. - 0.5 * (pguess - p) / (pguess + B));
  }
  // we have a rarefaction situation
  return 1. / (rho * cs) * std::pow(pguess / p, -0.5 * cst::GP1 * cst::ONEOVERGAMMA);
}

