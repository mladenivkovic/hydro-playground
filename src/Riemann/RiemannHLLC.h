/**
 * @file RiemannHLLC.h
 * @brief The HLLC Riemann solver.
 */

#pragma once

#include "Gas.h"
#include "RiemannBase.h"

/**
 * The HLLC Riemann solver.
 * See Section 3.4 in theory document.
 */
class RiemannHLLC: public RiemannBase {

  //! Left wave speed
  Float _SL;

  //! Right wave speed
  Float _SR;

  //! Star state wave speed
  Float _Sstar;

  //! Compute q_{L,R} needed for the wave speed estimate.
  Float _qLR(Float pstar, Float pLR);

  //! Compute the wave speeds SL, SR, Sstar
  void computeWaveSpeedEstimates();

  //! Compute the star states.
  void computeStarCStates(ConservedState& UStarL, ConservedState& UStarR);

  //! Sample the solution.
  ConservedFlux sampleHLLCSolution();

public:
  RiemannHLLC(PrimitiveState& l, PrimitiveState& r, const size_t dimension):
    RiemannBase(l, r, dimension),
    _SL(0.),
    _SR(0.),
    _Sstar(0.) {};

  ~RiemannHLLC() = default;

  //! Call the actual solver.
  ConservedFlux solve() override;
};

