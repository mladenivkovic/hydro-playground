/**
 * @file RiemannHLLC.h
 * @brief The HLLC Riemann solver.
 */

#pragma once

#include "Gas.h"
#include "RiemannBase.h"

namespace riemann {


  /**
   * The HLLC Riemann solver.
   */
  class RiemannHLLC : public RiemannBase {

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
    void computeStarCStates(idealGas::ConservedState& UStarL, idealGas::ConservedState& UStarR);

    //! Sample the solution.
    idealGas::PrimitiveState sampleSolution();

  public:

    RiemannHLLC(
        idealGas::PrimitiveState& l, idealGas::PrimitiveState& r,
        const size_t dimension) :
      RiemannBase(l, r, dimension),
      _SL(0.),
      _SR(0.),
      _Sstar(0.) {};

    ~RiemannHLLC() = default;

    //! Call the actual solver.
    idealGas::PrimitiveState solve() override;

  };

} // namespace riemann

