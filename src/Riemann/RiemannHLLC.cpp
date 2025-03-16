#include "RiemannHLLC.h"

#include <cmath>

#include "Constants.h"
#include "Gas.h"
#include "Timer.h"

/**
 * @brief solve the Riemann problem with the HLLC solver.
 *
 * @return the intercell flux of conserved variables corresponding to the
 * solution sampled at x=0.
 */
idealGas::ConservedFlux riemann::RiemannHLLC::solve() {

  timer::Timer tick(timer::Category::Riemann);

  if (hasVacuum()) {
    idealGas::PrimitiveState vac = solveVacuum();
    idealGas::ConservedFlux  sol(vac, _dim);
    return sol;
  }

  computeWaveSpeedEstimates();

  return sampleHLLCSolution();
}


/**
 * Compute q_{L,R} needed for the wave speed estimate.
 * TODO: Add equation in theory
 *
 * pstar:   (estimated) pressure of the star state
 * pLR:     left or right pressure, depending whether
 *          you want q_L or q_R
 */
inline Float riemann::RiemannHLLC::_qLR(Float pstar, Float pLR) {
  if (pstar > pLR) {
    // shock relation
    return (std::sqrt(1. + 0.5 * (cst::GAMMA + 1.) / cst::GAMMA * (pstar / pLR - 1.)));
  }
  // Else: rarefaction relation
  return 1.;
}


/**
 * Compute the wave speed (estimates) SL, SR, and Sstar.
 */
void riemann::RiemannHLLC::computeWaveSpeedEstimates() {

  /* Start by computint the simple primitive variable speed estimate */
  /* --------------------------------------------------------------- */

  Float aL = _left.getSoundSpeed();
  Float aR = _right.getSoundSpeed();

  Float rhoL = _left.getRho();
  Float rhoR = _right.getRho();

  Float vL = _left.getV(_dim);
  Float vR = _right.getV(_dim);

  Float pL = _left.getP();
  Float pR = _right.getP();


  Float temp = 0.25 * (rhoL + rhoR) * (aL + aR);
  Float PPV  = 0.5 * (pL + pR) + 0.5 * (vL - vR) * temp;

  if (PPV < 0.)
    PPV = cst::SMALLP;

  Float pstar = PPV;
  Float vstar = 0.5 * (vL + vR) + 0.5 * (pL - pR) / temp;

  // defined in Config.h
#ifdef HLLC_USE_ADAPTIVE_SPEED_ESTIMATE

  // use the adaptive wave speed estimate
  // ------------------------------------------------

  // find ratio Q = pmax/pmin, where pmax, pmin are pL and pR
  Float pmin = pL > pR ? pR : pL;
  Float pmax = pL > pR ? pL : pR;
  Float qmax = pmax / pmin;

  // if the ratio pmax/pmin isn't too big, and the primitive variable pressure
  // is between left and right pressure, then PPV approximation is fine
  if (qmax <= 2. and (pmin <= PPV and PPV <= pmax)) {
    pstar = PPV;
    vstar = 0.5 * (vL + vR) + 0.5 * (pL - pR) / temp;
  } else {

    if (PPV <= pmin) {
      // Primitive variable approximation isn't good enough.
      // if we expect rarefaction, use the TRRS solver

      Float aLinv   = 1. / aL;
      Float aRinv   = 1. / aR;
      Float pLRbeta = std::pow(pL / pR, cst::BETA);

      vstar = ((pLRbeta - 1.) / cst::GM1HALF + vL * aLinv * pLRbeta + vR * aRinv)
              / (aRinv + aLinv * pLRbeta);

      pstar
        = 0.5
          * (pR * std::pow((1. + aRinv * cst::GM1HALF * (vstar - vR)), 1. / cst::BETA) + pL * std::pow((1. + aLinv * cst::GM1HALF * (vL - vstar)), 1. / cst::BETA));
    }

    else {
      // If not rarefactions, you'll encounter shocks, so use TSRS solver

      Float AL = cst::TWOOVERGAMMAP1 / rhoL;
      Float AR = cst::TWOOVERGAMMAP1 / rhoR;
      Float BL = cst::GM1OGP1 * pL;
      Float BR = cst::GM1OGP1 * pR;

      Float gL = std::sqrt(AL / (PPV + BL));
      Float gR = std::sqrt(AR / (PPV + BR));

      pstar = (gL * pL + gR * pR - (vR - vL)) / (gL + gR);
      vstar = 0.5 * (vR + vL + (pstar - pR) * gR - (pstar - pL) * gL);
    }
  }
#endif /* adaptive solution */

  _SL    = vL - aL * _qLR(pstar, pL);
  _SR    = vR + aR * _qLR(pstar, pR);
  _Sstar = vstar;
}


/**
 * Compute the !conserved! star states UstarL and UstarR of the HLLC solution
 * of the Riemann problem.
 * Assumes that the wave speed estimates have been computed already.
 */
void riemann::RiemannHLLC::computeStarCStates(
  idealGas::ConservedState& UStarL, idealGas::ConservedState& UStarR
) {

#if DEBUG_LEVEL > 0
  if (_SL == 0. and _SR == 0. and _Sstar == 0.)
    warning("Suspicious wave speed estimates.");
#endif

  size_t other = (_dim + 1) % 2;

  Float rhoL = _left.getRho();
  Float rhoR = _right.getRho();

  Float vLdim   = _left.getV(_dim);
  Float vRdim   = _right.getV(_dim);
  Float vLother = _left.getV(other);
  Float vRother = _right.getV(other);

  Float pL = _left.getP();
  Float pR = _right.getP();

  Float SLMUL = _SL - vLdim;
  Float SRMUR = _SR - vRdim;

  // compute left and right conserved star states

  Float lcomp = rhoL * SLMUL / (_SL - _Sstar);
  UStarL.setRho(lcomp);
  UStarL.setRhov(_dim, lcomp * _Sstar);
  UStarL.setRhov(other, lcomp * vLother);

  Float EL    = 0.5 * rhoL * _left.getVSquared() + pL * cst::ONEOVERGAMMAM1;
  Float EnewL = lcomp * ((EL / rhoL) + (_Sstar - vLdim) * (_Sstar + pL / (rhoL * SLMUL)));
  UStarL.setE(EnewL);

  Float rcomp = rhoR * SRMUR / (_SR - _Sstar);
  UStarR.setRho(rcomp);
  UStarR.setRhov(_dim, rcomp * _Sstar);
  UStarR.setRhov(other, rcomp * vRother);

  Float ER    = 0.5 * rhoR * _right.getVSquared() + pR * cst::ONEOVERGAMMAM1;
  Float EnewR = rcomp * ((ER / rhoR) + (_Sstar - vRdim) * (_Sstar + pR / (rhoR * SRMUR)));
  UStarR.setE(EnewR);
}


/**
 * Sample the solution. This assumes the wave speeds SL, SR, and Sstar
 * have been computed already.
 * We need to do this differently for the HLLC solver because it delivers us
 * directly with a conserved state, whereas other Riemann solvers give us
 * primitive states.
 *
 * @return the intercell flux of primitive variables
 */
idealGas::ConservedFlux riemann::RiemannHLLC::sampleHLLCSolution() {
#if DEBUG_LEVEL > 0
  if (_SL == 0. and _SR == 0. and _Sstar == 0.)
    warning("Suspicious wave speed estimates.");
#endif

  // x / t. We always center the problem at x=0, but to sample the solution in
  // general, we need to adapt this value.
  constexpr Float xovert = 0.;


  // compute left and right star !conserved! states

  idealGas::ConservedState UL;
  UL.fromPrim(_left);

  idealGas::ConservedState UR;
  UR.fromPrim(_right);

  idealGas::ConservedState UStarL;
  idealGas::ConservedState UStarR;

  computeStarCStates(UStarL, UStarR);

  // Compute left and right fluxes

  idealGas::ConservedState FL;
  FL.getCFluxFromCstate(UL, _dim);

  idealGas::ConservedState FR;
  FR.getCFluxFromCstate(UR, _dim);


  // Compute left and right star fluxes

  Float                    rhoL   = FL.getRho() + _SL * (UStarL.getRho() - UL.getRho());
  Float                    rhovLx = FL.getRhov(0) + _SL * (UStarL.getRhov(0) - UL.getRhov(0));
  Float                    rhovLy = FL.getRhov(1) + _SL * (UStarL.getRhov(1) - UL.getRhov(1));
  Float                    EL     = FL.getE() + _SL * (UStarL.getE() - UL.getE());
  idealGas::ConservedState FstarL(rhoL, rhovLx, rhovLy, EL);

  Float                    rhoR   = FR.getRho() + _SR * (UStarR.getRho() - UR.getRho());
  Float                    rhovRx = FR.getRhov(0) + _SR * (UStarR.getRhov(0) - UR.getRhov(0));
  Float                    rhovRy = FR.getRhov(1) + _SR * (UStarR.getRhov(1) - UR.getRhov(1));
  Float                    ER     = FR.getE() + _SR * (UStarR.getE() - UR.getE());
  idealGas::ConservedState FstarR(rhoR, rhovRx, rhovRy, ER);


  // finally, sample the solution

  Float rho_sol   = NAN;
  Float rhovx_sol = NAN;
  Float rhovy_sol = NAN;
  Float E_sol     = NAN;

  if (xovert <= _SL) {
    // solution is F_L
    rho_sol   = FL.getRho();
    rhovx_sol = FL.getRhov(0);
    rhovy_sol = FL.getRhov(1);
    E_sol     = FL.getE();
  } else if (xovert <= _Sstar) {
    // solution is F*_L
    rho_sol   = FstarL.getRho();
    rhovx_sol = FstarL.getRhov(0);
    rhovy_sol = FstarL.getRhov(1);
    E_sol     = FstarL.getE();
  } else if (xovert <= _SR) {
    // solution is F*_R
    rho_sol   = FstarR.getRho();
    rhovx_sol = FstarR.getRhov(0);
    rhovy_sol = FstarR.getRhov(1);
    E_sol     = FstarR.getE();
  } else {
    // solution is F_R
    rho_sol   = FR.getRho();
    rhovx_sol = FR.getRhov(0);
    rhovy_sol = FR.getRhov(1);
    E_sol     = FR.getE();
  }


#if DEBUG_LEVEL > 0
  assert(not std::isnan(rho_sol));
  assert(not std::isnan(rhovx_sol));
  assert(not std::isnan(rhovy_sol));
  assert(not std::isnan(E_sol));
#endif

  idealGas::ConservedFlux Fsol(rho_sol, rhovx_sol, rhovy_sol, E_sol);
  return Fsol;
}
