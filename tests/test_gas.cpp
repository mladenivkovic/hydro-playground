#include <cassert>
#include <cmath>
#include <random>
#include <sstream>
#include <string>

#include "Config.h"
#include "Constants.h"
#include "Gas.h"
#include "Logging.h"


using ps = PrimitiveState;
using cs = ConservedState;


constexpr int nrepeat = 10000;

#if PRECISION == SINGLE_PRECISION
// how many digits to check for
constexpr Float TOLERANCE_RHO = 1.e-4;
constexpr Float TOLERANCE_V   = 1.e-4;
constexpr Float TOLERANCE_P   = 1.e-4;
#elif PRECISION == DOUBLE_PRECISION
constexpr Float TOLERANCE_RHO = 1.e-12;
constexpr Float TOLERANCE_V   = 1.e-12;
constexpr Float TOLERANCE_P   = 1.e-12;
#else
#error invalid presicion defined
#endif


bool isEqual(Float a, Float b, Float rel_tol, Float abs_tol = 0.) {

  bool rel = (1. - std::abs(a / b)) < rel_tol;
  if (abs_tol == 0.)
    return rel;

  bool abs = std::abs(a - b) < abs_tol;

  return (rel or abs);
}


void testConversion(std::mt19937& generator) {

  // Careful with ranges here. Big ones lead to troube with precision.
  std::uniform_real_distribution<Float> positive(cst::SMALLRHO, 1.);
  std::uniform_real_distribution<Float> uniform(-1., 1.);

  // Convert prim -> cons -> prim -> cons
  for (int i = 0; i < nrepeat; i++) {
    Float rho = positive(generator);
    Float vx  = uniform(generator);
    Float vy  = uniform(generator);
    Float p   = positive(generator);

    // Determine tolerance. In prim->cons, we multiply rho * v.
    // This can give you several orders of magnitude of difference.
    // When converting back, in particularly for the pressure, we
    // can get heavy precision errors.
    Float vmax = std::max(std::abs(vx), std::abs(vy));
    Float mag  = std::ceil(
      std::max(std::abs(std::log10(rho / vmax)), std::abs(std::log10(vmax / rho)))
    );
    Float ABS_TOL_P = TOLERANCE_P * std::pow(10., mag * mag) * p;

    ps prim(rho, vx, vy, p);

    cs cons;
    cons.fromPrim(prim);

    ps prim2;
    prim2.fromCons(cons);

    cs cons2;
    cons2.fromPrim(prim2);

    if (not isEqual(prim2.getRho(), prim.getRho(), TOLERANCE_RHO)) {
      std::stringstream msg;
      msg << prim2.getRho() << ", ";
      msg << prim.getRho() << ", ";
      msg << 1. - prim2.getRho() / prim.getRho();
      error(msg.str());
    }

    if (not isEqual(prim2.getV(0), prim.getV(0), TOLERANCE_V)) {
      std::stringstream msg;
      msg << prim2.getV(0) << ", ";
      msg << prim.getV(0) << ", ";
      msg << 1. - prim2.getV(0) / prim.getV(0);
      error(msg.str());
    }

    if (not isEqual(prim2.getV(1), prim.getV(1), TOLERANCE_V)) {
      std::stringstream msg;
      msg << prim2.getV(1) << ", ";
      msg << prim.getV(1) << ", ";
      msg << 1. - prim2.getV(1) / prim.getV(1);
      error(msg.str());
    }

    if (not isEqual(prim2.getP(), prim.getP(), TOLERANCE_P, ABS_TOL_P)) {
      std::stringstream msg;
      msg << prim2.getP() << ", ";
      msg << prim.getP() << ", ";
      msg << 1. - prim2.getP() / prim.getP();
      error(msg.str());
    }

    if (not isEqual(cons2.getRho(), cons.getRho(), TOLERANCE_RHO)) {
      std::stringstream msg;
      msg << cons2.getRho() << ", ";
      msg << cons.getRho() << ", ";
      msg << 1. - cons2.getRho() / cons.getRho();
      error(msg.str());
    }

    if (not isEqual(cons2.getRhov(0), cons.getRhov(0), TOLERANCE_V)) {
      std::stringstream msg;
      msg << cons2.getRhov(0) << ", ";
      msg << cons.getRhov(0) << ", ";
      msg << 1. - cons2.getRhov(0) / cons.getRhov(0);
      error(msg.str());
    }

    if (not isEqual(cons2.getRhov(1), cons.getRhov(1), TOLERANCE_V)) {
      std::stringstream msg;
      msg << cons2.getRhov(1) << ", ";
      msg << cons.getRhov(1) << ", ";
      msg << 1. - cons2.getRhov(1) / cons.getRhov(1);
      error(msg.str());
    }

    if (not isEqual(cons2.getE(), cons.getE(), TOLERANCE_P)) {
      std::stringstream msg;
      msg << cons2.getE() << ", ";
      msg << cons.getE() << ", ";
      msg << 1. - cons2.getE() / cons.getE();
      error(msg.str());
    }


    // make sure you get the correct conserved fluxes

    ConservedFlux cf1x;
    cf1x.getCFluxFromPState(prim, 0);

    ConservedFlux cf1y;
    cf1y.getCFluxFromPState(prim, 1);

    ConservedFlux cf2x;
    cf2x.getCFluxFromCstate(cons, 0);

    ConservedFlux cf2y;
    cf2y.getCFluxFromCstate(cons, 1);

    // Float ABS_TOL_E = TOLERANCE_P * std::pow(10., mag * mag) * cons.getE();

    if (not isEqual(cf1x.getRho(), cf2x.getRho(), TOLERANCE_RHO)) {
      std::stringstream msg;
      msg << cf1x.getRho() << ", ";
      msg << cf2x.getRho() << ", ";
      msg << 1. - cf1x.getRho() / cf2x.getRho();
      error(msg.str());
    }

    if (not isEqual(cf1x.getRhov(0), cf2x.getRhov(0), TOLERANCE_V)) {
      std::stringstream msg;
      msg << cf1x.getRhov(0) << ", ";
      msg << cf2x.getRhov(0) << ", ";
      msg << 1. - cf1x.getRhov(0) / cf2x.getRhov(0);
      error(msg.str());
    }

    if (not isEqual(cf1x.getRhov(1), cf2x.getRhov(1), TOLERANCE_V)) {
      std::stringstream msg;
      msg << cf1x.getRhov(1) << ", ";
      msg << cf2x.getRhov(1) << ", ";
      msg << 1. - cf1x.getRhov(1) / cf2x.getRhov(1);
      error(msg.str());
    }

    if (not isEqual(cf1x.getE(), cf2x.getE(), TOLERANCE_P)) {
      std::stringstream msg;
      msg << cf1x.getE() << ", ";
      msg << cf2x.getE() << ", ";
      msg << 1. - cf1x.getE() / cf2x.getE();
      error(msg.str());
    }

    if (not isEqual(cf1y.getRho(), cf2y.getRho(), TOLERANCE_RHO)) {
      std::stringstream msg;
      msg << cf1y.getRho() << ", ";
      msg << cf2y.getRho() << ", ";
      msg << 1. - cf1y.getRho() / cf2y.getRho();
      error(msg.str());
    }

    if (not isEqual(cf1y.getRhov(0), cf2y.getRhov(0), TOLERANCE_V)) {
      std::stringstream msg;
      msg << cf1y.getRhov(0) << ", ";
      msg << cf2y.getRhov(0) << ", ";
      msg << 1. - cf1y.getRhov(0) / cf2y.getRhov(0);
      error(msg.str());
    }

    if (not isEqual(cf1y.getRhov(1), cf2y.getRhov(1), TOLERANCE_V)) {
      std::stringstream msg;
      msg << cf1y.getRhov(1) << ", ";
      msg << cf2y.getRhov(1) << ", ";
      msg << 1. - cf1y.getRhov(1) / cf2y.getRhov(1);
      error(msg.str());
    }

    if (not isEqual(cf1y.getE(), cf2y.getE(), TOLERANCE_P)) {
      std::stringstream msg;
      msg << cf1y.getE() << ", ";
      msg << cf2y.getE() << ", ";
      msg << 1. - cf1y.getE() / cf2y.getE();
      error(msg.str());
    }
  }
}


int main() {

  logging::setStage(logging::LogStage::Test);

  std::mt19937 generator(666);

  testConversion(generator);

  message("Done.");

  return 0;
}
