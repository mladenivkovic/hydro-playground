#include <cassert>
#include <cmath>
#include <random>
#include <sstream>
#include <string>

#include "Config.h"
#include "Constants.h"
#include "Gas.h"
#include "Logging.h"


using ps = idealGas::PrimitiveState;
using cs = idealGas::ConservedState;


constexpr int nrepeat = 10000;

#if PRECISION == SINGLE_PRECISION
constexpr Float TOLERANCE = 1.e-4;
#elif PRECISION == DOUBLE_PRECISION
constexpr Float TOLERANCE = 1.e-8;
#else
#error invalid presicion defined
#endif


bool isEqual(Float a, Float b, Float tolerance = TOLERANCE) {
  return (1. - std::abs(a / b)) < tolerance;
}


void testConversion(std::mt19937& generator) {

  std::uniform_real_distribution<Float> positive(cst::SMALLRHO, 1.e3);
  std::uniform_real_distribution<Float> uniform(-1.e3, 1.e3);

  // Convert prim -> cons -> prim -> cons
  for (int i = 0; i < nrepeat; i++) {
    Float rho = positive(generator);
    Float vx  = uniform(generator);
    Float vy  = uniform(generator);
    Float p   = positive(generator);

    ps prim(rho, vx, vy, p);

    cs cons;
    cons.fromPrim(prim);

    ps prim2;
    prim2.fromCons(cons);

    cs cons2;
    cons2.fromPrim(prim2);


    if (not isEqual(prim2.getRho(), prim.getRho())) {
      std::stringstream msg;
      msg << prim2.getRho() << ", ";
      msg << prim.getRho() << ", ";
      msg << 1. - prim2.getRho() / prim.getRho();
      error(msg.str());
    }

    if (not isEqual(prim2.getV(0), prim.getV(0))) {
      std::stringstream msg;
      msg << prim2.getV(0) << ", ";
      msg << prim.getV(0) << ", ";
      msg << 1. - prim2.getV(0) / prim.getV(0);
      error(msg.str());
    }

    if (not isEqual(prim2.getV(1), prim.getV(1))) {
      std::stringstream msg;
      msg << prim2.getV(1) << ", ";
      msg << prim.getV(1) << ", ";
      msg << 1. - prim2.getV(1) / prim.getV(1);
      error(msg.str());
    }

    if (not isEqual(prim2.getP(), prim.getP())) {
      std::stringstream msg;
      msg << prim2.getP() << ", ";
      msg << prim.getP() << ", ";
      msg << 1. - prim2.getP() / prim.getP();
      error(msg.str());
    }

    if (not isEqual(cons2.getRho(), cons.getRho())) {
      std::stringstream msg;
      msg << cons2.getRho() << ", ";
      msg << cons.getRho() << ", ";
      msg << 1. - cons2.getRho() / cons.getRho();
      error(msg.str());
    }

    if (not isEqual(cons2.getRhov(0), cons.getRhov(0))) {
      std::stringstream msg;
      msg << cons2.getRhov(0) << ", ";
      msg << cons.getRhov(0) << ", ";
      msg << 1. - cons2.getRhov(0) / cons.getRhov(0);
      error(msg.str());
    }

    if (not isEqual(cons2.getRhov(1), cons.getRhov(1))) {
      std::stringstream msg;
      msg << cons2.getRhov(1) << ", ";
      msg << cons.getRhov(1) << ", ";
      msg << 1. - cons2.getRhov(1) / cons.getRhov(1);
      error(msg.str());
    }

    if (not isEqual(cons2.getE(), cons.getE())) {
      std::stringstream msg;
      msg << cons2.getE() << ", ";
      msg << cons.getE() << ", ";
      msg << 1. - cons2.getE() / cons.getE();
      error(msg.str());
    }
  }
}


int main() {

  logging::setStage(logging::LogStage::Test);

  std::mt19937 generator(666);

  testConversion(generator);


  return 0;
}
