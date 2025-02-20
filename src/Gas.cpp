#include "Gas.h"

#include <cmath>

#include "Constants.h"

// Stuff for primitive state

IdealGas::PrimitiveState::PrimitiveState():
  // initialiser list
  rho(0.),
  u({0., 0.}),
  p(0.) { /* empty body... */ };

void IdealGas::PrimitiveState::ConservedToPrimitive(const ConservedState& c) {
  if (c.getRho() <= SMALLRHO) {
    // execption handling for vacuum
    setRho(SMALLRHO);
    setU(0, SMALLU);
    setU(1, SMALLU);
    setP(SMALLP);
  } else {
    setRho(c.getRho());
    setU(0, c.getRhou(0) / c.getRho());
    setU(1, c.getRhou(1) / c.getRho());
    setP(GM1 * c.getE() - 0.5 * c.getRhoUSquared() / c.getRho());

    // handle negative pressure
    if (getP() <= SMALLP)
      setP(SMALLP);
  }
}

float_t IdealGas::PrimitiveState::getSoundSpeed() {
  return std::sqrt(GAMMA * getP() / getRho());
}


float_t IdealGas::PrimitiveState::getEnergy() {
  return 0.5 * getRho() * getUSquared() + getP() / GM1;
}

// getters and setters for PrimitiveState

void IdealGas::PrimitiveState::setRho(const float_t val) {
  rho = val;
}

float_t IdealGas::PrimitiveState::getRho() const {
  return rho;
}

void IdealGas::PrimitiveState::setU(const int index, const float_t val) {
  u[index] = val;
}

float_t IdealGas::PrimitiveState::getU(const int index) const {
  return u[index];
}

float_t IdealGas::PrimitiveState::getUSquared() const {
  return u[0] * u[0] + u[1] * u[1];
}

void IdealGas::PrimitiveState::setP(const float_t val) {
  p = val;
}

float_t IdealGas::PrimitiveState::getP() const {
  return p;
}


// Stuff for conserved state

IdealGas::ConservedState::ConservedState():
  // initialiser list
  rho(0.),
  rhou({0., 0.}),
  E(0.) { /* empty body... */ };


void IdealGas::ConservedState::PrimitiveToConserved(const PrimitiveState& p) {
  setRho(p.getRho());
  setRhou(0, p.getRho() * p.getU(0));
  setRhou(1, p.getRho() * p.getU(1));
  setE(0.5 * p.getRho() * p.getUSquared() + p.getP() / GM1);
}


void IdealGas::ConservedState::GetCFluxFromPstate(const PrimitiveState& p, int dimension) {
  /* -----------------------------------------------------------
   * Compute the flux of conserved variables of the Euler
   * equations given a primitive state vector
   *
   * The flux is not an entire tensor for 3D Euler equations, but
   * correpsonds to the dimensionally split vectors F, G as
   * described in the "Euler equations in 2D" section of the
   * documentation TeX files.
   * That's why you need to specify the dimension.
   *
   * Everything about this is copied from the original C code.
   * For now, just need to get the machinery in before making
   * it C++ style.
   * ----------------------------------------------------------- */

  setRho(p.getRho() * p.getU(dimension));
  setRhou(dimension, p.getRho() * p.getU(dimension) * p.getU(dimension) + p.getP());
  setRhou((dimension + 1) % 2, p.getRho() * p.getU(0) * p.getU(1));

  float_t tempE = 0.5 * p.getRho() * p.getUSquared() + p.getP() / GM1;
  setE(tempE + p.getP() * p.getU(dimension));
}

void IdealGas::ConservedState::GetCFluxFromCstate(const ConservedState& c, int dimension) {
  /* -----------------------------------------------------------
   * Compute the flux of conserved variables of the Euler
   * equations given a conserved state vector
   *
   * The flux is not an entire tensor for 3D Euler equations, but
   * correpsonds to the dimensionally split vectors F, G as
   * described in the "Euler equations in 2D" section of the
   * documentation TeX files.
   * That's why you need to specify the dimension.
   *
   * Everything about this is copied from the original C code.
   * For now, just need to get the machinery in before making
   * it C++ style.
   * ----------------------------------------------------------- */
  setRho(c.getRhou(dimension));
  if (c.getRho() > 0) {
    float_t v = c.getRhou(dimension) / c.getRho();
    float_t p = GM1 * c.getRhoUSquared() / c.getRho();

    setRhou(dimension, c.getRho() * v * v + p);
    setRhou((dimension + 1) % 2, c.getRhou((dimension + 1) % 2) * v);
    setE((c.getE() + p) * v);
  } else {
    setRhou(0, 0);
    setRhou(0, 1);
    setE(0);
  }
}

/* Getters and Setters */

void IdealGas::ConservedState::setRhou(const size_t index, const float_t val) {
  rhou[index] = val;
}

float_t IdealGas::ConservedState::getRhou(const size_t index) const {
  return rhou[index];
}

float_t IdealGas::ConservedState::getRhoUSquared() const {
  return rhou[0] * rhou[0] + rhou[1] * rhou[1];
}

void IdealGas::ConservedState::setE(const float_t val) {
  E = val;
}

float_t IdealGas::ConservedState::getE() const {
  return E;
}

float_t IdealGas::ConservedState::getRho() const {
  return rho;
}

void IdealGas::ConservedState::setRho(const float_t val) {
  rho = val;
}
