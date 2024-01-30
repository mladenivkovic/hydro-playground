#include "Gas.h"

/*
Stuff for primitive state
*/

IdealGas::PrimitiveState::PrimitiveState():
  // initialiser list
  rho(0),
  u({0, 0}),
  p(0) // empty body...
  {};

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

Precision IdealGas::PrimitiveState::getSoundSpeed() { return std::sqrt(GAMMA * getP() / getRho()); }


Precision IdealGas::PrimitiveState::getEnergy() {
  return 0.5 * getRho() * getUSquared() + getP() / GM1;
}

/* getters and setters for PrimitiveState  */

void IdealGas::PrimitiveState::setRho(Precision val) { rho = val; }

Precision IdealGas::PrimitiveState::getRho() const { return rho; }

void IdealGas::PrimitiveState::setU(int index, const Precision val) { u[index] = val; }

Precision IdealGas::PrimitiveState::getU(int index) const { return u[index]; }

Precision IdealGas::PrimitiveState::getUSquared() const { return u[0] * u[0] + u[1] * u[1]; }

void IdealGas::PrimitiveState::setP(const Precision val) { p = val; }

Precision IdealGas::PrimitiveState::getP() const { return p; }


/*
Stuff for conserved state
*/

IdealGas::ConservedState::ConservedState():
  // initialiser list
  rho(0),
  rhou({0, 0}),
  E(0){};


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

  Precision tempE = 0.5 * p.getRho() * p.getUSquared() + p.getP() / GM1;
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
    Precision v = c.getRhou(dimension) / c.getRho();
    Precision p = GM1 * c.getRhoUSquared() / c.getRho();

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

void      IdealGas::ConservedState::setRhou(int index, const Precision val) { rhou[index] = val; }
Precision IdealGas::ConservedState::getRhou(int index) const { return rhou[index]; }

Precision IdealGas::ConservedState::getRhoUSquared() const {
  return rhou[0] * rhou[0] + rhou[1] * rhou[1];
}

void      IdealGas::ConservedState::setE(const Precision val) { E = val; }
Precision IdealGas::ConservedState::getE() const { return E; }

Precision IdealGas::ConservedState::getRho() const { return rho; }
void      IdealGas::ConservedState::setRho(Precision val) { rho = val; }
