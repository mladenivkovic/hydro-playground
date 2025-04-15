#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <string>

#include "Config.h"
#include "Constants.h"
#include "Logging.h"
#include "Utils.h"


class ConservedState;
class PrimitiveState;

// Aliases for clarity. The states and fluxes will have the same components,
// so we can use the same data structure. But this aliasing should make things
// more clear.
using ConservedFlux = ConservedState;
using PrimitiveFlux = PrimitiveState;


/**
 * @brief Holds a primitive state (density, velocity, pressure)
 */
class PrimitiveState {
private:
  //! density
  Float _rho;

  //! velocity
  // std::array<Float, Dimensions> _v;

  // really sorry if you are reading this Mladen
  Float _v[Dimensions];

  //! pressure
  Float _p;


public:
  PrimitiveState();
  PrimitiveState(const Float rho, const std::array<Float, Dimensions> vel, const Float p);
  PrimitiveState(const Float rho, const Float vx, const Float p);
  PrimitiveState(const Float rho, const Float vx, const Float vy, const Float p);

  /**
   * Clear out contents.
   */
  void clear() {
    *this = PrimitiveState();
  }

  /**
   * Set the current primitive state vector to equivalend of given conserved
   * state.
   */
  void fromCons(const ConservedState& cons);

  //! Get the local soundspeed given a primitive state
  [[nodiscard]] Float getSoundSpeed() const;

  //! Get the total gas energy from a primitive state
  __host__ __device__ [[nodiscard]] Float getE() const;

  //! Get a string of the state.
  [[nodiscard]] std::string toString() const;


  // Getters and setters!

  // Setter for Rho
  __host__ __device__ void                setRho(const Float val);
  __host__ __device__ [[nodiscard]] Float getRho() const;

  // same for u
  __host__ __device__ void                setV(const std::size_t index, const Float val);
  __host__ __device__ [[nodiscard]] Float getV(const std::size_t index) const;

  // used a lot, made a function for it
  __host__ __device__ [[nodiscard]] Float getVSquared() const;

  __host__ __device__ void                setP(const Float val);
  __host__ __device__ [[nodiscard]] Float getP() const;
};


/**
 * @brief Holds a conserved state (density, momentum, energy)
 */
class ConservedState {
private:
  //! Density
  Float _rho;

  //! Momentum: rho * v
  // std::array<Float, Dimensions> _rhov;
  Float _rhov[Dimensions];

  //! Energy
  Float _energy;

public:
  // Standard constructor, init variables to 0
  ConservedState();
  ConservedState(const Float rho, const Float rhovx, const Float rhovy, const Float E);
  explicit ConservedState(const PrimitiveState& prim, const size_t dimension);

  /**
   * Clear out contents.
   */
  void clear() {
    *this = ConservedState();
  }


  /**
   * Set the current conserved state vector to equivalent of given primitive
   * state.
   */
   __host__ __device__ void fromPrim(const PrimitiveState& prim);


  /**
   * Compute the flux of conserved variables of the Euler
   * equations given a primitive variable state vector
   */
  void getCFluxFromPState(const PrimitiveState& pstate, const std::size_t dimension);


  /**
   * Compute the flux of conserved variables of the Euler
   * equations given a conserved state vector
   */
  void getCFluxFromCstate(const ConservedState& cstate, const std::size_t dimension);


  //! Get a string of the state.
  [[nodiscard]] std::string toString() const;


  // Getters and setters!
  __host__ __device__               void  setRho(const Float val);
  __host__ __device__ [[nodiscard]] Float getRho() const;

  // same for u
  __host__ __device__ void                setRhov(const std::size_t index, const Float val);
  __host__ __device__ [[nodiscard]] Float getRhov(const std::size_t index) const;
  __host__ __device__ [[nodiscard]] Float getRhoVSquared() const;

  __host__ __device__ void                setE(const Float val);
  __host__ __device__ [[nodiscard]] Float getE() const;

  __host__ __device__ [[nodiscard]] Float getP() const;
};


// --------------------------------------------------------
// Definitions
// --------------------------------------------------------

// Primitive State Stuff
// --------------------------

__host__ __device__ inline void PrimitiveState::setRho(const Float val) {
  // These checks will fail because we (ab)use the PrimitiveState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(val >= 0.);
  // #endif
  _rho = val;
}


__host__ __device__ inline Float PrimitiveState::getRho() const {
  // These checks will fail because we (ab)use the PrimitiveState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(_rho >= 0.);
  // #endif
  return _rho;
}


__host__ __device__ inline void PrimitiveState::setV(const size_t index, const Float val) {
#if __CUDA_ARCH__
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
#endif
  _v[index] = val;
}


__host__ __device__ inline Float PrimitiveState::getV(const size_t index) const {
#if __CUDA_ARCH__
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
#endif
  return _v[index];
}


__host__ __device__ inline Float PrimitiveState::getVSquared() const {
  if (Dimensions == 1)
    return _v[0] * _v[0];

  if (Dimensions == 2)
    return _v[0] * _v[0] + _v[1] * _v[1];

  // error("Not implemented");
  // return 0.;
}


inline void PrimitiveState::setP(const Float val) {
  // These checks will fail because we (ab)use the PrimitiveState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(val >= 0.);
  // #endif
  _p = val;
}


inline Float PrimitiveState::getP() const {
  // These checks will fail because we (ab)use the PrimitiveState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(_p >= 0.);
  // #endif
  return _p;
}


/**
 * Compute the local sound speed given a primitive state.
 * Eq. 6
 */
inline Float PrimitiveState::getSoundSpeed() const {
  return std::sqrt(cst::GAMMA * getP() / getRho());
}


/**
 * Get the total gas energy from a primitive state.
 * Eq. 18
 */
inline Float PrimitiveState::getE() const {

  return 0.5 * getRho() * getVSquared() + getP() * cst::ONEOVERGM1;
}


// Conserved State Stuff
// --------------------------

__host__ __device__ inline void ConservedState::setRhov(const size_t index, const Float val) {
#if __CUDA_ARCH__
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
#endif
  _rhov[index] = val;
}


__host__ __device__ inline Float ConservedState::getRhov(const size_t index) const {
#if __CUDA_ARCH__
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
#endif
  return _rhov[index];
}


__host__ __device__ inline Float ConservedState::getRhoVSquared() const {
  return _rhov[0] * _rhov[0] + _rhov[1] * _rhov[1];
}


inline void ConservedState::setE(const Float val) {
  // These checks will fail because we (ab)use the ConservedState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(val >= 0.);
  // #endif
  _energy = val;
}


inline Float ConservedState::getE() const {
  // These checks will fail because we (ab)use the ConservedState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(_energy >= 0.);
  // #endif
  return _energy;
}


inline Float ConservedState::getRho() const {
  // These checks will fail because we (ab)use the ConservedState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(_rho >= 0.);
  // #endif
  return _rho;
}


inline void ConservedState::setRho(const Float val) {
  // These checks will fail because we (ab)use the ConservedState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(val >= 0.);
  // #endif
  _rho = val;
}

inline Float ConservedState::getP() const {
  // this makes prim->cons->prim conversion worse due to roundoff errors.
  // Float one_over_rho = 1. / rho;
  // Float rv2 = cons.getRhoVSquared() * one_over_rho;
  // this also makes it worse.
  // return (cst::GM1 * getE() - cst::GM1 * 0.5 * rv2);

  Float rho          = getRho();
  Float one_over_rho = 1. / rho;
  Float vx           = getRhov(0) * one_over_rho;
  Float vy           = getRhov(1) * one_over_rho;
  Float rv2          = rho * (vx * vx + vy * vy);
  return cst::GM1 * (getE() - 0.5 * rv2);
}



/**
 * Compute the conserved state vector of a given primitive state.
 *
 * See eqns. 16 - 18 in theory document.
 */
__host__ __device__ inline void ConservedState::fromPrim(const PrimitiveState& p) {
  setRho(p.getRho());
  setRhov(0, p.getRho() * p.getV(0));
  setRhov(1, p.getRho() * p.getV(1));
  setE(p.getE());
}

