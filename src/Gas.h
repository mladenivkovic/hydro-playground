#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <string>

#include "Config.h"
#include "Constants.h"


namespace idealGas {
  // forward declaration, ConservedToPrimitive doesn't work without it
  class ConservedState;
  class PrimitiveState;

  /**
   * @brief Holds a primitive state (density, velocity, pressure)
   */
  class PrimitiveState {
  private:
    //! density
    Float rho;

    //! velocity
    std::array<Float, Dimensions> v;

    //! pressure
    Float p;


  public:
    PrimitiveState();
    PrimitiveState(const Float rho, const std::array<Float, Dimensions> vel, const Float p);
    PrimitiveState(const Float rho, const Float vx, const Float p);
    PrimitiveState(const Float rho, const Float vx, const Float vy, const Float p);

    // copy assignment
    // TODO(mivkov): This doesn't compile. Check with boundary conditions
    // PrimitiveState& operator=(const PrimitiveState& other) = default;

    // putting this in just in case it's needed
    void resetToInitialState() {
      *this = PrimitiveState();
    }

    /**
     * Convert a conserved state to a (this) primitive state.
     * Overwrites the contents of this primitive state.
     */
    void ConservedToPrimitive(const ConservedState& conservedState);

    //! Get the local soundspeed given a primitive state
    [[nodiscard]] Float getSoundSpeed() const;

    //! Get the total gas energy from a primitive state
    [[nodiscard]] Float getE() const;

    //! Get a string of the state.
    [[nodiscard]] std::string toString() const;


    // Getters and setters!

    // Setter for Rho
    void                  setRho(const Float val);
    [[nodiscard]] Float getRho() const;

    // same for u
    void                  setV(const std::size_t index, const Float val);
    [[nodiscard]] Float getV(const std::size_t index) const;

    // used a lot, made a function for it
    [[nodiscard]] Float getVSquared() const;

    void                  setP(const Float val);
    [[nodiscard]] Float getP() const;
  };


  /**
   * @brief Holds a conserved state (density, momentum, energy)
   */
  class ConservedState {
  private:
    //! Density
    Float rho;

    //! Momentum: rho * v
    std::array<Float, Dimensions> rhov;

    //! Energy
    Float E;

  public:
    // Standard constructor, init variables to 0
    ConservedState();

    // putting this in in case it's needed
    // TODO(mivkov): is this needed?
    void resetToInitialState() {
      *this = ConservedState();
    }

    //! Compute the conserved state vector of a given primitive state.
    void PrimitiveToConserved(const PrimitiveState& primState);

    /**
     * Compute the flux of conserved variables of the Euler
     * equations given a primitive state vector
     */
    void GetCFluxFromPstate(const PrimitiveState& pstate, const std::size_t dimension);

    /**
     * Compute the flux of conserved variables of the Euler
     * equations given a conserved state vector
     */
    void GetCFluxFromCstate(const ConservedState& cstate, const std::size_t dimension);

    //! Get a string of the state.
    [[nodiscard]] std::string toString() const;


    // Getters and setters!
    void                  setRho(const Float val);
    [[nodiscard]] Float getRho() const;

    // same for u
    void                  setRhov(const std::size_t index, const Float val);
    [[nodiscard]] Float getRhov(const std::size_t index) const;
    [[nodiscard]] Float getRhoVSquared() const;

    void                  setE(const Float val);
    [[nodiscard]] Float getE() const;
  };
} // namespace idealGas


// --------------------------------------------------------
// Definitions
// --------------------------------------------------------

// Primitive State Stuff
// --------------------------

inline void idealGas::PrimitiveState::setRho(const Float val) {
#if DEBUG_LEVEL > 0
  assert(val >= 0.);
#endif
  rho = val;
}


inline Float idealGas::PrimitiveState::getRho() const {
#if DEBUG_LEVEL > 0
  assert(rho >= 0.);
#endif
  return rho;
}


inline void idealGas::PrimitiveState::setV(const size_t index, const Float val) {
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
  v[index] = val;
}


inline Float idealGas::PrimitiveState::getV(const size_t index) const {
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
  return v[index];
}


inline Float idealGas::PrimitiveState::getVSquared() const {
  return v[0] * v[0] + v[1] * v[1];
}


inline void idealGas::PrimitiveState::setP(const Float val) {
#if DEBUG_LEVEL > 0
  assert(val >= 0.);
#endif
  p = val;
}


inline Float idealGas::PrimitiveState::getP() const {
#if DEBUG_LEVEL > 0
  assert(p >= 0.);
#endif
  return p;
}


/**
 * Compute the local sound speed given a primitive state
 */
inline Float idealGas::PrimitiveState::getSoundSpeed() const {
  return std::sqrt(GAMMA * getP() / getRho());
}


/**
 * Get the total gas energy from a primitive state
 */
inline Float idealGas::PrimitiveState::getE() const {
  return 0.5 * getRho() * getVSquared() + getP() / GM1;
}


// Conserved State Stuff
// --------------------------

inline void idealGas::ConservedState::setRhov(const size_t index, const Float val) {
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
  rhov[index] = val;
}


inline Float idealGas::ConservedState::getRhov(const size_t index) const {
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
  return rhov[index];
}


inline Float idealGas::ConservedState::getRhoVSquared() const {
  return rhov[0] * rhov[0] + rhov[1] * rhov[1];
}


inline void idealGas::ConservedState::setE(const Float val) {
#if DEBUG_LEVEL > 0
  assert(val >= 0.);
#endif
  E = val;
}


inline Float idealGas::ConservedState::getE() const {
#if DEBUG_LEVEL > 0
  assert(E >= 0.);
#endif
  return E;
}


inline Float idealGas::ConservedState::getRho() const {
#if DEBUG_LEVEL > 0
  assert(rho >= 0.);
#endif
  return rho;
}


inline void idealGas::ConservedState::setRho(const Float val) {
#if DEBUG_LEVEL > 0
  assert(val >= 0.);
#endif
  rho = val;
}
