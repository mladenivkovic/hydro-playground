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
    float_t rho;

    //! velocity
    std::array<float_t, Dimensions> v;

    //! pressure
    float_t p;


  public:
    PrimitiveState();
    PrimitiveState(const float_t rho, const std::array<float_t, Dimensions> vel, const float_t p);
    PrimitiveState(const float_t rho, const float_t vx, const float_t p);
    PrimitiveState(const float_t rho, const float_t vx, const float_t vy, const float_t p);

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
    [[nodiscard]] float_t getSoundSpeed() const;

    //! Get the total gas energy from a primitive state
    [[nodiscard]] float_t getE() const;

    //! Get a string of the state.
    [[nodiscard]] std::string toString() const;


    // Getters and setters!

    // Setter for Rho
    void                  setRho(const float_t val);
    [[nodiscard]] float_t getRho() const;

    // same for u
    void                  setV(const std::size_t index, const float_t val);
    [[nodiscard]] float_t getV(const std::size_t index) const;

    // used a lot, made a function for it
    [[nodiscard]] float_t getVSquared() const;

    void                  setP(const float_t val);
    [[nodiscard]] float_t getP() const;
  };


  /**
   * @brief Holds a conserved state (density, momentum, energy)
   */
  class ConservedState {
  private:
    //! Density
    float_t rho;

    //! Momentum: rho * v
    std::array<float_t, Dimensions> rhov;

    //! Energy
    float_t E;

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
    void                  setRho(const float_t val);
    [[nodiscard]] float_t getRho() const;

    // same for u
    void                  setRhov(const std::size_t index, const float_t val);
    [[nodiscard]] float_t getRhov(const std::size_t index) const;
    [[nodiscard]] float_t getRhoVSquared() const;

    void                  setE(const float_t val);
    [[nodiscard]] float_t getE() const;
  };
} // namespace idealGas


// --------------------------------------------------------
// Definitions
// --------------------------------------------------------

// Primitive State Stuff
// --------------------------

inline void idealGas::PrimitiveState::setRho(const float_t val) {
#if DEBUG_LEVEL > 0
  assert(val >= 0.);
#endif
  rho = val;
}


inline float_t idealGas::PrimitiveState::getRho() const {
#if DEBUG_LEVEL > 0
  assert(rho >= 0.);
#endif
  return rho;
}


inline void idealGas::PrimitiveState::setV(const size_t index, const float_t val) {
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
  v[index] = val;
}


inline float_t idealGas::PrimitiveState::getV(const size_t index) const {
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
  return v[index];
}


inline float_t idealGas::PrimitiveState::getVSquared() const {
  return v[0] * v[0] + v[1] * v[1];
}


inline void idealGas::PrimitiveState::setP(const float_t val) {
#if DEBUG_LEVEL > 0
  assert(val >= 0.);
#endif
  p = val;
}


inline float_t idealGas::PrimitiveState::getP() const {
#if DEBUG_LEVEL > 0
  assert(p >= 0.);
#endif
  return p;
}


/**
 * Compute the local sound speed given a primitive state
 */
inline float_t idealGas::PrimitiveState::getSoundSpeed() const {
  return std::sqrt(GAMMA * getP() / getRho());
}


/**
 * Get the total gas energy from a primitive state
 */
inline float_t idealGas::PrimitiveState::getE() const {
  return 0.5 * getRho() * getVSquared() + getP() / GM1;
}


// Conserved State Stuff
// --------------------------

inline void idealGas::ConservedState::setRhov(const size_t index, const float_t val) {
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
  rhov[index] = val;
}


inline float_t idealGas::ConservedState::getRhov(const size_t index) const {
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
  return rhov[index];
}


inline float_t idealGas::ConservedState::getRhoVSquared() const {
  return rhov[0] * rhov[0] + rhov[1] * rhov[1];
}


inline void idealGas::ConservedState::setE(const float_t val) {
#if DEBUG_LEVEL > 0
  assert(val >= 0.);
#endif
  E = val;
}


inline float_t idealGas::ConservedState::getE() const {
#if DEBUG_LEVEL > 0
  assert(E >= 0.);
#endif
  return E;
}


inline float_t idealGas::ConservedState::getRho() const {
#if DEBUG_LEVEL > 0
  assert(rho >= 0.);
#endif
  return rho;
}


inline void idealGas::ConservedState::setRho(const float_t val) {
#if DEBUG_LEVEL > 0
  assert(val >= 0.);
#endif
  rho = val;
}
