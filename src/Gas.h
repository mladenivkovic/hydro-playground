#pragma once

#include <array>
#include <math.h>

#include "Config.h"
#include "Constants.h"


namespace IdealGas {
  // hacky forward declaration, ConservedToPrimitive doesn't work without it
  class ConservedState;
  class PrimitiveState;

  class PrimitiveState {
  private:
    float_t                rho; /* density */
    std::array<float_t, 2> u;   /* velocity vector. u[0] = ux, u[1] = uy */
    float_t                p;   /* pressure */


  public:
    /* Standard constructor, init variables to 0 */
    PrimitiveState();

    /* putting this in just in case it's needed */
    void resetToInitialState() { *this = PrimitiveState(); }

    void ConservedToPrimitive(const ConservedState& conservedState);

    float_t getSoundSpeed();
    float_t getEnergy();

    /*
    Getters and setters!
    */
    /* Setter for Rho */
    void      setRho(float_t val);
    float_t getRho() const;

    /* same for u */
    void      setU(int index, const float_t val);
    float_t getU(const int index) const;

    /*used a lot, made a function for it*/
    float_t getUSquared() const;

    void      setP(const float_t val);
    float_t getP() const;
  };

  class ConservedState {
  private:
    float_t                rho;
    std::array<float_t, 2> rhou;
    float_t                E;

  public:
    /* Standard constructor, init variables to 0 */
    ConservedState();

    /* putting this in in case it's needed */
    void resetToInitialState() { *this = ConservedState(); }

    void PrimitiveToConserved(const PrimitiveState& primState);
    void GetCFluxFromPstate(const PrimitiveState& p, int dimension);
    void GetCFluxFromCstate(const ConservedState& c, int dimension);

    /*
    Getters and setters!
    */
    void      setRho(const float_t& val);
    float_t getRho() const;

    /* same for u */
    void      setRhou(int index, const float_t val);
    float_t getRhou(int index) const;
    float_t getRhoUSquared() const;

    void      setE(const float_t val);
    float_t getE() const;
  };
} // namespace IdealGas
