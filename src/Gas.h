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
    Precision                rho; /* density */
    std::array<Precision, 2> u;   /* velocity vector. u[0] = ux, u[1] = uy */
    Precision                p;   /* pressure */


  public:
    /* Standard constructor, init variables to 0 */
    PrimitiveState();

    /* putting this in just in case it's needed */
    void resetToInitialState() { *this = PrimitiveState(); }

    void ConservedToPrimitive(const ConservedState& conservedState);

    Precision getSoundSpeed();
    Precision getEnergy();

    /*
    Getters and setters!
    */
    /* Setter for Rho */
    void      setRho(const Precision val);
    Precision getRho() const;

    /* same for u */
    void      setU(int index, const Precision val);
    Precision getU(const int index) const;

    /*used a lot, made a function for it*/
    Precision getUSquared() const;

    void      setP(const Precision val);
    Precision getP() const;
  };

  class ConservedState {
  private:
    Precision                rho;
    std::array<Precision, 2> rhou;
    Precision                E;

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
    void      setRho(const Precision& val);
    Precision getRho() const;

    /* same for u */
    void      setRhou(int index, const Precision val);
    Precision getRhou(int index) const;
    Precision getRhoUSquared() const;

    void      setE(const Precision val);
    Precision getE() const;
  };
} // namespace IdealGas
