#pragma once

#include <string>

#include "Config.h"
#include "Gas.h"


namespace cell {

  class Cell {

  public:

    //! Standard constructor
    Cell();

    //! copy assignment, for copying boundary data
    //! Return reference to this, for chaining calls
    // TODO: This doesn't compile
    // Cell& operator=(const Cell& other) = default;

    void CopyBoundaryData(const Cell* real);

    void CopyBoundaryDataReflective(const Cell* real, const size_t dimension);

    //! Calls conserved to primitive on the members
    void ConservedToPrimitive() {
      _prim.ConservedToPrimitive(_cons);
    };
    //! Calls primitive to conserved on the members
    void PrimitiveToConserved() {
      _cons.PrimitiveToConserved(_prim);
    };

  private:
    //! Cell ID
    int _id;

    //! x position of cell centre
    float_t _x;

    //! y position of cell centre
    float_t _y;

    //! Primitive gas state
    IdealGas::PrimitiveState _prim;

    //! Conserved gas state
    IdealGas::ConservedState _cons;

    //! Fluxes of primitive state
    IdealGas::PrimitiveState _pflux;

    //! Fluxes of conserved state
    IdealGas::ConservedState _cflux;

    //! Acceleration
    // std::array<float_t, Dimensions> _acc;

  public:
    // leaving these for now
    // std::string getIndexString();

    // getters and setters
    void setX(float_t x);
    void setY(float_t y);

    void                      setId(const int id);
    int                       getID() const;

    //! Get cell index(es) in grid
    std::pair<size_t, size_t> getIJ();


    //! return refs to the above
    IdealGas::PrimitiveState& getPrim() {
      return _prim;
    }
    IdealGas::ConservedState& getCons() {
      return _cons;
    }
    IdealGas::PrimitiveState& getPFlux() {
      return _pflux;
    }
    IdealGas::ConservedState& getCFlux() {
      return _cflux;
    }

    // const versions to shush the compiler
    const IdealGas::PrimitiveState& getPrim() const {
      return _prim;
    }
    const IdealGas::ConservedState& getCons() const {
      return _cons;
    }
  };

} // namespace cell
