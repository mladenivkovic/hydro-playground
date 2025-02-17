#pragma once

#include <string>
#include <vector>

#include "Config.h"
#include "Gas.h"


namespace cell {
  class Cell;

  class Cell {
  public:
    //! Standard constructor
    Cell();
    //! copy assignment, for copying boundary data
    //! Return reference to this, for chaining calls
    // TODO: This doesn't compile
    // Cell& operator=(const Cell& other) = default;

    //! Should be called from within the ghost
    void CopyBoundaryData(const Cell* real);
    //! Should be called from within the ghost
    void CopyBoundaryDataReflective(const Cell* real, const int dimension);

    //! Calls conserved to primitive on the members
    void ConservedToPrimitive() {
      _prim.ConservedToPrimitive(_cons);
    };
    //! Calls primitive to conserved on the members
    void PrimitiveToConserved() {
      _cons.PrimitiveToConserved(_prim);
    };

  private:
    int _id;

    /*
    Positions of centres
    */
    float_t _x;
    float_t _y;

    IdealGas::PrimitiveState _prim;
    IdealGas::ConservedState _cons;

    IdealGas::PrimitiveState _pflux;
    IdealGas::ConservedState _cflux;

    std::array<float_t, Dimensions> _acc;

  public:
    // leaving these for now
    std::string getIndexString();

    // getters and setters
    void setX(float_t x);
    void setY(float_t y);

    void                      setId(const int id);
    int                       getID() const;
    std::pair<size_t, size_t> getIJ();


    // return refs to the above
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
