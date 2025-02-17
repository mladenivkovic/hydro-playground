#pragma once

#include <string>
#include <vector>

#include "Config.h"
#include "Gas.h"


namespace cell {
  class Cell;

  // template <int Dimensions>
  class Grid {
  private:
    std::vector<Cell> _cells;

  public:
    Grid();
    Cell& getCell(size_t i);
    Cell& getCell(size_t i, size_t j);

    void InitGrid();
    /**
     * @brief get the total mass of the grid.
     */
    float_t GetTotalMass();

    //! Pass in vector of initial vals, to be read from IC file.
    //! In 1d this should be:
    //! [density, velocity, pressure]
    //! In 2d this should be:
    //! [density, velocity_x, velocity_y, pressure]
    //!
    void SetInitialConditions(int position, std::vector<float_t> vals);

    void getCStatesFromPstates();
    void getPStatesFromCstates();
    void resetFluxes();

    void setBoundary();
    void realToGhost(
      std::vector<Cell*>,
      std::vector<Cell*>,
      std::vector<Cell*>,
      std::vector<Cell*>,
      int dimension = 0
    );

    // static copy for global access
    static Grid  Instance;
    static Grid& getInstance() {
      return Instance;
    }
  };

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
