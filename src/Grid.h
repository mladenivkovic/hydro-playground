#pragma once

#include <vector>

#include "Cell.h"
#include "Logging.h"
#include "Parameters.h"


namespace grid {

  class Grid {
  private:

    //! Cell array.
    cell::Cell* _cells;

    //! number of cells to use (in each dimension)
    size_t _nx;

    //! number of mesh points, including boundary cells
    size_t _nxTot;

    //! cell size
    float_t _dx;

    //! Number of Ghost cells at each edge
    size_t _nbc;

    //! boundary condition
    BC::BoundaryCondition _boundaryType;

    //! Initialised?
    bool _initialised;

  public:
    Grid();
    ~ Grid();
    // TODO(mivkov): Other constructors/operators needed here for completeness.
    // At least delete.


    /**
     * Get a cell by its index. Here 1D and 2D versions.
     */
    cell::Cell& getCell(const size_t i);
    cell::Cell& getCell(const size_t i, const size_t j);


    /**
     * @brief Initialise the grid.
     */
    void initGrid(const parameters::Parameters& pars);


    /**
     * @brief get the total mass of the grid.
     */
    float_t getTotalMass();


    /**
     * @brief Pass in vector of initial vals, to be read from IC file.
     * In 1d this should be:
     *   [density, velocity, pressure]
     * In 2d this should be:
     * [density, velocity_x, velocity_y, pressure]
     */
    void setInitialConditions(
      size_t position, std::vector<float_t> vals, const parameters::Parameters& pars
    );


    //! Run through the grid and get cstates from pstates
    void getCStatesFromPstates();


    //! Run through the grid and get pstates from cstates
    void getPStatesFromCstates();


    //! Reset all fluxes
    void resetFluxes();


    //! enforce boundary conditions.
    void setBoundary();


    //! Apply the boundary conditions from real to ghost cells.
    void realToGhost(
      std::vector<cell::Cell*> realLeft,
      std::vector<cell::Cell*> realRight,
      std::vector<cell::Cell*> ghostLeft,
      std::vector<cell::Cell*> ghostRight,
      const size_t             dimension = 0
    );

    // Getters and setters

    /**
     * @brief Get the number of cells with actual content per dimension
     */
    [[nodiscard]] size_t getNx() const;
    void                 setNx(const size_t nx);


    /**
     * @brief Get the type of boundary condition used
     */
    [[nodiscard]] BC::BoundaryCondition getBoundaryType() const;
    void                            setBoundaryType(BC::BoundaryCondition boundary);


    /**
     * @brief Get the number of boundary cells on each side of the box
     */
    [[nodiscard]] size_t getNBC() const;
    void                 setNBC(size_t nbc);


    /**
     * @brief get the total number of boundary cells per dimension.
     */
    [[nodiscard]] size_t getNBCTot() const;


    /**
     * @brief get the total number of cells per dimension. This includes
     * boundary cells.
     * @TODO: what to do with replication
     */
    [[nodiscard]] size_t getNxTot() const;


    /**
     * @brief Get the cell size
     */
    [[nodiscard]] float_t getDx() const;
    void                  setDx(const float_t dx);

  }; // class Grid

} // namespace grid


// --------------------------------------------------------
// Definitions
// --------------------------------------------------------


/**
 * Get (reference to) a cell by its index.
 * This is for the 1D grid.
 */
inline cell::Cell& grid::Grid::getCell(const size_t i) {

#if DEBUG_LEVEL > 0
  if (Dimensions != 1) {
    error("This function is for 1D only!")
  }
#endif
  return _cells[i];
}


/**
 * Get (reference to) a cell by its index.
 * This is for the 2D grid.
 */
inline cell::Cell& grid::Grid::getCell(const size_t i, const size_t j) {

#if DEBUG_LEVEL > 1
  if (_cells == nullptr)
    error("Cells array not allocated.");

#endif


  size_t nxTot = getNxTot();

#if DEBUG_LEVEL > 0
  if (Dimensions != 2) {
    error("This function is for 2D only!")
  }
#endif
  return _cells[i + j * nxTot];
}


inline size_t grid::Grid::getNx() const {
  return _nx;
}


inline void grid::Grid::setNx(const size_t nx) {
  _nx = nx;
}


inline BC::BoundaryCondition grid::Grid::getBoundaryType() const {
  return _boundaryType;
}


inline void grid::Grid::setBoundaryType(BC::BoundaryCondition boundaryType) {
  _boundaryType = boundaryType;
}


inline size_t grid::Grid::getNBC() const {
  return _nbc;
}


inline void grid::Grid::setNBC(const size_t bc) {
  _nbc = bc;
}


inline size_t grid::Grid::getNBCTot() const {
  return 2 * getNBC();
}


inline size_t grid::Grid::getNxTot() const {
  return getNx() + 2 * getNBC();
}


inline float_t grid::Grid::getDx() const {
  return _dx;
}


inline void grid::Grid::setDx(const float_t dx) {
  _dx = dx;
}


