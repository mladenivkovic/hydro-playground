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

    //! number of cells to use (i.e. excluding boundaries, including replications)
    //! (in each dimension)
    size_t _nx;

    //! number of cells to use (i.e. excluding boundaries) in single replicated
    //! region (in each dimension)
    size_t _nx_norep;

    //! cell size
    Float _dx;

    //! box size
    Float _boxsize;

    //! Number of Ghost cells at each edge
    size_t _nbc;

    //! Are we replicating the box?
    size_t _replicate;

    //! boundary condition
    BC::BoundaryCondition _boundaryType;

    //! Initialised?
    bool _initialised;


    /**
     * @brief get the total number of boundary cells per dimension.
     */
    [[nodiscard]] size_t _getNBCTot() const;


    //! Fetch the desired quantity for printing given a cell index
    Float _getQuanityForPrintout(cell::Cell& cell, std::string& quantity);


  public:
    Grid();
    ~Grid();

    // The grid is never intended to be used via several instances.
    // I could write all of this out, but I see no point.
    Grid(const Grid& other) = delete;
    Grid& operator=(const Grid& other) = delete;

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
     * @brief Initialise (and allocate) the cells.
     */
    void initCells();


    /**
     * @brief get the total mass of the grid.
     */
    Float getTotalMass();


    /**
     * @brief Replicate the initial conditions in every dimension.
     */
    void replicateICs();


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


    //! Print out the grid.
    void printGrid(bool boundaries = true, bool conserved = false);
    void printGrid(const char* quantity, bool boundaries = true);


    /**
     * @brief get the total number of cells per dimension. This includes
     * boundary cells and replicated cells.
     */
    [[nodiscard]] size_t getNxTot() const;


    /**
     * @brief get the index of the first actual (= non boundary/ghost) cell
     */
    [[nodiscard]] size_t getFirstCellIndex() const;


    /**
     * @brief get the index of the first actual (= non boundary/ghost) cell
     */
    [[nodiscard]] size_t getLastCellIndex() const;


    // Getters and setters


    /**
     * @brief Get the number of cells with actual content per dimension
     * i.e. excluding boundary cells, including replications
     */
    [[nodiscard]] size_t getNx() const;
    void                 setNx(const size_t nx);


    /**
     * @brief Get the number of cells per dimension
     * excluding boundaries and excluding replications
     */
    [[nodiscard]] size_t getNxNorep() const;
    void                 setNxNorep(const size_t nx);


    /**
     * @brief Get the type of boundary condition used
     */
    [[nodiscard]] BC::BoundaryCondition getBoundaryType() const;
    void                                setBoundaryType(BC::BoundaryCondition boundary);


    /**
     * @brief Get the number of boundary cells on each side of the box
     */
    [[nodiscard]] size_t getNBC() const;
    void                 setNBC(size_t nbc);


    /**
     * @brief Are we replicating the box?
     */
    [[nodiscard]] size_t getReplicate() const;
    void                 setReplicate(size_t replicate);


    /**
     * @brief Get the cell size
     */
    [[nodiscard]] Float getDx() const;
    void                setDx(const Float dx);


    /**
     * @brief Get the simulation box size
     */
    [[nodiscard]] Float getBoxsize() const;
    void                setBoxsize(const Float boxsize);
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
    error("This function is for 1D only!");
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

#if DEBUG_LEVEL > 0
  if (Dimensions != 2) {
    error("This function is for 2D only!");
  }
#endif

  size_t nxTot = getNxTot();
  return _cells[i + j * nxTot];
}


inline size_t grid::Grid::getNx() const {
  return _nx;
}


inline void grid::Grid::setNx(const size_t nx) {
  _nx = nx;
}


inline size_t grid::Grid::getNxNorep() const {
  return _nx_norep;
}


inline void grid::Grid::setNxNorep(const size_t nx) {
  _nx_norep = nx;
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


inline size_t grid::Grid::getReplicate() const {
  return _replicate;
}


inline void grid::Grid::setReplicate(const size_t replicate) {
  _replicate = replicate;
}


inline Float grid::Grid::getDx() const {
  return _dx;
}


inline void grid::Grid::setDx(const Float dx) {
  _dx = dx;
}


inline Float grid::Grid::getBoxsize() const {
  return _boxsize;
}


inline void grid::Grid::setBoxsize(const Float boxsize) {
  _boxsize = boxsize;
}


inline size_t grid::Grid::_getNBCTot() const {
  return 2 * getNBC();
}


inline size_t grid::Grid::getNxTot() const {
  return getNx() + _getNBCTot();
}


inline size_t grid::Grid::getFirstCellIndex() const {
  return getNBC();
}


inline size_t grid::Grid::getLastCellIndex() const {
  return getNx() + getNBC();
}
