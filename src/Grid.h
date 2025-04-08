#pragma once

#include <vector>

#include "Cell.h"
#include "Logging.h"
#include "Parameters.h"
#include "Utils.h"


class Grid {
private:
  //! Cell array.
  Cell* _cells;

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
  BC::BoundaryCondition _boundary_type;


  /**
   * @brief get the total number of boundary cells per dimension.
   */
  [[nodiscard]] size_t _getNBCTot() const;


  //! Fetch the desired quantity for printing given a cell index
  //! TODO will not work on device
  Float _getQuanityForPrintout(Cell& cell, std::string& quantity);


public:
  HOST explicit Grid(const Parameters& params);
  HOST_DEVICE  ~Grid();

  // The grid is never intended to be used via several instances.
  // I could write all of this out, but I see no point.
  Grid(const Grid& other)            = delete;
  Grid& operator=(const Grid& other) = delete;

  /**
   * Get a cell by its index. Here 1D and 2D versions.
   */
  HOST_DEVICE  Cell& getCell(const size_t i);
  HOST_DEVICE  Cell& getCell(const size_t i, const size_t j);


  /**
   * @brief Initialise (and allocate) the cells.
   */
  HOST void initCells();


  /**
   * @brief get the total mass of the grid.
   */
  HOST_DEVICE Float collectTotalMass();


  /**
   * @brief Replicate the initial conditions in every dimension.
   */
  HOST void replicateICs();


  //! Run through the grid and get cstates from pstates
  HOST_DEVICE void convertPrim2Cons();


  //! Run through the grid and get pstates from cstates
  void convertCons2Prim();


  //! Reset all fluxes
  void resetFluxes();


  //! Apply boundary conditions.
  void applyBoundaryConditions();


  //! Apply the boundary conditions from real to ghost cells.
  void realToGhost(
    std::vector<Cell*> real_left,
    std::vector<Cell*> real_right,
    std::vector<Cell*> ghost_left,
    std::vector<Cell*> ghost_right,
    const size_t       dimension = 0
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
  HOST_DEVICE [[nodiscard]] size_t getNx() const;
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


// --------------------------------------------------------
// Definitions
// --------------------------------------------------------


/**
 * Get (reference to) a cell by its index.
 * This is for the 1D grid.
 */
inline Cell& Grid::getCell(const size_t i) {

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
inline Cell& Grid::getCell(const size_t i, const size_t j) {

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


HOST_DEVICE inline size_t Grid::getNx() const {
  return _nx;
}


inline void Grid::setNx(const size_t nx) {
  _nx = nx;
}


inline size_t Grid::getNxNorep() const {
  return _nx_norep;
}


inline void Grid::setNxNorep(const size_t nx) {
  _nx_norep = nx;
}


inline BC::BoundaryCondition Grid::getBoundaryType() const {
  return _boundary_type;
}


inline void Grid::setBoundaryType(BC::BoundaryCondition boundaryType) {
  _boundary_type = boundaryType;
}


inline size_t Grid::getNBC() const {
  return _nbc;
}


inline void Grid::setNBC(const size_t bc) {
  _nbc = bc;
}


inline size_t Grid::getReplicate() const {
  return _replicate;
}


inline void Grid::setReplicate(const size_t replicate) {
  _replicate = replicate;
}


inline Float Grid::getDx() const {
  return _dx;
}


inline void Grid::setDx(const Float dx) {
  _dx = dx;
}


inline Float Grid::getBoxsize() const {
  return _boxsize;
}


inline void Grid::setBoxsize(const Float boxsize) {
  _boxsize = boxsize;
}


inline size_t Grid::_getNBCTot() const {
  return 2 * getNBC();
}


inline size_t Grid::getNxTot() const {
  return getNx() + _getNBCTot();
}


inline size_t Grid::getFirstCellIndex() const {
  return getNBC();
}


inline size_t Grid::getLastCellIndex() const {
  return getNx() + getNBC();
}
