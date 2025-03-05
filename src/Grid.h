#pragma once

#include <vector>

#include "Cell.h"
#include "Logging.h"
#include "Parameters.h"


namespace grid {

  class Grid {
  private:
    std::vector<cell::Cell> _cells;

  public:
    // Constructors et al.
    // Forbid copying and moving - we want a singleton.
    Grid()                        = default; // constructor
    Grid(const Grid&)             = delete;  // copy constructor
    Grid(const Grid&&)            = delete;  // move constructor
    Grid  operator=(const Grid&)  = delete;  // copy operator
    Grid& operator=(const Grid&&) = delete;  // move assignment operator
    // TODO(mivkov): Deallocate grid here
    ~Grid() = default;

    //! Get a cell by its index. Here 1D and 2D versions.
    cell::Cell& getCell(const size_t i);
    cell::Cell& getCell(const size_t i, const size_t j, const parameters::Parameters& pars);

    //! Initialise the grid.
    void initGrid( const parameters::Parameters& pars );


    /**
     * @brief get the total mass of the grid.
     */
    float_t getTotalMass( const parameters::Parameters& pars );

    /**
     * @brief Pass in vector of initial vals, to be read from IC file.
     * In 1d this should be:
     *   [density, velocity, pressure]
     * In 2d this should be:
     * [density, velocity_x, velocity_y, pressure]
     */
    void setInitialConditions(size_t position, std::vector<float_t> vals,     const parameters::Parameters& pars);


    //! Run through the grid and get cstates from pstates
    void getCStatesFromPstates(const parameters::Parameters& pars);

    //! Run through the grid and get pstates from cstates
    void getPStatesFromCstates(const parameters::Parameters& pars);

    //! Reset all fluxes
    void resetFluxes(const parameters::Parameters& pars);

    //! enforce boundary conditions.
    void setBoundary(const parameters::Parameters& pars);

    //! Apply the boundary conditions from real to ghost cells.
    static void realToGhost(
      std::vector<cell::Cell*>,
      std::vector<cell::Cell*>,
      std::vector<cell::Cell*>,
      std::vector<cell::Cell*>,
      const parameters::Parameters& pars,
      const size_t dimension = 0
    );

    // static copy for global access
    static Grid  Instance;
    static Grid& getInstance() {
      return Instance;
    }
  };

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
inline cell::Cell& grid::Grid::getCell(const size_t i, const size_t j,
    const parameters::Parameters& pars
    ) {

  static size_t nxTot = pars.getNxTot();

#if DEBUG_LEVEL > 0
  if (Dimensions != 2) {
    error("This function is for 2D only!")
  }
#endif
  return _cells[i + j * nxTot];
}
