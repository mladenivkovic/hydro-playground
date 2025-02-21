#pragma once

#include <vector>

#include "Cell.h"


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
    cell::Cell& getCell(size_t i);
    cell::Cell& getCell(size_t i, size_t j);

    //! Initialise the grid.
    void initGrid();

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
    void setInitialConditions(size_t position, std::vector<float_t> vals);

    //! Run through the grid and get cstates from pstates
    void getCStatesFromPstates();

    //! Run through the grid and get pstates from cstates
    void getPStatesFromCstates();

    //! Reset all fluxes
    void resetFluxes();

    //! enforce boundary conditions.
    void setBoundary();

    //! Apply the boundary conditions from real to ghost cells.
    static void realToGhost(
      std::vector<cell::Cell*>,
      std::vector<cell::Cell*>,
      std::vector<cell::Cell*>,
      std::vector<cell::Cell*>,
      const size_t dimension = 0
    );

    // static copy for global access
    static Grid  Instance;
    static Grid& getInstance() {
      return Instance;
    }
  };

} // namespace grid
