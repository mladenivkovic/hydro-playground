#pragma once

#include <vector>

#include "Cell.h"


namespace grid {

  // template <int Dimensions>
  class Grid {
  private:
    std::vector<cell::Cell> _cells;

  public:
    Grid() = default;
    // Forbid copying and moving.
    Grid(const Grid& ) = delete;
    void operator=(const Grid&) = delete;
    Grid(const Grid&& ) = delete;
    ~Grid() = default;

    cell::Cell& getCell(size_t i);
    cell::Cell& getCell(size_t i, size_t j);

    void InitGrid();
    /**
     * @brief get the total mass of the grid.
     */
    float_t GetTotalMass();

    /**
     * @brief Pass in vector of initial vals, to be read from IC file.
     * In 1d this should be:
     *   [density, velocity, pressure]
     * In 2d this should be:
     * [density, velocity_x, velocity_y, pressure]
     */
    void SetInitialConditions(int position, std::vector<float_t> vals);

    void getCStatesFromPstates();
    void getPStatesFromCstates();
    void resetFluxes();

    void setBoundary();
    void realToGhost(
      std::vector<cell::Cell*>,
      std::vector<cell::Cell*>,
      std::vector<cell::Cell*>,
      std::vector<cell::Cell*>,
      int dimension = 0
    );

    // static copy for global access
    static Grid  Instance;
    static Grid& getInstance() {
      return Instance;
    }
  };

} // namespace grid

