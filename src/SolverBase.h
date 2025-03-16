/**
 * @file SolverBase.h
 * @brief Base class for hydro solvers.
 */

#pragma once

#include "Config.h"
#include "Grid.h"
#include "Parameters.h"


namespace solver {


  class SolverBase {

  protected:
    //! Current time
    Float t;

    //! Current time step
    Float dt;

    //! Previous time step
    Float dt_old;

    //! Current step
    size_t step_count;

    //! Total mass in grid.
    Float total_mass_init;
    Float total_mass_current;

    //! Reference to runtime parameters
    parameters::Parameters& params;

    //! Reference to the grid.
    grid::Grid& grid;

    //! Compute current time step size.
    void computeDt();

    //! Apply the actual time integration step.
    void integrateHydro(const size_t dim);

    //! Do we still need to run?
    bool keepRunning();

    //! Write a log to screen, if requested.
    void writeLog(const std::string& timingstr);

    //! Write the log header to screen
    void writeLogHeader();

    //! Apply the time update for a pair of cells.
    static void applyTimeUpdate(cell::Cell& left, cell::Cell& right, const Float dtdx);

  public:
    SolverBase(parameters::Parameters& params_, grid::Grid& grid_);
    ~SolverBase() = default;

    //! Call the actual solver.
    void solve();

    //! Run a single step.
    virtual void step() {
      // Virtual functions need a definition too somewhere.
      error("This should never be called.");
    };
  };

} // namespace solver
