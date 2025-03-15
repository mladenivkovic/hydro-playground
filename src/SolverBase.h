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

    //! Current step
    size_t stepCount;

    //! Reference to runtime parameters
    parameters::Parameters& params;

    //! Reference to the grid.
    grid::Grid& grid;

    //! Compute current time step size.
    void computeDt();

  public:

    SolverBase(parameters::Parameters& params_, grid::Grid& grid_);
    ~SolverBase() = default;

    //! Call the actual solver.
    void solve();

    //! Run a single step.
    virtual void step(){
      // Virtual functions need a definition too somewhere.
      error("This should never be called.");
    };

  };

} // namespace solver

