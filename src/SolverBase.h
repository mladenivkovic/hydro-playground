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


  public:

    SolverBase();
    ~SolverBase() = default;

    //! Call the actual solver.
    void solve(parameters::Parameters& params, grid::Grid& grid);

    //! Run a single step.
    virtual void step(){
      // Virtual functions need a definition too somewhere.
      error("This should never be called.");
    };

  };

} // namespace solver

