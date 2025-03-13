/**
 * @file BoundaryConditions
 * @brief contains boundary condition type enum.
 */

#pragma once

namespace BC {

  //! Boundary condition types
  enum class BoundaryCondition {
    Periodic     = 0,
    Reflective   = 1,
    Transmissive = 2
  };

} // namespace BC
