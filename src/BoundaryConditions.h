/**
 * @file BoundaryConditions
 * @brief contains boundary condition type enum.
 */

#pragma once

namespace BC {

  //! Boundary condition types
  enum BoundaryCondition {
    Periodic     = 0,
    Reflective   = 1,
    Transmissive = 2,
    Undefined,
    Count
  };


  //! Get a name for your boundary condition.
  inline const char* getBoundaryConditionName(const enum BoundaryCondition bc){

    switch(bc){
      case BoundaryCondition::Periodic:
        return "Periodic";
      case BoundaryCondition::Reflective:
        return "Reflective";
      case BoundaryCondition::Transmissive:
        return "Transmissive";
      case Count:
        return "Count";
      case BoundaryCondition::Undefined:
      default:
        return "Undefined";
    }
  }

} // namespace BC
