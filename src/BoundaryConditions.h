/**
 * @file BoundaryConditions
 * @brief contains boundary condition type enum.
 */

#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include "Cell.h"


/**
 * A simple container for boundary cells. Just think of it as an array of
 * pointers to cells.
 *
 * I added this trying to avoid "<data type> is not trivially copyable and not
 * guaranteed to be mapped correctly" warnings with OpenMP offloading, but this
 * didn't really solve it. But it lets us avoid std::vectors and is simple enough,
 * so let's keep it for now.
 */
class Boundary {

private:
  size_t _n;
  Cell** _cells;

public:
  explicit Boundary(const size_t N):
    _n(N),
    _cells(new Cell*[N]) {};

  ~Boundary() {
    delete[] _cells;
  }

  // Copy operator
  Boundary(const Boundary& other):
    _n(other._n) {
    for (size_t i = 0; i < _n; i++)
      _cells[i] = other._cells[i];
  }

  // Copy assignment
  Boundary& operator=(const Boundary& other) {
    this->_n = other._n;
    for (size_t i = 0; i < _n; i++)
      this->_cells[i] = other._cells[i];
    return *this;
  }

  // Move operator
  Boundary(const Boundary&& other) noexcept:
    _n(other._n) {
    _cells = std::move(other._cells);
  }

  // Move assignment
  Boundary& operator=(Boundary&& other) noexcept {
    this->_n     = other._n;
    this->_cells = other._cells;
    return *this;
  }

  // overload [] operator for access
  Cell*& operator[](size_t index) {
#if DEBUG_LEVEL > 0
    if (index > _n)
      error("Invalid index:" + std::to_string(index));
#endif
    return _cells[index];
  }

  /**
   * Get the size of the cell pointer array.
   */
  size_t size() {
    return _n;
  }
};

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
  inline const char* getBoundaryConditionName(const enum BoundaryCondition bc) {

    switch (bc) {
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


  using BoundaryFunctionPtr = void(*)(Boundary&, Boundary&, Boundary&, Boundary&, const size_t, const size_t);
} // namespace BC


