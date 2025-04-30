/**
 * @file RiemannBase.h
 * @brief Base class for Riemann solvers.
 */

#pragma once

#include "Config.h"
#include "Gas.h"
#include "Logging.h"
#include "Utils.h"


class RiemannBase {

protected:
  //! The left state
  PrimitiveState& _left;

  //! The right state
  PrimitiveState& _right;

  //! Star state pressure
  Float _pstar;

  //! Star state velocity
  Float _vstar;

  //! In which dimension/direction to solve the problem.
  size_t _dim;

  //! Do we have vacuum generating conditions?
  __host__ __device__ bool hasVacuum();

  //! Get the vacuum solution
  template <Device>
  __host__ __device__ PrimitiveState solveVacuum();

  //! Sample the solved Riemann problem.
  template <Device>
  __host__ __device__ ConservedFlux sampleSolution();

public:
  __host__ __device__
  RiemannBase(PrimitiveState& l, PrimitiveState& r, const size_t dimension):
    _left(l),
    _right(r),
    _dim(dimension) {};
  ~RiemannBase() = default;

  //! Call the actual solver. Make it pure virtual
  virtual ConservedFlux solve() = 0;

  //! Usual template trick will not work here since this is a virtual function.
  //! Virtualisation is resolved at runtime and therefore not compatible with
  //! templates.
  __device__ virtual ConservedFlux solveOnGpu() = 0;
};


/**
* Do we have a vacuum or vacuum generating conditions?
*
* Section 3.5 in theory document, and eq. 86
*/
inline bool RiemannBase::hasVacuum() {

  if (_left.getRho() <= cst::SMALLRHO)
    return true;
  if (_right.getRho() <= cst::SMALLRHO)
    return true;

  Float delta_v = _right.getV(_dim) - _left.getV(_dim);
  Float v_crit  = cst::TWOOVERGM1 * (_left.getSoundSpeed() + _right.getSoundSpeed());

  return delta_v >= v_crit;
}

