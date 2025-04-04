#include "BoundaryConditions.h"

/**
 * Apply the periodic boundary conditions.
 *
 * @param realL:     array of pointers to real cells with lowest index
 * @param realR:     array of pointers to real cells with highest index
 * @param ghostL:    array of pointers to ghost cells with lowest index
 * @param ghostR:    array of pointers to ghost cells with highest index
 * @param nbc:       number of boundary cells.
 * @param dimension: dimension integer. 0 for x, 1 for y. Needed for
 *                   reflective boundary conditions.
 */
void BC::periodic(
        Boundary&    real_left,
        Boundary&    real_right,
        Boundary&    ghost_left,
        Boundary&    ghost_right,
        const size_t nbc,
        const size_t dimension){

#if DEBUG_LEVEL > 0
  assert(real_left.size() == nbc);
  assert(real_right.size() == nbc);
  assert(ghost_left.size() == nbc);
  assert(ghost_right.size() == nbc);
#endif

  for (size_t i = 0; i < nbc; i++) {
    ghost_left[i]->copyBoundaryData(real_right[i]);
    ghost_right[i]->copyBoundaryData(real_left[i]);
  }
}


/**
 * Apply the reflective boundary conditions.
 *
 * @param realL:     array of pointers to real cells with lowest index
 * @param realR:     array of pointers to real cells with highest index
 * @param ghostL:    array of pointers to ghost cells with lowest index
 * @param ghostR:    array of pointers to ghost cells with highest index
 * @param nbc:       number of boundary cells.
 * @param dimension: dimension integer. 0 for x, 1 for y. Needed for
 *                   reflective boundary conditions.
 */
void BC::reflective(
        Boundary&    real_left,
        Boundary&    real_right,
        Boundary&    ghost_left,
        Boundary&    ghost_right,
        const size_t nbc,
        const size_t dimension){

#if DEBUG_LEVEL > 0
  assert(real_left.size() == nbc);
  assert(real_right.size() == nbc);
  assert(ghost_left.size() == nbc);
  assert(ghost_right.size() == nbc);
#endif

  for (size_t i = 0; i < nbc; i++) {
    ghost_left[i]->copyBoundaryDataReflective(real_left[real_left.size() - i - 1], dimension);
    ghost_right[i]->copyBoundaryDataReflective(real_right[real_right.size() - i - 1], dimension);
  }
}


/**
 * Apply the transmissive boundary conditions.
 *
 * @param realL:     array of pointers to real cells with lowest index
 * @param realR:     array of pointers to real cells with highest index
 * @param ghostL:    array of pointers to ghost cells with lowest index
 * @param ghostR:    array of pointers to ghost cells with highest index
 * @param nbc:       number of boundary cells.
 * @param dimension: dimension integer. 0 for x, 1 for y. Needed for
 *                   reflective boundary conditions.
 */
void BC::transmissive(
        Boundary&    real_left,
        Boundary&    real_right,
        Boundary&    ghost_left,
        Boundary&    ghost_right,
        const size_t nbc,
        const size_t dimension){

#if DEBUG_LEVEL > 0
  assert(real_left.size() == nbc);
  assert(real_right.size() == nbc);
  assert(ghost_left.size() == nbc);
  assert(ghost_right.size() == nbc);
#endif
  for (size_t i = 0; i < nbc; i++) {
    ghost_left[i]->copyBoundaryData(real_left[real_left.size() - i - 1]);
    ghost_right[i]->copyBoundaryData(real_right[real_right.size() - i - 1]);
  }
}



