
OpenMP Offloading
==================


AMD GPUs
-----------------------


Compile commands and flags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table:: Compile commands for LLVM
    :header: "Language", "Command", "Flag"

    C,       `amdclang` / `clang`,       `-fopenmp --offload-arch=<gfx###>`
    C++,     `amdclang++` / `clang++`,   `-fopenmp --offload-arch=<gfx###>`
    Fortran, `amdflang` / `flang`,       `-fopenmp --offload-arch=<gfx###>`


.. csv-table:: Compile commands for GCC
    :header: "Language", "Command", "Flag"

    C,       `gcc`,        `-fopenmp --foffload=-march=<gfx###>`
    C++,     `g++`,        `-fopenmp --foffload=-march=<gfx###>`
    Fortran, `gfortran`,   `-fopenmp --foffload=-march=<gfx###>`



.. csv-table:: Architecture targets
    :header: Offloading Target (CPU/GPU/GCD), Architecture `<gfx###>`

    AMD MI300 Series,       `gfx942`
    AMD MI200 Series,       `gfx90a`
    AMD MI100,              `gfx908`
    Native Host (CPU),      `-fopenmp-targets=amdgcn-amd-amdhsa`





Necessary Steps and Other Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- First step: Trying to get a simple pragma going for the first loop (zeroing out fluxes) using

```
#pragma omp target teams distribute parallel for
```

- **Problems**:

- Make sure you add the `-fopenmp` et al. flags to both the compiler *and the linker*.
- If the loop to be offloaded calls functions/subroutines, those functions/subroutines either need
  to be inlined, or you need to declare them a target using

.. code-block:: cpp

    #pragma omp declare target
    void your_function(){...}
    #pragma omp end declare target


- You need to do the same for data/variables you want to be available on the device:

.. code-block:: cpp

    #pragma omp declare target
    constexpr int Dimensions = 2;
    #pragma omp end declare target

- AMD's clang++ (v18.0.0 and v21.0.0) couldn't offload part of my code which uses
  `std::function(...)`. I used that to select the function to be applied to the cells to enforce
  boundary conditions. Several attempts with lambdas, inlining etc didn't work out.
  Instead, I switched to good old function pointers. That worked out.

- I keep getting a lot of `[-Wopenmp-mapping]` warnings in relation to having arrays of pointers.
  Having explicit copy, move etc constructors and operators didn't help. The AMD experts assure me
  that won't be an issue.

.. code-block:: bash

    warning: Type 'Boundary' is not trivially copyable and not guaranteed to be mapped correctly [-Wopenmp-mapping]

  


