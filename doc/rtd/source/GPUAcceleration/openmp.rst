
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

    C,       `amdclang` / `clang`,       `-fopenmp --foffload=-march=<gfx###>`
    C++,     `amdclang++` / `clang++`,   `-fopenmp --foffload=-march=<gfx###>`
    Fortran, `amdflang` / `flang`,       `-fopenmp --foffload=-march=<gfx###>`



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

- Make sure you add the `-fopenmp` et al. flags to both the compiler and the linker.
- If the loop to be offloaded calls functions/subroutines, those functions/subroutines either need to be inlined, or you need to declare them a target using

.. code-block:: cpp

    #pragma omp declare target
    void your_function(){...}
    #pragma omp end declare target


- You need to do the same for data/variables you want to be available on the device:

.. code-block:: cpp

    #pragma omp declare target
    constexpr int Dimensions = 2;
    #pragma omp end declare target


