
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
