# ----------------------------------
# Globals / Main Project
# ----------------------------------

cmake_minimum_required(VERSION 3.21)

project(hydro_playground VERSION 1.0)
set(LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)

# Include module for checking compiler flags
include(CheckCXXCompilerFlag)

# ----------------------------------
# Demand OpenMP support for pragmas
# ----------------------------------
find_package(OpenMP REQUIRED)

# ---------------------------
# Automatically set GPU offloading flags if available
# ---------------------------
# These flags are added if the compiler supports OpenMP GPU offloading.
# NVHPC, Clang, and GNU offloading flags are handled.
# set(GPU_OFFLOAD_FLAGS "")
# if(OpenMP_CXX_FOUND)
#   if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC")
#     # NVHPC offloading flag for GPU
#     set(GPU_OFFLOAD_FLAGS "-mp=gpu")
#   elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     # Clang offloading for NVIDIA GPUs using NVPTX
#     set(GPU_OFFLOAD_FLAGS "-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda")
#   elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
#     # Check if GCC supports offloading flag; this flag is only available in newer versions/builds
#     check_cxx_compiler_flag("-fopenmp-targets=nvptx-none" COMPILER_SUPPORTS_GCC_OFFLOADING)
#     if(COMPILER_SUPPORTS_GCC_OFFLOADING)
#       set(GPU_OFFLOAD_FLAGS "-fopenmp-targets=nvptx-none")
#     else()
#       message(FATAL_ERROR "GCC does not support OpenMP offloading to NVPTX. Offloading is required for this project.")
#     endif()
#   else()
#     message(FATAL_ERROR "Unknown compiler ${CMAKE_CXX_COMPILER_ID} or no GPU offloading flags defined. GPU OpenMP offloading is required.")
#   endif()
# endif()
#
# if(GPU_OFFLOAD_FLAGS STREQUAL "")
#   message(FATAL_ERROR "GPU OpenMP offloading support is required but not available with the current compiler.")
# endif()
#
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GPU_OFFLOAD_FLAGS}")
# message(STATUS "Using GPU offloading flags: ${GPU_OFFLOAD_FLAGS}")

string(REPLACE -O2 -O3 CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
   # using Clang
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp --offload-arch=gfx942 -fstrict-aliasing")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   # using GCC
   set(CMAKE_CXX_FLAGS 
 "${CMAKE_CXX_FLAGS} -fopenmp -foffload=-march=gfx942 -fstrict-aliasing -fopt-info-optimized-omp")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Cray")
endif()

#-------------------------------
# Set up build types
#-------------------------------

# Note: This needs to be done *BEFORE* adding libraries, targets etc.
# Otherwise, the compiler options won't be applied.

# specify with -DCMAKE_BUILD_TYPE=[Debug,Release,RelWithDebInfo]

if(NOT CMAKE_BUILD_TYPE)
    message("* No build type provided. Set with -DCMAKE_BUILD_TYPE=[Debug,Release,RelWithDebInfo].")
    message("* Setting to Release and continuing.")
    set(CMAKE_BUILD_TYPE Release
        CACHE STRING "Choose the type of build : [Debug,Release,RelWithDebInfo]."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# Notes:
#   - Wno-unused-parameter:
#       Ignore warnings about unused function parameters.



if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(
        -Werror -Wall -Wextra -Wpedantic -Wno-unused-parameter -O3
    )
    set(DEBUG_LEVEL 0)

    if (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        add_compile_options( -ipo )
        add_link_options(-ipo)
    endif()

elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(
        -Werror -Wall -Wextra -Wpedantic -Wno-unused-parameter -O3 -g
    )
    set(DEBUG_LEVEL 1)

    if (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        add_compile_options(-ipo)
        add_link_options(-ipo)
    endif()

elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
        -Werror -Wall -Wextra -Wpedantic -Wno-unused-parameter -O0 -g
    )
    set(DEBUG_LEVEL 2)

else()
    message(FATAL_ERROR
        "Invalid CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}. "
        "Options are [Debug,Release,RelWithDebInfo]. "
        "Pick one of those."
        )

endif()

message("* Current build type is : ${CMAKE_BUILD_TYPE}")





# ---------------------------
# Options
# ---------------------------

# Terminal colours

option(TERMINAL_COLORS "Enable colour logging outputs" OFF)
if (TERMINAL_COLORS)
    add_definitions(-DTERMINAL_COLORS)
endif(TERMINAL_COLORS)

message(STATUS "TERMINAL_COLORS is set to: ${TERMINAL_COLORS}")



# Float precision

set(PRECISION "DOUBLE" CACHE STRING "Set floating point precision (SINGLE DOUBLE)")
set_property(CACHE PRECISION PROPERTY STRINGS "SINGLE" "DOUBLE")
message(STATUS "PRECISION is set to: ${PRECISION}")

if(PRECISION STREQUAL "SINGLE")
    add_definitions(-DPRECISION=1)
elseif(PRECISION STREQUAL "DOUBLE")
    add_definitions(-DPRECISION=2)
else()
    message(FATAL_ERROR
        "Invalid PRECISION=${PRECISION}. "
        "Options are [SINGLE, DOUBLE]. "
        "Pick one of those."
        )
endif()


# Hydro Solver

set(SOLVER "MUSCL" CACHE STRING "Select hydro solver (MUSCL GODUNOV)")
set_property(CACHE SOLVER PROPERTY STRINGS "MUSCL" "GODUNOV")
message(STATUS "SOLVER is set to: ${SOLVER}")

if(SOLVER STREQUAL "MUSCL")
    add_definitions(-DSOLVER=11)
elseif(SOLVER STREQUAL "GODUNOV")
    add_definitions(-DSOLVER=12)
else()
    message(FATAL_ERROR
        "Invalid SOLVER=${SOLVER}. "
        "Options are [MUSCL, GODUNOV]. "
        "Pick one of those."
        )
endif()



# Riemann Solver

set(RIEMANN "EXACT" CACHE STRING "Select Riemann solver (HLLC EXACT)")
set_property(CACHE RIEMANN PROPERTY STRINGS "HLLC" "EXACT")
message(STATUS "RIEMANN is set to: ${RIEMANN}")

if(RIEMANN STREQUAL "HLLC")
    add_definitions(-DRIEMANN_SOLVER=21)
elseif(RIEMANN STREQUAL "EXACT")
    add_definitions(-DRIEMANN_SOLVER=22)
else()
    message(FATAL_ERROR
        "Invalid RIEMANN=${RIEMANN}. "
        "Options are [HLLC, EXACT]. "
        "Pick one of those."
        )
endif()




# Slope Limiter

set(LIMITER "VANLEER" CACHE STRING "Select slope limiter (MINMOD VANLEER)")
set_property(CACHE LIMITER PROPERTY STRINGS "MINMOD" "VANLEER")
message(STATUS "LIMITER is set to: ${LIMITER}")

if(LIMITER STREQUAL "MINMOD")
    add_definitions(-DLIMITER=31)
elseif(LIMITER STREQUAL "VANLEER")
    add_definitions(-DLIMITER=32)
else()
    message(FATAL_ERROR
        "Invalid LIMITER=${LIMITER}. "
        "Options are [MINMOD, VANLEER]. "
        "Pick one of those."
        )
endif()








# ---------------------------
# Main Target
# ---------------------------

configure_file(src/Config.h.in Config.h)


#  --------------------------------------------------------------------------------------------------------
file(GLOB_RECURSE CPP_SRC_FILES FOLLOW_SYMLINKS ${PROJECT_SOURCE_DIR}/src/*.cpp)
# file(GLOB_RECURSE CUDA_SRC_FILES FOLLOW_SYMLINKS ${PROJECT_SOURCE_DIR}/src/*.cu)
# file(GLOB_RECURSE HIP_SRC_FILES FOLLOW_SYMLINKS ${PROJECT_SOURCE_DIR}/src/*.hip)
file(GLOB_RECURSE MAIN_FILE FOLLOW_SYMLINKS ${PROJECT_SOURCE_DIR}/src/main.cpp)
list(REMOVE_ITEM CPP_SRC_FILES ${MAIN_FILE})

set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/src)                # allow subdirs to find main include files
set(COMMON_INCLUDES ${COMMON_INCLUDES} ${PROJECT_BINARY_DIR}) # include generated Config.h
include_directories(${COMMON_INCLUDES})


add_library(objects OBJECT ${CPP_SRC_FILES} ${PROJECT_BINARY_DIR}/Version.cpp)


# MAIN EXECUTABLE: hydro
add_executable(hydro ${MAIN_FILE})
target_link_libraries(hydro PUBLIC objects)
target_link_libraries(hydro PUBLIC OpenMP::OpenMP_CXX)




# ----------------------------------
# Tests
# ----------------------------------

add_subdirectory(tests)

enable_testing()
add_test(NAME logging COMMAND test_logging)
add_test(NAME io COMMAND test_io)
add_test(NAME gas COMMAND test_gas)



# ----------------------------------
# Git stuff
# ----------------------------------

# https://stackoverflow.com/questions/1435953/how-can-i-pass-git-sha1-to-compiler-as-definition-using-cmake

find_package(Git)

# the commit's SHA1, and whether the building workspace was dirty or not
execute_process(COMMAND
    "${GIT_EXECUTABLE}" describe --match=NeVeRmAtCh --always --abbrev=40 --dirty
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_SHA1
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# the date of the commit
execute_process(COMMAND
    "${GIT_EXECUTABLE}" log -1 --format=%ad --date=local
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_DATE
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# the current branch
execute_process(COMMAND
    "${GIT_EXECUTABLE}" rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_BRANCH
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# the subject of the commit
execute_process(COMMAND
    "${GIT_EXECUTABLE}" log -1 --format=%s
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_COMMIT_SUBJECT
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# generate Version.cpp
configure_file("${CMAKE_SOURCE_DIR}/src/Version.cpp.in" "${CMAKE_BINARY_DIR}/Version.cpp" @ONLY)

list(APPEND SOURCES "${CMAKE_BINARY_DIR}/Version.cpp" src/Version.h)
