# ----------------------------------
# Globals / Main Project
# ----------------------------------

cmake_minimum_required(VERSION 3.21)

project(hydro_playground VERSION 0.1)
set(LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)



#-------------------------------
# Set up build types
#-------------------------------

# Note: This needs to be done *BEFORE* adding libraries, targets etc.
# Otherwise, the compiler options won't be applied.

# specify with -DCMAKE_BUILD_TYPE=[Debug,Release,RelWithDebInfo]

if(NOT CMAKE_BUILD_TYPE)
    message("* No build type provided. Set with -DCMAKE_BUILD_TYPE=[Debug,Release,RelWithDebInfo].")
    message("* Setting to Debug and continuing.")
    set(CMAKE_BUILD_TYPE Debug
        CACHE STRING "Choose the type of build : [Debug,Release,RelWithDebInfo]."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)
# TODO: Move this back to release once dev has converged

# Notes:
#   - Wno-unused-parameter:
#       Ignore warnings about unused function parameters.

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(
        -Werror -Wall -Wextra -Wpedantic -Wno-unused-parameter -O3
    )
    set(DEBUG_LEVEL 0)

elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(
        -Werror -Wall -Wextra -Wpedantic -Wno-unused-parameter -O3 -g
    )
    set(DEBUG_LEVEL 1)

elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
        -Werror -Wall -Wextra -Wpedantic -Wno-unused-parameter -O0 -g
    )
    set(DEBUG_LEVEL 2)

else()
    message(FATAL_ERROR
        "Invalid CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}. "
        "Options are [Release, RelWithDebInfo, Debug]. "
        "Pick one of those."
        )

endif()

message("* Current build type is : ${CMAKE_BUILD_TYPE}")





# ---------------------------
# Options
# ---------------------------

option(TERMINAL_COLORS "Enable colour logging outputs" ON)
if (TERMINAL_COLORS)
    add_definitions(-DTERMINAL_COLORS)
endif(TERMINAL_COLORS)

message(STATUS "TERMINAL_COLORS is set to: ${TERMINAL_COLORS}")


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



# ---------------------------
# Main Target
# ---------------------------

add_executable(hydro src/main.cpp)
add_subdirectory(src)

target_include_directories(hydro PUBLIC "${PROJECT_BINARY_DIR}")

configure_file(src/Config.h.in Config.h)




# ----------------------------------
# Tests
# ----------------------------------

add_subdirectory(tests)

enable_testing()
add_test(NAME logging COMMAND test_logging)


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


