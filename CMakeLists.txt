# ----------------------------------
# Globals / Main Project
# ----------------------------------

cmake_minimum_required(VERSION 3.21)

project(hydro_playground VERSION 0.1)
set(LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
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

# Notes:
#   - Wno-unused-parameter:
#       Ignore warnings about unused function parameters.
#   - Wno-string-plus-int:
#       Avoid warning "adding 'int' to a string does not append to the string"
#       That occurs all the time because of my __FILENAME__ = __FILE__ + SOURCE_PATH_SIZE trick.

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(
        -Werror -Wall -Wextra -Wpedantic -Wno-unused-parameter -Wno-string-plus-int -O3
    )
    set(DEBUG_LEVEL 0)

elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(
        -Werror -Wall -Wextra -Wpedantic -Wno-unused-parameter -Wno-string-plus-int -O3 -g
    )
    set(DEBUG_LEVEL 1)

elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
        -Werror -Wall -Wextra -Wpedantic -Wno-unused-parameter -Wno-string-plus-int -O0 -g
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


# ---------------------------------------
# Truncate full path from __FILE__ macro
# ---------------------------------------

# The additional / is important to remove the last character from the path.
# Note that it does not matter if the OS uses / or \, because we are only
# saving the path size.
string(LENGTH "${CMAKE_SOURCE_DIR}/" SOURCE_PATH_SIZE)
add_definitions("-DSOURCE_PATH_SIZE=${SOURCE_PATH_SIZE}")

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


