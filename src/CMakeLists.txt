# OBJECT: create "object libraries".
# See https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#object-libraries

add_library(Version OBJECT "${CMAKE_BINARY_DIR}/Version.cpp" Version.h)
target_include_directories(Version PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(Logging OBJECT Logging.cpp Logging.h)
target_include_directories(Logging PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h

add_library(Parameters OBJECT Parameters.cpp Parameters.h)
target_include_directories(Parameters PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h

add_library(Utils OBJECT Utils.cpp Utils.h)
target_include_directories(Utils PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_link_libraries(Utils Version)
target_link_libraries(Utils Logging)

add_library(Gas OBJECT Gas.cpp Gas.h)
target_include_directories(Gas PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_link_libraries(Gas Logging)

# Linking to the main executable
# ------------------------------
# Order matters here! E.g. Utils needs Version and Logging. If you put them before Utils,
# Utils won't find its link dependencies during the executable linking step.
# The target_link_libraries of Utils above only applies to the compile step of the object library above.
target_link_libraries(hydro Utils)
target_link_libraries(hydro Parameters)
target_link_libraries(hydro Version)
target_link_libraries(hydro Logging)

