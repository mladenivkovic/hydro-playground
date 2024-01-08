
add_library(Version "${CMAKE_BINARY_DIR}/Version.cpp" Version.h)
target_include_directories(Version PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(Logging Logging.cpp Logging.h)
target_include_directories(Logging PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h

add_library(Utils Utils.cpp Utils.h)
target_include_directories(Utils PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_link_libraries(Utils Version)
target_link_libraries(Utils Logging)

add_library(Parameters Parameters.cpp Parameters.h)

add_library(Gas Gas.cpp Gas.h)
target_include_directories(Gas PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_link_libraries(Gas Logging)

add_library(Cell Cell.cpp Cell.h)
target_include_directories(Cell PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_link_libraries(Cell Version)
target_link_libraries(Cell Logging)

target_link_libraries(hydro Logging)
target_link_libraries(hydro Utils)
target_link_libraries(hydro Parameters)

