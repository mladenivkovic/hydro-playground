
add_library(Version "${CMAKE_BINARY_DIR}/Version.cpp" Version.h)
target_include_directories(Version PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(Logging Logging.cpp Logging.h)

add_library(Utils Utils.cpp Utils.h)
target_link_libraries(Utils Version)
target_link_libraries(Utils Logging)

add_library(Parameters Parameters.cpp Parameters.h)


target_link_libraries(hydro Logging)
target_link_libraries(hydro Utils)
target_link_libraries(hydro Parameters)

