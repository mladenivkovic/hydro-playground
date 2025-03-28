# OBJECT: create "object libraries".
# See https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#object-libraries

include_directories(Solver)
include_directories(Riemann)

add_library(Version OBJECT "${CMAKE_BINARY_DIR}/Version.cpp" Version.h)
target_include_directories(Version PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(Logging OBJECT Logging.cpp Logging.h)
target_include_directories(Logging PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h

add_library(Parameters OBJECT Parameters.cpp Parameters.h BoundaryConditions.h)
target_include_directories(Parameters PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h

add_library(Utils OBJECT Utils.cpp Utils.h)
target_include_directories(Utils PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_link_libraries(Utils Version)
target_link_libraries(Utils Logging)

add_library(Gas OBJECT Gas.cpp Gas.h)
target_include_directories(Gas PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_link_libraries(Gas Logging)

add_library(Cell OBJECT Cell.cpp Cell.h)
target_include_directories(Cell PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_link_libraries(Cell Logging)

add_library(Grid OBJECT Grid.cpp Grid.h BoundaryConditions.h)
target_include_directories(Grid PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_link_libraries(Grid Cell)
target_link_libraries(Grid Logging)

add_library(IO OBJECT IO.cpp IO.h BoundaryConditions.h Timer.h)
target_include_directories(IO PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_link_libraries(IO Logging)
target_link_libraries(IO Utils)

add_library(SolverBase OBJECT Solver.h Solver/SolverBase.h Solver/SolverBase.cpp)
target_include_directories(SolverBase PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_include_directories(SolverBase PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}") # we need this for Config.h
target_link_libraries(SolverBase Grid)
target_link_libraries(SolverBase IO)
target_link_libraries(SolverBase Logging)
target_link_libraries(SolverBase Parameters)

add_library(SolverMUSCL OBJECT Solver.h Solver/SolverMUSCL.h Solver/SolverMUSCL.cpp Limiter.h Limiter/LimiterMinmod.h Limiter/LimiterVanLeer.h)
target_include_directories(SolverMUSCL PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_include_directories(SolverMUSCL PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}") # we need this for Config.h
target_link_libraries(SolverMUSCL SolverBase)
target_link_libraries(SolverMUSCL RiemannBase)
target_link_libraries(SolverMUSCL RiemannHLLC)
target_link_libraries(SolverMUSCL RiemannExact)

add_library(SolverGodunov OBJECT Solver.h Solver/SolverGodunov.h Solver/SolverGodunov.cpp)
target_include_directories(SolverGodunov PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_include_directories(SolverGodunov PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}") # we need this for Config.h
target_link_libraries(SolverGodunov SolverBase)
target_link_libraries(SolverGodunov RiemannBase)
target_link_libraries(SolverGodunov RiemannHLLC)
target_link_libraries(SolverGodunov RiemannExact)


add_library(RiemannBase OBJECT Riemann.h Riemann/RiemannBase.h Riemann/RiemannBase.cpp)
target_include_directories(RiemannBase PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_include_directories(RiemannBase PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}") # we need this for Config.h
target_link_libraries(RiemannBase Logging)

add_library(RiemannExact OBJECT Riemann.h Riemann/RiemannExact.h Riemann/RiemannExact.cpp)
target_include_directories(RiemannExact PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_include_directories(RiemannExact PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}") # we need this for Config.h
target_link_libraries(RiemannExact RiemannBase)

add_library(RiemannHLLC OBJECT Riemann.h Riemann/RiemannHLLC.h Riemann/RiemannHLLC.cpp)
target_include_directories(RiemannHLLC PUBLIC "${PROJECT_BINARY_DIR}") # we need this for Config.h
target_include_directories(RiemannHLLC PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}") # we need this for Config.h
target_link_libraries(RiemannHLLC RiemannBase)









# Linking to the main executable
# ------------------------------
# Order matters here! E.g. Utils needs Version and Logging. If you put them before Utils,
# Utils won't find its link dependencies during the executable linking step.
# The target_link_libraries of Utils above only applies to the compile step of the object library above.
target_link_libraries(hydro SolverMUSCL)
target_link_libraries(hydro SolverGodunov)
target_link_libraries(hydro SolverBase)
target_link_libraries(hydro RiemannHLLC)
target_link_libraries(hydro RiemannExact)
target_link_libraries(hydro RiemannBase)
target_link_libraries(hydro IO)
target_link_libraries(hydro Grid)
target_link_libraries(hydro Cell)
target_link_libraries(hydro Gas)
target_link_libraries(hydro Utils)
target_link_libraries(hydro Parameters)
target_link_libraries(hydro Logging)
target_link_libraries(hydro Version)

