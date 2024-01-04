target_sources(hydro PUBLIC
    # Logging.cpp Logging.h
    # Utils.cpp Utils.h
    # Version.h
    )

add_library(Logging Logging.cpp Logging.h)

add_library(Utils Utils.cpp Utils.h)
target_sources(Utils PUBLIC "${CMAKE_BINARY_DIR}/Version.cpp" Version.h)
# target_include_directories(Utils PUBLIC src)



# target_sources(Logging PRIVATE
#     Logging.cpp Logging.h
    # Utils.cpp Utils.h
    # )
#
# add_executable(hydro main.cpp)
target_link_libraries(hydro Logging)
target_link_libraries(hydro Utils)
#


