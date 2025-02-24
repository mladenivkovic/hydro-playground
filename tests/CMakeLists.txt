add_compile_options(-O0)

add_executable(test_logging test_logging.cpp)
target_link_libraries(test_logging Logging)
target_include_directories(test_logging PUBLIC "${CMAKE_SOURCE_DIR}/src")


add_executable(test_io test_io.cpp)
target_link_libraries(test_io IO)
target_link_libraries(test_io Logging)
target_link_libraries(test_io Grid)
target_link_libraries(test_io Cell)
target_link_libraries(test_io Parameters)
target_link_libraries(test_io Gas)
target_include_directories(test_io PUBLIC "${CMAKE_SOURCE_DIR}/src")
