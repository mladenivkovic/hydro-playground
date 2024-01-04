
add_executable(test_logging test_logging.cpp)
target_link_libraries(test_logging Logging)
target_include_directories(test_logging PUBLIC "${CMAKE_SOURCE_DIR}/src")
