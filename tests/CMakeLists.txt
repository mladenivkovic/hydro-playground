add_compile_options(-O0)

add_executable(test_logging test_logging.cpp)
target_link_libraries(test_logging objects)


add_executable(test_io test_io.cpp)
target_link_libraries(test_io objects)


add_executable(test_gas test_gas.cpp)
target_link_libraries(test_gas objects)
