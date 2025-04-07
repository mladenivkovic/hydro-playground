add_compile_options(-O0)

add_executable(test_logging test_logging.cpp)
target_link_libraries(test_logging cpp_objects)


add_executable(test_io test_io.cpp)
target_link_libraries(test_io cpp_objects)


add_executable(test_gas test_gas.cpp)
target_link_libraries(test_gas cpp_objects)
