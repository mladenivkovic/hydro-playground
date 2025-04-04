add_compile_options(-O0)

add_executable(test_logging test_logging.cpp)
target_link_libraries(test_logging objects)
target_link_options(test_logging PUBLIC ${AMD_GPU_OMP_FLAGS})


add_executable(test_io test_io.cpp)
target_link_libraries(test_io objects)
target_link_options(test_io PUBLIC ${AMD_GPU_OMP_FLAGS})


add_executable(test_gas test_gas.cpp)
target_link_libraries(test_gas objects)
target_link_options(test_gas PUBLIC ${AMD_GPU_OMP_FLAGS})
