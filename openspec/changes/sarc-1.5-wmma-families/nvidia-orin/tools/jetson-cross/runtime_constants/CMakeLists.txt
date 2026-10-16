cmake_minimum_required(VERSION 3.24)
project(runtime_constants CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
add_executable(runtime_constants main.cpp)
target_link_libraries(
  runtime_constants PRIVATE extension_module extension_tensor
                            extension_data_loader executorch
)
