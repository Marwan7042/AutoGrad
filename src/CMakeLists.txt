cmake_minimum_required(VERSION 4.2.3)

add_library(vecore Tensor.cpp)
add_library(vecore::vecore ALIAS vecore)

target_include_directories(vecore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

set_source_files_properties(Tensor.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(vecore PUBLIC libvc::libvc BLAS::BLAS CUDA::cublas CUDA::cudart)
