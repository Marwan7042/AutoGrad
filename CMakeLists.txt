cmake_minimum_required(VERSION 3.10)
project(Micrograd CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Point the compiler at your MySTL headers
include_directories(${CMAKE_SOURCE_DIR}/../MySTL)

# Find OpenBLAS on your system
find_package(BLAS REQUIRED)
find_package(CUDAToolkit REQUIRED)

set_source_files_properties(Tensor.cpp PROPERTIES LANGUAGE CUDA)

add_executable(DeepEngine tests.cpp Tensor.cpp)
target_link_libraries(DeepEngine PUBLIC BLAS::BLAS CUDA::cublas CUDA::cudart)

add_executable(TrainEngine train.cpp Tensor.cpp)
target_link_libraries(TrainEngine PUBLIC BLAS::BLAS CUDA::cublas CUDA::cudart)

add_executable(TrainMLP train_mlp.cpp Tensor.cpp)
target_link_libraries(TrainMLP PUBLIC BLAS::BLAS CUDA::cublas CUDA::cudart)

add_executable(TrainMNIST train_mnist.cpp Tensor.cpp)
target_link_libraries(TrainMNIST PUBLIC BLAS::BLAS CUDA::cublas CUDA::cudart)