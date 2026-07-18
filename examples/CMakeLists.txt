cmake_minimum_required(VERSION 4.2.3)

add_executable(trainmlp train_mlp.cpp)
add_executable(trainmnist train_mnist.cpp)
add_executable(train train.cpp)

target_include_directories(trainmnist PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
target_compile_definitions(trainmnist PRIVATE PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

target_link_libraries(trainmlp PRIVATE micrograd::micrograd)
target_link_libraries(trainmnist PRIVATE micrograd::micrograd)
target_link_libraries(train PRIVATE micrograd::micrograd)

