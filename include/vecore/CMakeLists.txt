cmake_minimum_required(VERSION 4.2.3)

add_library(vecore
    Tensor.h
    nn.h
    optim.h
)

add_library(vecore::vecore ALIAS vecore)

target_link_libraries(PUBLIC libvc)

target_include_directories(vecore PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
