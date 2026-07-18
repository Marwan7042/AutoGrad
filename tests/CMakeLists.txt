cmake_minimum_required(VERSION 4.2.3)

add_executable(test tests.cpp)

target_link_libraries(test PRIVATE vecore::vecore)

