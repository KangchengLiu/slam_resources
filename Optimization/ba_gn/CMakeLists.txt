cmake_minimum_required(VERSION 2.8)
project(ba_gn)

set(CMAKE_CXX_STANDARD 11)

include_directories("/opt/eigen-3.3.3/include/eigen3")

include_directories( "/usr/local/include/" )

add_executable(${PROJECT_NAME} ba_gn.cpp)
