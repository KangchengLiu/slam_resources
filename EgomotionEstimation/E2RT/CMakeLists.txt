cmake_minimum_required(VERSION 2.8)
project(E2RT)

set(CMAKE_CXX_STANDARD 11)

include_directories("/opt/eigen-3.3.4/include/eigen3")

find_package(Sophus)
if(Sophus_FOUND)
	include_directories( /opt/Sophus/include/ )
endif()

add_executable(${PROJECT_NAME} E2Rt.cpp)
