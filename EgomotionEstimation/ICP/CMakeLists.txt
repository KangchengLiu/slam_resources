cmake_minimum_required(VERSION 2.8)
project(ICP)

set(CMAKE_CXX_STANDARD 11)

include_directories("/opt/eigen-3.3.4/include/eigen3")

find_package(Sophus)
if(Sophus_FOUND)
    include_directories( /opt/Sophus/include/ )
endif()

find_package( Pangolin )
if( Pangolin_FOUND )
    include_directories( ${Pangolin_INCLUDE_DIRS} )
    link_libraries( ${Pangolin_LIBRARIES} )
endif()

add_executable(${PROJECT_NAME} ICP.cpp)
