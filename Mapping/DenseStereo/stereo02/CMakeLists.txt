cmake_minimum_required(VERSION 2.8)
project(StereoDisparity)

set(CMAKE_CXX_STANDARD 11)

include_directories("/opt/eigen-3.3.4/include/eigen3")

find_package( Pangolin )
if( Pangolin_FOUND )
    include_directories( ${Pangolin_INCLUDE_DIRS} )
	link_libraries( ${Pangolin_LIBRARIES} )
endif()

find_package( OpenCV )
if( OpenCV_FOUND )
    include_directories( ${OpenCV_INCLUDE_DIRS} )
	link_libraries( ${OpenCV_LIBS} )
endif()

add_executable(${PROJECT_NAME} disparity.cpp)
