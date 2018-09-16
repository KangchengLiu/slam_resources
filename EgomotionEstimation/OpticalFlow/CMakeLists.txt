cmake_minimum_required(VERSION 2.8)
project(OpticalFlow)

set(CMAKE_CXX_STANDARD 11)

include_directories("/opt/eigen-3.3.3/include/eigen3")

find_package( OpenCV )
if( OpenCV_FOUND )
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	link_libraries( ${OpenCV_LIBS} )
endif()

add_executable( useLK useLK.cpp )

add_executable( optical_flow optical_flow.cpp )
