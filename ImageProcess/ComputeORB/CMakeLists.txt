cmake_minimum_required(VERSION 2.8)
project(ComputeORB)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV )
if( OpenCV_FOUND )
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	link_libraries( ${OpenCV_LIBS} )
endif()

add_executable( ${PROJECT_NAME} computeORB.cpp )
