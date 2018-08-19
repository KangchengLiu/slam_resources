cmake_minimum_required(VERSION 2.8)
project( UndistortImage )

find_package( OpenCV )
if( OpenCV_FOUND )
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	link_libraries( ${OpenCV_LIBS} )
endif()

add_executable( ${PROJECT_NAME} undistort_image.cpp )
