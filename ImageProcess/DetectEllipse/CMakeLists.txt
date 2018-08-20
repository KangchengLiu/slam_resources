cmake_minimum_required(VERSION 2.8)
project( DetectEllipse )

find_package( OpenCV 2 REQUIRED )
if( OpenCV_FOUND )
    include_directories( ${OpenCV_INCLUDE_DIRS} )
    link_libraries( ${OpenCV_LIBS} )
endif()

add_executable( ${PROJECT_NAME} main.cpp DetectEllipse.cpp)
