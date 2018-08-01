cmake_minimum_required(VERSION 2.8)
project(DenseStereo)

set(CMAKE_CXX_STANDARD 11)

find_package( Pangolin )
if( Pangolin_FOUND )
    include_directories( ${Pangolin_INCLUDE_DIRS} )
    link_libraries( ${Pangolin_LIBRARIES} )
endif()

find_package( OpenCV 2 )
if( OpenCV_FOUND )
    include_directories( ${OpenCV_INCLUDE_DIRS} )
    link_libraries( ${OpenCV_LIBS} )
endif()

add_executable(${PROJECT_NAME} main.cpp)
