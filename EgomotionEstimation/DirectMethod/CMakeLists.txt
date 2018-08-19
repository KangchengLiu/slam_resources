cmake_minimum_required(VERSION 2.8)
project(DirectMethod)

set(CMAKE_CXX_STANDARD 11)

include_directories("/opt/eigen-3.3.4/include/eigen3")

find_package( OpenCV )
if( OpenCV_FOUND )
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	link_libraries( ${OpenCV_LIBS} )
endif()

find_package(Sophus)
if(Sophus_FOUND)
    include_directories( /home/chenguang/projects/SLAM/slambook/3rdparty/Sophus )
    link_libraries( /home/chenguang/projects/SLAM/slambook/3rdparty/Sophus/build/libSophus.so )
endif()

find_package( Pangolin )
if( Pangolin_FOUND )
    include_directories( ${Pangolin_INCLUDE_DIRS} )
    link_libraries( ${Pangolin_LIBRARIES} )
endif()

add_executable(${PROJECT_NAME} direct_method.cpp )
