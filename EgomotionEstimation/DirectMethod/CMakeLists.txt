cmake_minimum_required(VERSION 2.8)
project(DirectMethod)

set(CMAKE_CXX_STANDARD 11)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

include_directories("/opt/eigen-3.3.4/include/eigen3")

find_package( OpenCV )
if( OpenCV_FOUND )
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	link_libraries( ${OpenCV_LIBS} )
endif()

find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )
link_libraries(Sophus)

find_package( Pangolin )
if( Pangolin_FOUND )
    include_directories( ${Pangolin_INCLUDE_DIRS} )
    link_libraries( ${Pangolin_LIBRARIES} )
endif()

find_package( G2O )
include_directories( ${G2O_INCLUDE_DIRS} )

set( G2O_LIBS g2o_core g2o_types_sba g2o_solver_csparse g2o_stuff g2o_csparse_extension )


add_executable( direct_method direct_method.cpp )

add_executable( direct_sparse direct_sparse.cpp )
target_link_libraries( direct_sparse  ${G2O_LIBS} )

add_executable( direct_semidense direct_semidense.cpp )
target_link_libraries( direct_semidense ${G2O_LIBS} )
