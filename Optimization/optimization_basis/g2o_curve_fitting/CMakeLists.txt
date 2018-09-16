find_package( G2O REQUIRED )
if(G2O_FOUND)
    include_directories( ${G2O_INCLUDE_DIRS} )
endif()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

add_executable( g2o_curve_fitting g2o_curve_fitting.cpp )
target_link_libraries( g2o_curve_fitting ${OpenCV_LIBS} g2o_core g2o_stuff )
