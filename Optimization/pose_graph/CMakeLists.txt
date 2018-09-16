cmake_minimum_required( VERSION 2.8 )
project( pose_graph )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# Eigen
find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

# sophus 
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

# g2o 
find_package( G2O REQUIRED )
include_directories( ${G2O_INCLUDE_DIR} )
link_directories( "/opt/g2o/lib" )

find_package( Cholmod REQUIRED )
include_directories( ${CHOLMOD_INCLUDE_DIR} )

# gtsam 
find_package( GTSAMCMakeTools )
find_package( GTSAM REQUIRED )
include_directories( ${GTSAM_INCLUDE_DIR} )


# Ceres 
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

add_executable( pose_graph_g2o_SE3 pose_graph_g2o_SE3.cpp )
target_link_libraries( pose_graph_g2o_SE3
    g2o_core g2o_stuff g2o_types_slam3d ${CHOLMOD_LIBRARIES}
)

add_executable( pose_graph_g2o_lie pose_graph_g2o_lie_algebra.cpp )
target_link_libraries( pose_graph_g2o_lie
    g2o_core g2o_stuff 
    ${CHOLMOD_LIBRARIES}
    #    ${Sophus_LIBRARIES}
    /home/chenguang/projects/SLAM/slambook/3rdparty/Sophus/build/libSophus.so
)

add_executable( pose_graph_gtsam pose_graph_gtsam.cpp )
target_link_libraries( pose_graph_gtsam
    ${CHOLMOD_LIBRARIES} gtsam
)
