project(vo_test)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(vo_rotation SHARED src/only_rotation.cpp src/visual_odometry.h)
target_link_libraries(vo_rotation ${OpenCV_LIBRARIES})
add_executable(vo_rot_test src/test_rotation.cpp)
target_link_libraries(vo_rot_test vo_rotation ${OpenCV_LIBRARIES})

add_library(visual_odometry SHARED src/odometry.cpp src/visual_odometry.h)
target_link_libraries(visual_odometry ${OpenCV_LIBRARIES})
add_executable(vis_odom_test src/test_odometry.cpp)
target_link_libraries(vis_odom_test visual_odometry ${OpenCV_LIBRARIES})
