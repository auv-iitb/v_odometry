project(vo_test)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_EXECUTABLE_DIR ${CMAKE_PROJECT_DIR}/bin)
set(CMAKE_LIBRARY_DIR ${CMAKE_PROJECT_DIR}/lib)

add_library(visual_odometry SHARED src/only_rotation.cpp)
add_executable(vis_odom_test src/test_rotation.cpp)
target_link_libraries(vis_odom_test visual_odometry)
target_link_libraries(vis_odom_test ${OpenCV_LIBRARIES})
