cmake_minimum_required(VERSION 3.0)
project(PointCloudToMesh)

find_package(PCL 1.12 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(pointcloud_to_mesh pointcloud_to_mesh.cpp)

target_link_libraries(pointcloud_to_mesh ${PCL_LIBRARIES})
