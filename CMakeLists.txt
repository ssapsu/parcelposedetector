cmake_minimum_required(VERSION 3.10)
project(ObjectPoseEstimation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Open3D directory
set(Open3D_DIR "/home/singleuav_causlab/open3d_install/lib/cmake/Open3D")

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Open3D
find_package(Open3D REQUIRED)
include_directories(${Open3D_INCLUDE_DIRS})

# Find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Include directories for your headers
include_directories(${CMAKE_SOURCE_DIR})

# Add source files
add_executable(ObjectPoseEstimation
    test.cpp
    Camera.cpp
)

# Link libraries
target_link_libraries(ObjectPoseEstimation PRIVATE
    Open3D::Open3D
    Eigen3::Eigen
    ${OpenCV_LIBS}
    tbb
)

# Copy the output.ply file to the build directory
configure_file(${CMAKE_SOURCE_DIR}/output.ply ${CMAKE_CURRENT_BINARY_DIR}/output.ply COPYONLY)
