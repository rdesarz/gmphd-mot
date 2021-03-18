cmake_minimum_required(VERSION 3.8)
project(motlib_ros LANGUAGES CXX)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()


#################
# Find packages #
#################

find_package(catkin REQUIRED COMPONENTS roscpp cv_bridge image_transport)
find_package(Eigen3 3.3 REQUIRED)
find_package(motlib REQUIRED)
find_package(OpenCV REQUIRED)

#################################
# catkin specific configuration #
#################################

catkin_package(CATKIN_DEPENDS)

#########
# Build #
#########

include_directories(${catkin_INCLUDE_DIRS})

link_libraries(${catkin_LIBRARIES})

add_subdirectory(src)
