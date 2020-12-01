cmake_minimum_required(VERSION 3.8)
project(motlib_ros LANGUAGES CXX)

include(CheckCXXCompilerFlag)
# CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
# if(COMPILER_SUPPORTS_MARCH_NATIVE)
#     
# endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")

#################
# Find packages #
#################

find_package(catkin REQUIRED COMPONENTS roscpp)
find_package(Eigen3 3.3 REQUIRED)
find_package(motlib REQUIRED)

#################################
# catkin specific configuration #
#################################

catkin_package(CATKIN_DEPENDS)

#########
# Build #
#########

include_directories(
        ${catkin_INCLUDE_DIRS}
        ${motlib_SRC_DIR})

link_libraries(
        ${catkin_LIBRARIES}
        Eigen3::Eigen)

add_subdirectory(src)
