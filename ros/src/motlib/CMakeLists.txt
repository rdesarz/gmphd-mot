cmake_minimum_required(VERSION 3.1)
project(motlib LANGUAGES CXX C)

if (POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif ()
if (POLICY CMP0074)
    cmake_policy(SET CMP0074 OLD)
endif ()

# digging into the root directory of motlib
get_filename_component(DEFAULT_motlib_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)
get_filename_component(DEFAULT_motlib_DIR ${DEFAULT_motlib_DIR} DIRECTORY)
get_filename_component(DEFAULT_motlib_DIR ${DEFAULT_motlib_DIR} DIRECTORY)

set(motlib_ROOT ${DEFAULT_motlib_DIR} CACHE STRING "Root directory of motlib")
message(STATUS "Root directory of motlib: ${motlib_ROOT}")

set(motlib_SRC_DIR ${DEFAULT_motlib_DIR}/src)
message(STATUS "Source directory of motlib: ${motlib_SRC_DIR}")
set(motlib_LIB_DIR ${DEFAULT_motlib_DIR}/build/lib CACHE STRING "Directory which contains the shared library of motlib")
message(STATUS "Library directory of motlib: ${motlib_LIB_DIR}")

##################
# Set build type #
##################

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

########################
# Set compiler options #
########################

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Og")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math")

set(BUILD_WITH_MARCH_NATIVE OFF CACHE BOOL "Enable architecture-aware optimization (-march=native)")

if (BUILD_WITH_MARCH_NATIVE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtune=native -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mtune=native -march=native")
    message(STATUS "Architecture-aware optimization (-march=native): ENABLED")
else ()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtune=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mtune=native")
    message(STATUS "Architecture-aware optimization (-march=native): DISABLED")
endif ()

#######################
# Check C++11 support #
#######################

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORTS_CXX17)
check_cxx_compiler_flag("-std=c++14" COMPILER_SUPPORTS_CXX14)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    message(STATUS "C++11 support: OK (-std=c++17)")
elseif (COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    message(STATUS "C++11 support: OK (-std=c++14)")
elseif (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    message(STATUS "C++11 support: OK (-std=c++11)")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    message(STATUS "C++11 support: OK (-std=c++0x)")
else ()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif ()

#################
# Find packages #
#################

find_package(catkin REQUIRED COMPONENTS roscpp)

# OpenVSLAM
find_library(motlib_LIB motlib HINTS ${motlib_LIB_DIR})

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
        ${motlib_LIB})

add_subdirectory(src)
