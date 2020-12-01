set(INSTALL_GTEST OFF CACHE BOOL "Install google-test" FORCE)

# Download google test
include(${PROJECT_SOURCE_DIR}/cmake/DownloadProject.cmake)
download_project(PROJ googletest
        URL https://github.com/google/googletest/archive/release-1.10.0.tar.gz
        URL_HASH SHA1=9c89be7df9c5e8cb0bc20b3c4b39bf7e82686770
        DOWNLOAD_NO_PROGRESS YES)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})

include(CTest)
include(GoogleTest)

add_executable(test_motlib
        ${CMAKE_CURRENT_SOURCE_DIR}/test_filter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_update_components.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_postprocessing.cpp
        )
target_link_libraries(test_motlib PRIVATE gtest_main ${PROJECT_NAME})
target_compile_features(test_motlib PRIVATE cxx_std_17)
add_test(NAME test_motlib COMMAND test_motlib)
gtest_discover_tests(test_motlib)