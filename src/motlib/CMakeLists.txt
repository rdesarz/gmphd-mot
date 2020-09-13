add_library(${PROJECT_NAME}
        SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/model.h
        ${CMAKE_CURRENT_SOURCE_DIR}/model.cpp)

# Set output directory of the library
set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_NAME motlib
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# ----- Configure OpenVSLAM library -----

# Include directories
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/>
        $<INSTALL_INTERFACE:include/>)

# Link to required libraries
target_link_libraries(${PROJECT_NAME})

# ----- Install configuration -----

# set(OPENVSLAM_INCLUDE_INSTALL_DIR ${INCLUDES_DESTINATION}/openvslam)
#
# # Install OpenVSLAM library
# install(TARGETS ${PROJECT_NAME}
#         EXPORT ${OPENVSLAM_TARGETS_EXPORT_NAME}
#         RUNTIME DESTINATION ${RUNTIME_DESTINATION}
#         LIBRARY DESTINATION ${LIBRARY_DESTINATION}
#         ARCHIVE DESTINATION ${ARCHIVE_DESTINATION})
#
# # Install OpenVSLAM headers
# file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
# install(FILES ${HEADERS}
#         DESTINATION ${OPENVSLAM_INCLUDE_INSTALL_DIR})
#
# # Install 3rd party headers
# install(FILES ${json_INCLUDE_DIR}/nlohmann/json.hpp ${json_INCLUDE_DIR}/nlohmann/json_fwd.hpp
#         DESTINATION ${OPENVSLAM_INCLUDE_INSTALL_DIR}/3rd/json/include/nlohmann)
# install(DIRECTORY ${spdlog_INCLUDE_DIR}
#         DESTINATION ${OPENVSLAM_INCLUDE_INSTALL_DIR}/3rd/spdlog)
#
# # ----- Append subdirectory -----
#
# add_subdirectory(camera)
# add_subdirectory(data)
# add_subdirectory(feature)
# add_subdirectory(initialize)
# add_subdirectory(io)
# add_subdirectory(match)
# add_subdirectory(module)
# add_subdirectory(optimize)
# add_subdirectory(publish)
# add_subdirectory(solve)
# add_subdirectory(util)