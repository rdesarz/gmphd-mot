# ----- Build example executables -----

set(EXECUTABLE_TARGETS "")

# add_executable(run_example run_example.cpp)
# list(APPEND EXECUTABLE_TARGETS run_example)

foreach(EXECUTABLE_TARGET IN LISTS EXECUTABLE_TARGETS)
    # Set output directory for executables
    set_target_properties(${EXECUTABLE_TARGET} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_BINARY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_BINARY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${PROJECT_BINARY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PROJECT_BINARY_DIR}")

    # Link OpenVSLAM
    target_link_libraries(${EXECUTABLE_TARGET}
            PRIVATE
            ${PROJECT_NAME})
endforeach()