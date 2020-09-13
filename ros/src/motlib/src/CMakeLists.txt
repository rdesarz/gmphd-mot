set(EXECUTABLE_TARGETS "")

add_executable(object_tracker object_tracker.cpp)
list(APPEND EXECUTABLE_TARGETS object_tracker)

foreach (EXECUTABLE_TARGET IN LISTS EXECUTABLE_TARGETS)
    add_dependencies(${EXECUTABLE_TARGET} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
endforeach ()