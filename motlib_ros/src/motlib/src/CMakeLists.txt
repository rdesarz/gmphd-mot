set(EXECUTABLE_TARGETS "")

add_executable(object_tracker object_tracker.cpp)
add_executable(image_publisher image_publisher.cpp)
add_executable(visual_tracker visual_tracker.cpp)
list(APPEND EXECUTABLE_TARGETS object_tracker)
list(APPEND EXECUTABLE_TARGETS image_publisher)
list(APPEND EXECUTABLE_TARGETS visual_tracker)

foreach (EXECUTABLE_TARGET IN LISTS EXECUTABLE_TARGETS)
    add_dependencies(${EXECUTABLE_TARGET} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
    target_link_libraries(${EXECUTABLE_TARGET} motlib ${OpenCV_LIBRARIES})
endforeach ()