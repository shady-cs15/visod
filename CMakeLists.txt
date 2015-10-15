cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(Odometry)
find_package(OpenCV 3.0 REQUIRED)

include_directories(/usr/local/include)
add_executable(odo odo.cpp)
target_link_libraries(odo ${OpenCV_LIBS})
