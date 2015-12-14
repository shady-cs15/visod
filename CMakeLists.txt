cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(Odometry)
find_package(OpenCV 3.0 REQUIRED)
FIND_PACKAGE(PCL 1.2 REQUIRED)

include_directories(/usr/local/include)
add_executable(odo odo.cpp)
target_link_libraries(odo ${OpenCV_LIBS})

INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
LINK_DIRECTORIES(${PCL_LIBRARY_DIRS})
ADD_DEFINITIONS(${PCL_DEFINITIONS})

set (LIBELAS_SRC_DIR libelas)
set (CMAKE_CXX_FLAGS "-msse3")

FILE (GLOB LIBELAS_SRC_FILES libelas/*.cpp)
add_executable(stereo disparity.cpp ${LIBELAS_SRC_FILES})
target_link_libraries(stereo ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(pcd_read readpcd.cpp)
target_link_libraries(pcd_read ${PCL_LIBRARIES})
