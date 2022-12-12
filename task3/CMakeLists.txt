cmake_minimum_required(VERSION 3.9)
project( opencv )
set(CMAKE_CXX_STANDARD 20)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( main main.cpp )
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -march=x86-64-v3")
set_property(TARGET main
             PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
target_link_libraries( main ${OpenCV_LIBS} )
