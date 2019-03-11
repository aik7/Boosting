cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(boosting)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
 endif()

add_executable(boosting ${SOURCES})
