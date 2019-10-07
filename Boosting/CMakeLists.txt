cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(boosting)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
 endif()

add_executable(boosting ${SOURCES})

target_link_libraries(boosting "${PEBBL_LIBRARIES}")
target_link_libraries(boosting "${GRB_LIBRARIES}")
target_link_libraries(boosting "${R_LIBRARIES}")
target_link_libraries(boosting "${Rcpp_LIBRARIES}")
target_link_libraries(boosting "${RInside_LIBRARIES}")
