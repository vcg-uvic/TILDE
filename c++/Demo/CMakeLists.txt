CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11)
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)


set(PROJECT_NAME "demo")
project(${PROJECT_NAME})


find_package(OpenCV REQUIRED)

FIND_PACKAGE( OpenMP QUIET)
if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

IF(UNIX)
  IF (NOT APPLE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DISLINUX")
  else()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DISMAC")
  endif()
endif()

if (OpenCV_FOUND)
 
  add_executable(${PROJECT_NAME} demo.cpp)
  target_link_libraries(${PROJECT_NAME} LINK_PUBLIC TILDE)


else()    
  set( OT_MISSING_DEPS "")

  if(NOT OPENCV_FOUND)
    set(OT_MISSING_DEPS "OpenCV, ${OT_MISSING_DEPS}")
  endif()

  message(STATUS "NOTICE: This project requires ${OT_MISSING_DEPS} and will not be compiled.")
endif()

