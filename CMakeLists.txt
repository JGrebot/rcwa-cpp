# cmake_minimum_required(VERSION 3.16 FATAL_ERROR) 
cmake_minimum_required(VERSION 3.15) 
project(
    rcwa-cpp
    VERSION 0.1
    DESCRIPTION "rcwa-cpp"
    LANGUAGES CXX)
	

# -----------------------------------------------------------------------------
# SET THE PROJECT OPTIONS
# -----------------------------------------------------------------------------

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)	
set(BLA_VENDOR OpenBLAS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_BUILD_TYPE "Release")
Set(BUILD_SHARED_LIBS OFF)

add_compile_definitions(PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}")

# Handling build type nicely Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(
    STATUS
      "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "${default_build_type}"
      CACHE STRING "Choose the type of build." FORCE)
endif()


# -----------------------------------------------------------------------------
# FIND / GET RELEVANT LIBRARIES
# -----------------------------------------------------------------------------
include(FetchContent)


# Find FFTW3 library
FetchContent_Declare(
    FFTW3
    URL      http://fftw.org/fftw-3.3.10.tar.gz
    URL_HASH MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c)

find_package(FFTW3 NO_MODULE)
if(NOT FFTW3)
    message("Fetching FFTW3 lib ...")
    FetchContent_MakeAvailable(FFTW3)
endif()


# Find Eigen3 library
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
  GIT_TAG "3.4.0"
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)

find_package(Eigen3 NO_MODULE)
if(NOT Eigen3_FOUND)
  set(EIGEN_BUILD_DOC OFF)
  set(BUILD_TESTING OFF)
  set(EIGEN_BUILD_PKGCONFIG OFF)
  message("Fetching Eigen3 lib ...")
  FetchContent_MakeAvailable(Eigen)
endif()


# Find fmt library
FetchContent_Declare(
  fmt
  GIT_REPOSITORY "https://github.com/fmtlib/fmt"
  GIT_TAG "9.0.0"
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)

find_package(fmt NO_MODULE)
if(NOT fmt)
  message("Fetching fmt lib ...")
  FetchContent_MakeAvailable(fmt)
endif()


find_package(Threads  REQUIRED)


 # Let's nicely support folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


# -----------------------------------------------------------------------------
# ADDING APPS AND SRC FOLDER
# -----------------------------------------------------------------------------
# The compiled library code is here
add_subdirectory(src)
# The executable code is here
add_subdirectory(apps)


MESSAGE(" All directories added.")
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
MESSAGE(STATUS "Library Type: " ${LIB_TYPE})
MESSAGE(STATUS "Compiler flags:" ${CMAKE_CXX_COMPILE_FLAGS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler cxx flags:" ${CMAKE_CXX_FLAGS})



