cmake_minimum_required(VERSION 3.10)
project(MachineLearningPlugins VERSION 1.0.0)

#===------------------------------------------------------------------------===
# Global settings some based on the external configuration settings
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_EXTENSIONS OFF )
set( CMAKE_CXX_VISIBILITY_PRESET hidden )
set( CMAKE_POSITION_INDEPENDENT_CODE True )
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
endif()

#===------------------------------------------------------------------------===
# Build information
string( TIMESTAMP BUILDDATE_YEAR_INTERNAL "%Y" )
string( TIMESTAMP BUILDDATE_MMDD_INTERNAL "%m%d" )
string( TIMESTAMP BUILDDATE_FULL_INTERNAL  "%Y-%m-%dT%H:%M:%S" )
string( TIMESTAMP BUILDDATE_STAMP "%Y.%m%d" )
string( REGEX REPLACE "^0" "" BUILDDATE_MMDD_INTERNAL ${BUILDDATE_MMDD_INTERNAL} )
set( BUILDDATE_YEAR "${BUILDDATE_YEAR_INTERNAL}" CACHE STRING "Year of the build: It will default to the current year." )
set( BUILDDATE_MMDD "${BUILDDATE_MMDD_INTERNAL}" CACHE STRING "Month and day of the build: It will default to the calendar month and day." )
set( BUILDDATE_FULL "${BUILDDATE_FULL_INTERNAL}" CACHE STRING "Exact time of the build." )

#===------------------------------------------------------------------------===
# Compile CMakeLists found in subdirectories
add_subdirectory(Plugins/Client)