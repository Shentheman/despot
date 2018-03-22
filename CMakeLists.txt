# https://github.com/AdaCompNUS/despot/blob/master/CMakeLists.txt

###################################
# 1. CMake Version
###################################
cmake_minimum_required(VERSION 2.8.3)


###################################
# 2. Package name
###################################
project(despot)


###################################
# 3. Append all the extra files from ./cmake
###################################


set(BINARY_INSTALL_PATH "bin" CACHE PATH "Binary install path")
set(LIBRARY_INSTALL_PATH "lib" CACHE PATH "Library install path")
set(INCLUDE_INSTALL_PATH "include" CACHE PATH "Include install path")
set(CONFIG_INSTALL_PATH "${LIBRARY_INSTALL_PATH}/${PROJECT_NAME}/cmake")

set(DESPOT_BUILD_EXAMPLES ON CACHE BOOL "Build C++ model examples")
set(DESPOT_BUILD_POMDPX ON CACHE BOOL "Build POMDPX example")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse")
# set(CMAKE_MODULE_PATH ${CMAKE_PREFIX_PATH} "${PROJECT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")


###################################
# 4. Find all the dependencies
###################################
# find_package is used for finding an external library 
# if CMake has a module that aids in finding this package 
# (https://cmake.org/Wiki/CMake:How_To_Find_Libraries)

find_package(catkin REQUIRED COMPONENTS)


###################################
# 5. Compile Options
###################################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
# https://jonasdevlieghere.com/a-better-youcompleteme-config/
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
#add_compile_options(-Wall -Wextra -Wpedantic)


###################################
# 6. execute setup.py to read the arguments to set up the devel space
###################################
# catkin_python_setup()


###################################
# 7. catkin specific configuration
###################################
# This must be done before building
# catkin_package() is a catkin-provided CMake macro. This is required to specify catkin-specific information to the build system which in turn is used to generate pkg-config and CMake files.
# This function must be called before declaring any targets with add_library() or add_executable(). The function has 5 optional arguments:
# INCLUDE_DIRS - The exported include paths (i.e. cflags) for the package
# LIBRARIES - The exported libraries from the project
# CATKIN_DEPENDS - Other catkin projects that this project depends on
# DEPENDS - Non-catkin CMake projects that this project depends on. For a better understanding
# CFG_EXTRAS - Additional configuration options
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  DEPENDS
  CFG_EXTRAS
)


###################################
# 8. Specifying Build Targets
###################################
# There is 1 target library = ${PROJECT_NAME}
# It is very important to note that the names of build targets in catkin must be unique regardless of the folders they are built/installed to.

# Include Paths - Where can header files be found for the code (most common in C/C++) being built
# The first argument "include" indicates that the include/ directory within the package is also part of the path.
include_directories(
  include 
)


# To specify an executable target that must be built
# This will build a target executable called main which is built from examples/main.cpp.
# Build example files
if(DESPOT_BUILD_EXAMPLES)
  add_subdirectory(examples/cpp_models/adventurer)
  add_subdirectory(examples/cpp_models/bridge)
  add_subdirectory(examples/cpp_models/chain)
  add_subdirectory(examples/cpp_models/navigation)
  add_subdirectory(examples/cpp_models/pocman)
  add_subdirectory(examples/cpp_models/reg_demo)
  add_subdirectory(examples/cpp_models/rock_sample)
  add_subdirectory(examples/cpp_models/simple_rock_sample)
  add_subdirectory(examples/cpp_models/tag)
  add_subdirectory(examples/cpp_models/tiger)
endif()

if(DESPOT_BUILD_POMDPX)
  add_subdirectory(examples/pomdpx_models)
endif()



# The add_library() CMake function is used to specify libraries to build. By default catkin builds shared libraries.

add_library("${PROJECT_NAME}" SHARED
  src/core/belief.cpp
  src/core/globals.cpp
  src/core/lower_bound.cpp
  src/core/mdp.cpp
  src/core/node.cpp
  src/core/policy.cpp
  src/core/pomdp.cpp
  src/core/solver.cpp
  src/core/upper_bound.cpp
  src/evaluator.cpp
  src/ippc/client.cpp
  src/pomdpx/parser/function.cpp
  src/pomdpx/parser/parser.cpp
  src/pomdpx/parser/variable.cpp
  src/pomdpx/pomdpx.cpp
  src/random_streams.cpp
  src/simple_tui.cpp
  src/solver/aems.cpp
  src/solver/despot.cpp
  src/solver/pomcp.cpp
  src/util/coord.cpp
  src/util/dirichlet.cpp
  src/util/exec_tracker.cpp
  src/util/floor.cpp
  src/util/gamma.cpp
  src/util/logging.cpp
  src/util/random.cpp
  src/util/seeds.cpp
  src/util/util.cpp
  src/util/tinyxml/tinystr.cpp
  src/util/tinyxml/tinyxml.cpp
  src/util/tinyxml/tinyxmlerror.cpp
  src/util/tinyxml/tinyxmlparser.cpp
)
target_link_libraries("${PROJECT_NAME}"
  ${TinyXML_LIBRARIES}
)
