cmake_minimum_required(VERSION 3.10)
project(yoloSegApp)
set(CMAKE_CXX_STANDARD 17)

# Required Packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Library Header Paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories("/usr/include/x86_64-linux-gnu")  # TensorRT headers
include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories(${OpenCV_INCLUDE_DIRS})  # OpenCV headers

# Linking to TensorRT Libraries
link_directories("/usr/lib/x86_64-linux-gnu")  # TensorRT libraries

# Collect all .cpp files in the src directory
file(GLOB MODULES "${CMAKE_SOURCE_DIR}/src/modules/*.cpp")
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB TEST_SOURCES "${CMAKE_SOURCE_DIR}/tests/*.cpp")

# Main Executable
add_executable(yoloSegApp ${SOURCES} ${MODULES})
target_link_libraries(yoloSegApp nvinfer nvinfer_plugin nvonnxparser ${OpenCV_LIBS} ${CUDA_LIBRARIES})

# Tests
set(TEST_OUTPUT_DIR "${CMAKE_BINARY_DIR}/tests/")
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE} ${MODULES})
    set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIR})
    target_link_libraries(${TEST_NAME} nvinfer nvinfer_plugin nvonnxparser ${OpenCV_LIBS} ${CUDA_LIBRARIES})
endforeach()

