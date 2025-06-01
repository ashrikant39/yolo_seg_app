cmake_minimum_required(VERSION 3.10)
project(yoloSegApp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

# === Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# === Find CUDA ===
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# === TensorRT include path ===
include_directories("$ENV{TRT_ROOT}/include")

# === TensorRT libraries ===
find_library(NVINFER_LIB nvinfer HINTS "$ENV{TRT_ROOT}/lib" "$ENV{TRT_ROOT}/lib64")
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS "$ENV{TRT_ROOT}/lib" "$ENV{TRT_ROOT}/lib64")
find_library(NVONNXPARSER_LIB nvonnxparser HINTS "$ENV{TRT_ROOT}/lib" "$ENV{TRT_ROOT}/lib64")

# Collect all .cpp files in the src directory
file(GLOB MODULES "${CMAKE_SOURCE_DIR}/src/modules/*.cpp")
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB TEST_SOURCES "${CMAKE_SOURCE_DIR}/tests/*.cpp")

# === Project Include Dir ===
include_directories("${CMAKE_SOURCE_DIR}/include")

# Main Executable
# add_executable(yoloSegApp ${SOURCES} ${MODULES})
# target_link_libraries(yoloSegApp ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB} ${NVONNXPARSER_LIB} ${OpenCV_LIBS} ${CUDA_LIBRARIES})

# Tests
set(TEST_OUTPUT_DIR "${CMAKE_BINARY_DIR}/tests/")
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE} ${MODULES})
    set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIR})
    target_link_libraries(${TEST_NAME} ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB} ${NVONNXPARSER_LIB} ${OpenCV_LIBS} ${CUDA_LIBRARIES})
endforeach()

