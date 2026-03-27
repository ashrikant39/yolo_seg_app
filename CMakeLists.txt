cmake_minimum_required(VERSION 3.10)
project(yoloSegApp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# === Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# === Find CUDA Toolkit ===
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# === TensorRT include/library discovery ===
# Ubuntu apt installs generally place headers/libs under /usr/include and /usr/lib/*.
# This project intentionally assumes standard system locations on Ubuntu.

set(TENSORRT_INCLUDE_HINTS
    /usr/include
    /usr/include/x86_64-linux-gnu
)
set(TENSORRT_LIBRARY_HINTS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu
)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_INCLUDE_HINTS} REQUIRED)
include_directories(${TENSORRT_INCLUDE_DIR})

# === TensorRT libraries ===
find_library(NVINFER_LIB nvinfer HINTS ${TENSORRT_LIBRARY_HINTS} REQUIRED)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TENSORRT_LIBRARY_HINTS} REQUIRED)
find_library(NVONNXPARSER_LIB nvonnxparser HINTS ${TENSORRT_LIBRARY_HINTS} REQUIRED)



# === cxxopts Header Only Library ===
find_package(cxxopts REQUIRED)

# Collect all .cpp files in the src directory
file(GLOB MODULES "${CMAKE_SOURCE_DIR}/src/modules/*.cpp")
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB TEST_SOURCES "${CMAKE_SOURCE_DIR}/tests/*.cpp")

# === Project Include Dir ===
include_directories("${CMAKE_SOURCE_DIR}/include")

# Main Executable
add_executable(yoloSegApp ${SOURCES} ${MODULES})
set_target_properties(yoloSegApp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
target_link_libraries(yoloSegApp 
${NVINFER_LIB} 
${NVINFER_PLUGIN_LIB} 
${NVONNXPARSER_LIB} 
${OpenCV_LIBS} 
CUDA::cudart
cxxopts::cxxopts
)

# --- NVTX: pick the available imported target, otherwise fallback ---
if (TARGET CUDA::nvToolsExt)
  target_link_libraries(yoloSegApp CUDA::nvToolsExt)
elseif (TARGET CUDA::nvtx3)
  target_link_libraries(yoloSegApp CUDA::nvtx3)
else()
  target_link_libraries(yoloSegApp nvToolsExt)  # -lnvToolsExt
  target_include_directories(yoloSegApp PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
endif()

# --- RPATH so binaries find TRT/CUDA without LD_LIBRARY_PATH tweaks (optional but handy) ---
# Add TRT lib dirs discovered by find_library to build/install rpath.
get_filename_component(NVINFER_LIB_DIR "${NVINFER_LIB}" DIRECTORY)
get_filename_component(NVINFER_PLUGIN_LIB_DIR "${NVINFER_PLUGIN_LIB}" DIRECTORY)
get_filename_component(NVONNXPARSER_LIB_DIR "${NVONNXPARSER_LIB}" DIRECTORY)
set(TENSORRT_LIB_DIRS "${NVINFER_LIB_DIR};${NVINFER_PLUGIN_LIB_DIR};${NVONNXPARSER_LIB_DIR}")
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH};${TENSORRT_LIB_DIRS}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
# (optional) keep rpath on install; adjust to your deployment policy
set(CMAKE_INSTALL_RPATH "${TENSORRT_LIB_DIRS}")

# =========================
# Component smoke tests
# =========================

enable_testing()

if(TEST_SOURCES)
    foreach(test_src ${TEST_SOURCES})
        get_filename_component(test_name "${test_src}" NAME_WE)

        if (test_name STREQUAL "char")
            continue()
        endif()

        # Build tests by compiling the same module sources as the main app
        # (but without `src/main.cpp`, since `MODULES` excludes it).
        add_executable(${test_name} ${test_src} ${MODULES})

        set_target_properties(${test_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )

        target_link_libraries(${test_name}
            ${NVINFER_LIB}
            ${NVINFER_PLUGIN_LIB}
            ${NVONNXPARSER_LIB}
            ${OpenCV_LIBS}
            CUDA::cudart
            cxxopts::cxxopts
        )

        # NVTX linking for the test targets
        if (TARGET CUDA::nvToolsExt)
            target_link_libraries(${test_name} CUDA::nvToolsExt)
        elseif (TARGET CUDA::nvtx3)
            target_link_libraries(${test_name} CUDA::nvtx3)
        else()
            target_link_libraries(${test_name} nvToolsExt)
            target_include_directories(${test_name} PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
        endif()

        add_test(NAME ${test_name} COMMAND ${CMAKE_BINARY_DIR}/bin/${test_name})
    endforeach()
endif()