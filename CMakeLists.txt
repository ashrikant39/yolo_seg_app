cmake_minimum_required(VERSION 3.10)
project(yoloSegApp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# === Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# === Find CUDA ===
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# === TensorRT include path ===
# Allow TRT_ROOT to be passed in or fallback to environment variable
if(NOT DEFINED TRT_ROOT)
    if(DEFINED ENV{TRT_ROOT})
        set(TRT_ROOT $ENV{TRT_ROOT})
    else()
        message(FATAL_ERROR "TRT_ROOT is not set! Please specify -DTRT_ROOT=/path/to/TensorRT or export TRT_ROOT.")
    endif()
endif()
include_directories("${TRT_ROOT}/include")

# === TensorRT libraries ===
find_library(NVINFER_LIB nvinfer HINTS "$ENV{TRT_ROOT}/lib" "$ENV{TRT_ROOT}/lib64")
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS "$ENV{TRT_ROOT}/lib" "$ENV{TRT_ROOT}/lib64")
find_library(NVONNXPARSER_LIB nvonnxparser HINTS "$ENV{TRT_ROOT}/lib" "$ENV{TRT_ROOT}/lib64")

# === Eigen Library ===
set(Eigen3_DIR "$ENV{Eigen_CMAKE_DIR}")
find_package (Eigen3 3.3 REQUIRED)


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
${CUDA_LIBRARIES} 
Eigen3::Eigen 
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
# Add TRT lib dirs to build and install rpath
set(TENSORRT_LIB_DIRS "${TRT_ROOT}/lib;${TRT_ROOT}/lib64")
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH};${TENSORRT_LIB_DIRS}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
# (optional) keep rpath on install; adjust to your deployment policy
set(CMAKE_INSTALL_RPATH "${TENSORRT_LIB_DIRS}")