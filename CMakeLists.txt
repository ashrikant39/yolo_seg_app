cmake_minimum_required(VERSION 3.16)
project(yoloSegApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(YOLO_BUILD_APP "Build the TensorRT/CUDA application" ON)
option(YOLO_BUILD_TESTS "Build initial CPU smoke tests" ON)

if(DEFINED ENV{OPENCV_INSTALL_PATH})
    set(OpenCV_DIR "$ENV{OPENCV_INSTALL_PATH}/lib/cmake/opencv4")
endif()
find_package(OpenCV REQUIRED)

add_library(yolo_cpu_utils
    src/post_process/utils/PostProcessUtils.cpp
)
target_include_directories(yolo_cpu_utils PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolo_cpu_utils PUBLIC ${OpenCV_LIBS})

if(YOLO_BUILD_TESTS)
    enable_testing()
    find_package(doctest REQUIRED)

    add_executable(initial_cpu_tests
        tests/initial_cpu_tests.cpp
    )
    target_link_libraries(initial_cpu_tests PRIVATE
        yolo_cpu_utils
        doctest::doctest
    )
    add_test(NAME initial_cpu_tests COMMAND initial_cpu_tests)

    add_executable(frame_source_tests
        tests/frame_source_tests.cpp
        src/logging/BaseLogger.cpp
    )
    target_include_directories(frame_source_tests PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(frame_source_tests PRIVATE
        ${OpenCV_LIBS}
        doctest::doctest
    )
    add_test(NAME frame_source_tests COMMAND frame_source_tests)
endif()

if(YOLO_BUILD_APP)
    find_package(CUDAToolkit REQUIRED)
    find_package(cxxopts REQUIRED)
    find_package(yaml-cpp REQUIRED)

    set(TENSORRT_INCLUDE_HINTS
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/cuda/include
    )
    set(TENSORRT_LIBRARY_HINTS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /usr/local/cuda/lib64
    )

    find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_INCLUDE_HINTS} REQUIRED)
    find_library(NVINFER_LIB nvinfer HINTS ${TENSORRT_LIBRARY_HINTS} REQUIRED)
    find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TENSORRT_LIBRARY_HINTS} REQUIRED)
    find_library(NVONNXPARSER_LIB nvonnxparser HINTS ${TENSORRT_LIBRARY_HINTS} REQUIRED)

    file(GLOB_RECURSE APP_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/src/application/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/backends/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/core/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/logging/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/memory_management/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/post_process/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/pre_process/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/sinks/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/source/*.cpp"
    )
    list(FILTER APP_SOURCES EXCLUDE REGEX ".*/src/pipelines/.*")

    add_executable(yoloSegApp
        src/main.cpp
        ${APP_SOURCES}
    )
    set_target_properties(yoloSegApp PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    target_include_directories(yoloSegApp PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        ${OpenCV_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
    )
    target_link_libraries(yoloSegApp PRIVATE
        ${NVINFER_LIB}
        ${NVINFER_PLUGIN_LIB}
        ${NVONNXPARSER_LIB}
        ${OpenCV_LIBS}
        CUDA::cudart
        cxxopts::cxxopts
        yaml-cpp
    )

    if(TARGET CUDA::nvToolsExt)
        target_link_libraries(yoloSegApp PRIVATE CUDA::nvToolsExt)
    elseif(TARGET CUDA::nvtx3)
        target_link_libraries(yoloSegApp PRIVATE CUDA::nvtx3)
    else()
        target_link_libraries(yoloSegApp PRIVATE nvToolsExt)
    endif()

    if(YOLO_BUILD_TESTS)
        add_executable(application_config_tests
            tests/application_config_tests.cpp
            src/core/yamlParser.cpp
        )
        target_include_directories(application_config_tests PRIVATE
            "${CMAKE_SOURCE_DIR}/include"
            ${OpenCV_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_link_libraries(application_config_tests PRIVATE
            ${OpenCV_LIBS}
            CUDA::cudart
            doctest::doctest
            yaml-cpp
        )
        add_test(NAME application_config_tests COMMAND application_config_tests)
    endif()

    get_filename_component(NVINFER_LIB_DIR "${NVINFER_LIB}" DIRECTORY)
    get_filename_component(NVINFER_PLUGIN_LIB_DIR "${NVINFER_PLUGIN_LIB}" DIRECTORY)
    get_filename_component(NVONNXPARSER_LIB_DIR "${NVONNXPARSER_LIB}" DIRECTORY)
    set(TENSORRT_LIB_DIRS "${NVINFER_LIB_DIR};${NVINFER_PLUGIN_LIB_DIR};${NVONNXPARSER_LIB_DIR}")
    set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH};${TENSORRT_LIB_DIRS}")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
    set(CMAKE_INSTALL_RPATH "${TENSORRT_LIB_DIRS}")
endif()
