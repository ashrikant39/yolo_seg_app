# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /teamspace/studios/this_studio/yolo_seg_app

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /teamspace/studios/this_studio/yolo_seg_app/build

# Include any dependencies generated for this target.
include CMakeFiles/yoloSegApp.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/yoloSegApp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yoloSegApp.dir/flags.make

CMakeFiles/yoloSegApp.dir/src/main.cpp.o: CMakeFiles/yoloSegApp.dir/flags.make
CMakeFiles/yoloSegApp.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teamspace/studios/this_studio/yolo_seg_app/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yoloSegApp.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yoloSegApp.dir/src/main.cpp.o -c /teamspace/studios/this_studio/yolo_seg_app/src/main.cpp

CMakeFiles/yoloSegApp.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yoloSegApp.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teamspace/studios/this_studio/yolo_seg_app/src/main.cpp > CMakeFiles/yoloSegApp.dir/src/main.cpp.i

CMakeFiles/yoloSegApp.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yoloSegApp.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teamspace/studios/this_studio/yolo_seg_app/src/main.cpp -o CMakeFiles/yoloSegApp.dir/src/main.cpp.s

CMakeFiles/yoloSegApp.dir/src/modules/logger.cpp.o: CMakeFiles/yoloSegApp.dir/flags.make
CMakeFiles/yoloSegApp.dir/src/modules/logger.cpp.o: ../src/modules/logger.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teamspace/studios/this_studio/yolo_seg_app/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/yoloSegApp.dir/src/modules/logger.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yoloSegApp.dir/src/modules/logger.cpp.o -c /teamspace/studios/this_studio/yolo_seg_app/src/modules/logger.cpp

CMakeFiles/yoloSegApp.dir/src/modules/logger.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yoloSegApp.dir/src/modules/logger.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teamspace/studios/this_studio/yolo_seg_app/src/modules/logger.cpp > CMakeFiles/yoloSegApp.dir/src/modules/logger.cpp.i

CMakeFiles/yoloSegApp.dir/src/modules/logger.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yoloSegApp.dir/src/modules/logger.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teamspace/studios/this_studio/yolo_seg_app/src/modules/logger.cpp -o CMakeFiles/yoloSegApp.dir/src/modules/logger.cpp.s

CMakeFiles/yoloSegApp.dir/src/modules/runner.cpp.o: CMakeFiles/yoloSegApp.dir/flags.make
CMakeFiles/yoloSegApp.dir/src/modules/runner.cpp.o: ../src/modules/runner.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teamspace/studios/this_studio/yolo_seg_app/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/yoloSegApp.dir/src/modules/runner.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yoloSegApp.dir/src/modules/runner.cpp.o -c /teamspace/studios/this_studio/yolo_seg_app/src/modules/runner.cpp

CMakeFiles/yoloSegApp.dir/src/modules/runner.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yoloSegApp.dir/src/modules/runner.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teamspace/studios/this_studio/yolo_seg_app/src/modules/runner.cpp > CMakeFiles/yoloSegApp.dir/src/modules/runner.cpp.i

CMakeFiles/yoloSegApp.dir/src/modules/runner.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yoloSegApp.dir/src/modules/runner.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teamspace/studios/this_studio/yolo_seg_app/src/modules/runner.cpp -o CMakeFiles/yoloSegApp.dir/src/modules/runner.cpp.s

# Object files for target yoloSegApp
yoloSegApp_OBJECTS = \
"CMakeFiles/yoloSegApp.dir/src/main.cpp.o" \
"CMakeFiles/yoloSegApp.dir/src/modules/logger.cpp.o" \
"CMakeFiles/yoloSegApp.dir/src/modules/runner.cpp.o"

# External object files for target yoloSegApp
yoloSegApp_EXTERNAL_OBJECTS =

yoloSegApp: CMakeFiles/yoloSegApp.dir/src/main.cpp.o
yoloSegApp: CMakeFiles/yoloSegApp.dir/src/modules/logger.cpp.o
yoloSegApp: CMakeFiles/yoloSegApp.dir/src/modules/runner.cpp.o
yoloSegApp: CMakeFiles/yoloSegApp.dir/build.make
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.2.0
yoloSegApp: /usr/local/cuda/lib64/libcudart_static.a
yoloSegApp: /usr/lib/x86_64-linux-gnu/librt.so
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0
yoloSegApp: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0
yoloSegApp: CMakeFiles/yoloSegApp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teamspace/studios/this_studio/yolo_seg_app/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable yoloSegApp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yoloSegApp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yoloSegApp.dir/build: yoloSegApp

.PHONY : CMakeFiles/yoloSegApp.dir/build

CMakeFiles/yoloSegApp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yoloSegApp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yoloSegApp.dir/clean

CMakeFiles/yoloSegApp.dir/depend:
	cd /teamspace/studios/this_studio/yolo_seg_app/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teamspace/studios/this_studio/yolo_seg_app /teamspace/studios/this_studio/yolo_seg_app /teamspace/studios/this_studio/yolo_seg_app/build /teamspace/studios/this_studio/yolo_seg_app/build /teamspace/studios/this_studio/yolo_seg_app/build/CMakeFiles/yoloSegApp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/yoloSegApp.dir/depend

