# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_COMMAND = /home/lhc/Downloads/clion-2019.2.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/lhc/Downloads/clion-2019.2.2/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lhc/work/CUDA_test_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lhc/work/CUDA_test_ws/src/cmake-build-debug

# Include any dependencies generated for this target.
include test_cuda/CMakeFiles/test_cuda_node.dir/depend.make

# Include the progress variables for this target.
include test_cuda/CMakeFiles/test_cuda_node.dir/progress.make

# Include the compile flags for this target's objects.
include test_cuda/CMakeFiles/test_cuda_node.dir/flags.make

test_cuda/CMakeFiles/test_cuda_node.dir/src/test_cuda_node_generated_test.cu.o: test_cuda/CMakeFiles/test_cuda_node.dir/src/test_cuda_node_generated_test.cu.o.depend
test_cuda/CMakeFiles/test_cuda_node.dir/src/test_cuda_node_generated_test.cu.o: test_cuda/CMakeFiles/test_cuda_node.dir/src/test_cuda_node_generated_test.cu.o.Debug.cmake
test_cuda/CMakeFiles/test_cuda_node.dir/src/test_cuda_node_generated_test.cu.o: ../test_cuda/src/test.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/lhc/work/CUDA_test_ws/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object test_cuda/CMakeFiles/test_cuda_node.dir/src/test_cuda_node_generated_test.cu.o"
	cd /home/lhc/work/CUDA_test_ws/src/cmake-build-debug/test_cuda/CMakeFiles/test_cuda_node.dir/src && /home/lhc/Downloads/clion-2019.2.2/bin/cmake/linux/bin/cmake -E make_directory /home/lhc/work/CUDA_test_ws/src/cmake-build-debug/test_cuda/CMakeFiles/test_cuda_node.dir/src/.
	cd /home/lhc/work/CUDA_test_ws/src/cmake-build-debug/test_cuda/CMakeFiles/test_cuda_node.dir/src && /home/lhc/Downloads/clion-2019.2.2/bin/cmake/linux/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Debug -D generated_file:STRING=/home/lhc/work/CUDA_test_ws/src/cmake-build-debug/test_cuda/CMakeFiles/test_cuda_node.dir/src/./test_cuda_node_generated_test.cu.o -D generated_cubin_file:STRING=/home/lhc/work/CUDA_test_ws/src/cmake-build-debug/test_cuda/CMakeFiles/test_cuda_node.dir/src/./test_cuda_node_generated_test.cu.o.cubin.txt -P /home/lhc/work/CUDA_test_ws/src/cmake-build-debug/test_cuda/CMakeFiles/test_cuda_node.dir/src/test_cuda_node_generated_test.cu.o.Debug.cmake

test_cuda/CMakeFiles/test_cuda_node.dir/src/main.cpp.o: test_cuda/CMakeFiles/test_cuda_node.dir/flags.make
test_cuda/CMakeFiles/test_cuda_node.dir/src/main.cpp.o: ../test_cuda/src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lhc/work/CUDA_test_ws/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object test_cuda/CMakeFiles/test_cuda_node.dir/src/main.cpp.o"
	cd /home/lhc/work/CUDA_test_ws/src/cmake-build-debug/test_cuda && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_cuda_node.dir/src/main.cpp.o -c /home/lhc/work/CUDA_test_ws/src/test_cuda/src/main.cpp

test_cuda/CMakeFiles/test_cuda_node.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_cuda_node.dir/src/main.cpp.i"
	cd /home/lhc/work/CUDA_test_ws/src/cmake-build-debug/test_cuda && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lhc/work/CUDA_test_ws/src/test_cuda/src/main.cpp > CMakeFiles/test_cuda_node.dir/src/main.cpp.i

test_cuda/CMakeFiles/test_cuda_node.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_cuda_node.dir/src/main.cpp.s"
	cd /home/lhc/work/CUDA_test_ws/src/cmake-build-debug/test_cuda && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lhc/work/CUDA_test_ws/src/test_cuda/src/main.cpp -o CMakeFiles/test_cuda_node.dir/src/main.cpp.s

# Object files for target test_cuda_node
test_cuda_node_OBJECTS = \
"CMakeFiles/test_cuda_node.dir/src/main.cpp.o"

# External object files for target test_cuda_node
test_cuda_node_EXTERNAL_OBJECTS = \
"/home/lhc/work/CUDA_test_ws/src/cmake-build-debug/test_cuda/CMakeFiles/test_cuda_node.dir/src/test_cuda_node_generated_test.cu.o"

devel/lib/test_cuda/test_cuda_node: test_cuda/CMakeFiles/test_cuda_node.dir/src/main.cpp.o
devel/lib/test_cuda/test_cuda_node: test_cuda/CMakeFiles/test_cuda_node.dir/src/test_cuda_node_generated_test.cu.o
devel/lib/test_cuda/test_cuda_node: test_cuda/CMakeFiles/test_cuda_node.dir/build.make
devel/lib/test_cuda/test_cuda_node: /usr/local/cuda-8.0/lib64/libcudart_static.a
devel/lib/test_cuda/test_cuda_node: /usr/lib/x86_64-linux-gnu/librt.so
devel/lib/test_cuda/test_cuda_node: /opt/ros/kinetic/lib/libroscpp.so
devel/lib/test_cuda/test_cuda_node: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
devel/lib/test_cuda/test_cuda_node: /usr/lib/x86_64-linux-gnu/libboost_signals.so
devel/lib/test_cuda/test_cuda_node: /opt/ros/kinetic/lib/librosconsole.so
devel/lib/test_cuda/test_cuda_node: /opt/ros/kinetic/lib/librosconsole_log4cxx.so
devel/lib/test_cuda/test_cuda_node: /opt/ros/kinetic/lib/librosconsole_backend_interface.so
devel/lib/test_cuda/test_cuda_node: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
devel/lib/test_cuda/test_cuda_node: /usr/lib/x86_64-linux-gnu/libboost_regex.so
devel/lib/test_cuda/test_cuda_node: /opt/ros/kinetic/lib/libxmlrpcpp.so
devel/lib/test_cuda/test_cuda_node: /opt/ros/kinetic/lib/libroscpp_serialization.so
devel/lib/test_cuda/test_cuda_node: /opt/ros/kinetic/lib/librostime.so
devel/lib/test_cuda/test_cuda_node: /opt/ros/kinetic/lib/libcpp_common.so
devel/lib/test_cuda/test_cuda_node: /usr/lib/x86_64-linux-gnu/libboost_system.so
devel/lib/test_cuda/test_cuda_node: /usr/lib/x86_64-linux-gnu/libboost_thread.so
devel/lib/test_cuda/test_cuda_node: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
devel/lib/test_cuda/test_cuda_node: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
devel/lib/test_cuda/test_cuda_node: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
devel/lib/test_cuda/test_cuda_node: /usr/lib/x86_64-linux-gnu/libpthread.so
devel/lib/test_cuda/test_cuda_node: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
devel/lib/test_cuda/test_cuda_node: test_cuda/CMakeFiles/test_cuda_node.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lhc/work/CUDA_test_ws/src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../devel/lib/test_cuda/test_cuda_node"
	cd /home/lhc/work/CUDA_test_ws/src/cmake-build-debug/test_cuda && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_cuda_node.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test_cuda/CMakeFiles/test_cuda_node.dir/build: devel/lib/test_cuda/test_cuda_node

.PHONY : test_cuda/CMakeFiles/test_cuda_node.dir/build

test_cuda/CMakeFiles/test_cuda_node.dir/clean:
	cd /home/lhc/work/CUDA_test_ws/src/cmake-build-debug/test_cuda && $(CMAKE_COMMAND) -P CMakeFiles/test_cuda_node.dir/cmake_clean.cmake
.PHONY : test_cuda/CMakeFiles/test_cuda_node.dir/clean

test_cuda/CMakeFiles/test_cuda_node.dir/depend: test_cuda/CMakeFiles/test_cuda_node.dir/src/test_cuda_node_generated_test.cu.o
	cd /home/lhc/work/CUDA_test_ws/src/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lhc/work/CUDA_test_ws/src /home/lhc/work/CUDA_test_ws/src/test_cuda /home/lhc/work/CUDA_test_ws/src/cmake-build-debug /home/lhc/work/CUDA_test_ws/src/cmake-build-debug/test_cuda /home/lhc/work/CUDA_test_ws/src/cmake-build-debug/test_cuda/CMakeFiles/test_cuda_node.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test_cuda/CMakeFiles/test_cuda_node.dir/depend

