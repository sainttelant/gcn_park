cmake_minimum_required(VERSION 3.14)
project(ParkingSlotDetection)

set(CMAKE_CXX_STANDARD 17)


set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# 添加下面的行为了禁止优化
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

# 查找OpenCV
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

message("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

# 设置TensorRT路径
set(TENSORRT_DIR /usr/include/x86_64-linux-gnu)  # 替换为实际路径
set(Tensorrt_LIB /usr/lib/x86_64-linux-gnu)  # 替换为实际路径
set(CUDA_DIR /usr/local/cuda-11.8/targets/x86_64-linux/include/)
set(CUDA_LIB /usr/local/cuda-11.8/targets/x86_64-linux/lib/)
# 包含TensorRT头文件
include_directories(${TENSORRT_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_DIR}
)

message("TensorRT_DIR: ${TENSORRT_DIR}")
message("project source dir: ${PROJECT_SOURCE_DIR}")
# 链接TensorRT库
link_directories(${Tensorrt_LIB} 
${CUDA_LIB}
)

# 添加可执行文件
add_executable(psdet_demo 
    psdetdemo.cpp 
    psdet.cpp
)

# 链接库
target_link_libraries(psdet_demo 
    ${OpenCV_LIBS}
    nvinfer
    nvonnxparser
    cudart
    cuda
)

# 设置输出目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)