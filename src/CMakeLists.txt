cmake_minimum_required(VERSION 3.14)
project(ParkingSlotDetection)

set(CMAKE_CXX_STANDARD 17)

set(ARCH 87)  # 设置GPU架构 for ada5000 gpu

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
set(TENSORRT_DIR /usr/include/aarch64-linux-gnu/)  # 替换为实际路径
set(Tensorrt_LIB /usr/lib/aarch64-linux-gnu/)  # 替换为实际路径
set(CUDA_DIR /usr/local/cuda-12.6/targets/aarch64-linux/include/)
set(CUDA_LIB /usr/local/cuda-12.6/targets/aarch64-linux/lib/)
# 包含TensorRT头文件
include_directories(${TENSORRT_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/plugin/grid_sampler
    ${PROJECT_SOURCE_DIR}/plugin/simpleplugin
    ${PROJECT_SOURCE_DIR}/common
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_DIR}
)

# 编一个cuda的动态库出来 

file(GLOB Project_ops_src  ${PROJECT_SOURCE_DIR}/common/*.cu ${PROJECT_SOURCE_DIR}/common/*.cpp ${PROJECT_SOURCE_DIR}/plugin/grid_sampler/*.cu ${PROJECT_SOURCE_DIR}/plugin/grid_sampler/*.cpp  ${PROJECT_SOURCE_DIR}/plugin/simpleplugin/*.cu  ${PROJECT_SOURCE_DIR}/plugin/simpleplugin/normalize.cu)


set(shared_target psdet_ops)

cuda_add_library(${shared_target} SHARED ${Project_ops_src} OPTIONS -arch=sm_${ARCH})

install (
    TARGETS ${shared_target}
    LIBRARY DESTINATION lib
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
    psdet_ops 
    cuda
)

# 设置输出目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)