cmake_minimum_required(VERSION 3.14)
project(ParkingSlotDetection LANGUAGES CXX CUDA)  # 声明CUDA语言

set(CMAKE_CXX_STANDARD 17)
set(ARCH 89)  # Tesla A100/Ada架构

# 构建配置
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")  # 合并标志
set(CMAKE_CUDA_ARCHITECTURES ${ARCH})  # 现代GPU架构设置[4](@ref)

# 依赖查找
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)  # 现代CUDA查找[5](@ref)


# TensorRT路径
set(TENSORRT_DIR /usr/include/x86_64-linux-gnu)
set(Tensorrt_LIB /usr/lib/x86_64-linux-gnu)



# 源文件集合
file(GLOB Project_ops_src  
    ${PROJECT_SOURCE_DIR}/common/*.cu 
    ${PROJECT_SOURCE_DIR}/common/*.cpp 
    ${PROJECT_SOURCE_DIR}/plugin/grid_sampler/*.cu 
    #${PROJECT_SOURCE_DIR}/plugin/grid_sampler/*.cpp  
    ${PROJECT_SOURCE_DIR}/plugin/simpleplugin/*.cu  
)

# message 所有源文件
message(STATUS "Project_ops_src: ${Project_ops_src}")

# CUDA动态库
set(shared_target psdet_ops)
add_library(${shared_target} SHARED ${Project_ops_src})
target_link_libraries(${shared_target} PUBLIC 
    CUDA::cudart
)
target_include_directories(${shared_target} PUBLIC
    ${TENSORRT_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/plugin
    ${PROJECT_SOURCE_DIR}/plugin/grid_sampler
    ${PROJECT_SOURCE_DIR}/plugin/simpleplugin
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(${shared_target} PUBLIC
    ${Tensorrt_LIB}
)

# 可执行文件
add_executable(psdet_demo 
    psdetdemo.cpp 
    psdet.cpp
)
target_link_libraries(psdet_demo PRIVATE
    ${OpenCV_LIBS}
    nvinfer
    nvonnxparser
    ${shared_target}
    CUDA::cudart  # 关键添加项
)

# 安装配置
install(TARGETS ${shared_target} LIBRARY DESTINATION lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)