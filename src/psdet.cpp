#include "psdet.hpp"
#include "gridSamplerKernel.h"
#include "normalize.h"
#include <algorithm>
#include <cassert>
#include <numeric>  // 修复：添加 std::iota 所需的头文件
using namespace psdet;

// 构造函数
PsDet::PsDet(const std::string& onnx_path, 
             const std::string& engine_path,
             int max_batch_size,
             int max_points,
             int max_slots,
             int input_width,
             int input_height)
    : onnx_path_(onnx_path),
      engine_path_(engine_path),
      max_batch_size_(max_batch_size),
      max_points_(max_points),
      max_slots_(max_slots),
      input_width_(input_width),
      input_height_(input_height) {
    
    input_h_.resize(max_batch_size_ * input_channels_ * input_height_ * input_width_);
    output_points_h_.resize(max_batch_size_ *3* points_3d_dim_ * points_4d_dim_);
    output_slots_h_.resize(max_batch_size_ *128* points_3d_dim_ * points_4d_dim_);

    // 初始化gnn相关的
    gnn_descriptors_h_.resize(max_points * 128);  // [1, 128, max_points] 实际是 [1,128,10]
    gnn_points_h_.resize(max_points * 2);          // [1, max_points, 2]   实际是 [1,10,2]
    gnn_edge_pred_h_.resize(max_points * max_points); // [1, 1, max_points*max_points] 实际是 [1,1,100]
    gnn_graph_output_h_.resize(max_points * 64);  // [1, 64, max_points]   实际是 [1,64,10]

}

// 析构函数 - 修复销毁顺序[8](@ref)
PsDet::~PsDet() {
    // 只destroy主要前端网络的buffers
    destroyBuffers();
    if (context_) {
        context_->destroy();
        context_ = nullptr;
    }
    if (engine_) {
        engine_->destroy();
        engine_ = nullptr;
    }
    if (runtime_) {
        runtime_->destroy();
        runtime_ = nullptr;
    }

     // 释放GNN资源
    if (gnn_context_) gnn_context_->destroy();
    if (gnn_engine_) gnn_engine_->destroy();
    
    // 释放CUDA内存
    if (gnn_descriptors_d_) cudaFree(gnn_descriptors_d_);
    if (gnn_points_d_) cudaFree(gnn_points_d_);
    if (gnn_edge_pred_d_) cudaFree(gnn_edge_pred_d_);
    if (gnn_graph_output_d_) cudaFree(gnn_graph_output_d_);
    
}

// 初始化CUDA缓冲区
bool PsDet::initBuffers() {
    destroyBuffers();
    
    cudaMalloc(&input_d_, input_h_.size() * sizeof(float));
    cudaMalloc(&output_points_d_, output_points_h_.size() * sizeof(float));
    cudaMalloc(&output_slots_d_, output_slots_h_.size() * sizeof(float));
    cudaStreamCreate(&stream_);
    
    return input_d_ && output_points_d_ && output_slots_d_ && stream_;
}

// 销毁CUDA缓冲区
void PsDet::destroyBuffers() {
    if (input_d_) cudaFree(input_d_);
    if (output_points_d_) cudaFree(output_points_d_);
    if (output_slots_d_) cudaFree(output_slots_d_);
    if (stream_) cudaStreamDestroy(stream_);
    
    input_d_ = output_points_d_ = output_slots_d_ = nullptr;
    stream_ = nullptr;
}

bool PsDet::build(bool fp16) {
    // 记录构建开始
    logger_.log(ILogger::Severity::kINFO, "Starting TensorRT engine build process...");
    logger_.log(ILogger::Severity::kINFO, ("ONNX Path: " + onnx_path_).c_str());
    logger_.log(ILogger::Severity::kINFO, ("Engine Output: " + engine_path_).c_str());

    // 1. 创建构建器（增加错误日志）
    IBuilder* builder = createInferBuilder(logger_);
    if (!builder) {
        logger_.log(ILogger::Severity::kERROR, "Failed to create TensorRT builder");
        return false;
    }

    // 2. 创建显式批处理网络
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    if (!network) {
        logger_.log(ILogger::Severity::kERROR, "Failed to create network definition");
        builder->destroy();
        return false;
    }

    // 3. 解析ONNX模型（增强错误处理）
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger_);
    if (!parser) {
        logger_.log(ILogger::Severity::kERROR, "Failed to create ONNX parser");
        network->destroy();
        builder->destroy();
        return false;
    }

    if (!parser->parseFromFile(onnx_path_.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        // 获取详细的解析错误信息
        const int numErrors = parser->getNbErrors();
        for (int i = 0; i < numErrors; ++i) {
            const auto* error = parser->getError(i);
            logger_.log(ILogger::Severity::kERROR, ("ONNX Parse Error: " + std::string(error->desc())).c_str());
        }
        parser->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }

    // 4. 创建构建配置
    IBuilderConfig* config = builder->createBuilderConfig();
    if (!config) {
        logger_.log(ILogger::Severity::kERROR, "Failed to create builder config");
        parser->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }

    // 5. 设置优化配置文件（关键调试点）
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (!profile) {
        logger_.log(ILogger::Severity::kERROR, "Failed to create optimization profile");
            //config->destroy();
        parser->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }

    // 获取输入信息并验证维度
    ITensor* input = network->getInput(0);
    if (!input) {
        logger_.log(ILogger::Severity::kERROR, "Network has no input tensors");
        //profile->destroy();
            //config->destroy();
        parser->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }

    const char* input_name = input->getName();
    Dims input_dims = input->getDimensions();
    
    // 调试输出：打印原始输入维度
    std::string dims_str = "Original Input Dimensions: [";
    for (int i = 0; i < input_dims.nbDims; ++i) {
        dims_str += std::to_string(input_dims.d[i]);
        if (i < input_dims.nbDims - 1) dims_str += ", ";
    }
    dims_str += "]";
    logger_.log(ILogger::Severity::kINFO, dims_str.c_str());

    // 验证并修复动态维度（核心调试增强）
    if (input_dims.nbDims != 4) {
        logger_.log(ILogger::Severity::kERROR, "Input must be 4-dimensional (NCHW format)");
        return false;
    }

    // 1. 打印所有输入张量信息（支持多输入）
        int num_inputs = network->getNbInputs();
        logger_.log(ILogger::Severity::kINFO, 
                    ("Network has " + std::to_string(num_inputs) + " inputs").c_str());

        for (int idx = 0; idx < num_inputs; ++idx) {
            ITensor* input_tensor = network->getInput(idx);
            const char* tensor_name = input_tensor->getName();
            Dims dims = input_tensor->getDimensions();

            // 构建详细维度描述字符串
            std::string dim_str = "Input[" + std::to_string(idx) + "]: " + tensor_name + " : [";
            for (int d = 0; d < dims.nbDims; ++d) {
                dim_str += (dims.d[d] == -1) ? "?" : std::to_string(dims.d[d]); // 动态维度显示为?
                if (d < dims.nbDims - 1) dim_str += ", ";
            }
            dim_str += "]";
            logger_.log(ILogger::Severity::kINFO, dim_str.c_str());
        }

    // 确保通道维度有效
    const int channels = input_dims.d[1] > 0 ? input_dims.d[1] : 3;
    if (channels != 1 && channels != 3) {
        logger_.log(ILogger::Severity::kWARNING, 
                   ("Unexpected channel dimension: " + std::to_string(channels) + 
                    ". Using default value 3").c_str());
    }

    // 设置优化范围（使用用户配置的尺寸）
    if (1)
    {
    const Dims min_dims = Dims4{1, channels, input_height_ / 2, input_width_ / 2};
    const Dims opt_dims = Dims4{max_batch_size_, channels, input_height_, input_width_};
    const Dims max_dims = Dims4{max_batch_size_, channels, input_height_ * 2, input_width_ * 2};

    // 调试输出：打印优化配置
    auto dimsToString = [](const Dims& d) {
        return "[" + std::to_string(d.d[0]) + ", " + std::to_string(d.d[1]) + ", " 
             + std::to_string(d.d[2]) + ", " + std::to_string(d.d[3]) + "]";
    };
    
    logger_.log(ILogger::Severity::kINFO, ("Setting optimization profile for: " + std::string(input_name)).c_str());
    logger_.log(ILogger::Severity::kINFO, ("  MIN shape: " + dimsToString(min_dims)).c_str());
    logger_.log(ILogger::Severity::kINFO, ("  OPT shape: " + dimsToString(opt_dims)).c_str());
    logger_.log(ILogger::Severity::kINFO, ("  MAX shape: " + dimsToString(max_dims)).c_str());

    // 应用优化配置
    profile->setDimensions(input_name, OptProfileSelector::kMIN, min_dims);
    profile->setDimensions(input_name, OptProfileSelector::kOPT, opt_dims);
    profile->setDimensions(input_name, OptProfileSelector::kMAX, max_dims);
    config->addOptimizationProfile(profile);
    }
    else
    {
        // fixed shape
        nvinfer1::ITensor* input = network->getInput(0);
        input->setDimensions(nvinfer1::Dims4(1, 3, 512, 512));
    }



    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30); // 1GB

    // 6. 设置计算精度
    if (fp16 && builder->platformHasFastFp16()) {
        config->setFlag(BuilderFlag::kFP16);
        logger_.log(ILogger::Severity::kINFO, "FP16 mode enabled");
    } else {
        logger_.log(ILogger::Severity::kINFO, "Using FP32 precision");
    } 

    

    // 7. 构建引擎（增加详细日志）
    printf("Building engine... (this may take several minutes) \n");
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    
    if (!engine) {
        logger_.log(ILogger::Severity::kERROR, "Engine build failed. Possible causes:");
        logger_.log(ILogger::Severity::kERROR, "1. Insufficient GPU memory");
        logger_.log(ILogger::Severity::kERROR, "2. Unsupported ONNX operator");
        logger_.log(ILogger::Severity::kERROR, "3. Invalid optimization profile settings");
        
        // 清理资源
       // profile->destroy();
            //config->destroy();
        parser->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }

    for (int i = 0; i < engine->getNbBindings(); ++i) {
        Dims dims = engine->getBindingDimensions(i);
        std::string type = (engine->bindingIsInput(i)) ? "Input" : "Output";
        std::cout << type << "[" << i << "] : ";
        for (int d = 0; d < dims.nbDims; ++d) {
            std::cout << (dims.d[d] == -1 ? "?" : std::to_string(dims.d[d])) << " ";
        }
        std::cout << std::endl;
    }


     printf("before serializing engine <<<<<<<<<<<<<<<<<<<\n");
    // 8. 序列化引擎
    IHostMemory* serialized_engine = engine->serialize();
    if (!serialized_engine) {
        logger_.log(ILogger::Severity::kERROR, "Failed to serialize engine");
        engine->destroy();
       // profile->destroy();
            //config->destroy();
        parser->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }
    printf("before saving engine <<<<<<<<<<<<<<<<<<<\n");
    // 9. 保存引擎文件
     std::ofstream ofs(engine_path_, std::ios::binary);
    if (ofs) {
        ofs.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
        logger_.log(ILogger::Severity::kINFO, 
                   ("Engine saved successfully. Size: " + 
                    std::to_string(serialized_engine->size() / (1024 * 1024)) + " MB").c_str());
    } else {
        logger_.log(ILogger::Severity::kERROR, 
                   ("Failed to open engine file for writing: " + engine_path_).c_str());
    } 

    printf("before destroy <<<<<<<<<<<<<<<<<<<\n");
    // 10. 资源清理（使用安全销毁模式）
    auto safeDestroy = [](auto* obj) { if (obj) obj->destroy(); };
    safeDestroy(serialized_engine);
    safeDestroy(engine);
    //safeDestroy(profile);
    safeDestroy(config);
    safeDestroy(parser);
    safeDestroy(network);
    safeDestroy(builder);

    return ofs.good();
   // logger_.log(ILogger::Severity::kINFO, "Engine build complete and save to " + engine_path_.c_str());
    printf("Engine build complete and save to %s\n", engine_path_.c_str());
}

// 加载引擎
bool PsDet::load() {
    std::ifstream engine_file(engine_path_, std::ios::binary);
    if (!engine_file) return false;
    
    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    engine_file.read(buffer.data(), size);
    
    runtime_ = createInferRuntime(logger_);
    if (!runtime_) return false;
    
    engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
    if (!engine_) {
        runtime_->destroy();
        runtime_ = nullptr;
        return false;
    }
    
    context_ = engine_->createExecutionContext();
    if (!context_ || !initBuffers()) {
        if (context_) context_->destroy();
        if (engine_) engine_->destroy();
        if (runtime_) runtime_->destroy();
        return false;
    }
    
    return true;
}

// 图像预处理
void PsDet::preprocess(const cv::Mat& image, float* input) {
    cv::Mat resized, rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(input_width_, input_height_));
    
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0/255.0);
    
    // CHW转换
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    
    size_t offset = 0;
    for (int c = 0; c < 3; ++c) {
        memcpy(input + offset, channels[c].data, input_height_ * input_width_ * sizeof(float));
        offset += input_height_ * input_width_;
    }
}


void PsDet::process_points(
    const float* points_data,
    std::vector<std::vector<KeyPoint>>& output_points,
    int batch_size) 
{
    
    output_points.clear();
    //output_points.resize(batch_size);
    
    std::vector<KeyPoint> points;
    for (int i = 0; i < 16; ++i) {       // 行循环
        for (int j = 0; j < 16; ++j) {   // 列循环
            const int base_idx = i * 16 + j;
            const float conf = points_data[0 * 256 + base_idx]; // 通道0：置信度
            const float offset_x = points_data[1 * 256 + base_idx]; // 通道1：x偏移
            const float offset_y = points_data[2 * 256 + base_idx]; // 通道2：y偏移

            if (conf >= point_thresh_) {
                const float xval = (j + offset_x) / 16.0f; // 归一化坐标计算
                const float yval = (i + offset_y) / 16.0f;

                // 边界检查（跳过边缘点）
                if (xval >= slot_thresh_ && xval <= (1 - slot_thresh_) &&
                    yval >= slot_thresh_ && yval <= (1 - slot_thresh_)) {
                    
                    // 创建KeyPoint实例并添加到容器
                    points.push_back(KeyPoint{conf, xval, yval});
                }
            }
        }
    }
    // 应用改进的NMS
    auto nms_points = applyNMS(points, nms_thresh_);
    output_points.push_back(nms_points);
    
}

 void PsDet::process_slots(
    const std::vector<std::vector<KeyPoint>>& points_list, 
    const float* descriptor_map,
    std::vector<std::vector<ParkingSlot>>& output_slots,
    int batch_size)
{
    output_slots.clear();
    output_slots.resize(batch_size);  // 确保有足够的batch空间
    
    // 描述符图的尺寸信息
    const int desc_channels = 128;
    const int desc_height = input_height_ / 32;
    const int desc_width = input_width_ / 32;
    
    // 检查描述符图尺寸合法性
    if (desc_height <= 0 || desc_width <= 0) {
        std::cerr << "Invalid descriptor map dimensions: " 
                 << desc_height << "x" << desc_width << std::endl;
        return;
    }

    std::cout << "Descriptor map dimensions: " << desc_height << "x" << desc_width << std::endl;

    // 批处理循环
    for (int b = 0; b < batch_size; ++b) {
        std::cout << "Processing batch: " << b << std::endl;

        // 1. 准备实际关键点（跳过固定大小的points_for_reserve）
        const auto& actual_points = points_list[b];
        // const int num_points = actual_points.size();

        // 使用10个点与python结果同步
        const int num_points = 10;
        const int actual_points_size = actual_points.size();
        
        std::cout << "Number of points in batch " << b << ": " << num_points << std::endl;

        // 跳过空点集
        if (num_points == 0) {
            std::cout << "No points in batch " << b << ", skipping." << std::endl;
            continue;
        }

        std::vector<ParkingSlot> batch_slots;
        std::vector<float> grid_points;
        grid_points.reserve(num_points * 2);  // 动态分配实际所需空间

        for (int i = 0; i < num_points; ++i) {
            if (i < actual_points_size) {
                float grid_x = std::clamp(actual_points[i].x * 2 - 1, -1.0f, 1.0f);
                float grid_y = std::clamp(actual_points[i].y * 2 - 1, -1.0f, 1.0f);
                grid_points.push_back(grid_x);
                grid_points.push_back(grid_y);
            }
            else {
                grid_points.push_back(-1.0f);
                grid_points.push_back(-1.0f);
            }       
        }


      
        std::cout << "Grid points prepared for batch " << b << ": " << grid_points.size() / 2 << " pairs" << std::endl;

        // 3. GPU描述符采样
        float* sampled_descriptors = nullptr;
        float* descriptor_map_gpu = nullptr;
     

       
        // 设置张量维度
        int input_dims[4] = {1, desc_channels, desc_height, desc_width};
        int grid_dims[4] = {1, 1, num_points, 2}; // [N, H_out, W_out, 2]

        int output_dims[4] = {1, desc_channels, 1, num_points}; // [N, C, H_out, W_out]

        printf("Input dims: %d %d %d %d\n", input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
        printf("Grid dims: %d %d %d %d\n", grid_dims[0], grid_dims[1], grid_dims[2], grid_dims[3]);
        printf("Output dims: %d %d %d %d\n", output_dims[0], output_dims[1], output_dims[2], output_dims[3]);

        size_t required_mem = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3] * sizeof(float);
        cudaError_t err = cudaMalloc((void**)&sampled_descriptors, required_mem);
        size_t required_mem2 = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3] * sizeof(float);
        cudaError_t err2 = cudaMalloc((void**)&descriptor_map_gpu, required_mem2);
        if (err != cudaSuccess) {
            std::cerr << "CUDA malloc error: " << cudaGetErrorString(err)
                     << " | Size: " << (required_mem / 1024) << " KB" << std::endl;
            continue;  // 跳过当前batch
        }

        if (err2 != cudaSuccess) {
            std::cerr << "CUDA malloc error: " << cudaGetErrorString(err2)
                     << " | Size: " << (required_mem / 1024) << " KB" << std::endl;
           
        }
        else{
            std::cout << "CUDA malloc success: " << required_mem / 1024 << " KB" << std::endl;
            cudaMemcpyAsync(descriptor_map_gpu, descriptor_map+b*desc_channels*desc_height*desc_width, required_mem2, cudaMemcpyHostToDevice, stream_);
        }

               
        // 分配设备内存
        float* d_grid_points = nullptr;
        size_t grid_bytes = num_points * 2 * sizeof(float); // 确保32字节
        cudaMalloc((void**)&d_grid_points, grid_bytes);
        cudaMemcpyAsync(d_grid_points, grid_points.data(), grid_bytes, 
                            cudaMemcpyHostToDevice, stream_);
        
        printf("before grid sample <<<<<<<<<<<<<<<<<<<<<<<\n");
        // 调用grid_sample内核
         grid_sample<float>(
            sampled_descriptors, 
            descriptor_map_gpu + b*desc_channels*desc_height*desc_width,
            d_grid_points,
            output_dims,
            input_dims,
            grid_dims,
            4,
            GridSamplerInterpolation::Bilinear,
            GridSamplerPadding::Zeros,
            false,
            stream_
        );  

        // 检查内核执行错误
        cudaStreamSynchronize(stream_);
        cudaError_t err_sample = cudaGetLastError();
        if (err_sample != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err_sample) 
                    << " at " << __FILE__ << ":" << __LINE__ << std::endl;
            exit(EXIT_FAILURE);
        }

        std::cout << "Grid sampling completed for batch " << b << std::endl;

        // 存放sampled_descriptors的结果到cpu中来看下：
        float* sampled_descriptors_cpu = nullptr;
        sampled_descriptors_cpu = (float*)malloc(required_mem);
        cudaMemcpyAsync(sampled_descriptors_cpu, sampled_descriptors, required_mem, cudaMemcpyDeviceToHost, stream_);

        // 保存结果到txt中 ,经过验证是相同的
    /*     std::ofstream file("images/predictions/descriptors_after_grid_sample_cpp.txt");
        for (int i = 0; i < required_mem / sizeof(float); i++) {
            file <<std::setprecision(6) <<sampled_descriptors_cpu[i] << " ";
            file << std::endl;
        }
        file.close(); */

  
        // 4. 归一化描述符



        float * out_normalized_d = nullptr;
        size_t out_normalized_d_bytes = num_points * desc_channels * sizeof(float);
        cudaMalloc((void**)&out_normalized_d, out_normalized_d_bytes);
         
       cuda_normalize<float>(sampled_descriptors, 
        out_normalized_d, 
        num_points,
        desc_channels,
        desc_channels,
        1e-12f,
        stream_);
           
        cudaStreamSynchronize(stream_);

        // 5. 构建数据字典
        SlotData data_dict;
        data_dict.descriptors.resize(num_points * desc_channels);
        data_dict.points.resize(2*num_points);   // 成对出现的。所以乘以2

        

        cudaMemcpy(data_dict.descriptors.data(), out_normalized_d, 
                  required_mem, cudaMemcpyDeviceToHost);

      /*   std::ofstream file("images/predictions/descriptors_after_normalize_cpp.txt");          
        for (int i =0; i < data_dict.descriptors.size(); i++) {
            file <<std::setprecision(6) <<data_dict.descriptors[i] << " ";
            file << std::endl;
        }
        file.close(); */

        data_dict.points = actual_points;

        std::cout << "Descriptors copied back to host for batch " << b << std::endl;



            // ===== 新增：GNN模型推理 =====
        // 1. 准备GNN输入数据 (descriptors)
        //const int num_points = actual_points.size();
          
        if (!actual_points.empty()) {
            for (int i = 0; i < num_points; ++i) {
            gnn_points_h_[i * 2] = actual_points[i].x;     // x坐标
            gnn_points_h_[i * 2 + 1] = actual_points[i].y; // y坐标
            if (i >3) // 与python的保持一致
            {
                gnn_points_h_[i * 2 + 1] = 0;
                gnn_points_h_[i * 2] = 0;
            }
            
            // 打印points数据
            //printf("points: %f, %f\n", gnn_points_h_[i * 2], gnn_points_h_[i * 2 + 1]);
        } 
        
        }
        else
        {
            printf("actual_points is empty, and continue...\n");
            continue;
        }
         // points输入: [1, num_points, 2]
        
        // 2. 传输数据到设备
        size_t desc_bytes = 1 * 128 * num_points * sizeof(float);
        cudaMemcpyAsync(gnn_descriptors_d_, data_dict.descriptors.data(),
                       desc_bytes, cudaMemcpyHostToDevice, stream_);
        
        size_t points_bytes = num_points * 2 * sizeof(float);
        cudaMemcpyAsync(gnn_points_d_, gnn_points_h_.data(),
                       points_bytes, cudaMemcpyHostToDevice, stream_);


        
        // 3. 设置动态维度 (实际点数)
        nvinfer1::Dims desc_dims;
        desc_dims.nbDims = 3;
        desc_dims.d[0] = 1;    // batch
        desc_dims.d[1] = 128; // 通道数
        desc_dims.d[2] = num_points; // 动态节点数

        // 2. 定义points输入维度 [1, num_points, 2]
        nvinfer1::Dims points_dims;
        points_dims.nbDims = 3;
        points_dims.d[0] = 1;    // batch
        points_dims.d[1] = num_points; // 节点数
        points_dims.d[2] = 2;    // 坐标维度(x,y)

        // 3. 绑定维度（增强错误检查）
        if (!gnn_context_->setBindingDimensions(0, desc_dims)) {
            auto engine_dims = gnn_engine_->getBindingDimensions(0);
            std::cerr << "Descriptors维度绑定失败! 引擎期望: ";
            for (int i=0; i<engine_dims.nbDims; ++i) 
                std::cerr << (engine_dims.d[i]==-1 ? "?" : std::to_string(engine_dims.d[i])) << " ";
            std::cerr << "\n实际设置: ";
            for (int i=0; i<desc_dims.nbDims; ++i) 
                std::cerr << desc_dims.d[i] << " ";
            continue; // 跳过当前batch
        }

        if (!gnn_context_->setBindingDimensions(1, points_dims)) {
            auto engine_dims = gnn_engine_->getBindingDimensions(1);
            std::cerr << "Points维度绑定失败! 引擎期望: ";
            for (int i=0; i<engine_dims.nbDims; ++i) 
                std::cerr << (engine_dims.d[i]==-1 ? "?" : std::to_string(engine_dims.d[i])) << " ";
            std::cerr << "\n实际设置: ";
            for (int i=0; i<points_dims.nbDims; ++i) 
                std::cerr << points_dims.d[i] << " ";
            continue;
        }
        
        // 4. 执行推理
        void* gnn_bindings[] = {
            gnn_descriptors_d_,   // binding 0: descriptors
            gnn_points_d_,        // binding 1: points
            gnn_graph_output_d_,  // binding 2: graph_output
            gnn_edge_pred_d_      // binding 3: edge_pred
        };
        
        if (!gnn_context_->enqueueV2(gnn_bindings, stream_, nullptr)) {
            std::cerr << "GNN inference failed for batch " << b << std::endl;
            continue;
        }
                
        // 5. 取回结果
        size_t edge_pred_bytes = 100*1 * sizeof(float);
        size_t graph_output_bytes = num_points * 64 * sizeof(float);
        
        cudaMemcpyAsync(gnn_edge_pred_h_.data(), gnn_edge_pred_d_,
                      edge_pred_bytes, cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(gnn_graph_output_h_.data(), gnn_graph_output_d_,
                      graph_output_bytes, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // 保存结果到txt
        
       /*  std::ofstream file_edge("images/predictions/edge_pred_cpp.txt");
        for(int i =0; i < 100; i++){
            file_edge << std::fixed << std::setprecision(6) << gnn_edge_pred_h_[i] << " ";
            file_edge << std::endl;
        }
        file_edge.close();

        std::ofstream file_graph("images/predictions/graph_output_cpp.txt");
        for (int i =0 ; i < 640; i++){
            file_graph << std::fixed << std::setprecision(6) << gnn_graph_output_h_[i] << " ";
            file_graph << std::endl;
        }
        file_graph.close(); */

       

        
        // 6. 使用edge_pred结果连接关键点， 4 和 10 是与python对齐的
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (i == j) continue;
                
                float edge_prob = gnn_edge_pred_h_[i * 10 + j];
                if (edge_prob > slot_thresh_) {
                    const auto& p1 = actual_points[i];
                    const auto& p2 = actual_points[j];
                    
                    ParkingSlot slot;
                    slot.confidence = edge_prob;
                    slot.coords = {p1.x, p1.y, p2.x, p2.y};
                    batch_slots.push_back(slot);
                }
            }
        }
        
       
        
        std::cout << "Slots processed for batch " << b << ": " << batch_slots.size() << " slots found" << std::endl;

        // 7. 资源清理
        cudaError_t free_err = cudaFree(sampled_descriptors);
        if (free_err != cudaSuccess) {
            std::cerr << "cudaFree error: " << cudaGetErrorString(free_err)
                     << " | Batch: " << b << std::endl;
        }

        cudaError_t free_err2 = cudaFree(d_grid_points);
        if (free_err2 != cudaSuccess) {
            std::cerr << "cudaFree error: " << cudaGetErrorString(free_err2)
                     << " | Batch: " << b << std::endl;
        }

        cudaError_t free_err3 = cudaFree(descriptor_map_gpu);
        if (free_err3 != cudaSuccess) {
            std::cerr << "cudaFree error: " << cudaGetErrorString(free_err3)
                     << " | Batch: " << b << std::endl;
        }
        descriptor_map_gpu = nullptr;
        
        output_slots[b] = std::move(batch_slots);  


        // 打印输出槽位信息
        for (auto res: output_slots[b]) {
            
            std::cout << res.coords[0] << " " << res.coords[1] << " " << res.coords[2] << " " << res.coords[3] << std::endl;
            std::cout << res.confidence << std::endl;

        }
        // 将结果写到txt中 
        /* std::ofstream file_slots("images/predictions/slots_pred_cpp.txt");
        for (auto res: output_slots[b]) {
            file_slots <<"confidence"<<"\t"<<"coords0"<<"\t"<<"coords1"<<"\t"<<"coords2"<<"\t"<<"coords3"<<"\n";
            file_slots << res.confidence << "\t" << res.coords[0] << "\t" << res.coords[1] << "\t" << res.coords[2] << "\t" << res.coords[3] << "\n";
            
        }
        file_slots.close(); */


    }
}



// 修改后的后处理函数
void PsDet::postprocess(std::vector<std::vector<KeyPoint>>& output_points,
                        std::vector<std::vector<ParkingSlot>>& output_slots) {
    // 获取实际批量大小（根据输入维度）
    Dims input_dims = context_->getBindingDimensions(0);
    int actual_batch_size = input_dims.d[0];
    
    // 处理点预测
    process_points(output_points_h_.data(), output_points, actual_batch_size);

    // write output_points to txt file 
    /* std::ofstream file("images/predictions/output_points_cpp.txt");
    
    if (!output_points.empty()) {
       for (const auto& points : output_points) {
           for (const auto& point : points) {
               file << std::fixed << std::setprecision(6) << point.conf << " " << point.x << " " << point.y << std::endl;
           }
       }
    }
 
    file.close(); */
    
    // 处理槽位预测（假设描述符图在output_slots_h_中）
    process_slots(output_points, output_slots_h_.data(), 
                 output_slots, actual_batch_size);  

                 
} 

// 改进的NMS实现（与Python版本一致）
std::vector<KeyPoint> PsDet::applyNMS(
    const std::vector<KeyPoint>& points, 
    float dist_thresh) {
    
    const size_t num_points = points.size();
    std::vector<bool> suppressed(num_points, false);
    std::vector<KeyPoint> result;
    
    // 遍历所有点对
    for (size_t i = 0; i < num_points; ++i) {
        for (size_t j = i + 1; j < num_points; ++j) {
            float dx = std::abs(points[i].x - points[j].x);
            float dy = std::abs(points[i].y - points[j].y);
            
            if (dx < dist_thresh && dy < dist_thresh) {
                if (points[i].conf < points[j].conf) {
                    suppressed[i] = true;
                } else {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    // 收集未被抑制的点
    for (size_t i = 0; i < num_points; ++i) {
        if (!suppressed[i]) {
            result.push_back(points[i]);
        }
    }
    
    return result;
}


// 执行推理
bool PsDet::infer(const cv::Mat& image, 
                  std::vector<std::vector<KeyPoint>>& output_points, 
                  std::vector<std::vector<ParkingSlot>>& output_slots) {
    if (!engine_ || !context_) return false;
    
    // 预处理
    preprocess(image, input_h_.data());
    
    // 数据传输
    cudaMemcpyAsync(input_d_, input_h_.data(), 
                   input_h_.size() * sizeof(float), 
                   cudaMemcpyHostToDevice, stream_);
    
    // 设置绑定
    void* bindings[] = {input_d_, output_points_d_, output_slots_d_};
    
    // 使用动态
    if (engine_->hasImplicitBatchDimension() == false) {
        std::cout << "引擎支持显式批处理" << std::endl;
        Dims4 input_dims{1, input_channels_, input_height_, input_width_};
        context_->setBindingDimensions(0, input_dims);
    }

    // 执行推理
    if (!context_->enqueueV2(bindings, stream_, nullptr)) {
        return false;
    }
    
    // 获取结果
    cudaMemcpyAsync(output_points_h_.data(), output_points_d_, 
                   output_points_h_.size() * sizeof(float), 
                   cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(output_slots_h_.data(), output_slots_d_, 
                   output_slots_h_.size() * sizeof(float), 
                   cudaMemcpyDeviceToHost, stream_);
    
    // 等待完成
    cudaStreamSynchronize(stream_);

    // 在执行推理后添加 for debug comparison with pth demo.py
    const Dims opt_dims = context_->getBindingDimensions(0);
    std::cout << "real inference Dims: ";
    for (int i=0; i<opt_dims.nbDims; ++i) 
    std::cout << opt_dims.d[i] << " ";
    Dims points_dims = context_->getBindingDimensions(1); // points输出索引
    Dims slots_dims = context_->getBindingDimensions(2);  // slots输出索引
    std::cout << ">Points output dims: ";
    for (int i = 0; i < points_dims.nbDims; ++i) 
        std::cout << points_dims.d[i] << " ";
    std::cout << "\nSlots output dims: ";
    for (int i = 0; i < slots_dims.nbDims; ++i)
        std::cout << slots_dims.d[i] << " ";

    // 在保存输出文件前添加
    float sum_points = std::accumulate(output_points_h_.begin(), output_points_h_.end(), 0.0f);
    float sum_slots = std::accumulate(output_slots_h_.begin(), output_slots_h_.end(), 0.0f);
    std::cout << "Points输出总和: " << sum_points << " | Slots输出总和: " << sum_slots;
    
   
    /* std::ofstream output_points_file("images/predictions/points_pred_cpp.txt"), \
    output_slots_file("images/predictions/output_slots_orig.txt");
    for (const auto& point : output_points_h_) {
        // 每存储一个就空行
        output_points_file << std::endl;
        output_points_file << std::fixed << std::setprecision(9) << point << " ";
    }

    output_points_file << std::endl;
    for (const auto& slot : output_slots_h_) {
        // 每存储一个就空行
        output_slots_file << std::endl;
        output_slots_file << std::fixed << std::setprecision(9) << slot << " ";
    }
    output_slots_file << std::endl;
    output_points_file.close();
    output_slots_file.close();  */
    
    // 检查CUDA错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream_);


    // 应该将output_points_h_，output_slots_h_分别转换成（3,16,16) 和 （128，16，16）的矩阵
    
    postprocess(output_points, output_slots);
    
    return true;
}

void PsDet::visualizeResults(cv::Mat & car,cv::Mat& image,
                            const std::vector<std::vector<KeyPoint>>& points,
                            const std::vector<std::vector<ParkingSlot>>& slots) 
{
    const int width = image.cols;
    const int height = image.rows;

    // ... [保持你原有的图像处理代码不变] ...

    // 距离阈值常量 (保持相同值)
    constexpr float VSLOT_MIN_DIST = 0.044771278151623496f;
    constexpr float VSLOT_MAX_DIST = 0.1099427457599304f;
    constexpr float HSLOT_MIN_DIST = 0.15057789144568634f;  // 新增水平车位阈值
    constexpr float HSLOT_MAX_DIST = 0.44449496544202816f;   // 新增水平车位阈值
    
    constexpr float SHORT_SEP_LEN = 0.199519231f;
    constexpr float LONG_SEP_LEN = 0.46875f;

    // 用于收集所有端点
    std::vector<cv::Point> junctions;
    
    // 1. 绘制停车位 (修正后的实现)
    for (const auto& batch_slots : slots) {
        for (const auto& slot : batch_slots) {
            // 坐标转换：归一化→像素
            cv::Point2f p0(
                width * slot.coords[0] - 0.5f,
                height * slot.coords[1] - 0.5f
            );
            cv::Point2f p1(
                width * slot.coords[2] - 0.5f,
                height * slot.coords[3] - 0.5f
            );
            
            // 计算方向向量
            cv::Point2f vec = p1 - p0;
            float length = std::sqrt(vec.x*vec.x + vec.y*vec.y);
            cv::Point2f unit_vec = (length > 0) ? vec / length : cv::Point2f(1, 0);
            
            // 计算平方距离 (与Python一致)
            float dist_sq = (slot.coords[0] - slot.coords[2]) * (slot.coords[0] - slot.coords[2]) + 
                           (slot.coords[1] - slot.coords[3]) * (slot.coords[1] - slot.coords[3]);
            
            // 确定分隔线长度 (添加水平车位判断)
            float sep_length;
            if (VSLOT_MIN_DIST <= dist_sq && dist_sq <= VSLOT_MAX_DIST) {
                sep_length = LONG_SEP_LEN;  // 垂直车位
            } else if (HSLOT_MIN_DIST <= dist_sq && dist_sq <= HSLOT_MAX_DIST) {
                sep_length = SHORT_SEP_LEN;  // 水平车位
            } else {
                sep_length = SHORT_SEP_LEN;  // 默认值
            }
            
            // 修正垂直延伸点的计算 (与Python完全一致)
            cv::Point2f p2(
                p0.x + height * sep_length * unit_vec.y,
                p0.y - width * sep_length * unit_vec.x
            );
            cv::Point2f p3(
                p1.x + height * sep_length * unit_vec.y,
                p1.y - width * sep_length * unit_vec.x
            );
            
            // 坐标取整
            cv::Point ip0(static_cast<int>(std::round(p0.x)), 
                         static_cast<int>(std::round(p0.y)));
            cv::Point ip1(static_cast<int>(std::round(p1.x)), 
                         static_cast<int>(std::round(p1.y)));
            cv::Point ip2(static_cast<int>(std::round(p2.x)), 
                         static_cast<int>(std::round(p2.y)));
            cv::Point ip3(static_cast<int>(std::round(p3.x)), 
                         static_cast<int>(std::round(p3.y)));
            
            // 绘制车位线 (改为统一蓝色)
            cv::line(image, ip0, ip1, cv::Scalar(255, 0, 0), 2);  // 主方向线
            cv::line(image, ip0, ip2, cv::Scalar(255, 0, 0), 2);  // 左侧分隔线
            cv::line(image, ip1, ip3, cv::Scalar(255, 0, 0), 2);  // 右侧分隔线
            
            // 收集端点 (与Python一致)
            junctions.push_back(ip0);
            junctions.push_back(ip1);
        }
    }
    
    // 绘制所有端点 (统一红色圆点)
    for (const auto& pt : junctions) {
        cv::circle(image, pt, 3, cv::Scalar(0, 0, 255), 4);
    }
    
    // 2. 绘制关键点（按批次不同颜色）
    const cv::Scalar colors[] = {
        cv::Scalar(0, 255, 255), // 黄色
        cv::Scalar(0, 255, 0),   // 绿色
        cv::Scalar(255, 0, 255), // 紫色
        cv::Scalar(0, 165, 255)  // 橙色
    };
    
    for (size_t batch = 0; batch < points.size(); ++batch) {
        const auto& batch_points = points[batch];
        cv::Scalar color = colors[batch % 4];
        
        for (size_t i = 0; i < batch_points.size(); i += 3) {
            const auto& kp = batch_points[i]; // 获取KeyPoint对象
            float conf = kp.conf;             // 直接访问成员变量
            float x = kp.x * width;           // 使用归一化坐标转换
            float y = kp.y * height;
            
            cv::Point pt(static_cast<int>(x), static_cast<int>(y));
            
            // 绘制实心点+黑边
            cv::circle(image, pt, 6, cv::Scalar(0, 0, 0), 2); // 黑边
            cv::circle(image, pt, 4, color, -1); // 彩色填充
            
            // 高置信度点添加文本
            if (conf > 0.3) {
                std::string label = cv::format("%.1f", conf);
                cv::putText(image, label, pt + cv::Point(-15, -15), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                           cv::Scalar(255, 255, 255), 2);
            }
        }
    }
    
    // 3. 添加统计信息
    int total_points = 0;
    for (auto& batch : points) total_points += batch.size() / 3;
    
    int total_slots = 0;
    for (auto& batch : slots) total_slots += batch.size();
    
    std::string stats = cv::format("Points: %d | Slots: %d", total_points, total_slots);
    cv::putText(image, stats, cv::Point(10, 30), 
               cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    // show image
    //cv::imshow("Parking Slot Detection", image);
    //cv::waitKey(50);
}

cv::Size PsDet::getInputSize() const {
    return cv::Size(input_width_, input_height_);
}

// 计算张量维度总元素数量
// 安全计算张量体积
int64_t PsDet::safe_volume(const nvinfer1::Dims& dims) {
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) { // 动态维度返回 -1
            return -1; // 标记为需运行时设置
        }
        v *= dims.d[i];
        if (v <= 0) { // 检查溢出
            throw std::runtime_error("Dimension overflow");
        }
    }
    return v;
}

bool PsDet::loadGNNModel(const std::string& gnn_engine_path) {
    std::ifstream engine_file(gnn_engine_path, std::ios::binary);
    if (!engine_file) {
        std::cerr << "Failed to open GNN engine file: " << gnn_engine_path << std::endl;
        return false;
    }

    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    engine_file.read(buffer.data(), size);

    // 使用相同的runtime
    gnn_engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
    if (!gnn_engine_) {
        std::cerr << "Failed to deserialize GNN engine" << std::endl;
        return false;
    }

    gnn_context_ = gnn_engine_->createExecutionContext();
    if (!gnn_context_) {
        std::cerr << "Failed to create GNN execution context" << std::endl;
        gnn_engine_->destroy();
        return false;
    }

    // === 核心修改1：删除OptimizationProfile配置（已存在于构建阶段）===
    Dims input_dims = gnn_engine_->getBindingDimensions(0);
    Dims output_dims = gnn_engine_->getBindingDimensions(1);
    
    // 打印实际维度
    std::cout << "GNN input dims: ";
    for (int i = 0; i < input_dims.nbDims; ++i) 
        std::cout << input_dims.d[i] << " ";
    std::cout << "\nGNN output dims: ";
    for (int i = 0; i < output_dims.nbDims; ++i) 
        std::cout << output_dims.d[i] << " ";
    std::cout << std::endl;

    // === 核心修改2：动态计算缓冲区大小 ===
   
    // 打印绑定信息， 包括模型的输入，和输出，以及dim的维度大小
    int nbBindings = gnn_engine_->getNbBindings();
    for (int i = 0; i < nbBindings; ++i) {
        Dims dims = gnn_engine_->getBindingDimensions(i);
        std::string name = gnn_engine_->getBindingName(i);
        bool isInput = gnn_engine_->bindingIsInput(i);
        
        std::cout << (isInput ? "Input" : "Output") 
                  << " " << i << ": " << name << " [";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << (dims.d[j] == -1 ? "?" : std::to_string(dims.d[j])) << " ";
        }
        std::cout << "]" << std::endl;
    }

     // 预分配设备内存
    cudaMalloc(&gnn_descriptors_d_, max_points_ * 128 * sizeof(float));
    cudaMalloc(&gnn_points_d_, max_points_ * 2 * sizeof(float));
    cudaMalloc(&gnn_edge_pred_d_, max_points_ * max_points_ * sizeof(float));
    cudaMalloc(&gnn_graph_output_d_, max_points_ * 64 * sizeof(float));
    
    
    return true;
}

