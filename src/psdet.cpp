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
}

// 析构函数 - 修复销毁顺序[8](@ref)
PsDet::~PsDet() {
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



/* void PsDet::process_slots(
     const std::vector<std::vector<KeyPoint>>& points_list, 
    const float* descriptor_map,
    std::vector<std::vector<ParkingSlot>>& output_slots,
    int batch_size)
{
    output_slots.clear();
    //output_slots.resize(batch_size);
    
    // 描述符图的尺寸信息
    const int desc_channels = 128;
    const int desc_height = input_height_ / 32;
    const int desc_width = input_width_ / 32;
    

    std::vector<KeyPoint> points_for_reserve;
    points_for_reserve.reserve(max_points_cfg);
    points_for_reserve.resize(max_points_cfg); // 使用 resize 而不是 reserve

        // 将 points_for_reserve 初始化为 max_points_cfg 个空的 KeyPoint
        for (int i = 0; i < max_points_cfg; ++i) {
            points_for_reserve[i] = KeyPoint{0.0f, 0.0f, 0.0f};
        }

        // 将 points_list[0] 复制到 points_for_reserve
        int points_to_copy = std::min(static_cast<int>(points_list[0].size()), max_points_cfg);
        for (int i = 0; i < points_to_copy; ++i) {
            points_for_reserve[i].conf = points_list[0][i].conf;
            points_for_reserve[i].x = points_list[0][i].x;
            points_for_reserve[i].y = points_list[0][i].y;
        }
   
       
        std::vector<ParkingSlot> batch_slots;
        

        // 1. 采样关键点的描述符
        std::vector<float> descriptors;
        std::vector<float> grid_points;
        grid_points.reserve(max_points_cfg);
        
        // 准备采样网格
        for (const auto& kp : points_for_reserve) {
            // 归一化坐标转网格坐标
            float grid_x = kp.x * 2 - 1;  // [-1, 1] 范围
            float grid_y = kp.y * 2 - 1;
            grid_points.push_back(grid_x);
            grid_points.push_back(grid_y);
        }
       

        int b = 1;
        // 在GPU上采样描述符
        const int num_points = points_for_reserve.size();
        float* sampled_descriptors= nullptr;
        cudaError_t err =cudaMalloc((void**)&sampled_descriptors, 1*128*1*10*sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaMalloc: " << cudaGetErrorString(err) << std::endl;
        }
        

        int input_dims[4] = {1, desc_channels, desc_height, desc_width}; // NCHW
        int grid_dims[4] = {1, num_points, 1, 2}; // [batch, num_points, 1, 2]
        int output_dims[4] = {1, desc_channels, num_points, 1}; // 输出维度

        // 网格坐标范围检查
        for (auto& coord : grid_points) {
            if (coord < -1.0f || coord > 1.0f) {
                std::cerr << "Invalid grid coordinate: " << coord 
                        << " at index " << &coord - grid_points.data() << std::endl;
                coord = std::clamp(coord, -1.0f, 1.0f);
            }
        }

        // 验证输入描述符维度
        std::cout << "Descriptor map actual size: " 
                << desc_height << "x" << desc_width << std::endl;
        std::cout << "Grid points count: " << grid_points.size() << std::endl;
        std::cout << "Expected grid points: " << num_points * 2 << std::endl;

        // 验证输出空间
        size_t required_mem = 1 * 128 * num_points * 1 * sizeof(float);
        if (required_mem > 1 * 128 * 1 * 10*sizeof(float)) {
            std::cerr << "Insufficient memory allocated! Required: " 
                    << required_mem << ", Allocated: "
                    << 1 * 128 * 1 * 10*sizeof(float) << std::endl;
        }

        // 调用grid_sample
        grid_sample<float>(
            sampled_descriptors, 
            &descriptor_map[b * desc_channels * desc_height * desc_width],
            grid_points.data(),
            output_dims,
            input_dims,
            grid_dims,
            4, // 4维数据（2D采样）
            GridSamplerInterpolation::Bilinear,
            GridSamplerPadding::Zeros,
            true, // align_corners
            stream_ // 使用类的CUDA流
        );

        cudaStreamSynchronize(stream_); // 等待内核完成
        cudaError_t err_grid = cudaGetLastError(); // 检查内核错误

        // 增强错误检查
        if (err_grid != cudaSuccess) {
            std::cerr << "CUDA error in grid_sample: " 
                    << cudaGetErrorString(err_grid) 
                    << " (Code: " << err_grid << ")" << std::endl;
            
            // 特定错误处理
            if (err_grid == cudaErrorInvalidValue) {
                std::cerr << "Likely cause: Invalid grid coordinates" << std::endl;
            } else if (err_grid == cudaErrorIllegalAddress) {
                std::cerr << "Likely cause: Memory access violation" << std::endl;
            }
        }
                
        // 归一化描述符
        normalize_cuda(sampled_descriptors, num_points, desc_channels, 2.0f, 1e-5f);
        
        // 2. 构建数据字典（模拟Python的data_dict）
      
        
        SlotData data_dict;
        
        // 将GPU数据拷贝回CPU
        data_dict.descriptors.resize(num_points * desc_channels);
        cudaMemcpy(data_dict.descriptors.data(), sampled_descriptors, 
                  num_points * desc_channels * sizeof(float), cudaMemcpyDeviceToHost);
        
        data_dict.points = points_for_reserve;
        
        // 3. 倾斜预测器（如果配置了）
        if (cfg_.use_slant_predictor) 
        {
            
            data_dict.slant_pred.resize(num_points * 2);  // 每个点有起点和终点倾斜
            for (int i = 0; i < num_points * 2; ++i) {
                data_dict.slant_pred[i] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
        
        // 4. 空位预测器（如果配置了）
        if (cfg_.use_vacant_predictor) {
            // 实际应用中应调用空位预测器模型
            data_dict.vacant_pred.resize(num_points);
            for (int i = 0; i < num_points; ++i) {
                data_dict.vacant_pred[i] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
        
        // 5. 图神经网络处理（如果配置了）
        if (cfg_.use_gnn) {
            // 实际应用中应调用GNN模型
            // 这里简化处理：添加随机噪声
            printf("data_dict.descriptors size is: %d\n", data_dict.descriptors.size());
            // 将data_dict.descriptors 的结果写入txt文件,小数点6位精度
            std::ofstream f("images/predictions/descriptors_after_grid_sample_cpp.txt");
            for (float val : data_dict.descriptors) {
                f << std::endl;
                f << std::fixed << std::setprecision(6) << val << " ";
            }
            for (float& val : data_dict.descriptors) {

                val += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
            }
        }
        
        // 6. 边缘预测器
        // 实际应用中应调用边缘预测器模型
        // 这里简化实现：连接空间上接近的点
        const float connection_threshold = 0.05f;  // 归一化距离阈值
        
        for (size_t i = 0; i < points_for_reserve.size(); ++i) {
            for (size_t j = i + 1; j < points_for_reserve.size(); ++j) {
                const auto& p1 = points_for_reserve[i];
                const auto& p2 = points_for_reserve[j];
                
                // 计算归一化距离
                float dx = p1.x - p2.x;
                float dy = p1.y - p2.y;
                float dist_sq = dx * dx + dy * dy;
                
                // 如果距离小于阈值且方向大致相同，则连接
                if (dist_sq < connection_threshold * connection_threshold) {
                    // 置信度基于距离和点的置信度
                    float conf = (p1.conf + p2.conf) * 0.5f * 
                                (1.0f - sqrt(dist_sq) / connection_threshold);
                    
                   if (conf > slot_thresh_) 
                        {
                            ParkingSlot slot;
                            slot.confidence = conf;
                            slot.coords = {p1.x, p1.y, p2.x, p2.y}; 
                            batch_slots.push_back(slot);
                        };
                }
            }
        }
        
        // 7. 清理资源
        if (sampled_descriptors == nullptr) {
            std::cerr << "Pointer is null!" << std::endl;
        }
        cudaError_t err_free =  cudaFree(sampled_descriptors);
       if (err_free != cudaSuccess) {
            std::cerr << "Error freeing sampled_descriptors: " 
                    << cudaGetErrorString(err_free) << " (Code: " << err_free << ")" << std::endl;
        }
        sampled_descriptors = nullptr;
        output_slots[b] = batch_slots;
    
} */

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

        // 保存结果到txt中
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
        cudaMemcpy(data_dict.descriptors.data(), out_normalized_d, 
                  required_mem, cudaMemcpyDeviceToHost);

        std::ofstream file("images/predictions/descriptors_after_normalize_cpp.txt");          
        for (int i =0; i < data_dict.descriptors.size(); i++) {
            file <<std::setprecision(6) <<data_dict.descriptors[i] << " ";
            file << std::endl;
        }
        file.close();

        data_dict.points = actual_points;

        std::cout << "Descriptors copied back to host for batch " << b << std::endl;

        // 6. 边缘预测 - 仅连接实际点
        const float connection_threshold = 0.05f;
        for (size_t i = 0; i < num_points; ++i) {
            for (size_t j = i + 1; j < num_points; ++j) {
                const auto& p1 = actual_points[i];
                const auto& p2 = actual_points[j];
                
                float dx = p1.x - p2.x;
                float dy = p1.y - p2.y;
                float dist_sq = dx * dx + dy * dy;
                
                if (dist_sq < connection_threshold * connection_threshold) {
                    float conf = (p1.conf + p2.conf) * 0.5f * 
                                (1.0f - sqrt(dist_sq) / connection_threshold);
                    
                    if (conf > slot_thresh_) {
                        ParkingSlot slot;
                        slot.confidence = conf;
                        slot.coords = {p1.x, p1.y, p2.x, p2.y}; 
                        batch_slots.push_back(slot);
                    }
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
    std::ofstream file("images/predictions/output_points_cpp.txt");
    
    if (!output_points.empty()) {
       for (const auto& points : output_points) {
           for (const auto& point : points) {
               file << std::fixed << std::setprecision(6) << point.conf << " " << point.x << " " << point.y << std::endl;
           }
       }
    }
 
    file.close();
    
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
    
   
   /*  std::ofstream output_points_file("images/predictions/output_points_orig.txt"), \
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
    output_slots_file.close(); */
    
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

    // resize the image to (input_width, input_height)
    cv::resize(image, image, cv::Size(input_width_, input_height_));

    cv::resize(car, car, cv::Size(input_width_, input_height_));
    // 设置指定区域为黑色
    cv::Rect roi(210, 145, 90, 220);

    roi.x = std::max(roi.x, 0);
    roi.y = std::max(roi.y, 0);
    roi.width = std::min(roi.width, image.cols - roi.x);
    roi.height = std::min(roi.height, image.rows - roi.y);

    // 将指定区域设置为0
    image(roi) = cv::Scalar(0, 0, 0);

    // 确保 roi 的大小与 car 图像的大小相同
    cv::resize(car, car, cv::Size(roi.width, roi.height));
    
   
    // 将 car 复制到 image 的指定区域
    cv::Mat image_roi = image(roi);
    if (car.channels() == 4 && image_roi.channels() == 3) {
        // 如果 car 是RGBA图像，将其转换为RGB图像
        cv::cvtColor(car, car, cv::COLOR_RGBA2RGB);
    }
    // 确保 car 和 image_roi 的大小相同
    if (car.size() == image_roi.size()) {
        cv::addWeighted(image_roi, 1.0, car, 1.0, 0.0, image_roi);
    } else {
        std::cerr << "Error: Sizes of car and image_roi do not match." << std::endl;
    }

 

    // 距离阈值常量
    constexpr float VSLOT_MIN_DIST = 0.04477f;
    constexpr float VSLOT_MAX_DIST = 0.10994f;
    constexpr float SHORT_SEP_LEN = 0.19952f;
    constexpr float LONG_SEP_LEN = 0.46875f;
    
    // 1. 绘制停车位
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
            
            // 确定分隔线长度
            float dist_sq = (slot.coords[0] - slot.coords[2])*(slot.coords[0] - slot.coords[2]) + 
                           (slot.coords[1] - slot.coords[3])*(slot.coords[1] - slot.coords[3]);
            
            float sep_length = (VSLOT_MIN_DIST <= dist_sq && dist_sq <= VSLOT_MAX_DIST)
                             ? LONG_SEP_LEN : SHORT_SEP_LEN;
            
            // 计算垂直方向延伸点
            cv::Point2f perpendicular(-unit_vec.y, unit_vec.x);
            cv::Point2f p2 = p0 + perpendicular * sep_length * height;
            cv::Point2f p3 = p1 + perpendicular * sep_length * height;
            
            // 坐标取整
            cv::Point ip0(static_cast<int>(std::round(p0.x)), 
                         static_cast<int>(std::round(p0.y)));
            cv::Point ip1(static_cast<int>(std::round(p1.x)), 
                         static_cast<int>(std::round(p1.y)));
            cv::Point ip2(static_cast<int>(std::round(p2.x)), 
                         static_cast<int>(std::round(p2.y)));
            cv::Point ip3(static_cast<int>(std::round(p3.x)), 
                         static_cast<int>(std::round(p3.y)));
            
            // 绘制车位线
            cv::line(image, ip0, ip1, cv::Scalar(0, 0, 255), 2);  // 主方向线（红色）
            cv::line(image, ip0, ip2, cv::Scalar(0, 255, 0), 2);  // 左侧分隔线（绿色）
            cv::line(image, ip1, ip3, cv::Scalar(0, 255, 0), 2);  // 右侧分隔线（绿色）
            
            // 绘制端点
            cv::circle(image, ip0, 5, cv::Scalar(255, 0, 0), -1); // 起点（蓝色）
            cv::circle(image, ip1, 5, cv::Scalar(0, 255, 255), -1); // 终点（黄色）
            
            // 显示置信度
            std::string conf_text = cv::format("%.2f", slot.confidence);
            cv::putText(image, conf_text, 
                       cv::Point((ip0.x + ip1.x)/2, (ip0.y + ip1.y)/2),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
        }
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
}

cv::Size PsDet::getInputSize() const {
    return cv::Size(input_width_, input_height_);
}