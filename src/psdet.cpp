#include "psdet.hpp"
#include <algorithm>
#include <cassert>

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
    output_points_h_.resize(max_batch_size_ * max_points_ * 3);
    output_slots_h_.resize(max_batch_size_ * max_slots_ * 5);
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

    // 确保通道维度有效
    const int channels = input_dims.d[1] > 0 ? input_dims.d[1] : 3;
    if (channels != 1 && channels != 3) {
        logger_.log(ILogger::Severity::kWARNING, 
                   ("Unexpected channel dimension: " + std::to_string(channels) + 
                    ". Using default value 3").c_str());
    }

    // 设置优化范围（使用用户配置的尺寸）
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

// 后处理 - 修复索引错误[6](@ref)
void PsDet::postprocess(std::vector<std::vector<float>>& output_points, 
                        std::vector<std::vector<ParkingSlot>>& output_slots) {
    output_points.clear();
    output_slots.clear();
    
    // 处理点预测
    for (int b = 0; b < max_batch_size_; ++b) {
        std::vector<float> batch_points;
        for (int i = 0; i < max_points_; ++i) {
            size_t offset = b * max_points_ * 3 + i * 3;
            float conf = output_points_h_[offset];
            if (conf > 0.1f) {
                batch_points.push_back(conf);
                batch_points.push_back(output_points_h_[offset + 1]); // x
                batch_points.push_back(output_points_h_[offset + 2]); // y
            }
        }
        output_points.push_back(batch_points);
    }
    
    // 处理槽位预测 - 修复索引
    for (int b = 0; b < max_batch_size_; ++b) {
        std::vector<ParkingSlot> batch_slots;
        for (int i = 0; i < max_slots_; ++i) {
            size_t offset = b * max_slots_ * 5 + i * 5;
            float conf = output_slots_h_[offset];
            if (conf > 0.1f) {
                ParkingSlot slot;
                slot.confidence = conf;
                slot.coords[0] = output_slots_h_[offset + 1]; // x1
                slot.coords[1] = output_slots_h_[offset + 2]; // y1
                slot.coords[2] = output_slots_h_[offset + 3]; // x2
                slot.coords[3] = output_slots_h_[offset + 4]; // y2 - 修复索引
                batch_slots.push_back(slot);
            }
        }
        output_slots.push_back(batch_slots);
    }
}

// 执行推理
bool PsDet::infer(const cv::Mat& image, 
                  std::vector<std::vector<float>>& output_points, 
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
    
    // 动态维度设置[6](@ref)
    if (engine_->hasImplicitBatchDimension() == false) {
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
    
    cudaStreamSynchronize(stream_);
    postprocess(output_points, output_slots);
    
    return true;
}

cv::Size PsDet::getInputSize() const {
    return cv::Size(input_width_, input_height_);
}