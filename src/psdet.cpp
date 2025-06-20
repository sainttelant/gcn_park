#include "psdet.hpp"
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

std::vector<KeyPoint> PsDet::applyNMS(
    const std::vector<KeyPoint>& points, 
    float dist_thresh
) {
    std::vector<bool> suppressed(points.size(), false);
    std::vector<KeyPoint> result;
    
    // 按置信度降序排序
    std::vector<size_t> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        return points[i].conf > points[j].conf;
    });
    
    // 遍历所有点
    for (size_t i = 0; i < indices.size(); ++i) {
        if (suppressed[indices[i]]) continue;
        result.push_back(points[indices[i]]);
        
        // 计算与后续点的距离
        for (size_t j = i + 1; j < indices.size(); ++j) {
            if (suppressed[indices[j]]) continue;
            
            float dx = points[indices[i]].x - points[indices[j]].x;
            float dy = points[indices[i]].y - points[indices[j]].y;
            float dist_sq = dx * dx + dy * dy;
            
            if (dist_sq < dist_thresh * dist_thresh) {
                suppressed[indices[j]] = true;
            }
        }
    }
    return result;
}

// 后处理 - 修复索引错误[6](@ref)
void PsDet::postprocess(std::vector<std::vector<KeyPoint>>& output_points,
                        std::vector<std::vector<ParkingSlot>>& output_slots) 
{
   output_points.clear();
    output_slots.clear();
    
    // 解析关键点输出 (shape: [batch, max_points, 3])
    for (int b = 0; b < max_batch_size_; ++b) {
        std::vector<KeyPoint> batch_points;
        for (int i = 0; i < max_points_; ++i) {
            const int base_idx = b * max_points_ * 3 + i * 3;
            const float conf = output_points_h_[base_idx];
            const float x = output_points_h_[base_idx + 1];
            const float y = output_points_h_[base_idx + 2];
            
            if (conf > point_thresh_) { // 置信度阈值过滤
                batch_points.push_back({conf, x, y});
            }
        }
        // 应用NMS（需实现）
        auto nms_points = applyNMS(batch_points, nms_thresh_);
        output_points.push_back(nms_points);
    }
    
    // 解析车位输出 (shape: [batch, max_slots, 5])
    for (int b = 0; b < max_batch_size_; ++b) {
        std::vector<ParkingSlot> batch_slots;
        for (int i = 0; i < max_slots_; ++i) {
            const int base_idx = b * max_slots_ * 5 + i * 5;
            const float conf = output_slots_h_[base_idx];
            const float x1 = output_slots_h_[base_idx + 1];
            const float y1 = output_slots_h_[base_idx + 2];
            const float x2 = output_slots_h_[base_idx + 3];
            const float y2 = output_slots_h_[base_idx + 4];
            
            if (conf > slot_thresh_) { // 车位置信度阈值
                batch_slots.push_back({conf, {x1, y1, x2, y2}});
            }
        }
        output_slots.push_back(batch_slots);
    }
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
    
    // write output_points_h_ and output_slots_h_ to txtfile 
    std::ofstream output_points_file("images/predictions/output_points_orig.txt"), \
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
    output_slots_file.close();
    
    // 检查CUDA错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return false;
    }




    cudaStreamSynchronize(stream_);
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