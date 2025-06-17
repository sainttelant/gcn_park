#ifndef PSDET_HPP
#define PSDET_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>

namespace psdet {

using namespace nvinfer1;

class Logger : public ILogger {
public:
    explicit Logger(Severity severity = Severity::kWARNING) : severity(severity) {}
    
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= this->severity) {
            std::cout << msg << std::endl;
        }
    }
    
private:
    Severity severity;
};

struct ParkingSlot {
    float confidence;
    float coords[4]; // x1, y1, x2, y2
};

class PsDet {
public:
    PsDet(const std::string& onnx_path, 
          const std::string& engine_path,
          int max_batch_size = 1,
          int max_points = 100,
          int max_slots = 50,
          int input_width = 512,
          int input_height = 512);

    ~PsDet();
    
    bool build(bool fp16 = true);
    bool load();
    void save();
    
    bool infer(const cv::Mat& image, 
               std::vector<std::vector<float>>& output_points, 
               std::vector<std::vector<ParkingSlot>>& output_slots);
    
    cv::Size getInputSize() const;

private:
    Logger logger_;
    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;

    std::string onnx_path_;
    std::string engine_path_;
    int max_batch_size_;
    int max_points_;
    int max_slots_;
    int input_width_;
    int input_height_;
    int input_channels_ = 3;
    
    void* input_d_ = nullptr;
    void* output_points_d_ = nullptr;
    void* output_slots_d_ = nullptr;
    
    std::vector<float> input_h_;
    std::vector<float> output_points_h_;
    std::vector<float> output_slots_h_;

    void preprocess(const cv::Mat& image, float* input);
    void postprocess(std::vector<std::vector<float>>& output_points, 
                     std::vector<std::vector<ParkingSlot>>& output_slots);
    
    bool initBuffers();
    void destroyBuffers();
    
    size_t getSizeByDim(const Dims& dims);
};

} // namespace psdet

#endif // PSDET_HPP