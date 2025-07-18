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
#include <array>

namespace psdet {

using namespace nvinfer1;

class Logger : public ILogger 
{
public:
    explicit Logger(Severity severity = Severity::kINFO) : severity(severity) {}
    
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= this->severity) {
            std::cout << msg << std::endl;
        }
    }
    
private:
    Severity severity;
};

struct KeyPoint {
    float conf; // 置信度
    float x, y; // 归一化坐标[0,1]
};

struct ParkingSlot {
    float confidence;
      std::array<float, 4> coords; 
};

struct Config {
    bool use_slant_predictor = false;
    bool use_vacant_predictor = false;
    bool use_gnn = true;
    
};

class PsDet {
public:
    PsDet(const std::string& onnx_path, 
          const std::string& engine_path,
          int max_batch_size = 1,
          int max_points = 50000,
          int max_slots = 50000,
          int input_width = 512,
          int input_height = 512);

    ~PsDet();
    
    bool build(bool fp16 = false);
    bool load();

    bool loadGNNModel(const std::string& gnn_engine_path);
    int64_t safe_volume(const nvinfer1::Dims& dims);
    void save();
    
    bool infer(const cv::Mat& image, 
               std::vector<std::vector<KeyPoint>>& output_points, 
               std::vector<std::vector<ParkingSlot>>& output_slots);
    
    cv::Size getInputSize() const;
    
     void visualizeResults(cv::Mat & car,cv::Mat& image,
                         const std::vector<std::vector<KeyPoint>>& points,
                         const std::vector<std::vector<ParkingSlot>>& slots);
    

       struct SlotData {
            std::vector<float> descriptors;
            std::vector<KeyPoint> points;
            std::vector<float> slant_pred;
            std::vector<float> vacant_pred;
            std::vector<float> graph_output;
            std::vector<float> edge_pred;
        };


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
    int points_3d_dim_ = 16;
    int points_4d_dim_ = 16;
    int input_width_;
    int input_height_;
    int input_channels_ = 3;
    
    void* input_d_ = nullptr;
    void* output_points_d_ = nullptr;
    void* output_slots_d_ = nullptr;
    std::vector<float> input_h_;
    std::vector<float> output_points_h_;
    std::vector<float> output_slots_h_;


    // gnn relative parameters
    nvinfer1::ICudaEngine* gnn_engine_ = nullptr;
    nvinfer1::IExecutionContext* gnn_context_ = nullptr;

    // 输入缓冲区
    float* gnn_descriptors_d_ = nullptr;   // descriptors输入(device)
    float* gnn_points_d_ = nullptr;        // points输入(device)
    std::vector<float> gnn_descriptors_h_; // descriptors输入(host)
    std::vector<float> gnn_points_h_;      // points输入(host)

    // 输出缓冲区
    float* gnn_edge_pred_d_ = nullptr;     // edge_pred输出(device)
    float* gnn_graph_output_d_ = nullptr;   // graph_output输出(device)
    std::vector<float> gnn_edge_pred_h_;   // edge_pred输出(host)
    std::vector<float> gnn_graph_output_h_; // graph_output输出(host)

    
    
    float point_thresh_ = 0.008f;
    float slot_thresh_ = 0.05f;
    float nms_thresh_ = 0.0625f;    
    int max_points_cfg = 10;
    
    Config cfg_;
  

    void preprocess(const cv::Mat& image, float* input);
    void postprocess(std::vector<std::vector<KeyPoint>>& output_points,
                     std::vector<std::vector<ParkingSlot>>& output_slots);

      void process_points(
        const float* points_data,
        std::vector<std::vector<KeyPoint>>& output_points,
        int batch_size
    );

      void process_slots(
         const std::vector<std::vector<KeyPoint>>& points_list, 
        const float* descriptor_map,
        std::vector<std::vector<ParkingSlot>>& output_slots,
        int batch_size
    );
    
    // NMS函数
    std::vector<KeyPoint> applyNMS(
    const std::vector<KeyPoint>& points, 
    float dist_thresh= 0.0625f
    );
    bool initBuffers();
    void destroyBuffers();
    
    size_t getSizeByDim(const Dims& dims);
};

} // namespace psdet

#endif // PSDET_HPP