#include "psdet.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>  // 用于路径操作

void drawResults(cv::Mat& image, 
                 std::vector<std::vector<float>>& points,
                 std::vector<std::vector<psdet::ParkingSlot>>& slots) 
{
    // ==================== 1. 初始化颜色方案 ====================
    const cv::Scalar POINT_COLORS[] = {
        cv::Scalar(0, 255, 255),   // 黄色：批次0
        cv::Scalar(0, 255, 0),     // 绿色：批次1
        cv::Scalar(255, 0, 255),   // 紫色：批次2
        cv::Scalar(0, 165, 255)    // 橙色：批次3
    };
    const cv::Scalar SLOT_COLOR(255, 0, 0); // 车位方向线：蓝色

    // ==================== 2. 绘制关键点（多批次） ====================
    for (size_t batch = 0; batch < points.size(); ++batch) {
        const auto& pt_batch = points[batch];
        cv::Scalar color = POINT_COLORS[batch % 4];

        for (size_t i = 0; i < pt_batch.size(); i += 3) {
            float conf = pt_batch[i];
            float x = pt_batch[i+1] * image.cols; // 归一化→像素坐标[10,11](@ref)
            float y = pt_batch[i+2] * image.rows;
            
            // 绘制实心点+黑边（增强对比度）
            cv::circle(image, cv::Point(x, y), 6, color, -1);
            cv::circle(image, cv::Point(x, y), 6, cv::Scalar(0,0,0), 1);
            
            // 高置信度点添加文本标注
            if (conf > 0.3) {
                std::string label = cv::format("%.1f", conf);
                cv::putText(image, label, cv::Point(x-15, y-15), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(30,30,30), 2); // 灰底
                cv::putText(image, label, cv::Point(x-15, y-15), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1); // 彩色字
            }
        }
    }

    // ==================== 3. 绘制停车位（多批次） ====================
    for (size_t batch = 0; batch < slots.size(); ++batch) {
        for (const auto& slot : slots[batch]) {
            // 坐标转换（归一化→像素）
            cv::Point pt1(slot.coords[0] * image.cols, 
                          slot.coords[1] * image.rows);
            cv::Point pt2(slot.coords[2] * image.cols, 
                          slot.coords[3] * image.rows);
            
            // 绘制方向线（起点→终点）
            cv::arrowedLine(image, pt1, pt2, SLOT_COLOR, 2, 8, 0, 0.1);
            
            // 标记端点（红起点，绿终点）
            cv::circle(image, pt1, 5, cv::Scalar(0,0,255), -1); // 起点红色
            cv::circle(image, pt2, 5, cv::Scalar(0,255,0), -1); // 终点绿色
            
            // 车位中点显示置信度
            cv::Point center((pt1.x + pt2.x)/2, (pt1.y + pt2.y)/2);
            std::string conf_text = cv::format("%.2f", slot.confidence);
            cv::rectangle(image, cv::Point(center.x-25, center.y-10), 
                          cv::Point(center.x+25, center.y+10), cv::Scalar(200,200,200), -1);
            cv::putText(image, conf_text, center + cv::Point(-20, 5), 
                        cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(0,100,0), 1);
        }
    }

    // ==================== 4. 添加统计信息（左上角） ====================
    int total_points = 0;
    for (auto& batch : points) total_points += batch.size() / 3;
    
    int total_slots = 0;
    for (auto& batch : slots) total_slots += batch.size();
    
    std::string stats = cv::format("Points: %d | Slots: %d", total_points, total_slots);
    cv::rectangle(image, cv::Point(10, 5), cv::Point(320, 40), cv::Scalar(50,50,50), -1);
    cv::putText(image, stats, cv::Point(20, 30), 
                cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(200, 200, 100), 2);
}


int main() {
    // 配置参数
    const std::string onnx_path = "/workspace/APA/gcn-parking-slot/cache/ps_gat/100/output_onnx/model_simplified.onnx";
    const std::string engine_path = "engine.trt";
    const int input_width = 512;
    const int input_height = 512;
    const int max_points = 100;
    const int max_slots = 50;

    // 图片目录设置
    const std::string image_dir = "/workspace/APA/gcn-parking-slot/datasets/parking_slot/testing/indoor-parking lot";
    
    // 获取所有JPG图片路径
    std::vector<std::string> image_paths;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(image_dir)) {
        if (entry.is_regular_file() && 
            (entry.path().extension() == ".jpg" || 
             entry.path().extension() == ".jpeg")) {
            image_paths.push_back(entry.path().string());
        }
    }
    
    if (image_paths.empty()) {
        std::cerr << "错误：未找到任何JPG图片！" << std::endl;
        return -1;
    }
    std::cout << "找到 " << image_paths.size() << " 张图片待处理" << std::endl;

    // 初始化检测器
    psdet::PsDet detector(onnx_path, engine_path, 1, max_points, max_slots, input_width, input_height);
    
    // 构建或加载引擎
    if (!detector.load()) {
        std::cout << "构建TensorRT引擎..." << std::endl;
        if (!detector.build(true)) {
            std::cerr << "引擎构建失败!" << std::endl;
            return 1;
        }
        if (!detector.load()) {
            std::cerr << "引擎加载失败!" << std::endl;
            return 1;
        }
    }
    
    // 遍历处理每张图片
    for (const auto& img_path : image_paths) {
        std::cout << "\n处理图片: " << img_path << std::endl;
        
        // 读取图片
        cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "  图片读取失败，跳过" << std::endl;
            continue;
        }
        
    
        // 执行推理
        std::vector<std::vector<float>> points;
        std::vector<std::vector<psdet::ParkingSlot>> slots;
        
        auto start = std::chrono::high_resolution_clock::now();
        bool success = detector.infer(image, points, slots);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (!success) {
            std::cerr << "  推理失败!" << std::endl;
            continue;
        }
        
        // 输出推理结果
        std::cout << "  推理时间: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                  << "ms" << std::endl;
        
        // 可视化结果（可选）
        if (!points.empty() && !points[0].empty()) {
            std::cout << "  检测到点位: " << points[0].size()/3 << "个" << std::endl;
            /* 这里可以添加可视化代码，如绘制点 */
            drawResults(image, points, slots);
        }
      
        
        // 保存结果图片（可选）
        std::string output_path = img_path + "_result.jpg";
        cv::imwrite(output_path, image);
    }
    
    std::cout << "\n处理完成！共处理 " << image_paths.size() << " 张图片" << std::endl;
    return 0;
}