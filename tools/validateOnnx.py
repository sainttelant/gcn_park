import cv2
import time
import torch
import numpy as np
from pathlib import Path
import onnxruntime as ort  # 添加ONNX Runtime依赖[2,3](@ref)

def draw(image, pred_dicts):
    """
    完全修复版停车位绘制函数
    - 解决数组布尔运算错误
    - 支持多维模型输出
    - 增强零向量防护
    """
    slots_pred = pred_dicts['slots_pred'][0]  # 移除批次维度
    height, width = image.shape[:2]
    
    # 常量定义
    VSLOT_MIN_DIST = 0.0448
    VSLOT_MAX_DIST = 0.1099
    SHORT_SEPARATOR_LENGTH = 0.1995
    LONG_SEPARATOR_LENGTH = 0.4688

    junctions = []
    
    for slot in slots_pred:
        # 关键修复：确保坐标提取为标量
        coords = slot[:4].cpu().numpy()
        
        # 处理多维输出（如batch维度）
        if coords.ndim > 1:
            coords = coords.squeeze()  # 移除多余维度
        
        # 显式转换为Python标量
        x0 = coords[0].item() if hasattr(coords[0], 'item') else float(coords[0])
        y0 = coords[1].item() if hasattr(coords[1], 'item') else float(coords[1])
        x1 = coords[2].item() if hasattr(coords[2], 'item') else float(coords[2])
        y1 = coords[3].item() if hasattr(coords[3], 'item') else float(coords[3])

        # 坐标映射
        p0_x = width * x0 - 0.5
        p0_y = height * y0 - 0.5
        p1_x = width * x1 - 0.5
        p1_y = height * y1 - 0.5

        # 向量计算（确保结果为标量）
        vec_x = float(p1_x - p0_x)
        vec_y = float(p1_y - p0_y)
        
        # 核心修复：标量化向量长度计算
        vec_length = np.sqrt(vec_x**2 + vec_y**2)
        
        # 标量检查（防止多维数组）
        if not np.isscalar(vec_length):
            vec_length = vec_length.item() if hasattr(vec_length, 'item') else float(vec_length)
        
        # 零向量防护
        if vec_length < 1e-5:
            vec_x, vec_y = 1.0, 0.0
            vec_length = 1.0
        else:
            vec_x /= vec_length
            vec_y /= vec_length

        # 选择分隔线类型
        distance = (x0 - x1)**2 + (y0 - y1)**2
        if VSLOT_MIN_DIST <= distance <= VSLOT_MAX_DIST:
            separating_length = LONG_SEPARATOR_LENGTH
        else:
            separating_length = SHORT_SEPARATOR_LENGTH

        # 计算垂直点
        p2_x = p0_x + height * separating_length * vec_y
        p2_y = p0_y - width * separating_length * vec_x
        p3_x = p1_x + height * separating_length * vec_y
        p3_y = p1_y - width * separating_length * vec_x

        # 转换为整数坐标
        points = [
            (int(round(p0_x)), int(round(p0_y))),
            (int(round(p1_x)), int(round(p1_y))),
            (int(round(p2_x)), int(round(p2_y))),
            (int(round(p3_x)), int(round(p3_y)))
        ]

        # 绘制停车位
        cv2.line(image, points[0], points[1], (255, 0, 0), 2)
        cv2.line(image, points[0], points[2], (0, 255, 0), 2)
        cv2.line(image, points[1], points[3], (0, 165, 255), 2)
        
        junctions.extend([points[0], points[1]])
    
    # 绘制连接点
    for pt in set(junctions):
        cv2.circle(image, pt, 6, (0, 0, 255), -1)
        
    return image



def validate_onnx_model(onnx_path, image_dir):
    # 初始化ONNX Runtime会话
    providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)  # 自动选择硬件加速[5](@ref)
    
    # 获取输入输出名称
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]  # 支持多输出模型[3,6](@ref)

    for img_path in Path(image_dir).glob('*.jpg'):
        # 保持原始预处理逻辑
        image = cv2.imread(str(img_path))
        image0 = cv2.resize(image, (512, 512))
        input_data = (image0 / 255.0).astype(np.float32)
        input_tensor = np.transpose(input_data, (2, 0, 1))[np.newaxis]  # HWC -> BCHW

        # ONNX推理
        start_time = time.time()
        outputs = session.run(output_names, {input_name: input_tensor})  # 执行推理[2,3](@ref)
        print(f"[DEBUG] ONNX输出形状:")
        print(f"  points_pred: {outputs[0].shape}") 
        print(f"  slots_pred: {outputs[1].shape}")
        
        # 打印第一个slot的数据类型
        slot_sample = outputs[1][0][0]
        print(f"Slot数据类型: {type(slot_sample)}, 值: {slot_sample}")
        sec_per_example = time.time() - start_time
        print(f'ONNX推理耗时: {sec_per_example:.4f}s')

        # 转换输出格式为PyTorch兼容格式
        pred_dicts = {
            'points_pred': [torch.from_numpy(outputs[0])],  # 假设第一个输出是points_pred
            'slots_pred': [torch.from_numpy(outputs[1])]    # 假设第二个输出是slots_pred
        }

        # 使用原始可视化函数验证结果
        result_image = draw(image0.copy(), pred_dicts)
        
        # 保存对比结果
        save_dir = Path(image_dir) / 'onnx_predictions'
        save_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(save_dir / f'{img_path.stem}_onnx.jpg'), result_image)

        ##############################
        # 精度验证关键步骤（新增）
        ##############################
        # 1. 数值精度对比
        print("ONNX points_pred统计: 总和={:.4f}, 最大={:.4f}, 最小={:.4f}".format(
            outputs[0].sum(), outputs[0].max(), outputs[0].min()
        ))
        
        # 2. 保存原始数据（与PyTorch输出对比）
        np.savetxt(save_dir / f'{img_path.stem}_points_onnx.txt', outputs[0].ravel())
        np.savetxt(save_dir / f'{img_path.stem}_slots_onnx.txt', outputs[1].ravel())

if __name__ == '__main__':
    onnx_path = "cache/ps_gat/100/output_onnx/model_simplified.onnx"
    image_dir = "images"  # 保持与原脚本一致
    validate_onnx_model(onnx_path, image_dir)
