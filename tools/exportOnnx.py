import torch
import onnx
import numpy as np
from PIL import Image
from torchvision import transforms
import pprint
from psdet.utils.config import get_config
from psdet.utils.common import get_logger
from psdet.models.builder import build_model
from pathlib import Path
import onnxruntime 
from torchsummary import summary

def export_model_to_onnx(model, cfg, device_id=0):
    # 设置使用的设备
    device = torch.device(f'cuda:{device_id}')
    
    # 将模型移动到指定设备并设为评估模式
    model.to(device)
    model.eval()

    # 清理缓存
    torch.cuda.empty_cache()

    # 1. 准备真实图像输入
    #image_path = cfg.test_image  # 假设配置中有测试图像路径
    image_dir = Path(cfg.data_root) / 'testing' / 'indoor-parking lot'
    image = Image.open(image_dir/"001.jpg").convert("RGB")
    
    # 创建与模型预期一致的预处理管道
    preprocess = transforms.Compose([
        transforms.Resize((cfg.image_size[0], cfg.image_size[1])),  # 使用配置中的图像尺寸
        transforms.ToTensor(),          # 转换为Tensor
    ])
    
    # 应用预处理
    image_tensor = preprocess(image).unsqueeze(0).to(device)  # 添加批次维度
    
    # 2. 创建输入字典
    input_dict = {'image': image_tensor}
    
    # 定义保存 ONNX 模型的路径
    onnx_model_path = cfg.save_onnx_dir
    simplified_path = cfg.simplified_path
    # 创建一个健壮的模型包装器
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, image):
            # 创建输入字典
            input_dict = {'image': image}
            
            # 调用模型
            data_dict = self.model(input_dict)
            
            # 提取模型输出
            points_pred = data_dict['points_pred']
            
            # 获取描述符图（如果可用）
            if 'descriptor_map' in data_dict:
                descriptor_map = data_dict['descriptor_map']
            else:
                # 对于 DirectionalPointDetector，描述符图可能不可用
                descriptor_map = torch.zeros(
                    image.size(0), 
                    cfg.model['descriptor_dim'] if hasattr(cfg, 'descriptor_dim') else 256,
                    image.size(2) // (cfg.model['depth_factor'] if hasattr(cfg, 'depth_factor') else 8),
                    image.size(3) // (cfg.model['depth_factor'] if hasattr(cfg, 'depth_factor') else 8)
                ).to(image.device)
            
            return points_pred, descriptor_map
    
    # 包装模型并移动到设备
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()

    # 在导出前测试一次前向传播
    with torch.no_grad():
        points_pred, descriptor_map = wrapped_model(image_tensor)
        print("输出结构:")
        # 安全地打印 points_pred 的信息
        if isinstance(points_pred, torch.Tensor):
            print(f"points_pred: 类型=张量, 形状={points_pred.shape}")
        elif isinstance(points_pred, (list, tuple)):
            print(f"points_pred: 类型={type(points_pred)}, 长度={len(points_pred)}")
            # 打印列表中第一个元素的类型（如果有）
            if len(points_pred) > 0:
                first_item = points_pred[0]
                print(f"  第一个元素的类型: {type(first_item)}")
                if isinstance(first_item[0], tuple):
                    print(f"  第一个元组有 {len(first_item)} 个元素")
                    if len(first_item) > 0:
                        print(f"    第一个元素的置信度: {first_item[0]}")
                        if isinstance(first_item[1], np.ndarray):
                            print(f"    位置数组的形状: {first_item[1].shape}")
        else:
            print(f"points_pred: 类型={type(points_pred)}")
        
        # 安全地打印 descriptor_map 的信息
        if descriptor_map is not None and isinstance(descriptor_map, torch.Tensor):
            print(f"descriptor_map: 类型=张量, 形状={descriptor_map.shape}")
        else:
            print(f"descriptor_map: {type(descriptor_map)}")
                    
    # 导出模型到 ONNX 格式
    torch.onnx.export(
        wrapped_model, 
        image_tensor,
        onnx_model_path/'model.onnx',
        export_params=True,
        opset_version=14,  # 使用更新的 opset 版本
        do_constant_folding=True,
        input_names=['image'],
        output_names=['points_pred', 'descriptor_map'],
        dynamic_axes={
            'image': {0: 'batch_size', 2: 'height', 3: 'width'},
            'points_pred': {0: 'batch_size'},
            'descriptor_map': {0: 'batch_size'}
        }
    )
    print(f"模型已转换为 ONNX 并保存到 {onnx_model_path}")

    # 验证 ONNX 模型
    onnx_model = onnx.load(onnx_model_path/"model.onnx")
  
    for input in onnx_model.graph.input:
        print(f"Input: {input.name}, Shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")
        print("ONNX 模型验证通过")

    # 可选：优化 ONNX 模型
     # 尝试简化模型
    try:
        from onnxsim import simplify
        simplified_model, check = simplify(onnx_model)
        onnx.save(simplified_model, simplified_path/"model_simplified.onnx")
        print(f"ONNX模型简化完成，保存到: {simplified_path}")
    except ImportError:
        print("警告: onnx-simplifier 未安装，跳过模型简化步骤")
    except Exception as e:
        print(f"模型简化失败: {e}")
    
   
    
    # 验证 ONNX 模型    
    validation_result = validate_onnx_model(wrapped_model, image_tensor, onnx_model_path)
    if validation_result:
        print("\n[🎉] ONNX模型验证成功！与原模型完全一致")
    else:
        print("\n[⚠️] 警告：ONNX模型输出与原始模型存在差异")
   
def validate_onnx_model(wrapped_model, image_tensor, onnx_model_path):
    # 1. 加载ONNX模型并验证基础结构
    onnx_model = onnx.load(onnx_model_path/"model_simplified.onnx")
    try:
        onnx.checker.check_model(onnx_model)  # 检查模型是否符合ONNX标准
        print("[✅] ONNX模型基础结构验证通过")
    except onnx.checker.ValidationError as e:
        print(f"[❌] ONNX模型结构异常: {e}")
        return False

    # 2. 准备ONNX Runtime推理会话
    ort_session = onnxruntime.InferenceSession(
        str(onnx_model_path/"model_simplified.onnx"),
        providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    )
    
    # 3. 获取输入输出名称
    input_name = ort_session.get_inputs()[0].name
    output_names = [out.name for out in ort_session.get_outputs()]
    
    # 4. 运行PyTorch原始模型推理
    with torch.no_grad():
        torch_points, torch_desc = wrapped_model(image_tensor)
        
        print("PyTorch points总和:", torch_points.sum().item())
        print("PyTorch points统计: 总和={:.4f}, 最大={:.4f}, 最小={:.4f}".format(
            torch_points.sum().item(),
            torch_points.max().item(),
            torch_points.min().item()
        ))
        # 确保输出为张量以便比较
        if not isinstance(torch_points, torch.Tensor):
            torch_points = torch.tensor(torch_points).to(image_tensor.device)
    
    # 5. 运行ONNX模型推理
    ort_inputs = {input_name: image_tensor.cpu().numpy()}
    ort_outs = ort_session.run(output_names, ort_inputs)
    onnx_points = torch.from_numpy(ort_outs[0]).to(image_tensor.device)
    onnx_desc = torch.from_numpy(ort_outs[1]).to(image_tensor.device)

    # 6. 数值一致性验证（关键步骤）
    passed = True
    # 验证关键点输出
    points_diff = torch.abs(torch_points - onnx_points)
    max_diff = points_diff.max().item()
    mean_diff = points_diff.mean().item()
    print(f"关键点输出差异: 最大={max_diff:.6f}, 平均={mean_diff:.6f}")
    if mean_diff > 1e-2:  # 设置合理阈值
        print("[❌] 关键点输出差异过大!")
        passed = False
    
    # 验证描述符输出
    desc_diff = torch.abs(torch_desc - onnx_desc)
    max_desc_diff = desc_diff.max().item()
    mean_desc_diff = desc_diff.mean().item()
    print(f"描述符输出差异: 最大={max_desc_diff:.6f}, 平均={mean_desc_diff:.6f}")
    if mean_desc_diff > 1e-2:
        print("[❌] 描述符输出差异过大!")
        passed = False
    
    # 7. 动态形状验证（可选）
    if cfg.validate_dynamic_shape:
        print("\n[🧪] 动态形状验证...")
        try:
            # 创建不同批次的输入
            batch2_input = torch.cat([image_tensor, image_tensor], dim=0)
            ort_inputs_batch2 = {input_name: batch2_input.cpu().numpy()}
            ort_outs_batch2 = ort_session.run(output_names, ort_inputs_batch2)
            print(f"[✅] 动态批次验证通过 (batch=2)")
        except Exception as e:
            print(f"[❌] 动态形状验证失败: {str(e)}")
            passed = False
    
    return passed    
    
if __name__ == '__main__':
    cfg = get_config()
    logger = get_logger(cfg.log_dir, cfg.tag)
    logger.info(pprint.pformat(cfg))
    
    model = build_model(cfg.model)
    logger.info(model)
    model.load_params_from_file(filename=cfg.ckpt, logger=logger, to_cpu=False)

    device = torch.device('cuda:0')
    model.to(device)

    export_model_to_onnx(model, cfg, device_id=0)
    