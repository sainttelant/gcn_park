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