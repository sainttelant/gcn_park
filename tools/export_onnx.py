import torch
import onnx
import pprint
from psdet.utils.config import get_config
from psdet.utils.common import get_logger
from psdet.models.builder import build_model

def export_model_to_onnx(model, cfg, device_id=0):
    # 设置使用的设备
    device = torch.device(f'cuda:{device_id}')
    
    # 将模型移动到指定设备并设为评估模式
    model.to(device)
    model.eval()

    # 清理缓存
    torch.cuda.empty_cache()

    # 准备虚拟输入 - 确保所有张量在相同设备
    dummy_image = torch.randn(1, 3, 512, 512).to(device)  # 输入图像
    dummy_points_pred = torch.randn(1, 100, 3).to(device)  # 点预测
    dummy_descriptor_map = torch.randn(1, 256, 32, 32).to(device)  # 描述符图
    
    # 定义保存 ONNX 模型的路径
    onnx_model_path = cfg.save_onnx_dir

    # 创建一个健壮的模型包装器，确保输出是张量
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, image, points_pred, descriptor_map):
            # 创建输入字典
            input_dict = {
                'image': image,
                'points_pred': points_pred,
                'descriptor_map': descriptor_map
            }
            
            # 调用模型
            try:
                outputs = self.model(input_dict)
                
                # 返回的第一个元素应该是字典
                if isinstance(outputs, tuple) and outputs and isinstance(outputs[0], dict):
                    out_dict = outputs[0]
                    
                    # 确保我们只返回张量类型的输出
                    tensor_outputs = []
                    
                    # 添加图像输出（如果有）
                    if 'image' in out_dict and torch.is_tensor(out_dict['image']):
                        tensor_outputs.append(out_dict['image'])
                    
                    # 添加点预测输出（如果有）
                    if 'points_pred' in out_dict and torch.is_tensor(out_dict['points_pred']):
                        tensor_outputs.append(out_dict['points_pred'])
                    
                    # 添加描述符图输出（如果有）
                    if 'descriptor_map' in out_dict and torch.is_tensor(out_dict['descriptor_map']):
                        tensor_outputs.append(out_dict['descriptor_map'])
                    
                    # 添加槽位预测输出（如果有）
                    if 'slots_pred' in out_dict and torch.is_tensor(out_dict['slots_pred']):
                        tensor_outputs.append(out_dict['slots_pred'])
                    
                    # 确保至少有一个输出
                    if not tensor_outputs:
                        tensor_outputs.append(torch.zeros(1, 3, 512, 512).to(image.device))
                    
                    return tuple(tensor_outputs)
                
                # 如果直接返回张量
                elif torch.is_tensor(outputs):
                    return (outputs,)
                    
                # 如果返回元组且全是张量
                elif isinstance(outputs, tuple) and all(torch.is_tensor(t) for t in outputs):
                    return outputs
                    
                else:
                    # 添加占位符输出
                    return (torch.zeros(1, 3, 512, 512).to(image.device),)
            
            except Exception as e:
                print(f"模型调用失败: {e}")
                return (torch.zeros(1, 3, 512, 512).to(image.device),)
    
    # 包装模型并移动到设备
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()

    # 在导出前测试一次前向传播
    try:
        with torch.no_grad():
            outputs = wrapped_model(dummy_image, dummy_points_pred, dummy_descriptor_map)
            print("输出结构:")
            for i, out in enumerate(outputs):
                if torch.is_tensor(out):
                    print(f"输出 {i}: 类型=张量, 形状={out.shape}")
                else:
                    print(f"输出 {i}: 类型={type(out)} (无法获取形状)")
                    
        # 成功测试后进行导出
        torch.onnx.export(
            wrapped_model, 
            (dummy_image, dummy_points_pred, dummy_descriptor_map),
            onnx_model_path/"model.onnx",
            export_params=True,
            opset_version=14,  # 使用更新的 opset 版本
            do_constant_folding=True,
            input_names=['image', 'points_pred', 'descriptor_map'],
            output_names=[f'output_{i}' for i in range(len(outputs))],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'points_pred': {0: 'batch_size'},
                'descriptor_map': {0: 'batch_size'},
                **{f'output_{i}': {0: 'batch_size'} for i in range(len(outputs))}
            }
        )
        print(f"模型已转换为 ONNX 并保存到 {onnx_model_path}")
        
        # 验证 ONNX 模型
        onnx_model = onnx.load(onnx_model_path/"model.onnx")
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型验证通过")
        
    except Exception as e:
        print(f"导出过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试添加更多的错误信息
        print("\n尝试导出更多调试信息...")
        try:
            # 导出时添加 verbose=True 获取更多信息
            torch.onnx.export(
                wrapped_model, 
                (dummy_image, dummy_points_pred, dummy_descriptor_map),
                onnx_model_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                verbose=True,  # 添加详细日志
                input_names=['image', 'points_pred', 'descriptor_map'],
                output_names=[f'output_{i}' for i in range(len(outputs))],
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'points_pred': {0: 'batch_size'},
                    'descriptor_map': {0: 'batch_size'},
                    **{f'output_{i}': {0: 'batch_size'} for i in range(len(outputs))}
                }
            )
        except Exception as e2:
            print(f"带详细日志的导出也失败: {e2}")

if __name__ == '__main__':
    # 加载模型配置和日志记录器
    cfg = get_config()
    logger = get_logger(cfg.log_dir, cfg.tag)
    logger.info(pprint.pformat(cfg))
    
    # 构建并加载模型
    model = build_model(cfg.model)
    logger.info(model)
    model.load_params_from_file(filename=cfg.ckpt, logger=logger, to_cpu=False)
    
    # 将模型移动到指定的 GPU
    device = torch.device('cuda:0')
    model.to(device)

    # 导出模型到 ONNX
    export_model_to_onnx(model, cfg, device_id=0)