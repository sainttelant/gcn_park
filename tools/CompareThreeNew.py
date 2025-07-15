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
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTEngineValidator:
    def __init__(self, engine_path):
        """初始化TensorRT引擎验证器"""
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def _load_engine(self, engine_path):
        """加载并反序列化TensorRT引擎"""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            runtime = trt.Runtime(TRT_LOGGER)
            return runtime.deserialize_cuda_engine(engine_data)
        except Exception as e:
            raise RuntimeError(f"引擎加载失败: {str(e)}") from e

    def _allocate_buffers(self):
        """分配输入/输出缓冲区（支持动态形状）"""
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for binding_idx in range(self.engine.num_bindings):
            binding_name = self.engine.get_tensor_name(binding_idx)
            dtype = self.engine.get_tensor_dtype(binding_name)
            shape = self.context.get_tensor_shape(binding_name)

            # 处理未指定形状的情况
            if -1 in shape:
                shape = [max(dim, 1) for dim in shape]  # 确保没有零值
                self.context.set_input_shape(binding_name, shape)

            # 计算缓冲区大小
            dtype_size = np.dtype(trt.nptype(dtype)).itemsize
            num_elements = int(np.prod(shape))
            size_bytes = int(num_elements * dtype_size)
            print(f"Allocating memory of size: {size_bytes}, type: {type(size_bytes)}")
            print(f"Binding name: {binding_name}, shape: {shape}, dtype: {dtype}")
            if size_bytes <= 0:
                raise ValueError(f"Invalid buffer size for binding {binding_name}")

            # 分配设备内存
            device_mem = cuda.mem_alloc(size_bytes)
            bindings.append(int(device_mem))

            # 分配主机内存（锁页内存加速传输）
            host_mem = cuda.pagelocked_empty(num_elements, trt.nptype(dtype))

            if self.engine.binding_is_input(binding_idx):
                inputs.append({'host': host_mem, 'device': device_mem, 'name': binding_name})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'name': binding_name})

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        """执行TensorRT推理"""
        # 验证输入形状
        input_shape = input_data.shape
        binding_shape = self.context.get_tensor_shape(self.inputs[0]['name'])
        if list(input_shape) != binding_shape:
            self.context.set_input_shape(self.inputs[0]['name'], input_shape)
            self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

        # 数据预处理（确保连续内存）
        input_data = np.ascontiguousarray(input_data)
        np.copyto(self.inputs[0]['host'], input_data.ravel())

        # 主机->设备传输
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 设备->主机传输
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)

        # 同步流
        self.stream.synchronize()

        # 重构输出形状
        results = []
        for output in self.outputs:
            shape = self.context.get_tensor_shape(output['name'])
            results.append(output['host'].reshape(shape))

        return results

    @staticmethod
    def compare_outputs(trt_outputs, ref_outputs, tolerance=1e-2):
        """对比TensorRT输出与参考输出"""
        report = {'passed': True, 'details': []}

        for i, (trt_out, ref_out) in enumerate(zip(trt_outputs, ref_outputs)):
            # 确保参考输出为numpy数组
            if isinstance(ref_out, torch.Tensor):
                ref_out = ref_out.detach().cpu().numpy()

            # 形状验证
            if trt_out.shape != ref_out.shape:
                report['passed'] = False
                report['details'].append({
                    'index': i,
                    'status': 'FAILED ❌',
                    'message': f"形状不匹配: TRT={trt_out.shape} vs REF={ref_out.shape}",
                    'max_diff': None,
                    'mean_diff': None
                })
                continue

            # 数值差异计算
            diff = np.abs(trt_out - ref_out)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            # 判断是否通过
            status = 'PASSED ✅' if mean_diff < tolerance else 'FAILED ❌'
            if status == 'FAILED ❌':
                report['passed'] = False

            report['details'].append({
                'index': i,
                'status': status,
                'message': f"最大差异: {max_diff:.6f}, 平均差异: {mean_diff:.6f}",
                'max_diff': max_diff,
                'mean_diff': mean_diff
            })

        return report

    def cleanup(self):
        """释放所有CUDA资源"""
        for buf in self.inputs + self.outputs:
            if 'device' in buf and buf['device']:
                buf['device'].free()
                
def export_gnn_edge_model(model, cfg, device_id=0):
    device = torch.device(f'cuda:{device_id}')
    
    # GNN边预测模型包装器
    class GNNEdgeWrapper(torch.nn.Module):
  
        def __init__(self, graph_encoder, edge_predictor):
            super().__init__()
            self.graph_encoder = graph_encoder
            self.edge_predictor = edge_predictor
            
        def forward(self, descriptors, points):
            data_dict = {
                'descriptors': descriptors,
                'points': points[:, :, :2]  # 仅取xy坐标[6](@ref)
            }
            
            # 处理GNN分支
            if self.graph_encoder is not None:
                data_dict = self.graph_encoder(data_dict)
                graph_output = data_dict['descriptors']
            else:
                graph_output = descriptors
            
            # 边预测
            data_dict = self.edge_predictor(data_dict)
            return graph_output, data_dict['edges_pred']
        
        # 创建模型实例
        # 创建包装器实例
    submodel = GNNEdgeWrapper(
        graph_encoder=model.model.graph_encoder if cfg.use_gnn else None,
        edge_predictor=model.model.edge_predictor  # 修正属性路径[1](@ref)
    ).to(device)
    submodel.eval()
        
        # 其余代码保持不变...
    gnn_out_dim = cfg.model.graph_encoder.gnn.proj_dim
    B, N = 1, cfg.model.max_points
    dummy_desc = torch.randn(B, gnn_out_dim, N).to(device)  # [1, 64, 10]
    dummy_points = torch.rand(B, N, 2).to(device)     
        
        # 导出ONNX
    torch.onnx.export(
            submodel,
            (dummy_desc, dummy_points),
            str(Path(cfg.save_onnx_dir) / "gnn_edge_model.onnx"),
            export_params=True,
            opset_version=14,
            input_names=['descriptors', 'points'],
            output_names=['graph_output', 'edge_pred'],
            dynamic_axes={
                'descriptors': {0: 'batch_size', 2: 'num_points'},
                'points': {0: 'batch_size', 1: 'num_points'},
                'graph_output': {0: 'batch_size', 2: 'num_points'},
                'edge_pred': {0: 'batch_size', 2: 'num_edges'}
        }
    )
    print(f"GNN边预测模型已导出到 {cfg.save_onnx_dir/'gnn_edge_model.onnx'}")


def export_model_to_onnx(model, cfg, device_id=0):
    # 设置使用的设备
    device = torch.device(f'cuda:{device_id}')

    # 将模型移动到指定设备并设为评估模式
    model.to(device)
    model.eval()

    # 清理缓存
    torch.cuda.empty_cache()

    # 1. 准备真实图像输入
    #image_dir = Path(cfg.data_root) / 'testing' / 'indoor-parking lot'
    
    image_dir = Path("images")
    image = Image.open(image_dir/"001.jpg").convert("RGB")

    # 创建与模型预期一致的预处理管道
    preprocess = transforms.Compose([
        transforms.Resize((cfg.image_size[0], cfg.image_size[1])),
        transforms.ToTensor(),
    ])

    # 应用预处理
    image_tensor = preprocess(image).unsqueeze(0).to(device)

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
            input_dict = {'image': image}
            data_dict = self.model(input_dict)
            points_pred = data_dict['points_pred']     
           

            if 'descriptor_map' in data_dict:
                descriptor_map = data_dict['descriptor_map']
            else:
                descriptor_map = torch.zeros(
                    image.size(0),
                    cfg.model.get('descriptor_dim', 256),
                    image.size(2) // cfg.model.get('depth_factor', 8),
                    image.size(3) // cfg.model.get('depth_factor', 8)
                ).to(image.device)

            return points_pred, descriptor_map

    # 包装模型并移动到设备
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()

    # 在导出前测试一次前向传播
    with torch.no_grad():
        points_pred, descriptor_map = wrapped_model(image_tensor)
        print(f"points_pred shape: {points_pred.shape}")
        print(f"descriptor_map shape: {descriptor_map.shape}")
       

    # 导出模型到 ONNX 格式
    torch.onnx.export(
        wrapped_model,
        image_tensor,
        onnx_model_path/'model.onnx',
        export_params=True,
        opset_version=14,
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
        a = input
        #print(a)
        for dim in input.type.tensor_type.shape.dim:
            aa = dim
            print(aa.dim_value)
        print(f"Input: {input.name}, Shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")
        print("ONNX 模型验证通过")

    # 优化 ONNX 模型
    try:
        from onnxsim import simplify
        simplified_model, check = simplify(onnx_model)
        onnx.save(simplified_model, simplified_path/"model_simplified.onnx")
        print(f"ONNX模型简化完成，保存到: {simplified_path}")
        
        model = onnx.load(simplified_path/"model.onnx")
        for input in model.graph.input:
            if input.name == "image":  # 替换为你的输入名称
                # 清除动态维度标记
                input.type.tensor_type.shape.dim[0].dim_value = 1  # batch
                input.type.tensor_type.shape.dim[2].dim_value = 512  # height
                input.type.tensor_type.shape.dim[3].dim_value = 512  # width
        onnx.save(model, simplified_path/"fixed_model.onnx")
        
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

    # 直接加载已有的TensorRT引擎进行验证
    validate_trt = False
    if validate_trt: 
        trt_validator = TensorRTEngineValidator(engine_path=onnx_model_path/"new.engine")
        trt_outputs = trt_validator.infer(image_tensor.cpu().numpy())

        # 获取PyTorch原始输出
        with torch.no_grad():
            torch_outputs = wrapped_model(image_tensor)

        # 转换为可比较格式
        torch_outputs = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in torch_outputs]

        # 比较结果
        report = trt_validator.compare_outputs(trt_outputs, torch_outputs)

        # 打印比较结果
        print("\n" + "="*50)
        print(f"TensorRT验证结果: {'通过' if report['passed'] else '失败'}")
        print("="*50)
        for detail in report['details']:
            print(f"输出 {detail['index']}: {detail['status']}")
            print(f"  → {detail['message']}")

        # 释放资源
        trt_validator.cleanup()

def validate_onnx_model(wrapped_model, image_tensor, onnx_model_path):
    # 1. 加载ONNX模型并验证基础结构
    onnx_model = onnx.load(onnx_model_path/"model_simplified.onnx")
    try:
        onnx.checker.check_model(onnx_model)
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
    
    ort_inputs = {input_name: image_tensor.cpu().numpy()}
    ort_outs = ort_session.run(output_names, ort_inputs)
    print(f"ONNX points_pred shape: {ort_outs[0].shape}")
    print(f"ONNX descriptor_map shape: {ort_outs[1].shape}")
   

    # 4. 运行PyTorch原始模型推理
    with torch.no_grad():
        torch_points, torch_desc = wrapped_model(image_tensor)

        print("PyTorch points总和:", torch_points.sum().item())
        print("PyTorch points统计: 总和={:.4f}, 最大={:.4f}, 最小={:.4f}".format(
            torch_points.sum().item(),
            torch_points.max().item(),
            torch_points.min().item()
        ))
        
        print("pytorch desc总和:", torch_desc.sum().item())
        print("pytorch desc统计: 总和={:.4f}, 最大={:.4f}, 最小={:.4f}".format(
            torch_desc.sum().item(),
            torch_desc.max().item(),
            torch_desc.min().item()
        ))
        if not isinstance(torch_points, torch.Tensor):
            torch_points = torch.tensor(torch_points).to(image_tensor.device)

    # 5. 运行ONNX模型推理
    ort_inputs = {input_name: image_tensor.cpu().numpy()}
    ort_outs = ort_session.run(output_names, ort_inputs)
    onnx_points = torch.from_numpy(ort_outs[0]).to(image_tensor.device)
    onnx_desc = torch.from_numpy(ort_outs[1]).to(image_tensor.device)

    # 6. 数值一致性验证
    passed = True
    # 验证关键点输出
    points_diff = torch.abs(torch_points - onnx_points)
    max_diff = points_diff.max().item()
    mean_diff = points_diff.mean().item()
    print(f"关键点输出差异: 最大={max_diff:.6f}, 平均={mean_diff:.6f}")
    if mean_diff > 1e-2:
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

    #export_model_to_onnx(model, cfg, device_id=0)
    export_gnn_edge_model(model, cfg, device_id=0)