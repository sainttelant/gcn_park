import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .gcn import GCNEncoder, EdgePredictor
from .post_process import calc_point_squre_dist, pass_through_third_point
from .post_process import get_predicted_points, get_predicted_directional_points, pair_marking_points
from .utils import define_halve_unit, define_detector_block, YetAnotherDarknet, vgg16, resnet18, resnet50
from copy import deepcopy
class PermissiveDict(dict):
    """支持属性访问的字典类型"""
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"No such attribute: {name}")
    
    def __setattr__(self, name, value):
        self[name] = value

class PointDetector(nn.modules.Module):
    """Detector for point without direction."""

    def __init__(self, cfg):
        super(PointDetector, self).__init__()
        self.cfg = cfg
        input_channel_size = cfg.input_channels
        depth_factor = cfg.depth_factor
        output_channel_size = cfg.output_channels
        
        self.point_loss_func = nn.MSELoss().cuda()
        
        if cfg.backbone == 'Darknet':
            self.feature_extractor = YetAnotherDarknet(input_channel_size, depth_factor)
        elif cfg.backbone == 'VGG16':
            self.feature_extractor = vgg16()
        elif cfg.backbone == 'resnet18':
            self.feature_extractor = resnet18()
        elif cfg.backbone == 'resnet50':
            self.feature_extractor = resnet50()
        else:
            raise ValueError('{} is not implemented!'.format(cfg.backbone))
        
        layers_points = []
        layers_points += define_detector_block(16 * depth_factor)
        layers_points += define_detector_block(16 * depth_factor)
        layers_points += [nn.Conv2d(32 * depth_factor, output_channel_size,
                                    kernel_size=1, stride=1, padding=0, bias=False)]
        self.point_predictor = nn.Sequential(*layers_points)

        layers_descriptor = []
        layers_descriptor += define_detector_block(16 * depth_factor)
        layers_descriptor += define_detector_block(16 * depth_factor)
        layers_descriptor += [nn.Conv2d(32 * depth_factor, cfg.descriptor_dim,
                                        kernel_size=1, stride=1, padding=0, bias=False)]
        self.descriptor_map = nn.Sequential(*layers_descriptor)

        
        base_dim = cfg.descriptor_dim
        if cfg.use_gnn:
            self.graph_encoder = GCNEncoder(cfg.graph_encoder)
            gnn_out_dim = cfg.graph_encoder.gnn.proj_dim
        else:
            gnn_out_dim = base_dim
        
  
        edge_cfg = PermissiveDict(cfg.edge_predictor.copy())  # 创建新实例
        edge_cfg.input_dim = 2 * gnn_out_dim  # 设置正确输入维度

        # 修复2: 使用正确的变量名
        print(f"GNN输出维度: {gnn_out_dim}, EdgePredictor输入维度: {edge_cfg.input_dim}")
        self.edge_predictor = EdgePredictor(edge_cfg)  # 使用edge_cfg而非cfg.edge_cfg

        if cfg.get('slant_predictor', None):
            self.slant_predictor = EdgePredictor(cfg.slant_predictor)

        if cfg.get('vacant_predictor', None):
            self.vacant_predictor = EdgePredictor(cfg.vacant_predictor)

    def forward(self, data_dict):
        img = data_dict['image']

        features = self.feature_extractor(img)  # [b, 1024, 16, 16]

        points_pred = self.point_predictor(features)
        points_pred = torch.sigmoid(points_pred)
        data_dict['points_pred'] = points_pred

        descriptor_map = self.descriptor_map(features)

        if self.training:
            marks = data_dict['marks']
            pred_dict = self.predict_slots(descriptor_map, marks[:, :, :2])
            data_dict.update(pred_dict)
        else:
            data_dict['descriptor_map'] = descriptor_map
            #data_dict['']

        return data_dict

    def predict_slots(self, descriptor_map, points):
        descriptors = self.sample_descriptors(descriptor_map, points)
        data_dict = {}
        data_dict['descriptors'] = descriptors
        data_dict['points'] = points

        if self.cfg.get('slant_predictor', None):
            pred_dict = self.slant_predictor(data_dict)
            data_dict['slant_pred'] = pred_dict['edges_pred']

        if self.cfg.get('vacant_predictor', None):
            pred_dict = self.vacant_predictor(data_dict)
            data_dict['vacant_pred'] = pred_dict['edges_pred']

        if self.cfg.use_gnn:
            data_dict = self.graph_encoder(data_dict) # 这个地方下采样了，从128 下采样到64，把descriptors从128下采样到64

        # save data_dict['descriptors'] into txt file
        import numpy as np
        descriptors_2d = data_dict['descriptors'].cpu().numpy().reshape(data_dict['descriptors'].shape[0] * data_dict['descriptors'].shape[1], -1)
        descriptors_2d = descriptors_2d.flatten()
        np.savetxt('images/predictions/graph_output_python.txt', descriptors_2d,fmt='%.6f')
        
        pred_dict = self.edge_predictor(data_dict)

        data_dict['edge_pred'] = pred_dict['edges_pred']
        
        edges_pred = data_dict['edge_pred'].cpu().numpy().reshape(data_dict['edge_pred'].shape[0] * data_dict['edge_pred'].shape[1], -1)
        edges_pred = edges_pred.flatten()
        np.savetxt('images/predictions/edge_pred_python.txt', edges_pred,fmt='%.6f')
        
        return data_dict

    def sample_descriptors(self, descriptors, keypoints):
        """ Interpolate descriptors at keypoint locations """
        b, c, h, w = descriptors.shape
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
        
        
        descriptors = torch.nn.functional.grid_sample(
            descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)

        # 使用npsavetxt save descriptors into txt
        import numpy as np
        descriptors_2d = descriptors.cpu().numpy().reshape(descriptors.shape[0] * descriptors.shape[1], -1)
        descriptors_2d = descriptors_2d.flatten()
        np.savetxt('images/predictions/descriptors_after_grid_sample_python.txt', descriptors_2d,fmt='%.6f')
        
        
        descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1)
        descriptors_cpu = descriptors.cpu().numpy().reshape(b, c, -1)
        descriptors_cpu = descriptors_cpu.flatten()
        np.savetxt('images/predictions/descriptors_after_normalize_python.txt', descriptors_cpu,fmt='%.6f')
        return descriptors

    def get_targets_points(self, data_dict):
        points_pred = data_dict['points_pred']
        marks_gt_batch = data_dict['marks']
        npoints = data_dict['npoints']

        b, c, h, w = points_pred.shape
        targets = torch.zeros(b, c, h, w).cuda()
        mask = torch.zeros_like(targets)
        mask[:, 0].fill_(1.)

        for batch_idx, marks_gt in enumerate(marks_gt_batch):
            n = npoints[batch_idx].long()
            for marking_point in marks_gt[:n]:
                x, y = marking_point[:2]
                col = math.floor(x * w)
                row = math.floor(y * h)
                # Confidence Regression
                targets[batch_idx, 0, row, col] = 1.
                # Offset Regression
                targets[batch_idx, 1, row, col] = x * w - col
                targets[batch_idx, 2, row, col] = y * h - row

                mask[batch_idx, 1:3, row, col].fill_(1.)
        return targets, mask


   


    def post_processing(self, data_dict):
        ret_dicts = {}
        pred_dicts = {}
        
        points_pred = data_dict['points_pred']
        descriptor_map = data_dict['descriptor_map']
        
        points_pred_batch = []
        slots_pred = []
  
        
        for b, marks in enumerate(points_pred):
            print('marks.shape:',marks.shape)
            print("b:",b)
            points_pred = get_predicted_points(marks, self.cfg.point_thresh, self.cfg.boundary_thresh)

            for i in range(len(points_pred)):
                points_pred[i] = (points_pred[i][0], points_pred[i][1][:2])
                score = points_pred[i][0]
                x_val = points_pred[i][1][0]
                y_val = points_pred[i][1][1]

                #output_data = np.column_stack((score, x_val, y_val))
                # 将数组写入文件
                #with open('images/predictions/output_points_python.txt', 'a') as f:
                    #np.savetxt(f, output_data, delimiter=' ', fmt="%.6f")
            points_pred_batch.append(points_pred)
         
            if len(points_pred) > 0:
                points_np = np.concatenate([p[1].reshape(1, -1) for p in points_pred], axis=0)
            else:
                points_np = np.zeros((self.cfg.max_points, 2))

            if points_np.shape[0] < self.cfg.max_points:
                points_full = np.zeros((self.cfg.max_points, 2))
                points_full[:len(points_pred)] = points_np
            else:
                points_full = points_np
            with open ('images/predictions/output_points_python.txt', 'a') as f:
                np.savetxt(f, points_full, delimiter=' ', fmt="%.6f")
            pred_dict = self.predict_slots(descriptor_map[b].unsqueeze(0), torch.Tensor(points_full).unsqueeze(0).cuda())
           
           
            edges = pred_dict['edges_pred'][0]
            n = points_np.shape[0]
            m = points_full.shape[0]
            
            slots = []
            for i in range(n):
                for j in range(n):
                    idx = i * m + j
                    score = edges[0, idx]
                    if score > 0.5:
                        x1, y1 = points_np[i,:2]
                        x2, y2 = points_np[j,:2]
                        slot = (score, np.array([x1, y1, x2, y2]))
                        slots.append(slot)

            slots_pred.append(slots)

        pred_dicts['points_pred'] = points_pred_batch
        pred_dicts['slots_pred'] = slots_pred
        # 修改后的代码
        with open('images/predictions/slots_pred_python.txt', 'a') as f:
            # 写入标题行
            header = "confidence coord0 coord1 coord2 coord3"
            f.write(header + "\n")  # 添加换行符[2,6](@ref)
            
            # 遍历所有batch的预测结果
            for batch_slots in slots_pred:
                # 遍历当前batch中的所有slot
                for slot in batch_slots:
                    # 提取置信度和坐标值
                    confidence = slot[0]
                    coords = slot[1]
                    
                    # 确保坐标有4个值
                    if len(coords) == 4:
                        # 格式化数据行：置信度+4个坐标值，保留6位小数
                        line = f"{confidence:.6f} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n"
                        f.write(line)  # 直接写入文件
                    else:
                        # 处理无效坐标数据
                        print(f"警告：跳过无效坐标数据（长度={len(coords)}）")
        return pred_dicts, ret_dicts
    
    def post_processing_onnx(self, data_dict):
        ret_dicts = {}
        pred_dicts = {}
        
        points_pred = data_dict['points_pred']
        descriptor_map = data_dict['descriptor_map']
        
        points_pred_batch = []
        slots_pred = []
        for b, marks in enumerate(points_pred):
            points_pred = get_predicted_points(marks, self.cfg.point_thresh, self.cfg.boundary_thresh)
            points_pred_batch.append(points_pred)
         
            if len(points_pred) > 0:
                points_np = np.concatenate([p[1].reshape(1, -1) for p in points_pred], axis=0)
            else:
                points_np = np.zeros((self.cfg.max_points, 2))

            if points_np.shape[0] < self.cfg.max_points:
                points_full = np.zeros((self.cfg.max_points, 2))
                points_full[:len(points_pred)] = points_np
            else:
                points_full = points_np
            
            device = torch.device('cuda:1')

            # 确保 descriptor_map[b] 在正确的设备上
            descriptor_map_b = descriptor_map[b].unsqueeze(0).to(device).float()

            # 确保 points_full 在正确的设备上
            points_full_tensor = torch.tensor(points_full).unsqueeze(0).to(device).float()

            # 调用 predict_slots 方法
            pred_dict = self.predict_slots(descriptor_map_b, points_full_tensor)
            #pred_dict = self.predict_slots(descriptor_map[b].unsqueeze(0), torch.Tensor(points_full).unsqueeze(0).cuda())
            edges = pred_dict['edges_pred'][0]
            n = points_np.shape[0]
            m = points_full.shape[0]
            
            slots = []
            for i in range(n):
                for j in range(n):
                    idx = i * m + j
                    score = edges[0, idx]
                    if score > 0.5:
                        x1, y1 = points_np[i,:2]
                        x2, y2 = points_np[j,:2]
                        slot = (score, np.array([x1, y1, x2, y2]))
                        slots.append(slot)

            slots_pred.append(slots)

        pred_dicts['points_pred'] = points_pred_batch
        pred_dicts['slots_pred'] = slots_pred
        return pred_dicts, ret_dicts

    def get_training_loss(self, data_dict):
        points_pred = data_dict['points_pred']
        targets, mask = self.get_targets_points(data_dict)

        disp_dict = {}
        
        loss_point = self.point_loss_func(points_pred * mask, targets * mask)
        
        edges_pred = data_dict['edges_pred']
        edges_target = torch.zeros_like(edges_pred)
        edges_mask = torch.zeros_like(edges_pred)

        match_targets = data_dict['match_targets']
        npoints = data_dict['npoints']

        for b in range(edges_pred.shape[0]):
            n = npoints[b].long()
            y = match_targets[b]
            m = y.shape[0]
            for i in range(n):
                t = y[i, 0]                
                for j in range(n):
                    idx = i * m + j
                    edges_mask[b, 0, idx] = 1
                    if j == t:
                        edges_target[b, 0, idx] = 1
                   
        loss_edge = F.binary_cross_entropy(edges_pred, edges_target, edges_mask)
        loss_all = self.cfg.losses.weight_point * loss_point + self.cfg.losses.weight_edge * loss_edge

        tb_dict = {
            'loss_all': loss_all.item(),
            'loss_point': loss_point.item(),
            'loss_edge': loss_edge.item()
        }
        return loss_all, tb_dict, disp_dict

class DirectionalPointDetector(nn.modules.Module):
    """Detector for point with direction."""

    def __init__(self, cfg):
        super(DirectionalPointDetector, self).__init__()
        self.cfg = cfg
        input_channel_size = cfg.input_channels
        depth_factor = cfg.depth_factor
        output_channel_size = cfg.output_channels
        self.feature_extractor = YetAnotherDarknet(input_channel_size, depth_factor)

        layers = []
        layers += define_detector_block(16 * depth_factor)
        layers += define_detector_block(16 * depth_factor)
        layers += [nn.Conv2d(32 * depth_factor, output_channel_size,
                             kernel_size=1, stride=1, padding=0, bias=False)]
        self.predict = nn.Sequential(*layers)
        
        self.loss_func = nn.MSELoss().cuda()

    def forward(self, data_dict):
        img = data_dict['image']
        prediction = self.predict(self.feature_extractor(img))
        # 4 represents that there are 4 value: confidence, shape, offset_x,
        # offset_y, whose range is between [0, 1].
        point_pred, angle_pred = torch.split(prediction, 4, dim=1)
        point_pred = torch.sigmoid(point_pred)
        angle_pred = torch.tanh(angle_pred)
        points_pred = torch.cat((point_pred, angle_pred), dim=1)

        data_dict['points_pred'] = points_pred
        return data_dict

    def get_targets(self, data_dict):
        marks_gt_batch = data_dict['marks']
        npoints = data_dict['npoints']
        batch_size = marks_gt_batch.size()[0]
        targets = torch.zeros(batch_size, self.cfg.output_channels,
                              self.cfg.feature_map_size,
                              self.cfg.feature_map_size).cuda()

        mask = torch.zeros_like(targets)
        mask[:, 0].fill_(1.)

        for batch_idx, marks_gt in enumerate(marks_gt_batch):
            n = npoints[batch_idx].long()
            for marking_point in marks_gt[:n]:
                x, y = marking_point[:2]
                col = math.floor(x * self.cfg.feature_map_size)
                row = math.floor(y * self.cfg.feature_map_size)
                # Confidence Regression
                targets[batch_idx, 0, row, col] = 1.
                # Makring Point Shape Regression
                targets[batch_idx, 1, row, col] = marking_point[3]  # shape
                # Offset Regression
                targets[batch_idx, 2, row, col] = x * 16 - col
                targets[batch_idx, 3, row, col] = y * 16 - row
                # Direction Regression
                direction = marking_point[2]
                targets[batch_idx, 4, row, col] = math.cos(direction)
                targets[batch_idx, 5, row, col] = math.sin(direction)

                mask[batch_idx, 1:6, row, col].fill_(1.)
        return targets, mask
    
    def get_training_loss(self, data_dict):
        points_pred = data_dict['points_pred']
        targets, mask = self.get_targets(data_dict)

        disp_dict = {}
        
        loss_all = self.loss_func(points_pred * mask, targets * mask)
        
        tb_dict = {
            'loss_all': loss_all.item()
        }
        return loss_all, tb_dict, disp_dict
    
    def post_processing(self, data_dict):
        ret_dicts = {}
        pred_dicts = {}
        
        points_pred = data_dict['points_pred']
        
        points_pred_batch = []
        slots_pred = []
        for b, marks in enumerate(points_pred):
            points_pred = get_predicted_directional_points(marks, self.cfg.point_thresh, self.cfg.boundary_thresh)
            points_pred_batch.append(points_pred)
         
            slots_infer = self.inference_slots(points_pred)
            slots_tmp = []
            for (i,j) in slots_infer:
                score = min(points_pred[i][0], points_pred[j][0])
                x1, y1 = points_pred[i][1][:2]
                x2, y2 = points_pred[j][1][:2]
                tmp = (score, np.array([x1, y1, x2, y2]))
                slots_tmp.append(tmp)

            slots_pred.append(slots_tmp)

        pred_dicts['points_pred'] = points_pred_batch
        pred_dicts['slots_pred'] = slots_pred
        return pred_dicts, ret_dicts

    def inference_slots(self, marking_points):
        """Inference slots based on marking points."""
        VSLOT_MIN_DIST = 0.044771278151623496
        VSLOT_MAX_DIST = 0.1099427457599304
        HSLOT_MIN_DIST = 0.15057789144568634
        HSLOT_MAX_DIST = 0.44449496544202816
        SLOT_SUPPRESSION_DOT_PRODUCT_THRESH = 0.8

        num_detected = len(marking_points)
        slots = []
        for i in range(num_detected - 1):
            for j in range(i + 1, num_detected):
                point_i = marking_points[i]
                point_j = marking_points[j]
                # Step 1: length filtration.
                distance = calc_point_squre_dist(point_i[1], point_j[1])
                if not (VSLOT_MIN_DIST <= distance <= VSLOT_MAX_DIST
                        or HSLOT_MIN_DIST <= distance <= HSLOT_MAX_DIST):
                    continue
                # Step 2: pass through filtration.
                if pass_through_third_point(marking_points, i, j, SLOT_SUPPRESSION_DOT_PRODUCT_THRESH):
                    continue
                result = pair_marking_points(point_i, point_j)
                if result == 1:
                    slots.append((i, j))
                elif result == -1:
                    slots.append((j, i))
        return slots
