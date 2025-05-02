import torch
import torch.nn.functional as F
from collections import OrderedDict
from scipy.ndimage import gaussian_filter
from arch.base import ModelBase
from torchvision import models

__all__ = ['SPADE']

class SPADE(ModelBase):
    def __init__(self, config):
        super(SPADE, self).__init__(config)
        self.config = config

        # 关键修改1：改用论文指定的Wide-ResNet50×2
        if self.config['net'] == 'wide_resnet50_2':
            self.net = models.wide_resnet50_2(pretrained=True, progress=True).to(self.device)
            
        self.features = []
        self.get_layer_features()
    
    def get_layer_features(self):
        def hook_t(module, input, output):
            self.features.append(output)
        
        # 注册layer1-3的最后一个残差块
        self.net.layer1[-1].register_forward_hook(hook_t)
        self.net.layer2[-1].register_forward_hook(hook_t)
        self.net.layer3[-1].register_forward_hook(hook_t)
        self.net.avgpool.register_forward_hook(hook_t)
    
    @staticmethod
    def cal_distance_matrix(x, y):

        return torch.cdist(x.unsqueeze(1), y.unsqueeze(0)).squeeze()

    def train_model(self, train_loader, task_id, inf=''):
        self.net.eval()
        self.train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

        for _ in range(self.config['num_epochs']):
            for batch in train_loader:
                img = batch['img'].to(self.device)
                self.features.clear()
                with torch.no_grad():
                    _ = self.net(img)
                
                for k,v in zip(self.train_outputs.keys(), self.features):
                    self.train_outputs[k].append(v.cpu())
                self.features = []

        for k in self.train_outputs:
            self.train_outputs[k] = torch.cat(self.train_outputs[k], 0)

    def prediction(self, valid_loader, task_id):
        self.net.eval()
        self.clear_all_list()
        self.test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

        # 关键修改2：动态尺寸处理
        for batch in valid_loader:
            img = batch['img'].to(self.device)
            mask = batch['mask'].to(self.device)
            label = batch['label'].to(self.device)
            mask[mask>=0.5] = 1
            mask[mask<0.5] = 0
            
            # 关键修复：添加以下三行
            self.img_gt_list.append(label.cpu().detach().numpy()[0])
            self.pixel_gt_list.append(mask.cpu().detach().numpy()[0,0])
            self.img_path_list.append(batch['img_src'])  # 确保路径被记录
            if img.shape[-2:] != (224, 224):
                img = F.interpolate(img, size=(224, 224), mode='bilinear')
            
            with torch.no_grad():
                self.features.clear()
                _ = self.net(img)
                
            for k,v in zip(self.test_outputs.keys(), self.features):
                self.test_outputs[k].append(v.cpu())
            self.features = []

        for k in self.test_outputs:
            self.test_outputs[k] = torch.cat(self.test_outputs[k], 0)

        # 关键修改3：优化K近邻计算
        test_feats = torch.flatten(self.test_outputs['avgpool'], 1)
        train_feats = torch.flatten(self.train_outputs['avgpool'], 1)
        dist_matrix = self.cal_distance_matrix(test_feats, train_feats)
        
        topk_values, topk_indexes = torch.topk(
            dist_matrix, 
            k=self.config['_top_k'],  # 确保配置K=50
            dim=1, 
            largest=False
        )

        self.img_pred_list = torch.mean(topk_values, 1).cpu().numpy()

        # 关键修改4：向量化像素级距离计算
        for t_idx in range(self.test_outputs['avgpool'].shape[0]):
            score_maps = []
            for layer_name in ['layer1', 'layer2', 'layer3']:
                test_feat = self.test_outputs[layer_name][t_idx]  # [C,H,W]
                train_feats_selected = self.train_outputs[layer_name][topk_indexes[t_idx]]  # [K,C,H,W]
                
                # 向量化计算
                C, H, W = test_feat.shape
                test_flat = test_feat.view(C, -1).T  # [H*W, C]
                train_flat = train_feats_selected.permute(0,2,3,1).reshape(-1, C)  # [K*H*W, C]
                
                dists = torch.cdist(test_flat.unsqueeze(0), train_flat.unsqueeze(0))  # [1, H*W, K*H*W]
                min_dists = torch.min(dists, dim=2)[0]  # [1, H*W]
                score_map = min_dists.view(1, 1, H, W)
                
                # 动态插值到原始尺寸
                score_map = F.interpolate(
                    score_map, 
                    size=img.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                score_maps.append(score_map)
            
            # 关键修改5：多尺度特征融合
            final_score = torch.mean(torch.cat(score_maps, 1), dim=1)
            final_score = gaussian_filter(final_score.squeeze().cpu().numpy(), sigma=4)
            self.pixel_pred_list.append(final_score)
