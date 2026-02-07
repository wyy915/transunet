# 图像分割与路径规划程序
# 训练时：只返回分割结果
# 推理时：可选择返回分割结果+规划路径
import torch.nn as nn
from .vit_seg_modeling import VisionTransformer as ViT_seg
from .path.path_planner import PathPlanner


class TransUNetWithPathPlanning(nn.Module):
    def __init__(self, config, img_size=224, num_classes=9):
        super().__init__()
        self.transunet = ViT_seg(config, img_size=img_size, num_classes=num_classes)
        self.path_planner = PathPlanner()

    def forward(self, x, return_paths=False):
        segmentation = self.transunet(x)

        if not self.training and return_paths:
            paths = self.path_planner.plan_paths(segmentation, x)
            return segmentation, paths

        return segmentation
