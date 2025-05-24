"""
速度编码器模块

负责将输入的速度场数据编码为低维潜在表示
"""

import torch
import torch.nn as nn
from typing import Optional
from .networks import ConvBlock, ConvBlockTanh


class VelocityEncoder(nn.Module):
    """速度场编码器
    
    将输入的速度场数据 (H, W) 编码为固定维度的潜在向量
    """
    
    def __init__(self, in_channels: int, 
                 dim1: int = 32, dim2: int = 64, dim3: int = 128, 
                 dim4: int = 256, dim5: int = 512):
        """
        Args:
            in_channels: 输入通道数
            dim1-dim5: 各层特征维度
        """
        super(VelocityEncoder, self).__init__()
        
        self.dim5 = dim5
        
        # 第一阶段：特征提取 (70x70 -> 16x16)
        self.stage1 = nn.ModuleList([
            ConvBlock(in_channels, dim1, kernel=1, stride=1, padding=0),  # 70x70
            ConvBlock(dim1, dim1, kernel=3, stride=1, padding=1),         # 70x70
            ConvBlock(dim1, dim2, kernel=3, stride=2, padding=1),         # 35x35
            ConvBlock(dim2, dim2, kernel=1, stride=1, padding=0),         # 35x35
            ConvBlock(dim2, dim2, kernel=3, stride=1, padding=1),         # 35x35
            ConvBlock(dim2, dim2, kernel=1, stride=1, padding=0),         # 35x35
            ConvBlock(dim2, dim3, kernel=3, stride=2, padding=1),         # 18x18
            ConvBlock(dim3, dim3, kernel=1, stride=1, padding=0),         # 18x18
            ConvBlock(dim3, dim3, kernel=3, stride=1, padding=0),         # 16x16
            ConvBlock(dim3, dim3, kernel=1, stride=1, padding=0),         # 16x16
        ])
        
        # 第二阶段：降维到潜在空间 (16x16 -> 1x1)
        self.stage2 = nn.ModuleList([
            ConvBlock(dim3, dim4, kernel=3, stride=2, padding=1),         # 8x8
            ConvBlock(dim4, dim4, kernel=1, stride=1, padding=0),         # 8x8
            ConvBlock(dim4, dim4, kernel=3, stride=2, padding=1),         # 4x4
            ConvBlock(dim4, dim4, kernel=3, stride=2, padding=1),         # 2x2
            ConvBlockTanh(dim4, dim5, kernel=3, stride=2, padding=1),     # 1x1
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            
        Returns:
            潜在表示，形状为 (B, dim5, 1, 1)
        """
        # 第一阶段特征提取
        for module in self.stage1:
            x = module(x)
        
        # 第二阶段降维
        for module in self.stage2:
            x = module(x)
        
        return x

    def forward_stage1(self, x: torch.Tensor) -> torch.Tensor:
        """仅执行第一阶段的前向传播
        
        用于需要中间特征的场景
        """
        for module in self.stage1:
            x = module(x)
        return x

    def forward_stage2(self, x: torch.Tensor) -> torch.Tensor:
        """仅执行第二阶段的前向传播
        
        Args:
            x: 第一阶段的输出，需要reshape为 (B, dim3, 16, 16)
        """
        # 确保输入形状正确
        if len(x.shape) == 2:  # 如果是扁平化的
            x = x.view(x.shape[0], -1, 16, 16)
        
        for module in self.stage2:
            x = module(x)
        return x

    def get_feature_dims(self) -> tuple:
        """返回各阶段的特征维度"""
        return {
            'stage1_output': (128, 16, 16),  # dim3, 16, 16
            'stage2_output': (self.dim5, 1, 1),  # dim5, 1, 1
        }

    def load_pretrained_weights(self, checkpoint_path: str, 
                              strict: bool = True) -> None:
        """加载预训练权重
        
        Args:
            checkpoint_path: 检查点文件路径
            strict: 是否严格匹配参数名
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理不同的检查点格式
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 提取编码器相关的权重
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                encoder_state_dict[new_key] = value
        
        self.load_state_dict(encoder_state_dict, strict=strict)
        print(f"已加载预训练编码器权重: {checkpoint_path}")

    def freeze_parameters(self) -> None:
        """冻结编码器参数，用于微调场景"""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        print("编码器参数已冻结")

    def unfreeze_parameters(self) -> None:
        """解冻编码器参数"""
        self.train()
        for param in self.parameters():
            param.requires_grad = True
        print("编码器参数已解冻") 