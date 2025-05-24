"""
基础网络层组件

包含各种基础的卷积块、残差块等网络组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


NORM_LAYERS = {
    'bn': nn.BatchNorm2d, 
    'in': nn.InstanceNorm2d, 
    'ln': nn.LayerNorm
}


class ConvBlock(nn.Module):
    """基础卷积块，包含卷积、规范化和激活函数"""
    
    def __init__(self, in_chan: int, out_chan: int, kernel: int = 3, 
                 stride: int = 1, padding: int = 1, norm: str = 'bn', 
                 dropout: Optional[float] = None):
        super(ConvBlock, self).__init__()

        layers = [nn.Conv2d(in_chan, out_chan, kernel, stride, padding)]
        
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_chan))
        
        layers.append(nn.ReLU())

        if dropout:
            layers.append(nn.Dropout2d(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DeconvBlock(nn.Module):
    """转置卷积块，用于上采样"""
    
    def __init__(self, in_chan: int, out_chan: int, kernel: int = 2, 
                 stride: int = 2, padding: int = 0, output_padding: int = 0, 
                 norm: str = 'bn', dropout: Optional[float] = None):
        super(DeconvBlock, self).__init__()

        layers = [nn.ConvTranspose2d(in_chan, out_chan, kernel_size=kernel,
                                   stride=stride, padding=padding, 
                                   output_padding=output_padding)]
        
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_chan))

        layers.append(nn.ReLU())

        if dropout:
            layers.append(nn.Dropout2d(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvBlockTanh(nn.Module):
    """使用Tanh激活的卷积块，通常用于输出层"""
    
    def __init__(self, in_chan: int, out_chan: int, kernel: int = 3, 
                 stride: int = 1, padding: int = 1, norm: str = 'bn'):
        super(ConvBlockTanh, self).__init__()

        layers = [nn.Conv2d(in_chan, out_chan, kernel, stride, padding)]
        
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_chan))
        
        layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RMSNorm(nn.Module):
    """RMS标准化层"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class Block1D(nn.Module):
    """1D卷积块，用于扩散模型"""
    
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, 
                scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """ResNet残差块"""
    
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: Optional[int] = None, 
                 groups: int = 8):
        super().__init__()
        self.has_time_emb = time_emb_dim is not None
        
        if self.has_time_emb:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 2)
            )

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        scale_shift = None
        
        if self.has_time_emb and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.unsqueeze(-1)  # b c -> b c 1
            scale, shift = torch.chunk(time_emb, 2, dim=1)
            scale_shift = (scale, shift)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


def Upsample1D(dim: int, dim_out: Optional[int] = None) -> nn.Sequential:
    """1D上采样层"""
    if dim_out is None:
        dim_out = dim
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv1d(dim, dim_out, 3, padding=1)
    )


def Downsample1D(dim: int, dim_out: Optional[int] = None) -> nn.Conv1d:
    """1D下采样层"""
    if dim_out is None:
        dim_out = dim
    return nn.Conv1d(dim, dim_out, 4, 2, 1) 