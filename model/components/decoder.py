"""
解码器模块

包含速度解码器和地震解码器的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .networks import ConvBlock, DeconvBlock, ConvBlockTanh
from .attention import VisionTransformer


class VelocityDecoder(nn.Module):
    """速度场解码器
    
    将潜在表示解码为速度场图像
    """
    
    def __init__(self, latent_dim: int, out_channels: int = 1,
                 dim1: int = 32, dim2: int = 64, dim3: int = 128, 
                 dim4: int = 256, dim5: int = 512):
        """
        Args:
            latent_dim: 潜在空间维度 
            out_channels: 输出通道数
            dim1-dim5: 各层特征维度
        """
        super(VelocityDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        
        # 从1x1逐步上采样到80x80
        self.blocks = nn.ModuleList([
            DeconvBlock(latent_dim, dim5, kernel=5),                    # 1x1 -> 5x5
            ConvBlock(dim5, dim5, kernel=3, stride=1),                  # 5x5
            DeconvBlock(dim5, dim4, kernel=4, stride=2, padding=1),     # 5x5 -> 10x10  
            ConvBlock(dim4, dim4, kernel=3, stride=1),                  # 10x10
            DeconvBlock(dim4, dim3, kernel=4, stride=2, padding=1),     # 10x10 -> 20x20
            ConvBlock(dim3, dim3, kernel=3, stride=1),                  # 20x20
            DeconvBlock(dim3, dim2, kernel=4, stride=2, padding=1),     # 20x20 -> 40x40
            ConvBlock(dim2, dim2, kernel=3, stride=1),                  # 40x40
            DeconvBlock(dim2, dim1, kernel=4, stride=2, padding=1),     # 40x40 -> 80x80
            ConvBlock(dim1, dim1, kernel=3, stride=1),                  # 80x80
        ])
        
        # 最终输出层，从80x80裁剪到70x70
        self.output_layer = ConvBlockTanh(dim1, out_channels, kernel=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 潜在表示，形状为 (B, latent_dim, 1, 1)
            
        Returns:
            重构的速度场，形状为 (B, out_channels, 70, 70)
        """
        # 通过所有解码层
        for block in self.blocks:
            x = block(x)
        
        # 从80x80裁剪到70x70
        x = F.pad(x, [-5, -5, -5, -5], mode='constant', value=0.0)
        
        # 输出层
        x = self.output_layer(x)
        
        return x

    def get_feature_shapes(self) -> dict:
        """返回各层的特征形状"""
        return {
            'input': (self.latent_dim, 1, 1),
            'after_block_0': (512, 5, 5),
            'after_block_2': (256, 10, 10), 
            'after_block_4': (128, 20, 20),
            'after_block_6': (64, 40, 40),
            'after_block_8': (32, 80, 80),
            'output': (self.out_channels, 70, 70)
        }


class SeismicDecoder(nn.Module):
    """地震数据解码器
    
    使用Vision Transformer将潜在表示解码为地震数据
    """
    
    def __init__(self, latent_dim: int, out_channels: int = 5, 
                 origin_h: int = 1000, origin_w: int = 70,
                 depth: int = 2, vit_latent_dim: int = 64, 
                 latent_h: int = 1, latent_w: int = 1):
        """
        Args:
            latent_dim: 输入潜在维度
            out_channels: 输出通道数  
            origin_h, origin_w: 输出的高度和宽度
            depth: Transformer深度
            vit_latent_dim: ViT内部特征维度
            latent_h, latent_w: 潜在空间的高度和宽度
        """
        super(SeismicDecoder, self).__init__()
        
        self.out_channels = out_channels
        self.origin_h = origin_h
        self.origin_w = origin_w
        self.vit_latent_dim = vit_latent_dim
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.depth = depth

        # 如果有深度，使用Vision Transformer
        if self.depth > 0:
            self.decoder = VisionTransformer(
                img_size=(self.latent_h, self.latent_w), 
                patch_size=1, 
                in_chans=latent_dim,
                depth=self.depth, 
                n_heads=8, 
                mlp_ratio=self.out_channels, 
                embed_dim=latent_dim
            )
        
        # 最终的MLP层，输出到目标尺寸
        self.mlp = nn.Linear(
            latent_dim, 
            self.origin_h * self.origin_w * self.out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 潜在表示，形状为 (B, latent_dim, latent_h, latent_w)
            
        Returns:
            地震数据，形状为 (B, out_channels, origin_h, origin_w)
        """
        batch_size = x.shape[0]
        
        # 如果有Transformer层
        if self.depth > 0:
            # ViT expects (B, C, H, W) input
            x = self.decoder(x, return_features=True)
            # 取CLS token的特征 
            x = x[:, 0, :]  # (B, latent_dim)
        else:
            # 直接展平
            x = x.view(batch_size, -1)
        
        # 通过MLP输出到目标维度
        x = self.mlp(x)
        
        # 重塑为目标形状
        x = x.view(batch_size, self.out_channels, self.origin_h, self.origin_w)
        
        return x

    def get_output_shape(self) -> tuple:
        """返回输出形状"""
        return (self.out_channels, self.origin_h, self.origin_w)


class LatentProjector(nn.Module):
    """潜在空间投影器
    
    将编码器输出投影到解码器所需的维度
    """
    
    def __init__(self, encoder_dim: int, decoder_dim: int, 
                 use_batch_norm: bool = True):
        """
        Args:
            encoder_dim: 编码器输出维度
            decoder_dim: 解码器输入维度  
            use_batch_norm: 是否使用批标准化
        """
        super(LatentProjector, self).__init__()
        
        self.fc = nn.Linear(encoder_dim, decoder_dim)
        
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm2d(decoder_dim)
        else:
            self.batch_norm = None
            
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 编码器输出，形状为 (B, encoder_dim) 或 (B, encoder_dim, 1, 1)
            
        Returns:
            投影后的特征，形状为 (B, decoder_dim, 1, 1)
        """
        # 展平输入
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        
        # 线性投影
        x = self.fc(x)
        
        # 重塑为4D张量
        x = x.view(x.shape[0], -1, 1, 1)
        
        # 批标准化
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        # 激活函数
        x = self.activation(x)
        
        return x


class DualDecoder(nn.Module):
    """双解码器架构
    
    同时包含速度解码器和地震解码器
    """
    
    def __init__(self, encoder_dim: int, 
                 velocity_channels: int = 1, seismic_channels: int = 5,
                 velocity_latent_dim: int = 128, seismic_latent_dim: int = 640,
                 seismic_h: int = 1000, seismic_w: int = 70):
        """
        Args:
            encoder_dim: 编码器输出维度
            velocity_channels: 速度场通道数
            seismic_channels: 地震数据通道数
            velocity_latent_dim: 速度解码器潜在维度
            seismic_latent_dim: 地震解码器潜在维度
            seismic_h, seismic_w: 地震数据的高度和宽度
        """
        super(DualDecoder, self).__init__()
        
        # 潜在空间投影器
        self.velocity_projector = LatentProjector(encoder_dim, velocity_latent_dim)
        self.seismic_projector = LatentProjector(encoder_dim, seismic_latent_dim)
        
        # 解码器
        self.velocity_decoder = VelocityDecoder(
            latent_dim=velocity_latent_dim,
            out_channels=velocity_channels
        )
        self.seismic_decoder = SeismicDecoder(
            latent_dim=seismic_latent_dim,
            out_channels=seismic_channels,
            origin_h=seismic_h,
            origin_w=seismic_w
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 编码器输出，形状为 (B, encoder_dim, 1, 1)
            
        Returns:
            (velocity, seismic): 重构的速度场和地震数据
        """
        # 投影到各自的潜在空间
        z_v = self.velocity_projector(x)
        z_s = self.seismic_projector(x)
        
        # 解码
        velocity = self.velocity_decoder(z_v)
        seismic = self.seismic_decoder(z_s)
        
        return velocity, seismic

    def freeze_velocity_decoder(self) -> None:
        """冻结速度解码器"""
        self.velocity_decoder.eval()
        self.velocity_projector.eval()
        
        for param in self.velocity_decoder.parameters():
            param.requires_grad = False
        for param in self.velocity_projector.parameters():
            param.requires_grad = False
        
        print("速度解码器已冻结")

    def freeze_seismic_decoder(self) -> None:
        """冻结地震解码器"""
        self.seismic_decoder.eval()
        self.seismic_projector.eval()
        
        for param in self.seismic_decoder.parameters():
            param.requires_grad = False
        for param in self.seismic_projector.parameters():
            param.requires_grad = False
        
        print("地震解码器已冻结") 