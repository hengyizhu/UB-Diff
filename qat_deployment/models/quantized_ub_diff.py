"""
量化版本的UB-Diff模型

专门为树莓派部署优化的量化模型
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Optional, Tuple, Dict, Any
import sys
import os

# 添加项目路径到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.components import (
    VelocityDecoder, 
    SeismicDecoder,
    Unet1D,
    GaussianDiffusion1DDefault,
    cosine_beta_schedule
)
from model.components.decoder import LatentProjector


class QuantizedDecoder(nn.Module):
    """量化的解码器模块
    
    包含速度解码器和地震解码器的量化版本
    """
    
    def __init__(self, 
                 encoder_dim: int = 512,
                 velocity_channels: int = 1,
                 seismic_channels: int = 5,
                 velocity_latent_dim: int = 128,
                 seismic_latent_dim: int = 640,
                 seismic_h: int = 1000,
                 seismic_w: int = 70,
                 quantize_velocity: bool = True,
                 quantize_seismic: bool = True):
        """
        Args:
            encoder_dim: 编码器输出维度
            velocity_channels: 速度场通道数
            seismic_channels: 地震数据通道数
            velocity_latent_dim: 速度解码器潜在维度
            seismic_latent_dim: 地震解码器潜在维度
            seismic_h, seismic_w: 地震数据尺寸
            quantize_velocity: 是否量化速度解码器
            quantize_seismic: 是否量化地震解码器
        """
        super(QuantizedDecoder, self).__init__()
        
        self.quantize_velocity = quantize_velocity
        self.quantize_seismic = quantize_seismic
        
        # 输入量化层
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()
        
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
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: 潜在表示 (B, encoder_dim, 1, 1)
            
        Returns:
            (velocity, seismic): 解码后的速度场和地震数据
        """
        # 量化输入
        x = self.quant(x)
        
        # 速度解码路径
        if self.quantize_velocity:
            z_v = self.velocity_projector(x)
            velocity = self.velocity_decoder(z_v)
            velocity = self.dequant(velocity)
        else:
            # 如果不量化速度解码器，先反量化
            x_v = self.dequant(x)
            z_v = self.velocity_projector(x_v)
            velocity = self.velocity_decoder(z_v)
        
        # 地震解码路径
        if self.quantize_seismic:
            z_s = self.seismic_projector(x)
            seismic = self.seismic_decoder(z_s)
            seismic = self.dequant(seismic)
        else:
            # 如果不量化地震解码器，先反量化
            x_s = self.dequant(x)
            z_s = self.seismic_projector(x_s)
            seismic = self.seismic_decoder(z_s)
        
        return velocity, seismic
    
    def freeze_velocity_path(self) -> None:
        """冻结速度解码路径"""
        for param in self.velocity_projector.parameters():
            param.requires_grad = False
        for param in self.velocity_decoder.parameters():
            param.requires_grad = False
        print("速度解码路径已冻结")
    
    def freeze_seismic_path(self) -> None:
        """冻结地震解码路径"""
        for param in self.seismic_projector.parameters():
            param.requires_grad = False
        for param in self.seismic_decoder.parameters():
            param.requires_grad = False
        print("地震解码路径已冻结")


class QuantizedUBDiff(nn.Module):
    """量化版本的UB-Diff模型
    
    用于树莓派部署的纯生成模型（不包含编码器）
    """
    
    def __init__(self,
                 encoder_dim: int = 512,
                 velocity_channels: int = 1,
                 seismic_channels: int = 5,
                 dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
                 time_steps: int = 256,
                 time_scale: int = 1,
                 objective: str = 'pred_v',
                 quantize_diffusion: bool = True,
                 quantize_decoder: bool = True):
        """
        Args:
            encoder_dim: 潜在空间维度
            velocity_channels: 速度场通道数
            seismic_channels: 地震数据通道数
            dim_mults: U-Net维度倍数
            time_steps: 扩散时间步数
            time_scale: 时间缩放因子
            objective: 扩散目标函数
            quantize_diffusion: 是否量化扩散模型
            quantize_decoder: 是否量化解码器
        """
        super(QuantizedUBDiff, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.quantize_diffusion = quantize_diffusion
        self.quantize_decoder = quantize_decoder
        
        # 量化/反量化层
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()
        
        # 1D U-Net用于扩散
        self.unet = Unet1D(
            dim=encoder_dim,
            channels=1,
            dim_mults=dim_mults
        )
        
        # 扩散过程
        betas = cosine_beta_schedule(timesteps=time_steps)
        self.diffusion = GaussianDiffusion1DDefault(
            model=self.unet,
            seq_length=encoder_dim,
            betas=betas,
            time_scale=time_scale,
            objective=objective,
            use_wandb=False
        )
        
        # 量化解码器
        self.decoder = QuantizedDecoder(
            encoder_dim=encoder_dim,
            velocity_channels=velocity_channels,
            seismic_channels=seismic_channels,
            quantize_velocity=quantize_decoder,
            quantize_seismic=quantize_decoder
        )
        
    def sample_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """从扩散过程采样潜在表示
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            采样的潜在表示 (B, encoder_dim)
        """
        # 从扩散模型采样
        z = self.diffusion.sample(batch_size)  # (B, 1, encoder_dim)
        z = z.squeeze(1)  # (B, encoder_dim)
        return z
    
    def generate(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成新的速度场和地震数据
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            (velocity, seismic): 生成的速度场和地震数据
        """
        # 采样潜在表示
        z = self.sample_latent(batch_size, device)
        
        # 重塑为解码器期望的格式
        z = z.view(batch_size, -1, 1, 1)
        
        # 解码
        velocity, seismic = self.decoder(z)
        
        return velocity, seismic
    
    def forward(self, z: Optional[torch.Tensor] = None, 
                batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播（用于导出）
        
        Args:
            z: 可选的潜在表示输入
            batch_size: 如果z为None，生成的批次大小
            
        Returns:
            (velocity, seismic): 生成的数据
        """
        if z is None:
            if batch_size is None:
                raise ValueError("必须提供z或batch_size")
            device = next(self.parameters()).device
            
            # 使用扩散模型采样
            z = self.diffusion.sample(batch_size)  # 使用内置采样方法
            z = z.squeeze(1) if z.dim() > 2 else z
        
        # 重塑并解码
        z = z.view(z.shape[0], -1, 1, 1)
        velocity, seismic = self.decoder(z)
        
        return velocity, seismic
    
    def load_pretrained_weights(self, checkpoint_path: str) -> None:
        """从预训练模型加载权重
        
        Args:
            checkpoint_path: 检查点路径
        """
        print(f"加载预训练权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 处理不同的检查点格式
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 加载扩散模型权重
        diffusion_dict = {}
        for key, value in state_dict.items():
            if key.startswith('diffusion.') or key.startswith('unet.'):
                diffusion_dict[key] = value
        
        # 加载解码器权重
        decoder_dict = {}
        for key, value in state_dict.items():
            if 'decoder' in key or 'projector' in key:
                # 调整键名以匹配新结构
                new_key = key.replace('dual_decoder.', 'decoder.')
                decoder_dict[new_key] = value
        
        # 应用权重
        self.load_state_dict({**diffusion_dict, **decoder_dict}, strict=False)
        print("预训练权重加载完成")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'encoder_dim': self.encoder_dim,
            'quantize_diffusion': self.quantize_diffusion,
            'quantize_decoder': self.quantize_decoder
        }


def create_quantized_model_for_deployment(checkpoint_path: str,
                                        quantize_all: bool = True) -> QuantizedUBDiff:
    """创建用于部署的量化模型
    
    Args:
        checkpoint_path: 预训练模型路径
        quantize_all: 是否量化所有组件
        
    Returns:
        准备部署的量化模型
    """
    # 创建模型
    model = QuantizedUBDiff(
        encoder_dim=512,
        velocity_channels=1,
        seismic_channels=5,
        dim_mults=(1, 2, 4, 8),
        time_steps=256,
        quantize_diffusion=quantize_all,
        quantize_decoder=quantize_all
    )
    
    # 加载预训练权重
    model.load_pretrained_weights(checkpoint_path)
    
    return model 