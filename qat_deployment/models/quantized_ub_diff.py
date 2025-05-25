"""
量化版本的UB-Diff模型

专门为树莓派部署优化的量化模型，集成改进的量化策略
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

# 导入改进的量化功能
from .improved_quantization import (
    analyze_model_quantizability,
    apply_improved_quantization,
    create_quantization_report,
    ImprovedQuantizationConfig
)


class Conv1DWrapper(nn.Module):
    """1D卷积包装器，转换为2D卷积以获得更好的量化支持"""
    
    def __init__(self, conv1d_layer):
        super().__init__()
        # 保存原始参数
        in_channels = conv1d_layer.in_channels
        out_channels = conv1d_layer.out_channels
        kernel_size = conv1d_layer.kernel_size[0]
        stride = conv1d_layer.stride[0]
        padding = conv1d_layer.padding[0]
        dilation = conv1d_layer.dilation[0]
        groups = conv1d_layer.groups
        bias = conv1d_layer.bias is not None
        
        # 创建等效的2D卷积
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            dilation=(1, dilation),
            groups=groups,
            bias=bias
        )
        
        # 复制权重
        with torch.no_grad():
            # 权重: (out, in, k) -> (out, in, 1, k)
            self.conv2d.weight.copy_(conv1d_layer.weight.unsqueeze(2))
            if bias:
                self.conv2d.bias.copy_(conv1d_layer.bias)
    
    def forward(self, x):
        # 输入: (B, C, L) -> (B, C, 1, L)
        if x.dim() == 3:
            x = x.unsqueeze(2)
        
        # 2D卷积
        x = self.conv2d(x)
        
        # 输出: (B, C, 1, L) -> (B, C, L)
        return x.squeeze(2)


def convert_conv1d_to_conv2d(module):
    """递归转换模块中的所有1D卷积为2D卷积"""
    converted_count = 0
    
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv1d):
            # 转换1D卷积
            setattr(module, name, Conv1DWrapper(child))
            converted_count += 1
        else:
            # 递归处理子模块
            converted_count += convert_conv1d_to_conv2d(child)
    
    return converted_count


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
        
        # 为每个路径创建独立的量化层
        self.velocity_quant = quant.QuantStub()
        self.velocity_dequant = quant.DeQuantStub()
        self.seismic_quant = quant.QuantStub()
        self.seismic_dequant = quant.DeQuantStub()
        
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
        # 速度解码路径
        if self.quantize_velocity:
            x_v = self.velocity_quant(x)
            z_v = self.velocity_projector(x_v)
            velocity = self.velocity_decoder(z_v)
            velocity = self.velocity_dequant(velocity)
        else:
            z_v = self.velocity_projector(x)
            velocity = self.velocity_decoder(z_v)
        
        # 地震解码路径  
        if self.quantize_seismic:
            x_s = self.seismic_quant(x)
            z_s = self.seismic_projector(x_s)
            seismic = self.seismic_decoder(z_s)
            seismic = self.seismic_dequant(seismic)
        else:
            z_s = self.seismic_projector(x)
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
    
    def unfreeze_velocity_path(self) -> None:
        """解冻速度解码路径"""
        for param in self.velocity_projector.parameters():
            param.requires_grad = True
        for param in self.velocity_decoder.parameters():
            param.requires_grad = True
        print("速度解码路径已解冻")
    
    def unfreeze_seismic_path(self) -> None:
        """解冻地震解码路径"""
        for param in self.seismic_projector.parameters():
            param.requires_grad = True
        for param in self.seismic_decoder.parameters():
            param.requires_grad = True
        print("地震解码路径已解冻")


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
    
    def apply_improved_quantization(self, backend: str = 'qnnpack', 
                                  convert_conv1d: bool = True,
                                  use_aggressive_config: bool = True) -> Dict[str, Any]:
        """应用改进的量化策略
        
        Args:
            backend: 量化后端
            convert_conv1d: 是否转换1D卷积
            use_aggressive_config: 是否使用激进的量化配置
            
        Returns:
            量化分析报告
        """
        print("=== 应用改进的量化策略 ===")
        
        # 分析原始模型
        original_analysis = analyze_model_quantizability(self)
        print(f"原始模型统计:")
        print(f"  总模块数: {original_analysis['total_modules']}")
        print(f"  可量化模块数: {original_analysis['quantizable_modules']}")
        print(f"  量化率: {original_analysis['quantization_ratio']:.2%}")
        print(f"  1D卷积数: {original_analysis['conv1d_modules']}")
        
        # 转换1D卷积
        if convert_conv1d:
            print("\n转换1D卷积为2D卷积...")
            converted_diffusion = convert_conv1d_to_conv2d(self.diffusion)
            converted_unet = convert_conv1d_to_conv2d(self.unet)
            total_converted = converted_diffusion + converted_unet
            print(f"✓ 成功转换 {total_converted} 个1D卷积层")
        
        # 设置量化后端
        torch.backends.quantized.engine = backend
        
        # 应用量化配置
        if self.quantize_diffusion:
            if use_aggressive_config:
                print("使用激进的量化配置...")
                # 应用改进的量化策略
                self.diffusion = apply_improved_quantization(
                    self.diffusion, backend=backend, convert_conv1d=False  # 已经转换过了
                )
                self.unet = apply_improved_quantization(
                    self.unet, backend=backend, convert_conv1d=False
                )
            else:
                print("使用标准量化配置...")
                qconfig = quant.get_default_qat_qconfig(backend)
                self.diffusion.qconfig = qconfig
                self.unet.qconfig = qconfig
                
                # 准备QAT
                self.train()
                quant.prepare_qat(self.diffusion, inplace=True)
                quant.prepare_qat(self.unet, inplace=True)
        
        # 分析改进后的模型
        improved_analysis = analyze_model_quantizability(self)
        print(f"\n改进后模型统计:")
        print(f"  总模块数: {improved_analysis['total_modules']}")
        print(f"  可量化模块数: {improved_analysis['quantizable_modules']}")
        print(f"  量化率: {improved_analysis['quantization_ratio']:.2%}")
        print(f"  1D卷积数: {improved_analysis['conv1d_modules']}")
        
        # 计算改进效果
        if original_analysis['quantization_ratio'] > 0:
            improvement = improved_analysis['quantization_ratio'] / original_analysis['quantization_ratio']
            print(f"\n🚀 量化率改进: {improvement:.1f}x")
        
        return {
            'original': original_analysis,
            'improved': improved_analysis,
            'converted_conv1d': total_converted if convert_conv1d else 0
        }
    
    def prepare_for_qat_training(self, backend: str = 'qnnpack') -> None:
        """为QAT训练准备模型"""
        print("=== 准备QAT训练 ===")
        
        # 应用改进的量化策略
        self.apply_improved_quantization(
            backend=backend,
            convert_conv1d=True,
            use_aggressive_config=True
        )
        
        # 设置训练模式
        self.train()
        print("✓ 模型已准备好进行QAT训练")
    
    def convert_to_quantized(self) -> 'QuantizedUBDiff':
        """转换为量化模型（用于部署）"""
        print("=== 转换为量化模型 ===")
        
        # 设置评估模式
        self.eval()
        
        # 转换扩散模型
        if self.quantize_diffusion:
            if hasattr(self.diffusion, 'qconfig'):
                self.diffusion = quant.convert(self.diffusion, inplace=False)
                print("✓ 扩散模型已转换为量化版本")
            
            if hasattr(self.unet, 'qconfig'):
                self.unet = quant.convert(self.unet, inplace=False)
                print("✓ U-Net已转换为量化版本")
        
        # 转换解码器
        if self.quantize_decoder:
            if hasattr(self.decoder, 'qconfig'):
                self.decoder = quant.convert(self.decoder, inplace=False)
                print("✓ 解码器已转换为量化版本")
        
        print("🎉 量化转换完成")
        return self
    
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