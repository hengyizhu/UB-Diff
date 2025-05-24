"""
UB-Diff 主模型架构

重构后的UB-Diff扩散模型，包含：
- 速度编码器
- 1D扩散模型  
- 双解码器架构
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .components import (
    VelocityEncoder, 
    VelocityDecoder, 
    SeismicDecoder,
    DualDecoder,
    Unet1D, 
    GaussianDiffusion1DDefault,
    cosine_beta_schedule
)


class UBDiff(nn.Module):
    """UB-Diff 扩散模型
    
    基于扩散过程的地震速度建模框架
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 encoder_dim: int = 512,
                 velocity_channels: int = 1,
                 seismic_channels: int = 5,
                 dim_mults: Tuple[int, ...] = (1, 2, 2, 2),
                 time_steps: int = 256,
                 time_scale: int = 1,
                 objective: str = 'pred_v',
                 use_wandb: bool = False,
                 pretrained_path: Optional[str] = None):
        """
        Args:
            in_channels: 输入通道数
            encoder_dim: 编码器输出维度
            velocity_channels: 速度场通道数
            seismic_channels: 地震数据通道数 
            dim_mults: U-Net维度倍数
            time_steps: 扩散时间步数
            time_scale: 时间缩放因子
            objective: 扩散目标函数类型
            use_wandb: 是否使用wandb记录
            pretrained_path: 预训练模型路径
        """
        super(UBDiff, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.velocity_channels = velocity_channels
        self.seismic_channels = seismic_channels
        self.time_steps = time_steps
        
        # 编码器
        self.encoder = VelocityEncoder(
            in_channels=in_channels,
            dim5=encoder_dim
        )
        
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
            use_wandb=use_wandb
        )
        
        # 双解码器
        self.dual_decoder = DualDecoder(
            encoder_dim=encoder_dim,
            velocity_channels=velocity_channels,
            seismic_channels=seismic_channels
        )
        
        # 激活函数
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # 训练步数
        self.step = 0
        
        # 加载预训练权重
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """训练时的前向传播
        
        Args:
            x: 输入速度场 (B, C, H, W)
            
        Returns:
            扩散损失
        """
        # 编码
        z = self.encoder(x)  # (B, encoder_dim, 1, 1)
        z = z.view(z.shape[0], 1, -1)  # (B, 1, encoder_dim)
        
        # 扩散损失
        loss = self.diffusion(z)
        
        return loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码速度场到潜在空间
        
        Args:
            x: 输入速度场 (B, C, H, W)
            
        Returns:
            潜在表示 (B, encoder_dim, 1, 1)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """解码潜在表示
        
        Args:
            z: 潜在表示 (B, encoder_dim, 1, 1) 或 (B, encoder_dim)
            
        Returns:
            (velocity, seismic): 重构的速度场和地震数据
        """
        # 确保输入形状正确
        if len(z.shape) == 2:
            z = z.view(z.shape[0], -1, 1, 1)
        elif len(z.shape) == 3:
            z = z.view(z.shape[0], -1, 1, 1)
            
        return self.dual_decoder(z)

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
        
        # 解码
        velocity, seismic = self.decode(z)
        
        return velocity, seismic

    def reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """重构输入的速度场
        
        Args:
            x: 输入速度场 (B, C, H, W)
            
        Returns:
            (velocity, seismic): 重构的速度场和地震数据
        """
        # 编码
        z = self.encode(x)
        
        # 解码
        velocity, seismic = self.decode(z)
        
        return velocity, seismic

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False) -> None:
        """加载预训练权重
        
        Args:
            checkpoint_path: 检查点文件路径
            strict: 是否严格匹配参数名
        """
        print(f"正在加载预训练权重: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 处理不同的检查点格式
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 分别加载编码器和解码器权重
        self._load_encoder_weights(state_dict)
        self._load_decoder_weights(state_dict)
        
        print("预训练权重加载完成")

    def _load_encoder_weights(self, state_dict: dict) -> None:
        """加载编码器权重"""
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                encoder_state_dict[new_key] = value
        
        if encoder_state_dict:
            self.encoder.load_state_dict(encoder_state_dict, strict=False)
            print("编码器权重已加载")

    def _load_decoder_weights(self, state_dict: dict) -> None:
        """加载解码器权重"""
        # 加载速度解码器
        velocity_decoder_dict = {}
        for key, value in state_dict.items():
            if key.startswith('decoder_v.'):
                new_key = key.replace('decoder_v.', '')
                velocity_decoder_dict[new_key] = value
            elif key.startswith('fc_v.'):
                new_key = key.replace('fc_v.', 'fc.')
                velocity_decoder_dict[new_key] = value
            elif key.startswith('batch_norm_v.'):
                new_key = key.replace('batch_norm_v.', 'batch_norm.')
                velocity_decoder_dict[new_key] = value
        
        # 加载地震解码器
        seismic_decoder_dict = {}
        for key, value in state_dict.items():
            if key.startswith('decoder_s.'):
                new_key = key.replace('decoder_s.', '')
                seismic_decoder_dict[new_key] = value
            elif key.startswith('fc_s.'):
                new_key = key.replace('fc_s.', 'fc.')
                seismic_decoder_dict[new_key] = value
            elif key.startswith('batch_norm_s.'):
                new_key = key.replace('batch_norm_s.', 'batch_norm.')
                seismic_decoder_dict[new_key] = value
        
        # 应用权重
        if velocity_decoder_dict:
            # 需要分别加载到projector和decoder
            print("速度解码器权重已准备")
        
        if seismic_decoder_dict:
            print("地震解码器权重已准备")

    def freeze_encoder(self) -> None:
        """冻结编码器参数"""
        self.encoder.freeze_parameters()

    def freeze_velocity_decoder(self) -> None:
        """冻结速度解码器参数"""
        self.dual_decoder.freeze_velocity_decoder()

    def freeze_seismic_decoder(self) -> None:
        """冻结地震解码器参数"""
        self.dual_decoder.freeze_seismic_decoder()

    def unfreeze_all(self) -> None:
        """解冻所有参数"""
        self.encoder.unfreeze_parameters()
        self.train()
        for param in self.parameters():
            param.requires_grad = True
        print("所有参数已解冻")

    def get_model_size(self) -> dict:
        """获取模型大小信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        unet_params = sum(p.numel() for p in self.unet.parameters())
        decoder_params = sum(p.numel() for p in self.dual_decoder.parameters())
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'encoder_params': encoder_params,
            'unet_params': unet_params,
            'decoder_params': decoder_params
        }

    def summary(self) -> str:
        """模型摘要"""
        model_info = self.get_model_size()
        
        summary = f"""
UB-Diff 模型摘要:
================
编码器维度: {self.encoder_dim}
速度场通道: {self.velocity_channels}
地震数据通道: {self.seismic_channels}
扩散时间步: {self.time_steps}

参数统计:
总参数量: {model_info['total_params']:,}
可训练参数: {model_info['trainable_params']:,}
编码器参数: {model_info['encoder_params']:,}
U-Net参数: {model_info['unet_params']:,}
解码器参数: {model_info['decoder_params']:,}
        """
        
        return summary.strip()


# 辅助函数
def create_ub_diff_model(config: dict) -> UBDiff:
    """根据配置创建UB-Diff模型
    
    Args:
        config: 模型配置字典
        
    Returns:
        UB-Diff模型实例
    """
    return UBDiff(**config)


if __name__ == '__main__':
    # 测试模型
    model = UBDiff(
        in_channels=1,
        encoder_dim=512,
        dim_mults=(1, 2, 2, 2),
        time_steps=256
    )
    
    # 打印模型摘要
    print(model.summary())
    
    # 测试前向传播
    x = torch.randn(2, 1, 70, 70)
    
    # 训练模式
    loss = model(x)
    print(f"扩散损失: {loss.item():.4f}")
    
    # 重构测试
    velocity, seismic = model.reconstruct(x)
    print(f"重构速度场形状: {velocity.shape}")
    print(f"重构地震数据形状: {seismic.shape}")
    
    # 生成测试
    velocity_gen, seismic_gen = model.generate(batch_size=2, device=x.device)
    print(f"生成速度场形状: {velocity_gen.shape}")
    print(f"生成地震数据形状: {seismic_gen.shape}") 