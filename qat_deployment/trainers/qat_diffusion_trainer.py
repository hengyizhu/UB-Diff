"""
扩散模型的量化感知训练器

训练量化的扩散模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import wandb
from tqdm import tqdm
import os
import sys

# 添加项目路径到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.data import SeismicVelocityDataset


class QATDiffusionTrainer:
    """扩散模型的量化感知训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 encoder_model: Optional[nn.Module] = None,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 gradient_clip: float = 1.0,
                 use_wandb: bool = False):
        """
        Args:
            model: 量化的UB-Diff模型
            encoder_model: 原始编码器（用于生成真实潜在表示）
            device: 训练设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            gradient_clip: 梯度裁剪值
            use_wandb: 是否使用WandB记录
        """
        self.model = model.to(device)
        self.encoder = encoder_model
        if self.encoder is not None:
            self.encoder = self.encoder.to(device)
            self.encoder.eval()
        
        self.device = device
        self.use_wandb = use_wandb
        self.gradient_clip = gradient_clip
        
        # 冻结解码器，只训练扩散模型
        self.model.decoder.eval()
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        
        # 优化器 - 只优化扩散模型参数
        diffusion_params = []
        for name, param in self.model.named_parameters():
            if 'diffusion' in name or 'unet' in name:
                diffusion_params.append(param)
        
        self.optimizer = optim.Adam(
            diffusion_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.global_step = 0
        
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              epochs: int = 100,
              checkpoint_dir: str = './checkpoints/qat_diffusion',
              save_every: int = 10,
              validate_every: int = 5):
        """训练扩散模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            checkpoint_dir: 检查点保存目录
            save_every: 保存检查点的频率
            validate_every: 验证的频率
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print("开始训练扩散模型（QAT）...")
        print(f"可训练参数数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 训练阶段
            train_loss = self._train_epoch(train_loader)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
            
            # 验证阶段
            if val_loader is not None and (epoch + 1) % validate_every == 0:
                val_loss = self._validate(val_loader)
                print(f"Validation Loss: {val_loss:.6f}")
                
                # 保存最佳模型
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_checkpoint(
                        os.path.join(checkpoint_dir, 'best_diffusion.pt'),
                        {'val_loss': val_loss}
                    )
            
            # 学习率调度
            self.scheduler.step()
            
            # 定期保存检查点
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, f'diffusion_epoch_{epoch+1}.pt'),
                    {'epoch': epoch+1}
                )
            
            # WandB记录
            if self.use_wandb:
                log_dict = {
                    'diffusion/train_loss': train_loss,
                    'diffusion/lr': self.scheduler.get_last_lr()[0],
                    'epoch': epoch
                }
                if val_loader is not None and (epoch + 1) % validate_every == 0:
                    log_dict['diffusion/val_loss'] = val_loss
                wandb.log(log_dict)
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        self.model.decoder.eval()  # 保持解码器在评估模式
        total_loss = 0.0
        
        for batch_idx, (seismic, velocity) in enumerate(tqdm(train_loader, desc="Training Diffusion")):
            velocity = velocity.to(self.device)
            
            # 获取真实潜在表示
            with torch.no_grad():
                if self.encoder is not None:
                    z = self.encoder(velocity)  # (B, encoder_dim, 1, 1)
                    z = z.view(z.shape[0], 1, -1)  # (B, 1, encoder_dim)
                else:
                    # 如果没有编码器，使用随机潜在表示
                    z = torch.randn(velocity.shape[0], 1, 512).to(self.device)
            
            # 计算扩散损失
            loss = self.model.diffusion(z)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # 定期记录
            if self.use_wandb and self.global_step % 100 == 0:
                wandb.log({
                    'diffusion/step_loss': loss.item(),
                    'global_step': self.global_step
                })
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        
        for seismic, velocity in val_loader:
            velocity = velocity.to(self.device)
            
            # 获取真实潜在表示
            if self.encoder is not None:
                z = self.encoder(velocity)
                z = z.view(z.shape[0], 1, -1)
            else:
                z = torch.randn(velocity.shape[0], 1, 512).to(self.device)
            
            # 计算扩散损失
            loss = self.model.diffusion(z)
            total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    @torch.no_grad()
    def generate_samples(self, num_samples: int = 4) -> Dict[str, torch.Tensor]:
        """生成样本用于可视化"""
        self.model.eval()
        
        # 生成数据
        velocity, seismic = self.model.generate(num_samples, self.device)
        
        return {
            'velocity': velocity.cpu(),
            'seismic': seismic.cpu()
        }
    
    def _save_checkpoint(self, path: str, metadata: Dict[str, Any]) -> None:
        """保存检查点"""
        # 只保存扩散模型的状态
        diffusion_state_dict = {}
        for name, param in self.model.state_dict().items():
            if 'diffusion' in name or 'unet' in name:
                diffusion_state_dict[name] = param
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),  # 保存完整模型状态
            'diffusion_state_dict': diffusion_state_dict,  # 单独保存扩散部分
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'global_step': self.global_step,
            **metadata
        }
        torch.save(checkpoint, path)
        print(f"检查点已保存: {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> Dict[str, Any]:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 加载模型状态
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'diffusion_state_dict' in checkpoint:
            # 只加载扩散部分
            self.model.load_state_dict(checkpoint['diffusion_state_dict'], strict=False)
        
        # 加载优化器状态
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"检查点已加载: {path}")
        return checkpoint 