"""
解码器的量化感知训练器

分阶段训练速度解码器和地震解码器，包含SSIM评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple
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
from .pytorch_ssim import SSIM


class QATDecoderTrainer:
    """解码器的量化感知训练器
    
    支持分阶段训练速度解码器和地震解码器，包含SSIM评估
    """
    
    def __init__(self,
                 model: nn.Module,
                 encoder_model: Optional[nn.Module] = None,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 use_wandb: bool = False):
        """
        Args:
            model: 量化的解码器模型
            encoder_model: 原始编码器（用于生成潜在表示）
            device: 训练设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            use_wandb: 是否使用WandB记录
        """
        self.model = model.to(device)
        self.encoder = encoder_model
        if self.encoder is not None:
            self.encoder = self.encoder.to(device)
            self.encoder.eval()  # 编码器始终处于评估模式
        
        self.device = device
        self.use_wandb = use_wandb
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.ssim_calculator = SSIM(window_size=11)
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_ssim = 0.0
        
    def train_velocity_decoder(self,
                             train_loader: DataLoader,
                             val_loader: Optional[DataLoader] = None,
                             epochs: int = 100,
                             checkpoint_dir: str = './checkpoints/qat_velocity',
                             val_every: int = 1):
        """训练速度解码器
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            checkpoint_dir: 检查点保存目录
            val_every: 每多少个epoch进行一次验证
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 冻结地震解码器路径
        self.model.freeze_seismic_path()
        
        # 重新创建优化器，只包含需要梯度的参数
        self._recreate_optimizer()
        
        print("开始训练速度解码器（QAT）...")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 训练阶段
            train_metrics = self._train_epoch_velocity(train_loader)
            
            # 验证阶段（根据val_every控制频率）
            if val_loader is not None and (epoch + 1) % val_every == 0:
                val_metrics = self._validate_velocity(val_loader)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_metrics['loss']:.4f}, Train SSIM: {train_metrics['ssim']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val SSIM: {val_metrics['ssim']:.4f}")
                
                # 学习率调度
                self.scheduler.step(val_metrics['loss'])
                
                # 保存最佳模型（基于loss）
                if val_metrics['loss'] < self.best_loss:
                    self.best_loss = val_metrics['loss']
                    self.best_ssim = val_metrics['ssim']
                    self._save_checkpoint(
                        os.path.join(checkpoint_dir, 'best_velocity.pt'),
                        {'type': 'velocity', 'val_loss': val_metrics['loss'], 'val_ssim': val_metrics['ssim']}
                    )
                    print(f"新的最佳速度解码器模型! Loss: {self.best_loss:.4f}")
                
                # WandB记录
                if self.use_wandb:
                    log_dict = {
                        'velocity/train_loss': train_metrics['loss'],
                        'velocity/train_ssim': train_metrics['ssim'],
                        'velocity/val_loss': val_metrics['loss'],
                        'velocity/val_ssim': val_metrics['ssim'],
                        'epoch': epoch
                    }
                    wandb.log(log_dict)
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_metrics['loss']:.4f}, Train SSIM: {train_metrics['ssim']:.4f}")
                
                # 只记录训练指标
                if self.use_wandb:
                    log_dict = {
                        'velocity/train_loss': train_metrics['loss'],
                        'velocity/train_ssim': train_metrics['ssim'],
                        'epoch': epoch
                    }
                    wandb.log(log_dict)
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, f'velocity_epoch_{epoch+1}.pt'),
                    {'type': 'velocity', 'epoch': epoch+1}
                )
    
    def train_seismic_decoder(self,
                            train_loader: DataLoader,
                            val_loader: Optional[DataLoader] = None,
                            epochs: int = 100,
                            checkpoint_dir: str = './checkpoints/qat_seismic',
                            freeze_velocity: bool = True,
                            val_every: int = 1):
        """训练地震解码器
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            checkpoint_dir: 检查点保存目录
            freeze_velocity: 是否冻结速度解码器
            val_every: 每多少个epoch进行一次验证
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 先解冻地震解码路径（如果之前被冻结）
        self.model.unfreeze_seismic_path()
        
        # 根据需要冻结速度解码器
        if freeze_velocity:
            self.model.freeze_velocity_path()
        
        # 重新创建优化器，只包含需要梯度的参数
        self._recreate_optimizer()
        
        print("开始训练地震解码器（QAT）...")
        
        # 重置最佳指标
        self.best_loss = float('inf')
        self.best_ssim = 0.0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 训练阶段
            train_metrics = self._train_epoch_seismic(train_loader)
            
            # 验证阶段（根据val_every控制频率）
            if val_loader is not None and (epoch + 1) % val_every == 0:
                val_metrics = self._validate_seismic(val_loader)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_metrics['loss']:.4f}, Train SSIM: {train_metrics['ssim']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val SSIM: {val_metrics['ssim']:.4f}")
                
                # 学习率调度
                self.scheduler.step(val_metrics['loss'])
                
                # 保存最佳模型（基于loss）
                if val_metrics['loss'] < self.best_loss:
                    self.best_loss = val_metrics['loss']
                    self.best_ssim = val_metrics['ssim']
                    self._save_checkpoint(
                        os.path.join(checkpoint_dir, 'best_seismic.pt'),
                        {'type': 'seismic', 'val_loss': val_metrics['loss'], 'val_ssim': val_metrics['ssim']}
                    )
                    print(f"新的最佳地震解码器模型! Loss: {self.best_loss:.4f}")
                
                # WandB记录
                if self.use_wandb:
                    log_dict = {
                        'seismic/train_loss': train_metrics['loss'],
                        'seismic/train_ssim': train_metrics['ssim'],
                        'seismic/val_loss': val_metrics['loss'],
                        'seismic/val_ssim': val_metrics['ssim'],
                        'epoch': epoch
                    }
                    wandb.log(log_dict)
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_metrics['loss']:.4f}, Train SSIM: {train_metrics['ssim']:.4f}")
                
                # 只记录训练指标
                if self.use_wandb:
                    log_dict = {
                        'seismic/train_loss': train_metrics['loss'],
                        'seismic/train_ssim': train_metrics['ssim'],
                        'epoch': epoch
                    }
                    wandb.log(log_dict)
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, f'seismic_epoch_{epoch+1}.pt'),
                    {'type': 'seismic', 'epoch': epoch+1}
                )
    
    def _train_epoch_velocity(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练速度解码器一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_ssim = 0.0
        num_batches = 0
        
        for batch_idx, (seismic, velocity) in enumerate(tqdm(train_loader, desc="Training Velocity")):
            seismic = seismic.to(self.device, dtype=torch.float32)
            velocity = velocity.to(self.device, dtype=torch.float32)
            
            # 获取潜在表示
            with torch.no_grad():
                if self.encoder is not None:
                    z = self.encoder(velocity)
                else:
                    z = torch.randn(velocity.shape[0], 512, 1, 1).to(self.device)
            
            # 前向传播
            velocity_pred, _ = self.model(z)
            
            # 计算损失
            loss = self.mse_loss(velocity_pred, velocity)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 计算当前batch的SSIM（避免存储大量数据）
            with torch.no_grad():
                batch_ssim = self.ssim_calculator(velocity / 2 + 0.5, velocity_pred / 2 + 0.5)
                total_ssim += batch_ssim.item()
                num_batches += 1
            
            # 定期清理显存以避免碎片化
            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
        
        return {
            'loss': total_loss / len(train_loader),
            'ssim': total_ssim / num_batches
        }
    
    def _train_epoch_seismic(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练地震解码器一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_ssim = 0.0
        num_batches = 0
        
        for batch_idx, (seismic, velocity) in enumerate(tqdm(train_loader, desc="Training Seismic")):
            seismic = seismic.to(self.device, dtype=torch.float32)
            velocity = velocity.to(self.device, dtype=torch.float32)
            
            # 获取潜在表示
            with torch.no_grad():
                if self.encoder is not None:
                    z = self.encoder(velocity)
                else:
                    z = torch.randn(velocity.shape[0], 512, 1, 1).to(self.device)
            
            # 前向传播
            _, seismic_pred = self.model(z)
            
            # 计算损失（只使用MSE进行反向传播）
            loss = self.mse_loss(seismic_pred, seismic)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 计算当前batch的SSIM（避免存储大量数据）
            with torch.no_grad():
                batch_ssim = self.ssim_calculator(seismic / 2 + 0.5, seismic_pred / 2 + 0.5)
                total_ssim += batch_ssim.item()
                num_batches += 1
            
            # 定期清理显存以避免碎片化
            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
        
        return {
            'loss': total_loss / len(train_loader),
            'ssim': total_ssim / num_batches
        }
    
    @torch.no_grad()
    def _validate_velocity(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证速度解码器"""
        self.model.eval()
        total_loss = 0.0
        total_ssim = 0.0
        num_batches = 0
        
        for seismic, velocity in val_loader:
            seismic = seismic.to(self.device, dtype=torch.float32)
            velocity = velocity.to(self.device, dtype=torch.float32)
            
            # 获取潜在表示
            if self.encoder is not None:
                z = self.encoder(velocity)
            else:
                z = torch.randn(velocity.shape[0], 512, 1, 1).to(self.device)
            
            # 前向传播
            velocity_pred, _ = self.model(z)
            
            # 计算损失
            loss = self.mse_loss(velocity_pred, velocity)
            total_loss += loss.item()
            
            # 计算当前batch的SSIM
            batch_ssim = self.ssim_calculator(velocity / 2 + 0.5, velocity_pred / 2 + 0.5)
            total_ssim += batch_ssim.item()
            num_batches += 1
        
        return {
            'loss': total_loss / len(val_loader),
            'ssim': total_ssim / num_batches
        }
    
    @torch.no_grad()
    def _validate_seismic(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证地震解码器"""
        self.model.eval()
        total_loss = 0.0
        total_ssim = 0.0
        num_batches = 0
        
        for seismic, velocity in val_loader:
            seismic = seismic.to(self.device, dtype=torch.float32)
            velocity = velocity.to(self.device, dtype=torch.float32)
            
            # 获取潜在表示
            if self.encoder is not None:
                z = self.encoder(velocity)
            else:
                z = torch.randn(velocity.shape[0], 512, 1, 1).to(self.device)
            
            # 前向传播
            _, seismic_pred = self.model(z)
            
            # 计算损失
            loss = self.mse_loss(seismic_pred, seismic)
            total_loss += loss.item()
            
            # 计算当前batch的SSIM
            batch_ssim = self.ssim_calculator(seismic / 2 + 0.5, seismic_pred / 2 + 0.5)
            total_ssim += batch_ssim.item()
            num_batches += 1
        
        return {
            'loss': total_loss / len(val_loader),
            'ssim': total_ssim / num_batches
        }
    
    def _save_checkpoint(self, path: str, metadata: Dict[str, Any]) -> None:
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'best_ssim': self.best_ssim,
            **metadata
        }
        torch.save(checkpoint, path)
        print(f"检查点已保存: {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_ssim = checkpoint.get('best_ssim', 0.0)
        print(f"检查点已加载: {path}")
        return checkpoint

    def _recreate_optimizer(self):
        """重新创建优化器，只包含需要梯度的参数"""
        # 获取当前学习率和权重衰减
        current_lr = self.optimizer.param_groups[0]['lr']
        current_weight_decay = self.optimizer.param_groups[0]['weight_decay']
        
        # 筛选出需要梯度的参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        print(f"重新创建优化器: {len(trainable_params)} 个可训练参数")
        
        # 重新创建优化器
        self.optimizer = optim.Adam(
            trainable_params,
            lr=current_lr,
            weight_decay=current_weight_decay
        )
        
        # 重新创建学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        ) 