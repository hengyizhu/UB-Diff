"""
编码器-解码器训练器

用于训练UB-Diff模型的编码器和解码器部分
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
from .pytorch_ssim import SSIM
import datetime

from ..ub_diff import UBDiff
from ..data import create_dataloaders, tonumpy_denormalize
from .utils import (
    MetricLogger, SmoothedValue, WarmupMultiStepLR, 
    setup_seed, save_checkpoint, load_checkpoint, count_parameters
)

try:
    import wandb
    _has_wandb = True
except ImportError:
    _has_wandb = False


class EncoderDecoderTrainer:
    """编码器-解码器训练器"""
    
    def __init__(self,
                 seismic_folder: str,
                 velocity_folder: str,
                 dataset_name: str,
                 output_path: str,
                 num_data: int = 24000,
                 paired_num: int = 5000,
                 batch_size: int = 64,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 lr_gamma: float = 0.98,
                 lr_milestones: Optional[list] = None,
                 warmup_epochs: int = 0,
                 num_workers: int = 4,
                 encoder_dim: int = 512,
                 lambda_g1v: float = 1.0,
                 lambda_g2v: float = 1.0,
                 use_wandb: bool = False,
                 wandb_project: str = "UB-Diff",
                 device: str = "cuda"):
        """
        Args:
            seismic_folder: 地震数据文件夹路径
            velocity_folder: 速度场数据文件夹路径
            dataset_name: 数据集名称
            output_path: 输出路径
            num_data: 训练数据数量
            paired_num: 配对数据数量
            batch_size: 批次大小
            learning_rate: 学习率
            weight_decay: 权重衰减
            lr_gamma: 学习率衰减因子
            lr_milestones: 学习率衰减里程碑
            warmup_epochs: 预热轮数
            num_workers: 数据加载线程数
            encoder_dim: 编码器维度
            lambda_g1v: L1损失权重
            lambda_g2v: L2损失权重
            use_wandb: 是否使用wandb记录
            wandb_project: wandb项目名称
            device: 训练设备
        """
        self.device = torch.device(device)
        self.output_path = output_path
        self.use_wandb = use_wandb and _has_wandb
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 加载数据
        self.train_loader, self.test_loader, self.paired_loader, self.dataset_ctx = create_dataloaders(
            seismic_folder=seismic_folder,
            velocity_folder=velocity_folder,
            dataset_name=dataset_name,
            num_data=num_data,
            paired_num=paired_num,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        # 创建模型
        self.model = UBDiff(
            in_channels=1,
            encoder_dim=encoder_dim,
            velocity_channels=1,
            seismic_channels=5
        ).to(self.device)
        
        # 损失函数
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        
        # 优化器和调度器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度
        if lr_milestones is None:
            lr_milestones = []
        
        warmup_iters = warmup_epochs * len(self.train_loader)
        lr_milestones_iter = [len(self.train_loader) * m for m in lr_milestones]
        
        self.scheduler = WarmupMultiStepLR(
            self.optimizer,
            milestones=lr_milestones_iter,
            gamma=lr_gamma,
            warmup_iters=warmup_iters,
            warmup_factor=1e-5
        )
        
        # 训练状态
        self.step = 0
        self.best_ssim = 0.0
        self.best_loss = float('inf')
        
        # 初始化wandb
        if self.use_wandb:
            wandb.init(project=wandb_project, name=f"encoder_decoder_{dataset_name}")
        
        print(f"编码器-解码器训练器初始化完成")
        print(f"模型参数统计: {count_parameters(self.model)}")

    def compute_loss(self, pred_velocity: torch.Tensor, pred_seismic: torch.Tensor,
                    gt_velocity: torch.Tensor, gt_seismic: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算损失"""
        # 速度损失
        velocity_l1 = self.l1_loss(pred_velocity, gt_velocity)
        velocity_l2 = self.l2_loss(pred_velocity, gt_velocity)
        velocity_loss = self.lambda_g1v * velocity_l1 + self.lambda_g2v * velocity_l2
        
        # 地震损失
        seismic_l1 = self.l1_loss(pred_seismic, gt_seismic)
        seismic_l2 = self.l2_loss(pred_seismic, gt_seismic)
        seismic_loss = self.lambda_g1v * seismic_l1 + self.lambda_g2v * seismic_l2
        
        # 总损失（只训练速度重构）
        total_loss = velocity_loss
        
        loss_dict = {
            'velocity_loss': velocity_loss.item(),
            'seismic_loss': seismic_loss.item(),
            'velocity_l1': velocity_l1.item(),
            'velocity_l2': velocity_l2.item(),
            'seismic_l1': seismic_l1.item(),
            'seismic_l2': seismic_l2.item()
        }
        
        return total_loss, loss_dict

    def train_one_epoch(self, epoch: int, print_freq: int = 50) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        # 冻结扩散部分，只训练编解码器
        for param in self.model.unet.parameters():
            param.requires_grad = False
        for param in self.model.diffusion.parameters():
            param.requires_grad = False
            
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
        metric_logger.add_meter('samples/s', SmoothedValue(window_size=10, fmt='{value:.3f}'))
        header = f'Epoch: [{epoch + 1}]'
        
        # 用于计算SSIM
        velocity_tensors = []
        pred_velocity_tensors = []
        
        for seismic, velocity in metric_logger.log_every(self.train_loader, print_freq, header):
            start_time = time.time()
            
            seismic = seismic.to(self.device, dtype=torch.float)
            velocity = velocity.to(self.device, dtype=torch.float)
            
            # 前向传播 - 重构
            pred_velocity, pred_seismic = self.model.reconstruct(velocity)
            
            # 计算损失
            loss, loss_dict = self.compute_loss(pred_velocity, pred_seismic, velocity, seismic)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # 记录指标
            batch_size = velocity.shape[0]
            metric_logger.update(
                loss=loss.item(),
                lr=self.optimizer.param_groups[0]['lr'],
                **loss_dict
            )
            metric_logger.meters['samples/s'].update(batch_size / (time.time() - start_time))
            
            # 收集张量用于SSIM计算
            velocity_tensors.append(velocity.detach())
            pred_velocity_tensors.append(pred_velocity.detach())
            
            # 记录到wandb
            if self.use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "train/step": self.step,
                    **{f"train/{k}": v for k, v in loss_dict.items()}
                })
            
            self.step += 1
        
        # 计算SSIM
        all_velocity = torch.cat(velocity_tensors, dim=0)
        all_pred_velocity = torch.cat(pred_velocity_tensors, dim=0)
        ssim_loss = SSIM(window_size=11)
        ssim_value = ssim_loss(all_velocity / 2 + 0.5, all_pred_velocity / 2 + 0.5)
        
        epoch_metrics = {
            'train_loss': metric_logger.meters['loss'].global_avg,
            'train_ssim': ssim_value.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        print(f'训练 SSIM: {ssim_value.item():.4f}')
        
        return epoch_metrics

    def evaluate(self, epoch: int) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        total_loss = 0.0
        velocity_tensors = []
        pred_velocity_tensors = []
        
        with torch.no_grad():
            for seismic, velocity in self.test_loader:
                seismic = seismic.to(self.device, dtype=torch.float)
                velocity = velocity.to(self.device, dtype=torch.float)
                
                # 重构
                pred_velocity, pred_seismic = self.model.reconstruct(velocity)
                
                # 计算损失
                loss, _ = self.compute_loss(pred_velocity, pred_seismic, velocity, seismic)
                total_loss += loss.item()
                
                # 收集张量
                velocity_tensors.append(velocity)
                pred_velocity_tensors.append(pred_velocity)
        
        # 计算平均损失
        avg_loss = total_loss / len(self.test_loader)
        
        # 计算SSIM
        all_velocity = torch.cat(velocity_tensors, dim=0)
        all_pred_velocity = torch.cat(pred_velocity_tensors, dim=0)
        ssim_loss = SSIM(window_size=11)
        ssim_value = ssim_loss(all_velocity / 2 + 0.5, all_pred_velocity / 2 + 0.5)
        
        eval_metrics = {
            'val_loss': avg_loss,
            'val_ssim': ssim_value.item()
        }
        
        print(f'验证损失: {avg_loss:.4f}, 验证SSIM: {ssim_value.item():.4f}')
        
        if self.use_wandb:
            wandb.log({f"val/{k}": v for k, v in eval_metrics.items()})
        
        return eval_metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_ssim': self.best_ssim,
            'best_loss': self.best_loss
        }
        
        filepath = os.path.join(self.output_path, f'checkpoint_epoch_{epoch + 1}.pth')
        save_checkpoint(checkpoint, filepath, is_best)

    def train(self, epochs: int, val_every: int = 20, print_freq: int = 50) -> None:
        """训练主循环"""
        print("开始编码器-解码器训练...")
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\n{'='*50}")
            print(f"Epoch: {epoch + 1}/{epochs}")
            
            # 训练
            train_metrics = self.train_one_epoch(epoch, print_freq)
            
            # 验证
            if (epoch + 1) % val_every == 0:
                val_metrics = self.evaluate(epoch)
                
                # 检查是否是最佳模型
                is_best = val_metrics['val_ssim'] > self.best_ssim
                if is_best:
                    self.best_ssim = val_metrics['val_ssim']
                    self.best_loss = val_metrics['val_loss']
                    print(f"新的最佳模型! SSIM: {self.best_ssim:.4f}")
                
                # 保存检查点
                self.save_checkpoint(epoch, is_best)
                
                print(f"当前最佳SSIM: {self.best_ssim:.4f}")
        
        # 训练完成
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'\n训练完成! 总时间: {total_time_str}')
        
        if self.use_wandb:
            wandb.finish()


def create_trainer_from_args(args) -> EncoderDecoderTrainer:
    """从命令行参数创建训练器"""
    return EncoderDecoderTrainer(
        seismic_folder=args.train_data,
        velocity_folder=args.train_label,
        dataset_name=args.dataset,
        output_path=args.output_path,
        num_data=args.num_data,
        paired_num=args.paired_num,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_gamma=args.lr_gamma,
        lr_milestones=args.lr_milestones,
        warmup_epochs=args.lr_warmup_epochs,
        num_workers=args.workers,
        encoder_dim=args.encoder_dim,
        lambda_g1v=args.lambda_g1v,
        lambda_g2v=args.lambda_g2v,
        use_wandb=args.use_wandb,
        wandb_project=args.proj_name
    ) 