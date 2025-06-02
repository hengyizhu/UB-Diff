"""
微调训练器

专门用于微调UB-Diff模型的地震解码器部分
"""

import os
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
from .pytorch_ssim import SSIM

from ..ub_diff import UBDiff
from ..data import create_dataloaders
from .utils import (
    MetricLogger, SmoothedValue, WarmupMultiStepLR, 
    setup_seed, save_checkpoint, load_checkpoint, count_parameters
)

try:
    import wandb
    _has_wandb = True
except ImportError:
    _has_wandb = False


class FinetuneTrainer:
    """微调训练器
    
    专门用于微调地震解码器，在预训练的编码器基础上训练地震数据生成
    """
    
    def __init__(self,
                 seismic_folder: str,
                 velocity_folder: str,
                 dataset_name: str,
                 output_path: str,
                 checkpoint_path: str,
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
                 device: str = "cuda",
                 preload: bool = True,
                 preload_workers: int = 8,
                 cache_size: int = 32,
                 use_memmap: bool = False):
        """
        Args:
            seismic_folder: 地震数据文件夹路径
            velocity_folder: 速度场数据文件夹路径
            dataset_name: 数据集名称
            output_path: 输出路径
            checkpoint_path: 预训练模型检查点路径
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
            preload: 是否预加载数据
            preload_workers: 预加载使用的线程数
            cache_size: LRU缓存大小
            use_memmap: 是否使用内存映射
        """
        self.device = torch.device(device)
        self.output_path = output_path
        self.use_wandb = use_wandb and _has_wandb
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 加载数据（优化版本）
        self.train_loader, self.test_loader, self.paired_loader, self.dataset_ctx = create_dataloaders(
            seismic_folder=seismic_folder,
            velocity_folder=velocity_folder,
            dataset_name=dataset_name,
            num_data=num_data,
            paired_num=paired_num,
            batch_size=batch_size,
            num_workers=num_workers,
            preload=preload,
            preload_workers=preload_workers,
            cache_size=cache_size,
            use_memmap=use_memmap,
            prefetch_factor=4,  # 增加预取因子以减少IO等待
            persistent_workers=True  # 使用持久worker减少初始化开销
        )
        
        # 创建模型并加载预训练权重
        self.model = UBDiff(
            in_channels=1,
            encoder_dim=encoder_dim,
            velocity_channels=1,
            seismic_channels=5,
            pretrained_path=checkpoint_path
        ).to(self.device)
        
        # 冻结编码器和速度解码器，只训练地震解码器
        self.model.freeze_encoder()
        self.model.freeze_velocity_decoder()
        
        # 冻结扩散部分
        for param in self.model.unet.parameters():
            param.requires_grad = False
        for param in self.model.diffusion.parameters():
            param.requires_grad = False
        
        # 损失函数
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        
        # 优化器和调度器（只优化地震解码器参数）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度
        if lr_milestones is None:
            lr_milestones = []
        
        warmup_iters = warmup_epochs * len(self.paired_loader)
        lr_milestones_iter = [len(self.paired_loader) * m for m in lr_milestones]
        
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
            wandb.init(project=wandb_project, name=f"finetune_{dataset_name}")
        
        print(f"微调训练器初始化完成")
        print(f"模型参数统计: {count_parameters(self.model)}")
        print(f"可训练参数统计: {sum(p.numel() for p in trainable_params)}")
        
        # 详细参数冻结状态检查
        self._verify_parameter_freezing()

    def _verify_parameter_freezing(self):
        """验证参数冻结状态"""
        print("\n" + "="*50)
        print("参数冻结状态验证")
        print("="*50)
        
        # 统计各组件的可训练参数
        encoder_trainable = 0
        velocity_decoder_trainable = 0
        seismic_decoder_trainable = 0
        other_trainable = 0
        
        problematic_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                
                if 'encoder' in name:
                    encoder_trainable += param_count
                    problematic_params.append(f"编码器参数: {name}")
                elif 'velocity' in name and ('decoder' in name or 'projector' in name):
                    velocity_decoder_trainable += param_count
                    problematic_params.append(f"速度解码器参数: {name}")
                elif 'seismic' in name and ('decoder' in name or 'projector' in name):
                    seismic_decoder_trainable += param_count
                else:
                    other_trainable += param_count
        
        print(f"编码器可训练参数: {encoder_trainable:,}")
        print(f"速度解码器可训练参数: {velocity_decoder_trainable:,}")
        print(f"地震解码器可训练参数: {seismic_decoder_trainable:,}")
        print(f"其他可训练参数: {other_trainable:,}")
        
        # 检查是否有问题
        if encoder_trainable > 0 or velocity_decoder_trainable > 0:
            print("\n❌ 检测到参数冻结问题!")
            for param_name in problematic_params:
                if 'encoder' in param_name or 'velocity' in param_name:
                    print(f"  {param_name}")
            
            # 强制重新冻结
            print("\n🔧 强制重新冻结参数...")
            self._force_freeze_parameters()
        else:
            print("\n✅ 参数冻结状态正确")

    def _force_freeze_parameters(self):
        """强制冻结应该被冻结的参数"""
        # 强制冻结编码器
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        
        # 强制冻结速度解码器和投影器
        for param in self.model.dual_decoder.velocity_decoder.parameters():
            param.requires_grad = False
        for param in self.model.dual_decoder.velocity_projector.parameters():
            param.requires_grad = False
        
        # 强制冻结扩散部分
        for param in self.model.unet.parameters():
            param.requires_grad = False
        for param in self.model.diffusion.parameters():
            param.requires_grad = False
        
        # 重新创建优化器，只包含真正需要训练的参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.optimizer.param_groups[0]['lr'],
            weight_decay=self.optimizer.param_groups[0]['weight_decay']
        )
        
        print(f"✅ 重新创建优化器，可训练参数: {sum(p.numel() for p in trainable_params):,}")

    def _ensure_frozen_modules_eval(self):
        """确保冻结的模块处于eval模式"""
        # 编码器
        self.model.encoder.eval()
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        
        # 速度解码器和投影器
        self.model.dual_decoder.velocity_decoder.eval()
        self.model.dual_decoder.velocity_projector.eval()
        
        for param in self.model.dual_decoder.velocity_decoder.parameters():
            param.requires_grad = False
        for param in self.model.dual_decoder.velocity_projector.parameters():
            param.requires_grad = False

    def compute_loss(self, pred_velocity: torch.Tensor, pred_seismic: torch.Tensor,
                    gt_velocity: torch.Tensor, gt_seismic: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算损失"""
        # 速度损失（用于监控，不参与训练）
        velocity_l1 = self.l1_loss(pred_velocity, gt_velocity)
        velocity_l2 = self.l2_loss(pred_velocity, gt_velocity)
        velocity_loss = self.lambda_g1v * velocity_l1 + self.lambda_g2v * velocity_l2
        
        # 地震损失（主要训练目标）
        seismic_l1 = self.l1_loss(pred_seismic, gt_seismic)
        seismic_l2 = self.l2_loss(pred_seismic, gt_seismic)
        seismic_loss = self.lambda_g1v * seismic_l1 + self.lambda_g2v * seismic_l2
        
        # 总损失（只训练地震重构）
        total_loss = seismic_loss
        
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
        
        # 强制确保冻结的模块保持冻结状态
        self._ensure_frozen_modules_eval()
                
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
        metric_logger.add_meter('samples/s', SmoothedValue(window_size=10, fmt='{value:.3f}'))
        header = f'Finetune Epoch: [{epoch + 1}]'
        
        # 用于计算SSIM
        velocity_tensors = []
        pred_velocity_tensors = []
        
        # 每个epoch开始时验证参数状态（仅第一个epoch和每10个epoch）
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1} 参数状态检查:")
            self._quick_param_check()
        
        # 使用配对数据进行训练
        for batch_idx, (seismic, velocity) in enumerate(metric_logger.log_every(self.paired_loader, print_freq, header)):
            start_time = time.time()
            
            seismic = seismic.to(self.device, dtype=torch.float)
            velocity = velocity.to(self.device, dtype=torch.float)
            
            # 前向传播 - 重构
            pred_velocity, pred_seismic = self.model.reconstruct(velocity)
            
            # 计算损失
            loss, loss_dict = self.compute_loss(pred_velocity, pred_seismic, velocity, seismic)
            
            # 反向传播前再次确保参数冻结
            if batch_idx == 0:  # 只在第一个batch检查
                self._ensure_frozen_modules_eval()
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 检查梯度（可选，仅调试时）
            if batch_idx == 0 and epoch == 0:
                self._check_gradients()
            
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
            'train_seismic_loss': metric_logger.meters['seismic_loss'].global_avg,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        print(f'微调训练 SSIM: {ssim_value.item():.4f}')
        print(f'微调训练 地震损失: {metric_logger.meters["seismic_loss"].global_avg:.4f}')
        print(f'微调训练 速度损失: {metric_logger.meters["velocity_loss"].global_avg:.4f}')
        
        return epoch_metrics

    def _quick_param_check(self):
        """快速参数状态检查"""
        velocity_trainable = sum(p.numel() for name, p in self.model.named_parameters() 
                               if p.requires_grad and 'velocity' in name and ('decoder' in name or 'projector' in name))
        encoder_trainable = sum(p.numel() for name, p in self.model.named_parameters() 
                              if p.requires_grad and 'encoder' in name)
        seismic_trainable = sum(p.numel() for name, p in self.model.named_parameters() 
                              if p.requires_grad and 'seismic' in name and ('decoder' in name or 'projector' in name))
        
        if velocity_trainable > 0 or encoder_trainable > 0:
            print(f"⚠️  参数泄漏检测: 编码器={encoder_trainable}, 速度解码器={velocity_trainable}")
            self._force_freeze_parameters()
        else:
            print(f"✅ 参数状态正常: 地震解码器={seismic_trainable}")

    def _check_gradients(self):
        """检查梯度状态（调试用）"""
        print("\n首次前向传播梯度检查:")
        velocity_grads = []
        seismic_grads = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.grad.norm() > 1e-8:
                if 'velocity' in name and ('decoder' in name or 'projector' in name):
                    velocity_grads.append(name)
                elif 'seismic' in name and ('decoder' in name or 'projector' in name):
                    seismic_grads.append(name)
        
        if velocity_grads:
            print(f"❌ 速度解码器有梯度的参数: {len(velocity_grads)}")
            for name in velocity_grads[:3]:  # 只显示前3个
                print(f"  {name}")
        else:
            print("✅ 速度解码器无梯度")
        
        print(f"✅ 地震解码器有梯度的参数: {len(seismic_grads)}")

    def evaluate(self, epoch: int) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        total_loss = 0.0
        total_seismic_loss = 0.0
        velocity_tensors = []
        pred_velocity_tensors = []
        
        with torch.no_grad():
            for seismic, velocity in self.test_loader:
                seismic = seismic.to(self.device, dtype=torch.float)
                velocity = velocity.to(self.device, dtype=torch.float)
                
                # 重构
                pred_velocity, pred_seismic = self.model.reconstruct(velocity)
                
                # 计算损失
                loss, loss_dict = self.compute_loss(pred_velocity, pred_seismic, velocity, seismic)
                total_loss += loss.item()
                total_seismic_loss += loss_dict['seismic_loss']
                
                # 收集张量
                velocity_tensors.append(velocity)
                pred_velocity_tensors.append(pred_velocity)
        
        # 计算平均损失
        avg_loss = total_loss / len(self.test_loader)
        avg_seismic_loss = total_seismic_loss / len(self.test_loader)
        
        # 计算SSIM
        all_velocity = torch.cat(velocity_tensors, dim=0)
        all_pred_velocity = torch.cat(pred_velocity_tensors, dim=0)
        ssim_loss = SSIM(window_size=11)
        ssim_value = ssim_loss(all_velocity / 2 + 0.5, all_pred_velocity / 2 + 0.5)
        
        eval_metrics = {
            'val_loss': avg_loss,
            'val_seismic_loss': avg_seismic_loss,
            'val_ssim': ssim_value.item()
        }
        
        print(f'验证损失: {avg_loss:.4f}, 地震损失: {avg_seismic_loss:.4f}, 验证SSIM: {ssim_value.item():.4f}')
        
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
        
        filepath = os.path.join(self.output_path, f'finetune_checkpoint_epoch_{epoch + 1}.pth')
        save_checkpoint(checkpoint, filepath, is_best)

    def train(self, epochs: int, val_every: int = 10, print_freq: int = 50) -> None:
        """训练主循环"""
        print("开始地震解码器微调训练...")
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\n{'='*50}")
            print(f"Finetune Epoch: {epoch + 1}/{epochs}")
            
            # 训练
            train_metrics = self.train_one_epoch(epoch, print_freq)
            
            # 验证
            if (epoch + 1) % val_every == 0:
                val_metrics = self.evaluate(epoch)
                
                # 检查是否是最佳模型（主要看地震损失）
                is_best = val_metrics['val_seismic_loss'] < self.best_loss
                if is_best:
                    self.best_loss = val_metrics['val_seismic_loss']
                    self.best_ssim = val_metrics['val_ssim']
                    print(f"新的最佳模型! 地震损失: {self.best_loss:.4f}")
                
                # 保存检查点
                self.save_checkpoint(epoch, is_best)
                
                print(f"当前最佳地震损失: {self.best_loss:.4f}")
        
        # 训练完成
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'\n微调训练完成! 总时间: {total_time_str}')
        
        if self.use_wandb:
            wandb.finish()


def create_trainer_from_args(args) -> FinetuneTrainer:
    """从命令行参数创建微调训练器"""
    return FinetuneTrainer(
        seismic_folder=args.train_data,
        velocity_folder=args.train_label,
        dataset_name=args.dataset,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint_path,
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
        wandb_project=args.proj_name,
        preload=args.preload,
        preload_workers=args.preload_workers,
        cache_size=args.cache_size,
        use_memmap=args.use_memmap
    ) 