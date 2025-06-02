"""
扩散模型训练器

用于训练UB-Diff模型的扩散部分
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import copy

from ..ub_diff import UBDiff
from ..data import create_diffusion_dataloader, tonumpy_denormalize, minmax_denormalize
from .utils import setup_seed, save_checkpoint, load_checkpoint, count_parameters

try:
    import wandb
    _has_wandb = True
except ImportError:
    _has_wandb = False


class DiffusionTrainer:
    """扩散模型训练器"""
    
    def __init__(self,
                 seismic_folder: str,
                 velocity_folder: str,
                 dataset_name: str,
                 num_data: int,
                 checkpoint_path: Optional[str] = None,
                 results_folder: str = './results',
                 batch_size: int = 16,
                 learning_rate: float = 8e-5,
                 num_steps: int = 150000,
                 gradient_accumulate_every: int = 1,
                 ema_decay: float = 0.995,
                 ema_update_every: int = 10,
                 save_and_sample_every: int = 30000,
                 num_samples: int = 25,
                 encoder_dim: int = 512,
                 time_steps: int = 256,
                 dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
                 objective: str = 'pred_v',
                 use_wandb: bool = False,
                 wandb_project: str = "UB-Diff",
                 device: str = "cuda",
                 num_workers: int = 4,
                 preload: bool = True,
                 preload_workers: int = 8,
                 cache_size: int = 32,
                 use_memmap: bool = False):
        """
        Args:
            seismic_folder: 地震数据文件夹路径
            velocity_folder: 速度场数据文件夹路径
            dataset_name: 数据集名称
            num_data: 训练数据数量
            checkpoint_path: 预训练编解码器检查点路径
            results_folder: 结果保存文件夹
            batch_size: 批次大小
            learning_rate: 学习率
            num_steps: 训练步数
            gradient_accumulate_every: 梯度累积步数
            ema_decay: EMA衰减率
            ema_update_every: EMA更新频率
            save_and_sample_every: 保存和采样频率
            num_samples: 采样数量
            encoder_dim: 编码器维度
            time_steps: 扩散时间步数
            dim_mults: U-Net维度倍数
            objective: 扩散目标
            use_wandb: 是否使用wandb记录
            wandb_project: wandb项目名称
            device: 训练设备
            num_workers: 数据加载线程数
            preload: 是否预加载数据
            preload_workers: 预加载使用的线程数
            cache_size: LRU缓存大小
            use_memmap: 是否使用内存映射
        """
        self.device = torch.device(device)
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.use_wandb = use_wandb and _has_wandb
        
        # 训练参数
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_and_sample_every = save_and_sample_every
        self.num_samples = num_samples
        
        # 加载数据（优化版本）
        self.dataloader, self.dataset_ctx = create_diffusion_dataloader(
            seismic_folder=seismic_folder,
            velocity_folder=velocity_folder,
            dataset_name=dataset_name,
            num_data=num_data,
            batch_size=batch_size,
            num_workers=num_workers,
            preload=preload,
            preload_workers=preload_workers,
            cache_size=cache_size,
            use_memmap=use_memmap,
            prefetch_factor=4,  # 增加预取因子以减少IO等待
            persistent_workers=True  # 使用持久worker减少初始化开销
        )
        
        # 创建循环数据加载器
        self.data_iter = self._cycle(self.dataloader)
        
        # 创建模型
        self.model = UBDiff(
            in_channels=1,
            encoder_dim=encoder_dim,
            time_steps=time_steps,
            dim_mults=dim_mults,
            objective=objective,
            use_wandb=use_wandb,
            pretrained_path=checkpoint_path
        ).to(self.device)
        
        # 冻结编解码器，只训练扩散模型
        self.model.freeze_encoder()
        self.model.freeze_velocity_decoder()
        self.model.freeze_seismic_decoder()
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        
        # EMA
        self.ema = EMA(self.model.unet, beta=ema_decay, update_every=ema_update_every)
        
        # 训练状态
        self.step = 0
        
        # 初始化wandb
        if self.use_wandb:
            wandb.init(project=wandb_project, name=f"diffusion_{dataset_name}")
        
        print(f"扩散模型训练器初始化完成")
        print(f"模型参数统计: {count_parameters(self.model)}")
        print(f"可训练参数统计: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def _cycle(self, dataloader):
        """创建无限循环的数据加载器"""
        while True:
            for data in dataloader:
                yield data

    def save(self, milestone: int, is_final: bool = False) -> None:
        """保存模型检查点"""
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict()
        }
        
        if is_final:
            # 保存最终最佳模型（使用EMA权重）
            torch.save(data, str(self.results_folder / 'best_diffusion_model.pt'))
            print(f"最终最佳模型已保存: {self.results_folder / 'best_diffusion_model.pt'}")
        else:
            # 正常的里程碑保存
            torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
            print(f"模型已保存: model-{milestone}.pt")

    def load(self, milestone: int) -> None:
        """加载模型检查点"""
        filepath = self.results_folder / f'model-{milestone}.pt'
        
        if not filepath.exists():
            print(f"检查点文件不存在: {filepath}")
            return
        
        data = torch.load(str(filepath), map_location=self.device)
        
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.optimizer.load_state_dict(data['optimizer'])
        self.ema.load_state_dict(data['ema'])
        
        print(f"模型已加载: {filepath}")

    def save_samples(self, milestone: int) -> None:
        """保存生成的样本"""
        self.ema.ema_model.eval()
        
        with torch.no_grad():
            # 生成样本
            velocity_gen, seismic_gen = self.model.generate(
                batch_size=self.num_samples, 
                device=self.device
            )
            
            # 反标准化
            velocity_denorm = minmax_denormalize(
                velocity_gen.cpu(), 
                self.dataset_ctx['label_min'], 
                self.dataset_ctx['label_max']
            )
            
            seismic_denorm = tonumpy_denormalize(
                seismic_gen,
                self.dataset_ctx['data_min'],
                self.dataset_ctx['data_max']
            )
            
            # 保存路径
            vel_folder = self.results_folder / 'velocity'
            seis_folder = self.results_folder / 'seismic'
            vel_folder.mkdir(exist_ok=True)
            seis_folder.mkdir(exist_ok=True)
            
            # 保存文件
            vel_path = vel_folder / f'generated_velocity_{milestone}.npy'
            seis_path = seis_folder / f'generated_seismic_{milestone}.npy'
            
            import numpy as np
            np.save(str(vel_path), velocity_denorm.numpy())
            np.save(str(seis_path), seismic_denorm)
            
            print(f"样本已保存: {vel_path}, {seis_path}")

    def train_step(self) -> float:
        """单步训练"""
        total_loss = 0.0
        
        for _ in range(self.gradient_accumulate_every):
            # 获取数据
            seismic, velocity = next(self.data_iter)
            velocity = velocity.to(self.device, dtype=torch.float)
            
            # 前向传播
            loss = self.model(velocity)
            loss = loss / self.gradient_accumulate_every
            total_loss += loss.item()
            
            # 反向传播
            loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # 优化器步骤
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # 更新EMA
        self.ema.update()
        
        return total_loss

    def train(self) -> None:
        """训练主循环"""
        print("开始扩散模型训练...")
        start_time = time.time()
        
        # 检查采样数量
        assert self._has_int_squareroot(self.num_samples), 'number of samples must have an integer square root'
        
        with tqdm(initial=self.step, total=self.num_steps) as pbar:
            while self.step < self.num_steps:
                # 训练步骤
                loss = self.train_step()
                
                # 更新进度条
                pbar.set_description(f'loss: {loss:.4f}')
                pbar.update(1)
                
                # 记录到wandb
                if self.use_wandb:
                    wandb.log({
                        "train/loss": loss,
                        "train/step": self.step
                    })
                
                # 保存和采样
                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    milestone = self.step // self.save_and_sample_every
                    
                    # 保存样本
                    self.save_samples(milestone)
                    
                    # 保存模型
                    self.save(milestone)
                
                self.step += 1
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n扩散模型训练完成! 总时间: {total_time:.2f}秒")
        
        # 保存最终最佳模型
        print("\n保存最终最佳模型...")
        final_milestone = self.step // self.save_and_sample_every
        self.save(final_milestone, is_final=True)
        
        if self.use_wandb:
            wandb.finish()

    def _has_int_squareroot(self, num: int) -> bool:
        """检查数字是否有整数平方根"""
        return int(math.sqrt(num)) ** 2 == num


class EMA:
    """指数移动平均"""
    
    def __init__(self, model, beta: float = 0.995, update_every: int = 10):
        super().__init__()
        self.beta = beta
        self.update_every = update_every
        
        self.model = model
        self.ema_model = self._copy_params_and_buffers(model)
        
        self.initted = False
        self.step = 0

    def _copy_params_and_buffers(self, model):
        """复制模型参数和缓冲区"""
        ema_model = copy.deepcopy(model)
        ema_model.requires_grad_(False)
        return ema_model

    def update(self):
        """更新EMA"""
        self.step += 1
        
        if (self.step % self.update_every) != 0:
            return
        
        if not self.initted:
            self._copy_from_model_to_ema()
            self.initted = True
            return
        
        self._update_moving_average()

    def _copy_from_model_to_ema(self):
        """从模型复制到EMA"""
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.copy_(param.data)

    def _update_moving_average(self):
        """更新移动平均"""
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.beta).add_(param.data, alpha=1 - self.beta)

    def state_dict(self):
        return {
            'ema_model': self.ema_model.state_dict(),
            'step': self.step,
            'initted': self.initted
        }

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.step = state_dict['step']
        self.initted = state_dict['initted']


def create_trainer_from_args(args) -> DiffusionTrainer:
    """从命令行参数创建训练器"""
    return DiffusionTrainer(
        seismic_folder=args.train_data,
        velocity_folder=args.train_label,
        dataset_name=args.dataset,
        num_data=args.num_data,
        checkpoint_path=args.checkpoint_path,
        results_folder=args.results_folder,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        save_and_sample_every=args.save_and_sample_every,
        encoder_dim=args.latent_dim,
        time_steps=args.time_steps,
        use_wandb=args.use_wandb,
        num_workers=args.workers,
        preload=not args.no_preload,
        preload_workers=args.preload_workers,
        cache_size=args.cache_size,
        use_memmap=args.use_memmap
    ) 