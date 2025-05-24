"""
UB-Diff数据生成器

整合编码器、解码器和扩散模型进行数据生成
"""

import os
import time
import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import matplotlib.pyplot as plt

from ..ub_diff import UBDiff
from ..data import tonumpy_denormalize, minmax_denormalize, DatasetConfig


class UBDiffGenerator:
    """UB-Diff模型生成器
    
    用于生成高质量的地震数据和速度场
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 dataset_name: str,
                 encoder_dim: int = 512,
                 time_steps: int = 256,
                 dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
                 objective: str = 'pred_v',
                 device: str = 'cuda'):
        """
        Args:
            checkpoint_path: 训练好的模型检查点路径
            dataset_name: 数据集名称（用于获取标准化参数）
            encoder_dim: 编码器维度
            time_steps: 扩散时间步数
            dim_mults: U-Net维度倍数
            objective: 扩散目标
            device: 生成设备
        """
        self.device = torch.device(device)
        self.dataset_name = dataset_name
        
        # 加载数据集配置
        self.config = DatasetConfig()
        self.dataset_ctx = self.config.get_dataset_info(dataset_name)
        
        # 创建模型
        self.model = UBDiff(
            in_channels=1,
            encoder_dim=encoder_dim,
            time_steps=time_steps,
            dim_mults=dim_mults,
            objective=objective,
            pretrained_path=checkpoint_path
        ).to(self.device)
        
        self.model.eval()
        print(f"UB-Diff生成器初始化完成，数据集: {dataset_name}")

    def generate_batch(self, 
                      batch_size: int = 16,
                      denormalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成一批数据
        
        Args:
            batch_size: 批次大小
            denormalize: 是否反标准化
            
        Returns:
            (velocity, seismic): 生成的速度场和地震数据
        """
        with torch.no_grad():
            velocity_gen, seismic_gen = self.model.generate(
                batch_size=batch_size,
                device=self.device
            )
            
            if denormalize:
                # 反标准化速度场
                velocity_gen = minmax_denormalize(
                    velocity_gen.cpu(),
                    self.dataset_ctx['label_min'],
                    self.dataset_ctx['label_max']
                )
                
                # 反标准化地震数据
                seismic_denorm = tonumpy_denormalize(
                    seismic_gen,
                    self.dataset_ctx['data_min'],
                    self.dataset_ctx['data_max']
                )
                seismic_gen = torch.from_numpy(seismic_denorm)
            
            return velocity_gen, seismic_gen

    def generate_and_save(self,
                         num_samples: int = 100,
                         batch_size: int = 16,
                         output_dir: str = './generated_data',
                         save_format: str = 'npy') -> None:
        """生成并保存大量数据
        
        Args:
            num_samples: 总样本数
            batch_size: 批次大小
            output_dir: 输出目录
            save_format: 保存格式 ('npy', 'pt')
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        velocity_dir = output_path / 'velocity'
        seismic_dir = output_path / 'seismic'
        velocity_dir.mkdir(exist_ok=True)
        seismic_dir.mkdir(exist_ok=True)
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        total_generated = 0
        
        print(f"开始生成 {num_samples} 个样本...")
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_samples - total_generated)
            
            # 生成数据
            velocity, seismic = self.generate_batch(
                batch_size=current_batch_size,
                denormalize=True
            )
            
            # 保存数据
            for i in range(current_batch_size):
                sample_idx = total_generated + i
                
                if save_format == 'npy':
                    np.save(
                        velocity_dir / f'velocity_{sample_idx:06d}.npy',
                        velocity[i].cpu().numpy()
                    )
                    np.save(
                        seismic_dir / f'seismic_{sample_idx:06d}.npy',
                        seismic[i].cpu().numpy()
                    )
                elif save_format == 'pt':
                    torch.save(
                        velocity[i].cpu(),
                        velocity_dir / f'velocity_{sample_idx:06d}.pt'
                    )
                    torch.save(
                        seismic[i].cpu(),
                        seismic_dir / f'seismic_{sample_idx:06d}.pt'
                    )
            
            total_generated += current_batch_size
            
            # 进度显示
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - start_time
                progress = total_generated / num_samples * 100
                print(f"进度: {progress:.1f}% ({total_generated}/{num_samples}) "
                      f"时间: {elapsed:.1f}s")
        
        print(f"生成完成! 保存位置: {output_path}")
        
        # 保存生成信息
        info = {
            'dataset_name': self.dataset_name,
            'num_samples': total_generated,
            'batch_size': batch_size,
            'save_format': save_format,
            'dataset_context': self.dataset_ctx,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        import json
        with open(output_path / 'generation_info.json', 'w') as f:
            json.dump(info, f, indent=2)

    def reconstruct_from_velocity(self, 
                                 velocity: torch.Tensor,
                                 denormalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """从速度场重构数据
        
        Args:
            velocity: 输入速度场
            denormalize: 是否反标准化
            
        Returns:
            (reconstructed_velocity, reconstructed_seismic): 重构的数据
        """
        with torch.no_grad():
            velocity = velocity.to(self.device)
            
            # 重构
            pred_velocity, pred_seismic = self.model.reconstruct(velocity)
            
            if denormalize:
                # 反标准化速度场
                pred_velocity = minmax_denormalize(
                    pred_velocity.cpu(),
                    self.dataset_ctx['label_min'],
                    self.dataset_ctx['label_max']
                )
                
                # 反标准化地震数据
                seismic_denorm = tonumpy_denormalize(
                    pred_seismic,
                    self.dataset_ctx['data_min'],
                    self.dataset_ctx['data_max']
                )
                pred_seismic = torch.from_numpy(seismic_denorm)
            
            return pred_velocity, pred_seismic

    def interpolate_in_latent_space(self,
                                   velocity1: torch.Tensor,
                                   velocity2: torch.Tensor,
                                   num_steps: int = 10) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """在潜在空间中进行插值
        
        Args:
            velocity1: 起始速度场
            velocity2: 结束速度场
            num_steps: 插值步数
            
        Returns:
            插值结果列表 [(velocity, seismic), ...]
        """
        with torch.no_grad():
            velocity1 = velocity1.to(self.device)
            velocity2 = velocity2.to(self.device)
            
            # 编码到潜在空间
            z1 = self.model.encoder(velocity1)
            z2 = self.model.encoder(velocity2)
            
            results = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                
                # 解码
                velocity_interp, seismic_interp = self.model.decode(z_interp)
                
                # 反标准化
                velocity_denorm = minmax_denormalize(
                    velocity_interp.cpu(),
                    self.dataset_ctx['label_min'],
                    self.dataset_ctx['label_max']
                )
                
                seismic_denorm = tonumpy_denormalize(
                    seismic_interp,
                    self.dataset_ctx['data_min'],
                    self.dataset_ctx['data_max']
                )
                seismic_denorm = torch.from_numpy(seismic_denorm)
                
                results.append((velocity_denorm, seismic_denorm))
            
            return results

    def evaluate_quality(self, 
                        real_velocity: torch.Tensor,
                        real_seismic: torch.Tensor,
                        num_generated: int = 100) -> Dict[str, float]:
        """评估生成质量
        
        Args:
            real_velocity: 真实速度场样本
            real_seismic: 真实地震数据样本
            num_generated: 用于评估的生成样本数
            
        Returns:
            质量评估指标字典
        """
        print(f"开始质量评估，生成 {num_generated} 个样本...")
        
        # 生成样本
        gen_velocity_list = []
        gen_seismic_list = []
        
        batch_size = 16
        num_batches = (num_generated + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_generated - len(gen_velocity_list))
            gen_vel, gen_seis = self.generate_batch(
                batch_size=current_batch_size,
                denormalize=True
            )
            gen_velocity_list.append(gen_vel)
            gen_seismic_list.append(gen_seis)
        
        gen_velocity = torch.cat(gen_velocity_list, dim=0)
        gen_seismic = torch.cat(gen_seismic_list, dim=0)
        
        # 计算统计指标
        metrics = {}
        
        # 速度场统计
        real_vel_mean = real_velocity.mean().item()
        real_vel_std = real_velocity.std().item()
        gen_vel_mean = gen_velocity.mean().item()
        gen_vel_std = gen_velocity.std().item()
        
        metrics['velocity_mean_diff'] = abs(real_vel_mean - gen_vel_mean)
        metrics['velocity_std_diff'] = abs(real_vel_std - gen_vel_std)
        
        # 地震数据统计
        real_seis_mean = real_seismic.mean().item()
        real_seis_std = real_seismic.std().item()
        gen_seis_mean = gen_seismic.mean().item()
        gen_seis_std = gen_seismic.std().item()
        
        metrics['seismic_mean_diff'] = abs(real_seis_mean - gen_seis_mean)
        metrics['seismic_std_diff'] = abs(real_seis_std - gen_seis_std)
        
        # 计算重构质量（如果可能）
        if len(real_velocity) > 0:
            sample_vel = real_velocity[:min(16, len(real_velocity))]
            sample_seis = real_seismic[:min(16, len(real_seismic))]
            
            # 标准化用于重构
            from ..data import MinMaxNormalize
            vel_normalizer = MinMaxNormalize(
                self.dataset_ctx['label_min'],
                self.dataset_ctx['label_max']
            )
            normalized_vel = vel_normalizer(sample_vel)
            
            recon_vel, recon_seis = self.reconstruct_from_velocity(
                normalized_vel, denormalize=True
            )
            
            metrics['reconstruction_velocity_mse'] = torch.nn.functional.mse_loss(
                recon_vel, sample_vel
            ).item()
            metrics['reconstruction_seismic_mse'] = torch.nn.functional.mse_loss(
                recon_seis, sample_seis
            ).item()
        
        print("质量评估完成:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")
        
        return metrics

    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要信息"""
        from ..trainers.utils import count_parameters
        
        param_counts = count_parameters(self.model)
        
        summary = {
            'model_type': 'UB-Diff',
            'dataset': self.dataset_name,
            'device': str(self.device),
            'parameters': param_counts,
            'dataset_context': self.dataset_ctx,
            'model_components': {
                'encoder_dim': self.model.encoder.dim5 if hasattr(self.model.encoder, 'dim5') else 'unknown',
                'time_steps': getattr(self.model.diffusion, 'num_timesteps', 'unknown'),
                'objective': getattr(self.model.diffusion, 'objective', 'unknown')
            }
        }
        
        return summary 