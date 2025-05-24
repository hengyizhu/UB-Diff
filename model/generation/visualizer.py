"""
模型可视化工具

用于展示生成结果、训练进度和模型性能
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import seaborn as sns


class ModelVisualizer:
    """模型可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Args:
            figsize: 图形大小
            dpi: 图形分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_velocity_field(self, 
                           velocity: np.ndarray,
                           title: str = "速度场",
                           save_path: Optional[str] = None,
                           show_colorbar: bool = True) -> None:
        """绘制速度场
        
        Args:
            velocity: 速度场数据 (H, W) 或 (C, H, W)
            title: 图标题
            save_path: 保存路径
            show_colorbar: 是否显示颜色条
        """
        if velocity.ndim == 3:
            velocity = velocity[0]  # 取第一个通道
        elif velocity.ndim > 3:
            velocity = velocity[0, 0]  # 取第一个样本的第一个通道
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        
        im = ax.imshow(velocity, cmap='seismic', aspect='auto')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('横向位置', fontsize=12)
        ax.set_ylabel('深度', fontsize=12)
        
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('速度 (m/s)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        plt.show()

    def plot_seismic_data(self,
                         seismic: np.ndarray,
                         title: str = "地震数据",
                         save_path: Optional[str] = None,
                         aspect_ratio: float = 0.5) -> None:
        """绘制地震数据
        
        Args:
            seismic: 地震数据 (C, T, X) 或 (T, X)
            title: 图标题
            save_path: 保存路径
            aspect_ratio: 纵横比
        """
        if seismic.ndim == 3:
            seismic = seismic[0]  # 取第一个通道
        elif seismic.ndim > 3:
            seismic = seismic[0, 0]  # 取第一个样本的第一个通道
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        
        im = ax.imshow(seismic, cmap='seismic', aspect=aspect_ratio, 
                      extent=[0, seismic.shape[1], seismic.shape[0], 0])
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('空间位置', fontsize=12)
        ax.set_ylabel('时间', fontsize=12)
        
        plt.colorbar(im, ax=ax, label='振幅')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        plt.show()

    def plot_comparison(self,
                       real_velocity: np.ndarray,
                       generated_velocity: np.ndarray,
                       real_seismic: np.ndarray,
                       generated_seismic: np.ndarray,
                       save_path: Optional[str] = None) -> None:
        """对比真实数据和生成数据
        
        Args:
            real_velocity: 真实速度场
            generated_velocity: 生成速度场
            real_seismic: 真实地震数据
            generated_seismic: 生成地震数据
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
        
        # 处理维度
        if real_velocity.ndim > 2:
            real_velocity = real_velocity[0] if real_velocity.ndim == 3 else real_velocity[0, 0]
        if generated_velocity.ndim > 2:
            generated_velocity = generated_velocity[0] if generated_velocity.ndim == 3 else generated_velocity[0, 0]
        if real_seismic.ndim > 2:
            real_seismic = real_seismic[0] if real_seismic.ndim == 3 else real_seismic[0, 0]
        if generated_seismic.ndim > 2:
            generated_seismic = generated_seismic[0] if generated_seismic.ndim == 3 else generated_seismic[0, 0]
        
        # 真实速度场
        im1 = axes[0, 0].imshow(real_velocity, cmap='seismic', aspect='auto')
        axes[0, 0].set_title('真实速度场', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('横向位置')
        axes[0, 0].set_ylabel('深度')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 生成速度场
        im2 = axes[0, 1].imshow(generated_velocity, cmap='seismic', aspect='auto')
        axes[0, 1].set_title('生成速度场', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('横向位置')
        axes[0, 1].set_ylabel('深度')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 真实地震数据
        im3 = axes[1, 0].imshow(real_seismic, cmap='seismic', aspect=0.5)
        axes[1, 0].set_title('真实地震数据', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('空间位置')
        axes[1, 0].set_ylabel('时间')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 生成地震数据
        im4 = axes[1, 1].imshow(generated_seismic, cmap='seismic', aspect=0.5)
        axes[1, 1].set_title('生成地震数据', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('空间位置')
        axes[1, 1].set_ylabel('时间')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        plt.show()

    def plot_interpolation(self,
                          interpolation_results: List[Tuple[np.ndarray, np.ndarray]],
                          save_path: Optional[str] = None) -> None:
        """绘制潜在空间插值结果
        
        Args:
            interpolation_results: 插值结果列表 [(velocity, seismic), ...]
            save_path: 保存路径
        """
        num_steps = len(interpolation_results)
        
        fig, axes = plt.subplots(2, num_steps, figsize=(3*num_steps, 8), dpi=self.dpi)
        
        if num_steps == 1:
            axes = axes.reshape(2, 1)
        
        for i, (velocity, seismic) in enumerate(interpolation_results):
            # 处理维度
            if velocity.ndim > 2:
                velocity = velocity[0] if velocity.ndim == 3 else velocity[0, 0]
            if seismic.ndim > 2:
                seismic = seismic[0] if seismic.ndim == 3 else seismic[0, 0]
            
            # 速度场
            im1 = axes[0, i].imshow(velocity, cmap='seismic', aspect='auto')
            axes[0, i].set_title(f'步骤 {i+1}', fontsize=12)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            
            # 地震数据
            im2 = axes[1, i].imshow(seismic, cmap='seismic', aspect=0.5)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
        
        # 添加行标签
        axes[0, 0].set_ylabel('速度场', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('地震数据', fontsize=14, fontweight='bold')
        
        plt.suptitle('潜在空间插值结果', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        plt.show()

    def plot_training_metrics(self,
                             metrics_history: Dict[str, List[float]],
                             save_path: Optional[str] = None) -> None:
        """绘制训练指标
        
        Args:
            metrics_history: 指标历史 {'loss': [...], 'ssim': [...], ...}
            save_path: 保存路径
        """
        num_metrics = len(metrics_history)
        
        if num_metrics == 0:
            print("没有指标数据可显示")
            return
        
        # 计算子图布局
        ncols = min(3, num_metrics)
        nrows = (num_metrics + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), dpi=self.dpi)
        
        if num_metrics == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes if num_metrics > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (metric_name, values) in enumerate(metrics_history.items()):
            if i < len(axes):
                axes[i].plot(values, linewidth=2, marker='o', markersize=3)
                axes[i].set_title(f'{metric_name}', fontsize=14, fontweight='bold')
                axes[i].set_xlabel('Epoch/Step', fontsize=12)
                axes[i].set_ylabel(metric_name, fontsize=12)
                axes[i].grid(True, alpha=0.3)
                
                # 添加最佳值标记
                if 'loss' in metric_name.lower():
                    best_idx = np.argmin(values)
                    best_val = values[best_idx]
                else:
                    best_idx = np.argmax(values)
                    best_val = values[best_idx]
                
                axes[i].plot(best_idx, best_val, 'ro', markersize=8, alpha=0.7)
                axes[i].annotate(f'最佳: {best_val:.4f}', 
                               xy=(best_idx, best_val),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # 隐藏多余的子图
        for i in range(len(metrics_history), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        plt.show()

    def plot_quality_metrics(self,
                            metrics: Dict[str, float],
                            save_path: Optional[str] = None) -> None:
        """绘制质量评估指标
        
        Args:
            metrics: 质量指标字典
            save_path: 保存路径
        """
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        
        bars = ax.bar(metric_names, metric_values, alpha=0.8)
        
        # 为每个条形添加数值标签
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title('模型质量评估指标', fontsize=16, fontweight='bold')
        ax.set_ylabel('指标值', fontsize=12)
        
        # 旋转x轴标签以避免重叠
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        plt.show()

    def plot_data_distribution(self,
                              real_data: np.ndarray,
                              generated_data: np.ndarray,
                              data_type: str = "数据",
                              save_path: Optional[str] = None) -> None:
        """绘制数据分布对比
        
        Args:
            real_data: 真实数据
            generated_data: 生成数据
            data_type: 数据类型名称
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=self.dpi)
        
        # 直方图对比
        axes[0].hist(real_data.flatten(), bins=50, alpha=0.7, label='真实数据', density=True)
        axes[0].hist(generated_data.flatten(), bins=50, alpha=0.7, label='生成数据', density=True)
        axes[0].set_title(f'{data_type}分布对比', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('数值', fontsize=12)
        axes[0].set_ylabel('密度', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q图
        from scipy import stats
        real_sorted = np.sort(real_data.flatten())
        gen_sorted = np.sort(generated_data.flatten())
        
        # 确保两个数组长度相同
        min_len = min(len(real_sorted), len(gen_sorted))
        real_quantiles = real_sorted[::len(real_sorted)//min_len][:min_len]
        gen_quantiles = gen_sorted[::len(gen_sorted)//min_len][:min_len]
        
        axes[1].scatter(real_quantiles, gen_quantiles, alpha=0.6, s=1)
        
        # 添加理想对角线
        min_val = min(real_quantiles.min(), gen_quantiles.min())
        max_val = max(real_quantiles.max(), gen_quantiles.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        axes[1].set_title(f'{data_type} Q-Q图', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('真实数据分位数', fontsize=12)
        axes[1].set_ylabel('生成数据分位数', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        plt.show() 