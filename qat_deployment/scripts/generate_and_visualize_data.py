#!/usr/bin/env python3
"""
使用curvefault-a模型生成并可视化数据

该脚本加载量化的TorchScript模型，生成速度场和地震数据，并进行详细的可视化分析。
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class CurveFaultVisualizer:
    """CurveFault模型专用可视化器"""
    
    def __init__(self, figsize=(15, 10), dpi=150):
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use('default')
        sns.set_palette("husl")
    
    def load_model(self, model_path):
        """加载TorchScript模型"""
        try:
            print(f"正在加载模型: {model_path}")
            model = torch.jit.load(model_path, map_location='cpu')
            model.eval()
            print("✅ 模型加载成功！")
            return model
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None
    
    def generate_data(self, model, num_samples=1):
        """使用模型生成数据"""
        print(f"正在生成 {num_samples} 个样本...")
        
        start_time = time.time()
        with torch.no_grad():
            # 根据部署指南，使用int32类型的tensor作为输入
            velocity, seismic = model(torch.tensor(num_samples, dtype=torch.int32))
        
        generation_time = time.time() - start_time
        
        print(f"✅ 生成完成！耗时: {generation_time:.3f}秒")
        print(f"   速度场形状: {velocity.shape}")  # 应该是 (num_samples, 1, 70, 70)
        print(f"   地震数据形状: {seismic.shape}")  # 应该是 (num_samples, 5, 1000, 70)
        
        return velocity.cpu().numpy(), seismic.cpu().numpy(), generation_time
    
    def plot_single_sample(self, velocity, seismic, sample_idx=0, save_path=None):
        """可视化单个样本的详细结果"""
        fig = plt.figure(figsize=(18, 12))
        
        # 获取单个样本数据
        vel_sample = velocity[sample_idx, 0]  # (70, 70)
        seismic_sample = seismic[sample_idx]  # (5, 1000, 70)
        
        # 1. 速度场
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(vel_sample, cmap='viridis', aspect='auto')
        ax1.set_title(f'速度场 (样本 {sample_idx+1})', fontsize=14, fontweight='bold')
        ax1.set_xlabel('横向位置 (网格点)')
        ax1.set_ylabel('深度 (网格点)')
        plt.colorbar(im1, ax=ax1, label='速度值')
        
        # 2. 速度场统计信息
        ax2 = plt.subplot(2, 3, 2)
        vel_flat = vel_sample.flatten()
        ax2.hist(vel_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('速度场数值分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('速度值')
        ax2.set_ylabel('频次')
        ax2.axvline(vel_flat.mean(), color='red', linestyle='--', 
                   label=f'均值: {vel_flat.mean():.3f}')
        ax2.axvline(np.median(vel_flat), color='orange', linestyle='--', 
                   label=f'中位数: {np.median(vel_flat):.3f}')
        ax2.legend()
        
        # 3. 地震数据 - 第一个通道
        ax3 = plt.subplot(2, 3, 3)
        im3 = ax3.imshow(seismic_sample[0], cmap='seismic', aspect='auto')
        ax3.set_title('地震数据 - 通道 1', fontsize=14, fontweight='bold')
        ax3.set_xlabel('空间位置 (网格点)')
        ax3.set_ylabel('时间 (采样点)')
        plt.colorbar(im3, ax=ax3, label='振幅')
        
        # 4. 地震数据 - 第三个通道
        ax4 = plt.subplot(2, 3, 4)
        im4 = ax4.imshow(seismic_sample[2], cmap='seismic', aspect='auto')
        ax4.set_title('地震数据 - 通道 3', fontsize=14, fontweight='bold')
        ax4.set_xlabel('空间位置 (网格点)')
        ax4.set_ylabel('时间 (采样点)')
        plt.colorbar(im4, ax=ax4, label='振幅')
        
        # 5. 地震数据所有通道的平均值
        ax5 = plt.subplot(2, 3, 5)
        seismic_mean = np.mean(seismic_sample, axis=0)  # 对通道维度求平均
        im5 = ax5.imshow(seismic_mean, cmap='seismic', aspect='auto')
        ax5.set_title('地震数据 - 通道平均', fontsize=14, fontweight='bold')
        ax5.set_xlabel('空间位置 (网格点)')
        ax5.set_ylabel('时间 (采样点)')
        plt.colorbar(im5, ax=ax5, label='平均振幅')
        
        # 6. 地震数据的时间序列（中间空间位置）
        ax6 = plt.subplot(2, 3, 6)
        middle_pos = seismic_sample.shape[2] // 2  # 中间位置
        for i, channel in enumerate(seismic_sample):
            ax6.plot(channel[:, middle_pos], label=f'通道 {i+1}', alpha=0.8)
        ax6.set_title(f'地震时间序列 (位置 {middle_pos})', fontsize=14, fontweight='bold')
        ax6.set_xlabel('时间 (采样点)')
        ax6.set_ylabel('振幅')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            print(f"详细可视化已保存到: {save_path}")
        
        plt.show()
    
    def plot_multi_samples(self, velocity, seismic, num_show=4, save_path=None):
        """可视化多个样本"""
        num_samples = min(num_show, velocity.shape[0])
        
        fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
        
        if num_samples == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(num_samples):
            vel_sample = velocity[i, 0]
            seismic_sample = seismic[i, 0]  # 只显示第一个通道
            
            # 速度场
            im1 = axes[0, i].imshow(vel_sample, cmap='viridis', aspect='auto')
            axes[0, i].set_title(f'速度场 - 样本 {i+1}', fontsize=12)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            
            # 地震数据
            im2 = axes[1, i].imshow(seismic_sample, cmap='seismic', aspect='auto')
            axes[1, i].set_title(f'地震数据 - 样本 {i+1}', fontsize=12)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
        
        # 添加颜色条
        plt.colorbar(im1, ax=axes[0, :], label='速度值', shrink=0.8)
        plt.colorbar(im2, ax=axes[1, :], label='振幅', shrink=0.8)
        
        plt.suptitle('多样本对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            print(f"多样本可视化已保存到: {save_path}")
        
        plt.show()
    
    def analyze_data_statistics(self, velocity, seismic):
        """分析生成数据的统计信息"""
        print("\n📊 数据统计分析:")
        print("=" * 50)
        
        # 速度场统计
        vel_stats = {
            '最小值': np.min(velocity),
            '最大值': np.max(velocity),
            '均值': np.mean(velocity),
            '标准差': np.std(velocity),
            '中位数': np.median(velocity)
        }
        
        print("🏔️  速度场统计:")
        for key, value in vel_stats.items():
            print(f"   {key}: {value:.6f}")
        
        # 地震数据统计
        seismic_stats = {
            '最小值': np.min(seismic),
            '最大值': np.max(seismic),
            '均值': np.mean(seismic),
            '标准差': np.std(seismic),
            '中位数': np.median(seismic)
        }
        
        print("\n🌊 地震数据统计:")
        for key, value in seismic_stats.items():
            print(f"   {key}: {value:.6f}")
        
        # 按通道分析地震数据
        print("\n📡 各通道地震数据统计:")
        for i in range(seismic.shape[1]):  # 5个通道
            channel_data = seismic[:, i, :, :]
            print(f"   通道 {i+1}: 均值={np.mean(channel_data):.6f}, "
                  f"标准差={np.std(channel_data):.6f}")
        
        return vel_stats, seismic_stats

def main():
    """主函数"""
    print("🚀 CurveFault-A 模型数据生成与可视化")
    print("=" * 60)
    
    # 初始化可视化器
    visualizer = CurveFaultVisualizer()
    
    # 模型路径
    model_path = "qat_deployment/exported_models/curvefault-a.pt"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 加载模型
    model = visualizer.load_model(model_path)
    if model is None:
        return
    
    # 创建输出目录
    output_dir = "generated_visualization_outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")
    
    # 生成数据
    print("\n" + "="*60)
    num_samples = 3  # 生成3个样本进行对比
    velocity, seismic, generation_time = visualizer.generate_data(model, num_samples)
    
    # 数据统计分析
    vel_stats, seismic_stats = visualizer.analyze_data_statistics(velocity, seismic)
    
    # 可视化
    print("\n" + "="*60)
    print("🎨 开始可视化...")
    
    # 1. 详细的单样本可视化
    single_sample_path = os.path.join(output_dir, "detailed_sample_analysis.png")
    visualizer.plot_single_sample(velocity, seismic, sample_idx=0, 
                                 save_path=single_sample_path)
    
    # 2. 多样本对比
    multi_sample_path = os.path.join(output_dir, "multi_sample_comparison.png")
    visualizer.plot_multi_samples(velocity, seismic, num_show=num_samples, 
                                 save_path=multi_sample_path)
    
    # 保存原始数据
    data_path = os.path.join(output_dir, "generated_data.npz")
    np.savez(data_path, 
             velocity=velocity, 
             seismic=seismic,
             generation_time=generation_time,
             velocity_stats=vel_stats,
             seismic_stats=seismic_stats)
    print(f"💾 原始数据已保存到: {data_path}")
    
    # 性能总结
    print("\n" + "="*60)
    print("📈 性能总结:")
    print(f"   生成样本数: {num_samples}")
    print(f"   总生成时间: {generation_time:.3f}秒")
    print(f"   平均每样本: {generation_time/num_samples:.3f}秒")
    print(f"   吞吐量: {num_samples/generation_time:.2f} 样本/秒")
    
    print("\n✅ 所有可视化完成！请查看生成的图像文件。")

if __name__ == "__main__":
    main() 