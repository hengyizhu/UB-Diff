#!/usr/bin/env python3
"""
测试curvefault-a量化模型的简化脚本
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def test_model():
    print("=== CurveFault-A 模型测试 ===")
    
    # 模型基本信息
    model_path = "qat_deployment/exported_models/curvefault-a.pt"
    save_dir = "test_results"
    
    # 创建输出目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cpu')  # 使用CPU避免GPU内存问题
    print(f"使用设备: {device}")
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 {model_path}")
        return
    
    file_size = os.path.getsize(model_path) / (1024**3)
    print(f"模型文件大小: {file_size:.2f} GB")
    
    # 加载模型
    print("\n正在加载模型...")
    start_time = time.time()
    
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        load_time = time.time() - start_time
        print(f"模型加载成功！耗时: {load_time:.2f}s")
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 模型信息
    print(f"\n模型类型: TorchScript (量化)")
    print(f"量化后端: qnnpack")
    
    # 测试推理
    print("\n开始模型推理测试...")
    num_samples = 2  # 减少样本数量以节省时间
    
    try:
        inference_start = time.time()
        
        with torch.no_grad():
            # 根据错误信息，需要传递tensor作为batch_size
            batch_size_tensor = torch.tensor(num_samples)
            velocity, seismic = model(batch_size_tensor)
        
        inference_time = time.time() - inference_start
        
        print(f"推理完成！")
        print(f"推理时间: {inference_time:.3f}s")
        print(f"平均每样本时间: {inference_time/num_samples:.3f}s")
        print(f"速度场输出形状: {velocity.shape}")
        print(f"地震数据输出形状: {seismic.shape}")
        
        # 保存数据
        save_data(velocity, seismic, save_dir, inference_time)
        
        # 可视化结果
        visualize_results(velocity, seismic, save_dir)
        
        # 简单性能测试
        performance_test(model, device)
        
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()

def save_data(velocity, seismic, save_dir, inference_time):
    """保存生成的数据"""
    print("\n保存数据...")
    
    save_dict = {
        'velocity': velocity.cpu().numpy(),
        'seismic': seismic.cpu().numpy(),
        'inference_time': inference_time,
        'velocity_stats': {
            'min': velocity.min().item(),
            'max': velocity.max().item(),
            'mean': velocity.mean().item(),
            'std': velocity.std().item()
        },
        'seismic_stats': {
            'min': seismic.min().item(),
            'max': seismic.max().item(),
            'mean': seismic.mean().item(),
            'std': seismic.std().item()
        }
    }
    
    save_path = os.path.join(save_dir, 'curvefault_results.npz')
    np.savez(save_path, **save_dict)
    print(f"数据已保存到: {save_path}")
    
    # 打印统计信息
    print("\n数据统计:")
    print(f"速度场 - 最小值: {save_dict['velocity_stats']['min']:.4f}, "
          f"最大值: {save_dict['velocity_stats']['max']:.4f}, "
          f"均值: {save_dict['velocity_stats']['mean']:.4f}")
    print(f"地震数据 - 最小值: {save_dict['seismic_stats']['min']:.4f}, "
          f"最大值: {save_dict['seismic_stats']['max']:.4f}, "
          f"均值: {save_dict['seismic_stats']['mean']:.4f}")

def visualize_results(velocity, seismic, save_dir):
    """可视化生成结果"""
    print("\n生成可视化图表...")
    
    num_samples = velocity.shape[0]
    
    # 创建综合可视化
    fig = plt.figure(figsize=(16, 10))
    
    for i in range(min(num_samples, 2)):  # 最多显示2个样本
        # 速度场可视化
        ax1 = plt.subplot(2, 4, i*4 + 1)
        vel_data = velocity[i, 0].cpu().numpy()
        im1 = ax1.imshow(vel_data, cmap='viridis', aspect='auto')
        ax1.set_title(f'样本 {i+1}: 速度场')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 地震数据可视化 - 显示多个通道
        for ch in range(min(3, seismic.shape[1])):  # 最多显示3个通道
            ax = plt.subplot(2, 4, i*4 + 2 + ch)
            seis_data = seismic[i, ch].cpu().numpy()
            im = ax.imshow(seis_data, cmap='seismic', aspect='auto', 
                          vmin=-np.abs(seis_data).max(), vmax=np.abs(seis_data).max())
            ax.set_title(f'样本 {i+1}: 地震数据 Ch{ch+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('时间')
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    
    # 保存综合图
    main_vis_path = os.path.join(save_dir, 'curvefault_visualization.png')
    plt.savefig(main_vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"主要可视化已保存到: {main_vis_path}")
    
    # 创建统计图表
    create_statistics_plot(velocity, seismic, save_dir)

def create_statistics_plot(velocity, seismic, save_dir):
    """创建数据统计可视化"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 速度场统计
    vel_data = velocity.cpu().numpy().flatten()
    axes[0, 0].hist(vel_data, bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('速度场值分布')
    axes[0, 0].set_xlabel('速度值')
    axes[0, 0].set_ylabel('频率')
    
    # 地震数据统计
    seis_data = seismic.cpu().numpy().flatten()
    axes[0, 1].hist(seis_data, bins=50, alpha=0.7, color='red')
    axes[0, 1].set_title('地震数据值分布')
    axes[0, 1].set_xlabel('振幅值')
    axes[0, 1].set_ylabel('频率')
    
    # 速度场空间平均
    vel_spatial_mean = velocity.mean(dim=(0, 1)).cpu().numpy()
    axes[0, 2].plot(vel_spatial_mean)
    axes[0, 2].set_title('速度场空间平均')
    axes[0, 2].set_xlabel('深度')
    axes[0, 2].set_ylabel('平均速度')
    
    # 地震数据时间平均
    seis_temporal_mean = seismic.mean(dim=(0, 2)).cpu().numpy()
    for ch in range(min(3, len(seis_temporal_mean))):
        axes[1, 0].plot(seis_temporal_mean[ch], label=f'通道 {ch+1}')
    axes[1, 0].set_title('地震数据时间平均')
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_ylabel('平均振幅')
    axes[1, 0].legend()
    
    # 数据质量指标
    quality_metrics = {
        '速度场方差': velocity.var().item(),
        '地震数据方差': seismic.var().item(),
        '速度场动态范围': (velocity.max() - velocity.min()).item(),
        '地震数据动态范围': (seismic.max() - seismic.min()).item()
    }
    
    metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in quality_metrics.items()])
    axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='center')
    axes[1, 1].set_title('数据质量指标')
    axes[1, 1].axis('off')
    
    # 模型输出尺寸信息
    info_text = f"""模型输出信息:
速度场形状: {velocity.shape}
地震数据形状: {seismic.shape}
数据类型: {velocity.dtype}
设备: {velocity.device}"""
    
    axes[1, 2].text(0.1, 0.5, info_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='center')
    axes[1, 2].set_title('输出信息')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    stats_path = os.path.join(save_dir, 'curvefault_statistics.png')
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"统计图表已保存到: {stats_path}")

def performance_test(model, device, num_runs=5):
    """简单的性能测试"""
    print(f"\n运行性能测试 ({num_runs} 次)...")
    
    times = []
    batch_size = 1
    
    # 预热
    with torch.no_grad():
        batch_size_tensor = torch.tensor(batch_size)
        _ = model(batch_size_tensor)
    
    # 性能测试
    for i in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            batch_size_tensor = torch.tensor(batch_size)
            _ = model(batch_size_tensor)
        
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"  运行 {i+1}: {times[-1]:.3f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n性能结果:")
    print(f"  平均时间: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"  吞吐量: {batch_size / avg_time:.2f} 样本/秒")
    
    return avg_time, std_time

if __name__ == "__main__":
    test_model() 