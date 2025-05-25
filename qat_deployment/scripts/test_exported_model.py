#!/usr/bin/env python3
"""
测试导出的TorchScript模型

用于验证模型在树莓派上的功能
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='测试导出的模型')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='TorchScript模型路径')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='生成样本数量')
    parser.add_argument('--save_dir', type=str, default='./test_outputs',
                        help='输出保存目录')
    parser.add_argument('--device', type=str, default='cpu',
                        help='运行设备')
    parser.add_argument('--benchmark', action='store_true',
                        help='运行性能基准测试')
    
    return parser.parse_args()


def visualize_results(velocity, seismic, save_path):
    """可视化生成结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 可视化速度场
    axes[0].imshow(velocity[0, 0].cpu().numpy(), cmap='viridis', aspect='auto')
    axes[0].set_title('Generated Velocity Field')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Z')
    
    # 可视化地震数据（第一个通道）
    axes[1].imshow(seismic[0, 0].cpu().numpy(), cmap='seismic', aspect='auto')
    axes[1].set_title('Generated Seismic Data (Channel 1)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Time')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"可视化结果已保存到: {save_path}")


def benchmark_model(model, device, num_runs=10):
    """基准测试模型性能"""
    print("\n运行性能基准测试...")
    
    # 预热
    print("预热中...")
    for _ in range(3):
        with torch.no_grad():
            _ = model(1)
    
    # 测试不同批次大小
    batch_sizes = [1, 2, 4] if device == 'cuda' else [1]
    
    results = {}
    for batch_size in batch_sizes:
        times = []
        
        for i in range(num_runs):
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(batch_size)
            
            torch.cuda.synchronize() if device == 'cuda' else None
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results[batch_size] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'throughput': batch_size / avg_time
        }
        
        print(f"\nBatch Size: {batch_size}")
        print(f"  平均时间: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"  吞吐量: {batch_size / avg_time:.2f} samples/s")
    
    return results


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    try:
        model = torch.jit.load(args.model_path, map_location=device)
        model.eval()
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 生成样本
    print(f"\n生成 {args.num_samples} 个样本...")
    start_time = time.time()
    
    with torch.no_grad():
        velocity, seismic = model(args.num_samples)
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    print(f"生成完成！耗时: {generation_time:.3f}s")
    print(f"速度场形状: {velocity.shape}")
    print(f"地震数据形状: {seismic.shape}")
    
    # 保存结果
    save_dict = {
        'velocity': velocity.cpu().numpy(),
        'seismic': seismic.cpu().numpy(),
        'generation_time': generation_time,
        'device': str(device)
    }
    
    save_path = os.path.join(args.save_dir, 'generated_data.npz')
    np.savez(save_path, **save_dict)
    print(f"\n数据已保存到: {save_path}")
    
    # 可视化第一个样本
    if args.num_samples > 0:
        vis_path = os.path.join(args.save_dir, 'visualization.png')
        visualize_results(velocity, seismic, vis_path)
    
    # 运行基准测试
    if args.benchmark:
        benchmark_results = benchmark_model(model, args.device)
        
        # 保存基准测试结果
        import json
        benchmark_path = os.path.join(args.save_dir, 'benchmark_results.json')
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"\n基准测试结果已保存到: {benchmark_path}")
    
    # 内存使用统计
    if args.device == 'cpu':
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"\n内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
    else:
        print(f"\nGPU内存使用: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
    
    print("\n测试完成！")


if __name__ == '__main__':
    main() 