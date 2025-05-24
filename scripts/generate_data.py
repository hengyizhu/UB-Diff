#!/usr/bin/env python3
"""
数据生成脚本

使用训练好的UB-Diff模型生成地震数据和速度场
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.generation import UBDiffGenerator, ModelVisualizer
from model.trainers.utils import setup_seed


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='UB-Diff数据生成')
    
    # 模型参数
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='训练好的完整模型检查点路径')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['flatvel-a', 'flatvel-b', 'curvefault-a', 'curvefault-b', 
                               'flatfault-a', 'flatfault-b'],
                       help='数据集名称')
    
    # 生成参数
    parser.add_argument('--num_samples', type=int, default=100,
                       help='生成样本数量')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='生成批次大小')
    parser.add_argument('--output_dir', type=str, default='./generated_data',
                       help='生成数据保存目录')
    parser.add_argument('--save_format', type=str, default='npy', 
                       choices=['npy', 'pt'],
                       help='保存格式')
    
    # 模型结构参数
    parser.add_argument('--encoder_dim', type=int, default=512,
                       help='编码器维度')
    parser.add_argument('--time_steps', type=int, default=256,
                       help='扩散时间步数')
    parser.add_argument('--dim_mults', type=int, nargs='+', default=[1, 2, 4, 8],
                       help='U-Net维度倍数')
    parser.add_argument('--objective', type=str, default='pred_v',
                       choices=['pred_noise', 'pred_x0', 'pred_v'],
                       help='扩散目标')
    
    # 可视化参数
    parser.add_argument('--visualize', action='store_true',
                       help='是否生成可视化图像')
    parser.add_argument('--num_visualize', type=int, default=5,
                       help='可视化样本数量')
    
    # 质量评估参数
    parser.add_argument('--evaluate_quality', action='store_true',
                       help='是否进行质量评估')
    parser.add_argument('--real_data_path', type=str,
                       help='真实数据路径（用于质量评估）')
    parser.add_argument('--real_label_path', type=str,
                       help='真实标签路径（用于质量评估）')
    parser.add_argument('--num_eval_samples', type=int, default=100,
                       help='用于质量评估的样本数')
    
    # 系统参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='生成设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    return parser.parse_args()


def load_real_data_for_evaluation(seismic_path: str, velocity_path: str, 
                                 num_samples: int = 100):
    """加载真实数据用于质量评估"""
    import numpy as np
    import torch
    from model.data import SeismicVelocityDataset, DatasetConfig
    
    # 简单加载一些样本用于评估
    seismic_files = sorted([f for f in os.listdir(seismic_path) if f.endswith('.npy')])[:5]
    velocity_files = sorted([f for f in os.listdir(velocity_path) if f.endswith('.npy')])[:5]
    
    seismic_data = []
    velocity_data = []
    
    for i, (s_file, v_file) in enumerate(zip(seismic_files, velocity_files)):
        if len(seismic_data) >= num_samples:
            break
            
        s_data = np.load(os.path.join(seismic_path, s_file))
        v_data = np.load(os.path.join(velocity_path, v_file))
        
        # 取前几个样本
        samples_to_take = min(num_samples - len(seismic_data), s_data.shape[0])
        seismic_data.append(torch.from_numpy(s_data[:samples_to_take]))
        velocity_data.append(torch.from_numpy(v_data[:samples_to_take]))
    
    return torch.cat(seismic_data, dim=0), torch.cat(velocity_data, dim=0)


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置随机种子
    setup_seed(args.seed)
    
    print("="*50)
    print("UB-Diff 数据生成")
    print("="*50)
    print(f"模型路径: {args.checkpoint_path}")
    print(f"数据集: {args.dataset}")
    print(f"生成样本数: {args.num_samples}")
    print(f"批次大小: {args.batch_size}")
    print(f"输出目录: {args.output_dir}")
    print(f"保存格式: {args.save_format}")
    print(f"设备: {args.device}")
    print("="*50)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 模型文件不存在: {args.checkpoint_path}")
        sys.exit(1)
    
    # 创建生成器
    generator = UBDiffGenerator(
        checkpoint_path=args.checkpoint_path,
        dataset_name=args.dataset,
        encoder_dim=args.encoder_dim,
        time_steps=args.time_steps,
        dim_mults=tuple(args.dim_mults),
        objective=args.objective,
        device=args.device
    )
    
    # 打印模型摘要
    summary = generator.get_model_summary()
    print("\n模型摘要:")
    print(f"  总参数数: {summary['parameters']['total_params']:,}")
    print(f"  可训练参数: {summary['parameters']['trainable_params']:,}")
    print(f"  编码器维度: {summary['model_components']['encoder_dim']}")
    print(f"  扩散时间步: {summary['model_components']['time_steps']}")
    print(f"  扩散目标: {summary['model_components']['objective']}")
    
    # 生成数据
    print("\n开始生成数据...")
    generator.generate_and_save(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_format=args.save_format
    )
    
    # 可视化
    if args.visualize:
        print("\n生成可视化图像...")
        visualizer = ModelVisualizer()
        
        # 生成一些样本用于可视化
        velocity_samples, seismic_samples = generator.generate_batch(
            batch_size=args.num_visualize,
            denormalize=True
        )
        
        # 创建可视化目录
        vis_dir = Path(args.output_dir) / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # 生成单独的图像
        for i in range(min(args.num_visualize, len(velocity_samples))):
            # 速度场
            visualizer.plot_velocity_field(
                velocity_samples[i].cpu().numpy(),
                title=f"生成速度场 #{i+1}",
                save_path=str(vis_dir / f'velocity_{i+1}.png')
            )
            
            # 地震数据
            visualizer.plot_seismic_data(
                seismic_samples[i].cpu().numpy(),
                title=f"生成地震数据 #{i+1}",
                save_path=str(vis_dir / f'seismic_{i+1}.png')
            )
        
        print(f"可视化图像已保存到: {vis_dir}")
    
    # 质量评估
    if args.evaluate_quality:
        if not args.real_data_path or not args.real_label_path:
            print("警告: 需要提供真实数据路径进行质量评估")
        else:
            print("\n开始质量评估...")
            try:
                real_seismic, real_velocity = load_real_data_for_evaluation(
                    args.real_data_path, 
                    args.real_label_path,
                    args.num_eval_samples
                )
                
                metrics = generator.evaluate_quality(
                    real_velocity=real_velocity,
                    real_seismic=real_seismic,
                    num_generated=args.num_eval_samples
                )
                
                # 保存评估结果
                eval_path = Path(args.output_dir) / 'quality_evaluation.json'
                import json
                with open(eval_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                print(f"质量评估结果已保存到: {eval_path}")
                
            except Exception as e:
                print(f"质量评估失败: {e}")
    
    print(f"\n数据生成完成! 查看结果: {args.output_dir}")


if __name__ == "__main__":
    main()