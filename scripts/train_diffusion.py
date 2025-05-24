#!/usr/bin/env python3
"""
扩散模型训练脚本

用于训练UB-Diff模型的扩散部分
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.trainers import DiffusionTrainer, setup_seed


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='UB-Diff扩散模型训练')
    
    # 数据相关参数
    parser.add_argument('--train_data', type=str, required=True,
                       help='训练地震数据文件夹路径')
    parser.add_argument('--train_label', type=str, required=True,
                       help='训练速度场数据文件夹路径')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['flatvel-a', 'flatvel-b', 'curvefault-a', 'curvefault-b', 
                               'flatfault-a', 'flatfault-b'],
                       help='数据集名称')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='预训练编码器-解码器检查点路径')
    parser.add_argument('--num_data', type=int, default=24000,
                       help='训练数据数量')
    
    # 训练参数
    parser.add_argument('--num_steps', type=int, default=150000,
                       help='训练步数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=8e-5,
                       help='学习率')
    parser.add_argument('--gradient_accumulate_every', type=int, default=1,
                       help='梯度累积步数')
    
    # EMA参数
    parser.add_argument('--ema_decay', type=float, default=0.995,
                       help='EMA衰减率')
    parser.add_argument('--ema_update_every', type=int, default=10,
                       help='EMA更新频率')
    
    # 保存和采样参数
    parser.add_argument('--save_and_sample_every', type=int, default=1000,
                       help='保存和采样频率')
    parser.add_argument('--num_samples', type=int, default=25,
                       help='采样数量（必须是完全平方数）')
    
    # 模型参数
    parser.add_argument('--latent_dim', type=int, default=512,
                       help='潜在空间维度')
    parser.add_argument('--time_steps', type=int, default=256,
                       help='扩散时间步数')
    parser.add_argument('--dim_mults', type=int, nargs='+', default=[1, 2, 4, 8],
                       help='U-Net维度倍数')
    parser.add_argument('--objective', type=str, default='pred_v',
                       choices=['pred_noise', 'pred_x0', 'pred_v'],
                       help='扩散目标')
    
    # 输出参数
    parser.add_argument('--results_folder', type=str, default='./results/diffusion',
                       help='结果保存文件夹')
    
    # 系统参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # wandb参数
    parser.add_argument('--use_wandb', action='store_true',
                       help='是否使用wandb记录')
    parser.add_argument('--proj_name', type=str, default='UB-Diff',
                       help='wandb项目名称')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置随机种子
    setup_seed(args.seed)
    
    print("="*50)
    print("UB-Diff 扩散模型训练")
    print("="*50)
    print(f"数据集: {args.dataset}")
    print(f"训练数据: {args.train_data}")
    print(f"标签数据: {args.train_label}")
    print(f"预训练模型: {args.checkpoint_path}")
    print(f"训练步数: {args.num_steps}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"潜在维度: {args.latent_dim}")
    print(f"时间步数: {args.time_steps}")
    print(f"扩散目标: {args.objective}")
    print(f"结果文件夹: {args.results_folder}")
    print(f"设备: {args.device}")
    print("="*50)
    
    # 检查预训练模型是否存在
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 预训练模型文件不存在: {args.checkpoint_path}")
        print("请先运行编码器-解码器训练脚本生成预训练模型")
        sys.exit(1)
    
    # 验证采样数量是否为完全平方数
    import math
    sqrt_samples = int(math.sqrt(args.num_samples))
    if sqrt_samples * sqrt_samples != args.num_samples:
        print(f"错误: 采样数量 {args.num_samples} 不是完全平方数")
        print("请选择一个完全平方数，如: 4, 9, 16, 25, 36, 49, 64, 81, 100等")
        sys.exit(1)
    
    # 创建训练器
    trainer = DiffusionTrainer(
        seismic_folder=args.train_data,
        velocity_folder=args.train_label,
        dataset_name=args.dataset,
        num_data=args.num_data,
        checkpoint_path=args.checkpoint_path,
        results_folder=args.results_folder,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        ema_update_every=args.ema_update_every,
        save_and_sample_every=args.save_and_sample_every,
        num_samples=args.num_samples,
        encoder_dim=args.latent_dim,
        time_steps=args.time_steps,
        dim_mults=tuple(args.dim_mults),
        objective=args.objective,
        use_wandb=args.use_wandb,
        wandb_project=args.proj_name,
        device=args.device
    )
    
    # 开始训练
    trainer.train()
    
    print("扩散模型训练完成!")


if __name__ == "__main__":
    main() 