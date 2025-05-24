#!/usr/bin/env python3
"""
编码器-解码器训练脚本

用于训练UB-Diff模型的编码器和解码器部分
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.trainers import EncoderDecoderTrainer, setup_seed


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='UB-Diff编码器-解码器训练')
    
    # 数据相关参数
    parser.add_argument('--train_data', type=str, required=True,
                       help='训练地震数据文件夹路径')
    parser.add_argument('--train_label', type=str, required=True,
                       help='训练速度场数据文件夹路径')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['flatvel-a', 'flatvel-b', 'curvefault-a', 'curvefault-b', 
                               'flatfault-a', 'flatfault-b'],
                       help='数据集名称')
    parser.add_argument('--num_data', type=int, default=24000,
                       help='训练数据数量')
    parser.add_argument('--paired_num', type=int, default=5000,
                       help='配对数据数量')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=300,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--lr_gamma', type=float, default=0.98,
                       help='学习率衰减因子')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[],
                       help='学习率衰减里程碑')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0,
                       help='学习率预热轮数')
    
    # 模型参数
    parser.add_argument('--encoder_dim', type=int, default=512,
                       help='编码器维度')
    parser.add_argument('--lambda_g1v', type=float, default=1.0,
                       help='L1损失权重')
    parser.add_argument('--lambda_g2v', type=float, default=1.0,
                       help='L2损失权重')
    
    # 输出和日志
    parser.add_argument('--output_path', type=str, default='./checkpoints/encoder_decoder',
                       help='模型保存路径')
    parser.add_argument('--val_every', type=int, default=20,
                       help='验证频率（每n个epoch）')
    parser.add_argument('--print_freq', type=int, default=50,
                       help='打印频率')
    
    # 系统参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备')
    parser.add_argument('--workers', type=int, default=4,
                       help='数据加载线程数')
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
    print("UB-Diff 编码器-解码器训练")
    print("="*50)
    print(f"数据集: {args.dataset}")
    print(f"训练数据: {args.train_data}")
    print(f"标签数据: {args.train_label}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"输出路径: {args.output_path}")
    print(f"设备: {args.device}")
    print("="*50)
    
    # 创建训练器
    trainer = EncoderDecoderTrainer(
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
        wandb_project=args.proj_name,
        device=args.device
    )
    
    # 开始训练
    trainer.train(
        epochs=args.epochs,
        val_every=args.val_every,
        print_freq=args.print_freq
    )
    
    print("训练完成!")


if __name__ == "__main__":
    main() 