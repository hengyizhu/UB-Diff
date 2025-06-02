 #!/usr/bin/env python3
"""
地震解码器微调脚本

专门用于微调UB-Diff模型的地震解码器部分
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.trainers import FinetuneTrainer, setup_seed


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='UB-Diff地震解码器微调')
    
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
    parser.add_argument('--num_data', type=int, default=48000,
                       help='训练数据数量')
    parser.add_argument('--paired_num', type=int, default=5000,
                       help='配对数据数量')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='微调轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='学习率（通常比初始训练更小）')
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
    parser.add_argument('--output_path', type=str, default='./checkpoints/finetune',
                       help='模型保存路径')
    parser.add_argument('--val_every', type=int, default=10,
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
    
    # 数据加载优化参数
    parser.add_argument('--preload_workers', type=int, default=8,
                       help='预加载使用的线程数')
    parser.add_argument('--cache_size', type=int, default=32,
                       help='LRU缓存大小（当不预加载时使用）')
    parser.add_argument('--use_memmap', action='store_true',
                       help='是否使用内存映射（对大文件有效）')
    parser.add_argument('--no_preload', action='store_true',
                       help='禁用数据预加载（节省内存）')
    
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
    print("UB-Diff 地震解码器微调")
    print("="*50)
    print(f"数据集: {args.dataset}")
    print(f"训练数据: {args.train_data}")
    print(f"标签数据: {args.train_label}")
    print(f"预训练模型: {args.checkpoint_path}")
    print(f"微调轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"输出路径: {args.output_path}")
    print(f"设备: {args.device}")
    print("="*50)
    
    # 检查预训练模型是否存在
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 预训练模型文件不存在: {args.checkpoint_path}")
        print("请先运行编码器-解码器训练脚本生成预训练模型")
        sys.exit(1)
    
    # 创建训练器
    trainer = FinetuneTrainer(
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
        device=args.device,
        preload=not args.no_preload,
        preload_workers=args.preload_workers,
        cache_size=args.cache_size,
        use_memmap=args.use_memmap
    )
    
    # 开始微调
    trainer.train(
        epochs=args.epochs,
        val_every=args.val_every,
        print_freq=args.print_freq
    )
    
    print("微调完成!")


if __name__ == "__main__":
    main()