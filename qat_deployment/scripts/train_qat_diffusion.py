#!/usr/bin/env python3
"""
量化感知训练 - 扩散模型训练脚本

训练量化的扩散模型
"""

import os
import sys
import argparse
import torch
import torch.quantization as quant

# 添加项目路径到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.ub_diff import UBDiff
from model.data import create_dataloaders
from qat_deployment.models import QuantizedUBDiff, prepare_qat_model, convert_to_quantized
from qat_deployment.trainers import QATDiffusionTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='QAT扩散模型训练')
    
    # 数据参数
    parser.add_argument('--train_data', type=str, required=True,
                        help='训练地震数据路径')
    parser.add_argument('--train_label', type=str, required=True,
                        help='训练速度场数据路径')
    parser.add_argument('--val_data', type=str, default=None,
                        help='验证地震数据路径')
    parser.add_argument('--val_label', type=str, default=None,
                        help='验证速度场数据路径')
    parser.add_argument('--dataset', type=str, default='flatvel-a',
                        help='数据集名称')
    
    # 模型参数
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='预训练扩散模型路径')
    parser.add_argument('--decoder_checkpoint', type=str, required=True,
                        help='QAT解码器检查点路径')
    parser.add_argument('--encoder_dim', type=int, default=512,
                        help='编码器维度')
    parser.add_argument('--time_steps', type=int, default=256,
                        help='扩散时间步数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='梯度裁剪')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    
    # 量化参数
    parser.add_argument('--quantize_diffusion', action='store_true',
                        help='是否量化扩散模型')
    parser.add_argument('--backend', type=str, default='qnnpack',
                        choices=['qnnpack', 'fbgemm'],
                        help='量化后端')
    
    # 其他参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/qat_diffusion',
                        help='检查点保存目录')
    parser.add_argument('--save_every', type=int, default=10,
                        help='保存检查点频率')
    parser.add_argument('--validate_every', type=int, default=5,
                        help='验证频率')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备')
    parser.add_argument('--use_wandb', action='store_true',
                        help='是否使用WandB记录')
    parser.add_argument('--wandb_project', type=str, default='ub-diff-qat',
                        help='WandB项目名称')
    
    return parser.parse_args()


def load_encoder(pretrained_path: str, device: str) -> torch.nn.Module:
    """加载预训练的编码器"""
    model = UBDiff(
        in_channels=1,
        encoder_dim=512,
        velocity_channels=1,
        seismic_channels=5,
        dim_mults=(1, 2, 2, 2)
    )
    
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    encoder = model.encoder
    encoder.eval()
    
    return encoder.to(device)


def create_simple_dataloader(seismic_path: str, velocity_path: str, 
                            dataset_name: str, batch_size: int = 32,
                            shuffle: bool = True, num_workers: int = 4):
    """创建简化的数据加载器"""
    from model.data import DatasetConfig, SeismicVelocityDataset
    from torch.utils.data import DataLoader
    
    # 获取数据集配置
    config = DatasetConfig()
    ctx = config.get_dataset_info(dataset_name)
    seismic_transform, velocity_transform = config.get_transforms(dataset_name, k=1.0)
    
    # 判断是否为断层族数据集
    fault_family = dataset_name in ['flatfault-a', 'curvefault-a', 'flatfault-b', 'curvefault-b']
    
    # 创建数据集
    dataset = SeismicVelocityDataset(
        seismic_folder=seismic_path,
        velocity_folder=velocity_path,
        seismic_transform=seismic_transform,
        velocity_transform=velocity_transform,
        fault_family=fault_family,
        preload=False
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    
    # 初始化WandB
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name='qat-diffusion',
            config=vars(args)
        )
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader = create_simple_dataloader(args.train_data, args.train_label, args.dataset, args.batch_size, True, args.num_workers)
    
    val_loader = None
    if args.val_data and args.val_label:
        val_loader = create_simple_dataloader(args.val_data, args.val_label, args.dataset, args.batch_size, False, args.num_workers)
    
    # 加载编码器（用于生成真实潜在表示）
    print("加载编码器...")
    encoder = load_encoder(args.pretrained_path, device)
    
    # 创建量化模型
    print("创建量化UB-Diff模型...")
    model = QuantizedUBDiff(
        encoder_dim=args.encoder_dim,
        velocity_channels=1,
        seismic_channels=5,
        dim_mults=(1, 2, 4, 8),
        time_steps=args.time_steps,
        quantize_diffusion=args.quantize_diffusion,
        quantize_decoder=True  # 解码器已经量化
    )
    
    # 加载预训练的扩散模型权重
    print("加载扩散模型权重...")
    checkpoint = torch.load(args.pretrained_path, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 加载扩散相关权重
    diffusion_state = {}
    for key, value in state_dict.items():
        if 'diffusion' in key or 'unet' in key:
            diffusion_state[key] = value
    
    model.load_state_dict(diffusion_state, strict=False)
    
    # 加载QAT解码器权重
    print("加载QAT解码器权重...")
    decoder_checkpoint = torch.load(args.decoder_checkpoint, map_location='cpu')
    decoder_state = decoder_checkpoint['model_state_dict']
    
    # 映射解码器权重
    for key, value in decoder_state.items():
        if not key.startswith('decoder.'):
            decoder_state[f'decoder.{key}'] = value
            del decoder_state[key]
    
    model.load_state_dict(decoder_state, strict=False)
    
    # 准备扩散模型的QAT
    if args.quantize_diffusion:
        print(f"准备扩散模型的量化感知训练，后端: {args.backend}")
        # 只对扩散部分准备QAT
        torch.backends.quantized.engine = args.backend
        qconfig = quant.get_default_qat_qconfig(args.backend)
        
        # 为扩散模型设置量化配置
        model.diffusion.qconfig = qconfig
        model.unet.qconfig = qconfig
        
        # 准备QAT
        model.train()
        quant.prepare_qat(model.diffusion, inplace=True)
        quant.prepare_qat(model.unet, inplace=True)
    
    # 创建训练器
    trainer = QATDiffusionTrainer(
        model=model,
        encoder_model=encoder,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        use_wandb=args.use_wandb
    )
    
    # 训练扩散模型
    print("\n=== 训练量化扩散模型 ===")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        validate_every=args.validate_every
    )
    
    # 保存最终模型
    print("\n保存最终QAT模型...")
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'args': vars(args)
    }
    torch.save(
        final_checkpoint,
        os.path.join(args.checkpoint_dir, 'final_qat_model.pt')
    )
    
    print("训练完成！")


if __name__ == '__main__':
    main() 