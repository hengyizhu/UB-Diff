#!/usr/bin/env python3
"""
量化感知训练 - 解码器训练脚本

分阶段训练速度解码器和地震解码器
"""

import os
import sys
import argparse
import torch

# 添加项目路径到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.ub_diff import UBDiff
from model.data import SeismicVelocityDataset
from qat_deployment.models import QuantizedDecoder, prepare_qat_model, fuse_modules_for_qat
from qat_deployment.trainers import QATDecoderTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='QAT解码器训练')
    
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
                        help='预训练模型路径')
    parser.add_argument('--encoder_dim', type=int, default=512,
                        help='编码器维度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--velocity_epochs', type=int, default=50,
                        help='速度解码器训练轮数')
    parser.add_argument('--seismic_epochs', type=int, default=50,
                        help='地震解码器训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    
    # 量化参数
    parser.add_argument('--quantize_velocity', action='store_true',
                        help='是否量化速度解码器')
    parser.add_argument('--quantize_seismic', action='store_true',
                        help='是否量化地震解码器')
    parser.add_argument('--backend', type=str, default='qnnpack',
                        choices=['qnnpack', 'fbgemm'],
                        help='量化后端')
    
    # 其他参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/qat_decoders',
                        help='检查点保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备')
    parser.add_argument('--use_wandb', action='store_true',
                        help='是否使用WandB记录')
    parser.add_argument('--wandb_project', type=str, default='ub-diff-qat',
                        help='WandB项目名称')
    
    return parser.parse_args()


def load_encoder(pretrained_path: str, device: str) -> torch.nn.Module:
    """加载预训练的编码器"""
    print(f"加载预训练模型: {pretrained_path}")
    
    # 创建原始模型
    model = UBDiff(
        in_channels=1,
        encoder_dim=512,
        velocity_channels=1,
        seismic_channels=5,
        dim_mults=(1, 2, 4, 8)
    )
    
    # 加载权重
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    
    # 只返回编码器
    encoder = model.encoder
    encoder.eval()
    
    return encoder.to(device)


def create_simple_dataloader(seismic_path: str, velocity_path: str, 
                            dataset_name: str, batch_size: int = 32,
                            shuffle: bool = True, num_workers: int = 4):
    """创建简化的数据加载器"""
    from model.data import DatasetConfig
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
            name='qat-decoders',
            config=vars(args)
        )
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader = create_simple_dataloader(args.train_data, args.train_label, args.dataset, args.batch_size, True, args.num_workers)
    
    val_loader = None
    if args.val_data and args.val_label:
        val_loader = create_simple_dataloader(args.val_data, args.val_label, args.dataset, args.batch_size, False, args.num_workers)
    
    # 加载编码器
    encoder = load_encoder(args.pretrained_path, device)
    
    # 创建量化解码器
    print("创建量化解码器...")
    decoder = QuantizedDecoder(
        encoder_dim=args.encoder_dim,
        velocity_channels=1,
        seismic_channels=5,
        quantize_velocity=args.quantize_velocity,
        quantize_seismic=args.quantize_seismic
    )
    
    # 从预训练模型加载解码器权重
    print("加载解码器权重...")
    checkpoint = torch.load(args.pretrained_path, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 映射权重
    decoder_state = {}
    for key, value in state_dict.items():
        if 'dual_decoder.velocity_projector' in key:
            new_key = key.replace('dual_decoder.velocity_projector', 'velocity_projector')
            decoder_state[new_key] = value
        elif 'dual_decoder.seismic_projector' in key:
            new_key = key.replace('dual_decoder.seismic_projector', 'seismic_projector')
            decoder_state[new_key] = value
        elif 'dual_decoder.velocity_decoder' in key:
            new_key = key.replace('dual_decoder.velocity_decoder', 'velocity_decoder')
            decoder_state[new_key] = value
        elif 'dual_decoder.seismic_decoder' in key:
            new_key = key.replace('dual_decoder.seismic_decoder', 'seismic_decoder')
            decoder_state[new_key] = value
    
    decoder.load_state_dict(decoder_state, strict=False)
    
    # 融合模块
    print("融合模块以提高效率...")
    decoder = fuse_modules_for_qat(decoder)
    
    # 准备QAT
    print(f"准备量化感知训练，后端: {args.backend}")
    decoder = prepare_qat_model(decoder, backend=args.backend)
    
    # 创建训练器
    trainer = QATDecoderTrainer(
        model=decoder,
        encoder_model=encoder,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_wandb=args.use_wandb
    )
    
    # 阶段1：训练速度解码器
    if args.quantize_velocity and args.velocity_epochs > 0:
        print("\n=== 阶段1：训练速度解码器 ===")
        trainer.train_velocity_decoder(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.velocity_epochs,
            checkpoint_dir=os.path.join(args.checkpoint_dir, 'velocity')
        )
    
    # 阶段2：训练地震解码器
    if args.quantize_seismic and args.seismic_epochs > 0:
        print("\n=== 阶段2：训练地震解码器 ===")
        trainer.train_seismic_decoder(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.seismic_epochs,
            checkpoint_dir=os.path.join(args.checkpoint_dir, 'seismic'),
            freeze_velocity=True
        )
    
    # 保存最终模型
    print("\n保存最终QAT模型...")
    final_checkpoint = {
        'model_state_dict': decoder.state_dict(),
        'args': vars(args)
    }
    torch.save(
        final_checkpoint,
        os.path.join(args.checkpoint_dir, 'final_qat_decoders.pt')
    )
    
    print("训练完成！")


if __name__ == '__main__':
    main() 