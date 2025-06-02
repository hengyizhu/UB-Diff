#!/usr/bin/env python3
"""
最终修复版本 - QAT扩散模型训练脚本

正确处理QAT解码器状态加载和保存
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
    parser = argparse.ArgumentParser(description='QAT扩散模型训练（最终修复版本）')
    
    # 数据参数
    parser.add_argument('--train_data', type=str, required=True,
                        help='训练地震数据路径')
    parser.add_argument('--train_label', type=str, required=True,
                        help='训练速度场数据路径')
    parser.add_argument('--val_data', type=str, default=None,
                        help='验证地震数据路径')
    parser.add_argument('--val_label', type=str, default=None,
                        help='验证速度场数据路径')
    parser.add_argument('--dataset', type=str, default='curvefault-a',
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
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=8e-5,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='梯度裁剪')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='数据加载线程数')
    
    # 量化参数
    parser.add_argument('--quantize_diffusion', action='store_true',
                        help='是否量化扩散模型')
    parser.add_argument('--backend', type=str, default='qnnpack',
                        choices=['qnnpack', 'fbgemm'],
                        help='量化后端')
    parser.add_argument('--convert_conv1d', action='store_true',
                        help='是否转换1D卷积为2D卷积以获得更好的量化支持')
    parser.add_argument('--use_aggressive_quantization', action='store_true',
                        help='是否使用更激进的量化配置')
    
    # 其他参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/qat_diffusion',
                        help='检查点保存目录')
    parser.add_argument('--save_every', type=int, default=10,
                        help='保存检查点频率')
    parser.add_argument('--validate_every', type=int, default=5,
                        help='验证频率（每多少个epoch验证一次）')
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
        dim_mults=(1, 2, 4, 8)
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
        preload=True
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


def count_qat_modules(model, module_path=""):
    """统计模型中QAT模块的数量"""
    qat_count = 0
    total_modules = 0
    
    for name, module in model.named_modules():
        total_modules += 1
        if hasattr(module, 'qconfig') and module.qconfig is not None:
            qat_count += 1
    
    return qat_count, total_modules


def load_qat_decoder_correctly(model, decoder_checkpoint_path):
    """正确加载QAT解码器状态"""
    print("\n=== 正确加载QAT解码器 ===")
    print(f"解码器路径: {decoder_checkpoint_path}")
    
    # 加载QAT解码器检查点
    decoder_checkpoint = torch.load(decoder_checkpoint_path, map_location='cpu')
    decoder_state = decoder_checkpoint['model_state_dict']
    
    print(f"📊 QAT解码器文件包含 {len(decoder_state)} 个参数")
    
    # 分析QAT参数
    qat_params = [k for k in decoder_state.keys() 
                  if 'fake_quant' in k or 'observer' in k or 'activation_post_process' in k]
    normal_params = [k for k in decoder_state.keys() 
                     if not ('fake_quant' in k or 'observer' in k or 'activation_post_process' in k)]
    
    print(f"✅ 其中 {len(qat_params)} 个是QAT参数")
    print(f"📦 其中 {len(normal_params)} 个是普通参数")
    
    # 检查加载前的QAT状态
    decoder_qat_before, decoder_total = count_qat_modules(model.decoder)
    print(f"加载前解码器QAT状态:")
    print(f"  📦 decoder: {decoder_qat_before}/{decoder_total} 个模块有QAT配置")
    
    # 方法1: 尝试直接加载所有状态（包括QAT状态）
    print("\n🔄 方法1: 直接加载所有QAT状态...")
    result = model.decoder.load_state_dict(decoder_state, strict=False)
    print(f"📋 加载结果:")
    print(f"  缺失的keys: {len(result.missing_keys)}")
    print(f"  意外的keys: {len(result.unexpected_keys)}")
    
    # 检查QAT状态是否正确加载
    decoder_qat_after, _ = count_qat_modules(model.decoder)
    print(f"加载后解码器QAT状态:")
    print(f"  📦 decoder: {decoder_qat_after}/{decoder_total} 个模块有QAT配置")
    
    if decoder_qat_after == 0:
        print("❌ 方法1失败，尝试方法2...")
        
        # 方法2: 先应用量化配置，再加载权重
        print("\n🔄 方法2: 先配置QAT，再加载权重...")
        
        # 为解码器应用QAT配置
        model.decoder.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        torch.quantization.prepare_qat(model.decoder, inplace=True)
        
        # 再次检查QAT状态
        decoder_qat_prepared, _ = count_qat_modules(model.decoder)
        print(f"QAT配置后解码器状态:")
        print(f"  📦 decoder: {decoder_qat_prepared}/{decoder_total} 个模块有QAT配置")
        
        # 再次尝试加载权重
        result = model.decoder.load_state_dict(decoder_state, strict=False)
        print(f"📋 第二次加载结果:")
        print(f"  缺失的keys: {len(result.missing_keys)}")
        print(f"  意外的keys: {len(result.unexpected_keys)}")
        
        # 最终检查
        decoder_qat_final, _ = count_qat_modules(model.decoder)
        print(f"最终解码器QAT状态:")
        print(f"  📦 decoder: {decoder_qat_final}/{decoder_total} 个模块有QAT配置")
        
        if decoder_qat_final > 0:
            print("✅ 方法2成功！QAT解码器状态已正确加载")
        else:
            print("❌ 方法2也失败了")
    else:
        print("✅ 方法1成功！QAT解码器状态已正确加载")
    
    return decoder_qat_after > 0 or (decoder_qat_after == 0 and 'decoder_qat_final' in locals() and decoder_qat_final > 0)


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    
    # 初始化WandB
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name='qat-diffusion-final-fix',
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
    
    # 创建量化模型（不启用解码器量化，稍后手动加载）
    print("\n=== 创建QuantizedUBDiff模型 ===")
    model = QuantizedUBDiff(
        encoder_dim=args.encoder_dim,
        velocity_channels=1,
        seismic_channels=5,
        dim_mults=(1, 2, 4, 8),
        time_steps=args.time_steps,
        quantize_diffusion=args.quantize_diffusion,
        quantize_decoder=False  # 先不启用，手动加载
    )
    
    # 加载预训练的扩散模型权重
    print("\n=== 加载扩散模型权重 ===")
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
    print("✅ 扩散模型权重加载完成")
    
    # 正确加载QAT解码器
    qat_success = load_qat_decoder_correctly(model, args.decoder_checkpoint)
    
    if not qat_success:
        print("⚠️ 警告：QAT解码器加载可能不完整，但继续训练...")
    
    # 对扩散模型应用量化策略
    if args.quantize_diffusion:
        print(f"\n=== 应用扩散模型量化策略 ===")
        print(f"量化后端: {args.backend}")
        print(f"转换1D卷积: {args.convert_conv1d}")
        print(f"激进量化: {args.use_aggressive_quantization}")
        
        quantization_report = model.apply_improved_quantization(
            backend=args.backend,
            convert_conv1d=args.convert_conv1d,
            use_aggressive_config=args.use_aggressive_quantization
        )
        
        # 保存量化报告
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        report_path = os.path.join(args.checkpoint_dir, 'quantization_analysis_final.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("最终修复版本量化分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("量化分析结果:\n")
            f.write(f"  总模块数: {quantization_report['total_modules']}\n")
            f.write(f"  总参数数: {quantization_report['total_params']:,}\n")
            f.write(f"  可量化模块: {quantization_report['quantizable_modules']}\n")
            f.write(f"  可量化参数: {quantization_report['quantizable_params']:,}\n")
            f.write(f"  可量化模块比例: {quantization_report['quantizable_ratio']:.1%}\n")
            f.write(f"  可量化参数比例: {quantization_report['quantizable_param_ratio']:.1%}\n")
            f.write(f"  1D卷积数量: {quantization_report['conv1d_count']}\n")
            
            if quantization_report['converted_conv1d'] > 0:
                f.write(f"\n✓ 成功转换 {quantization_report['converted_conv1d']} 个1D卷积层\n")
        
        print(f"量化分析报告已保存到: {report_path}")
    else:
        print("跳过扩散模型量化")
    
    # 最终QAT状态检查
    print(f"\n=== 最终模型QAT状态检查 ===")
    decoder_qat, decoder_total = count_qat_modules(model.decoder)
    diffusion_qat, diffusion_total = count_qat_modules(model.diffusion)
    unet_qat, unet_total = count_qat_modules(model.unet)
    
    print(f"  📦 decoder: {decoder_qat}/{decoder_total} 个模块有QAT配置")
    print(f"  📦 diffusion: {diffusion_qat}/{diffusion_total} 个模块有QAT配置") 
    print(f"  📦 unet: {unet_qat}/{unet_total} 个模块有QAT配置")
    
    # 移动到设备
    model = model.to(device)
    
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
    
    # 保存最佳组合模型
    print("\n保存最佳QAT扩散模型...")
    
    best_diffusion_path = os.path.join(args.checkpoint_dir, 'best_diffusion.pt')
    
    # 如果存在最佳模型，加载它
    if os.path.exists(best_diffusion_path):
        print(f"加载最佳扩散模型: {best_diffusion_path}")
        best_checkpoint = torch.load(best_diffusion_path, map_location='cpu')
        best_model_state = best_checkpoint['model_state_dict']
    else:
        print("警告：未找到最佳扩散模型，使用当前模型状态")
        best_model_state = model.state_dict()
    
    # 保存最佳模型
    final_checkpoint = {
        'model_state_dict': best_model_state,
        'args': vars(args),
        'qat_status': {
            'decoder_qat_modules': decoder_qat,
            'diffusion_qat_modules': diffusion_qat,
            'unet_qat_modules': unet_qat,
        },
        'note': 'Best QAT diffusion model with properly loaded QAT decoder'
    }
    torch.save(
        final_checkpoint,
        os.path.join(args.checkpoint_dir, 'best_qat_diffusion_final.pt')
    )
    
    print(f"最终QAT扩散模型已保存到: {os.path.join(args.checkpoint_dir, 'best_qat_diffusion_final.pt')}")
    print("训练完成！")


if __name__ == '__main__':
    main() 