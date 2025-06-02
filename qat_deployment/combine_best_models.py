#!/usr/bin/env python3
"""
组合最佳QAT解码器模型

将最佳速度解码器和最佳地震解码器的权重组合成一个完整的模型
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# 添加项目路径到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_checkpoint(path: str) -> dict:
    """加载检查点文件"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"检查点文件不存在: {path}")
    
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint


def combine_decoder_weights(velocity_checkpoint: dict, seismic_checkpoint: dict) -> dict:
    """组合速度解码器和地震解码器的权重
    
    Args:
        velocity_checkpoint: 最佳速度解码器检查点
        seismic_checkpoint: 最佳地震解码器检查点
        
    Returns:
        组合后的模型状态字典
    """
    print("开始组合解码器权重...")
    
    velocity_state = velocity_checkpoint['model_state_dict']
    seismic_state = seismic_checkpoint['model_state_dict']
    
    # 创建组合后的状态字典
    combined_state = {}
    
    # 添加所有权重，地震解码器的权重会覆盖重复的键（如共享层）
    print("合并速度解码器权重...")
    for key, value in velocity_state.items():
        combined_state[key] = value
        if 'velocity' in key.lower():
            print(f"  ✓ {key}")
    
    print("合并地震解码器权重...")
    for key, value in seismic_state.items():
        combined_state[key] = value
        if 'seismic' in key.lower():
            print(f"  ✓ {key}")
    
    # 统计权重数量
    velocity_keys = [k for k in velocity_state.keys() if 'velocity' in k.lower()]
    seismic_keys = [k for k in seismic_state.keys() if 'seismic' in k.lower()]
    shared_keys = [k for k in combined_state.keys() if 'velocity' not in k.lower() and 'seismic' not in k.lower()]
    
    print(f"\n权重统计:")
    print(f"  速度解码器专用权重: {len(velocity_keys)} 个")
    print(f"  地震解码器专用权重: {len(seismic_keys)} 个")
    print(f"  共享权重: {len(shared_keys)} 个")
    print(f"  总权重数: {len(combined_state)} 个")
    
    return combined_state


def create_combined_checkpoint(velocity_checkpoint: dict, 
                             seismic_checkpoint: dict,
                             combined_state: dict) -> dict:
    """创建组合后的检查点"""
    
    # 提取元数据
    velocity_loss = velocity_checkpoint.get('val_loss', 'N/A')
    velocity_ssim = velocity_checkpoint.get('val_ssim', 'N/A')
    seismic_loss = seismic_checkpoint.get('val_loss', 'N/A')
    seismic_ssim = seismic_checkpoint.get('val_ssim', 'N/A')
    
    combined_checkpoint = {
        'model_state_dict': combined_state,
        'velocity_metrics': {
            'val_loss': velocity_loss,
            'val_ssim': velocity_ssim,
            'epoch': velocity_checkpoint.get('epoch', 'N/A')
        },
        'seismic_metrics': {
            'val_loss': seismic_loss,
            'val_ssim': seismic_ssim,
            'epoch': seismic_checkpoint.get('epoch', 'N/A')
        },
        'combination_info': {
            'velocity_source': 'checkpoints/qat_decoders/velocity/best_velocity.pt',
            'seismic_source': 'checkpoints/qat_decoders/seismic/best_seismic.pt',
            'note': 'Combined best velocity and seismic decoders from separate training sessions'
        }
    }
    
    return combined_checkpoint


def main():
    parser = argparse.ArgumentParser(description='组合最佳QAT解码器模型')
    parser.add_argument('--velocity_checkpoint', type=str, 
                       default='checkpoints/qat_decoders/velocity/best_velocity.pt',
                       help='最佳速度解码器检查点路径')
    parser.add_argument('--seismic_checkpoint', type=str,
                       default='checkpoints/qat_decoders/seismic/best_seismic.pt', 
                       help='最佳地震解码器检查点路径')
    parser.add_argument('--output_path', type=str,
                       default='checkpoints/qat_decoders/best_combined_qat_decoders.pt',
                       help='输出组合模型路径')
    parser.add_argument('--force', action='store_true',
                       help='强制覆盖已存在的输出文件')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("QAT解码器模型组合工具")
    print("=" * 60)
    
    # 检查输入文件
    print(f"\n检查输入文件:")
    print(f"速度解码器: {args.velocity_checkpoint}")
    print(f"地震解码器: {args.seismic_checkpoint}")
    
    if not os.path.exists(args.velocity_checkpoint):
        print(f"❌ 错误: 速度解码器文件不存在: {args.velocity_checkpoint}")
        return False
    else:
        print(f"✅ 速度解码器文件存在")
        
    if not os.path.exists(args.seismic_checkpoint):
        print(f"❌ 错误: 地震解码器文件不存在: {args.seismic_checkpoint}")
        return False
    else:
        print(f"✅ 地震解码器文件存在")
    
    # 检查输出文件
    if os.path.exists(args.output_path) and not args.force:
        print(f"❌ 错误: 输出文件已存在: {args.output_path}")
        print("使用 --force 参数强制覆盖")
        return False
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载检查点
        print(f"\n加载检查点...")
        velocity_checkpoint = load_checkpoint(args.velocity_checkpoint)
        seismic_checkpoint = load_checkpoint(args.seismic_checkpoint)
        
        print(f"✅ 速度解码器检查点加载成功")
        print(f"   - 验证损失: {velocity_checkpoint.get('val_loss', 'N/A')}")
        print(f"   - 验证SSIM: {velocity_checkpoint.get('val_ssim', 'N/A')}")
        
        print(f"✅ 地震解码器检查点加载成功")
        print(f"   - 验证损失: {seismic_checkpoint.get('val_loss', 'N/A')}")
        print(f"   - 验证SSIM: {seismic_checkpoint.get('val_ssim', 'N/A')}")
        
        # 组合权重
        combined_state = combine_decoder_weights(velocity_checkpoint, seismic_checkpoint)
        
        # 创建组合检查点
        combined_checkpoint = create_combined_checkpoint(
            velocity_checkpoint, seismic_checkpoint, combined_state
        )
        
        # 保存组合模型
        print(f"\n保存组合模型到: {args.output_path}")
        torch.save(combined_checkpoint, args.output_path)
        
        print(f"✅ 组合模型保存成功!")
        
        # 显示最终信息
        print(f"\n" + "=" * 60)
        print("组合完成!")
        print("=" * 60)
        print(f"输出文件: {args.output_path}")
        print(f"文件大小: {os.path.getsize(args.output_path) / 1024 / 1024:.2f} MB")
        
        print(f"\n模型性能信息:")
        print(f"速度解码器 - Loss: {velocity_checkpoint.get('val_loss', 'N/A')}, SSIM: {velocity_checkpoint.get('val_ssim', 'N/A')}")
        print(f"地震解码器 - Loss: {seismic_checkpoint.get('val_loss', 'N/A')}, SSIM: {seismic_checkpoint.get('val_ssim', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return False


if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1) 