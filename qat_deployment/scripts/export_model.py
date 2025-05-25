#!/usr/bin/env python3
"""
导出量化模型为TorchScript格式

用于树莓派部署
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Tuple

# 添加项目路径到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from qat_deployment.models import (
    QuantizedUBDiff, 
    convert_to_quantized,
    export_quantized_torchscript,
    check_model_quantizable
)


def parse_args():
    parser = argparse.ArgumentParser(description='导出量化模型')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='QAT模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='./exported_models',
                        help='输出目录')
    parser.add_argument('--model_name', type=str, default='ub_diff_quantized',
                        help='导出模型名称')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='导出时的批次大小')
    parser.add_argument('--backend', type=str, default='qnnpack',
                        choices=['qnnpack', 'fbgemm'],
                        help='量化后端')
    parser.add_argument('--test_generation', action='store_true',
                        help='是否测试生成功能')
    parser.add_argument('--optimize_for_mobile', action='store_true',
                        help='是否为移动设备优化')
    
    return parser.parse_args()


class DeploymentModel(nn.Module):
    """用于部署的简化模型
    
    只包含生成功能，不包含训练相关的代码
    """
    
    def __init__(self, quantized_model: QuantizedUBDiff):
        super(DeploymentModel, self).__init__()
        self.model = quantized_model
        
    def forward(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成数据
        
        Args:
            batch_size: 生成的批次大小
            
        Returns:
            (velocity, seismic): 生成的速度场和地震数据
        """
        # 直接使用模型的生成方法
        velocity, seismic = self.model.generate(batch_size, next(self.model.parameters()).device)
        return velocity, seismic


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置量化后端
    torch.backends.quantized.engine = args.backend
    print(f"使用量化后端: {args.backend}")
    
    # 加载QAT模型
    print(f"加载QAT模型: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    
    # 创建模型
    model = QuantizedUBDiff(
        encoder_dim=512,
        velocity_channels=1,
        seismic_channels=5,
        dim_mults=(1, 2, 2, 2),
        time_steps=256,
        quantize_diffusion=True,
        quantize_decoder=True
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 检查模型可量化性
    print("\n检查模型量化情况...")
    quantization_info = check_model_quantizable(model)
    print(f"总模块数: {quantization_info['total_modules']}")
    print(f"可量化模块数: {quantization_info['quantizable_modules']}")
    print(f"量化率: {quantization_info['quantization_ratio']:.2%}")
    
    if quantization_info['warnings']:
        print("\n警告:")
        for warning in quantization_info['warnings']:
            print(f"  - {warning}")
    
    # 转换为量化模型
    print("\n转换为量化模型...")
    model.eval()
    
    # 准备校准数据
    calibration_data = torch.randn(1, 1, 512).cpu()
    
    # 转换模型
    quantized_model = convert_to_quantized(model, calibration_data)
    
    # 创建部署模型
    print("\n创建部署模型...")
    deployment_model = DeploymentModel(quantized_model)
    deployment_model.eval()
    
    # 测试生成
    if args.test_generation:
        print("\n测试生成功能...")
        with torch.no_grad():
            velocity, seismic = deployment_model(batch_size=args.batch_size)
            print(f"生成速度场形状: {velocity.shape}")
            print(f"生成地震数据形状: {seismic.shape}")
    
    # 导出TorchScript
    print("\n导出TorchScript模型...")
    
    # 创建示例输入
    example_batch_size = torch.tensor(args.batch_size, dtype=torch.int32)
    
    # 追踪模型
    with torch.no_grad():
        traced_model = torch.jit.trace(
            deployment_model,
            example_batch_size,
            check_trace=False
        )
    
    # 优化模型
    if args.optimize_for_mobile:
        print("为移动设备优化模型...")
        traced_model = torch.jit.optimize_for_mobile(traced_model)
    
    # 保存模型
    output_path = os.path.join(args.output_dir, f"{args.model_name}.pt")
    traced_model.save(output_path)
    print(f"模型已导出到: {output_path}")
    
    # 计算模型大小
    model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"模型大小: {model_size:.2f} MB")
    
    # 保存模型信息
    info_path = os.path.join(args.output_dir, f"{args.model_name}_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"模型名称: {args.model_name}\n")
        f.write(f"量化后端: {args.backend}\n")
        f.write(f"模型大小: {model_size:.2f} MB\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"编码器维度: 512\n")
        f.write(f"速度场输出: (1, 70, 70)\n")
        f.write(f"地震数据输出: (5, 1000, 70)\n")
        f.write(f"\n量化信息:\n")
        f.write(f"总模块数: {quantization_info['total_modules']}\n")
        f.write(f"可量化模块数: {quantization_info['quantizable_modules']}\n")
        f.write(f"量化率: {quantization_info['quantization_ratio']:.2%}\n")
    
    print(f"模型信息已保存到: {info_path}")
    
    # 创建部署指南
    deploy_guide_path = os.path.join(args.output_dir, "deployment_guide.md")
    with open(deploy_guide_path, 'w') as f:
        f.write("# 树莓派部署指南\n\n")
        f.write("## 1. 环境准备\n\n")
        f.write("```bash\n")
        f.write("# 安装PyTorch (树莓派版本)\n")
        f.write("pip install torch==1.13.0\n")
        f.write("```\n\n")
        f.write("## 2. 加载模型\n\n")
        f.write("```python\n")
        f.write("import torch\n\n")
        f.write("# 加载模型\n")
        f.write(f"model = torch.jit.load('{args.model_name}.pt')\n")
        f.write("model.eval()\n\n")
        f.write("# 生成数据\n")
        f.write("with torch.no_grad():\n")
        f.write("    velocity, seismic = model(1)  # batch_size=1\n")
        f.write("```\n\n")
        f.write("## 3. 性能优化\n\n")
        f.write("- 使用单批次推理以减少内存使用\n")
        f.write("- 考虑使用半精度（fp16）进一步减少内存\n")
        f.write("- 可以调整扩散步数以加快生成速度\n")
    
    print(f"部署指南已保存到: {deploy_guide_path}")
    
    print("\n导出完成！")


if __name__ == '__main__':
    main() 