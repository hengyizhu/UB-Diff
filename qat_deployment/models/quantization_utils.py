"""
量化感知训练工具函数

提供模型量化的辅助功能
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Dict, Any, Optional


def get_qat_qconfig() -> quant.QConfig:
    """获取量化感知训练的配置
    
    Returns:
        适用于ARM设备的QAT配置
    """
    # 使用适合ARM设备的量化配置
    activation = quant.FakeQuantize.with_args(
        observer=quant.MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False
    )
    
    weight = quant.FakeQuantize.with_args(
        observer=quant.MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False
    )
    
    return quant.QConfig(activation=activation, weight=weight)


def prepare_qat_model(model: nn.Module, 
                      qconfig: Optional[quant.QConfig] = None,
                      backend: str = 'qnnpack') -> nn.Module:
    """准备模型进行量化感知训练
    
    Args:
        model: 要量化的模型
        qconfig: 量化配置，如果为None则使用默认配置
        backend: 量化后端，默认使用qnnpack（适合ARM）
        
    Returns:
        准备好进行QAT的模型
    """
    # 设置量化后端
    torch.backends.quantized.engine = backend
    
    # 设置量化配置
    if qconfig is None:
        qconfig = get_qat_qconfig()
    
    model.qconfig = qconfig
    
    # 为子模块设置量化配置
    for name, module in model.named_modules():
        # 跳过某些不适合量化的层
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            module.qconfig = None
        else:
            module.qconfig = qconfig
    
    # 准备QAT
    model.train()
    quant.prepare_qat(model, inplace=True)
    
    print(f"模型已准备进行量化感知训练，后端: {backend}")
    return model


def convert_to_quantized(model: nn.Module, 
                        calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
    """将QAT模型转换为量化模型
    
    Args:
        model: 经过QAT训练的模型
        calibration_data: 可选的校准数据
        
    Returns:
        量化后的模型
    """
    model.eval()
    
    # 如果提供了校准数据，运行一次前向传播
    if calibration_data is not None:
        with torch.no_grad():
            _ = model(calibration_data)
    
    # 转换为量化模型
    quantized_model = quant.convert(model, inplace=False)
    
    print("模型已转换为量化版本")
    return quantized_model


def fuse_modules_for_qat(model: nn.Module) -> nn.Module:
    """融合模块以提高量化效率
    
    Args:
        model: 要融合的模型
        
    Returns:
        融合后的模型
    """
    # 保存当前训练模式
    was_training = model.training
    
    # 设置为eval模式进行融合
    model.eval()
    
    # 融合常见的模式
    patterns_to_fuse = []
    
    # 查找Conv-BatchNorm-ReLU模式
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            for i in range(len(module) - 1):
                # Conv-BatchNorm融合
                if isinstance(module[i], (nn.Conv1d, nn.Conv2d)) and \
                   isinstance(module[i+1], (nn.BatchNorm1d, nn.BatchNorm2d)):
                    if i + 2 < len(module) and isinstance(module[i+2], (nn.ReLU, nn.LeakyReLU)):
                        patterns_to_fuse.append([f'{name}.{i}', f'{name}.{i+1}', f'{name}.{i+2}'])
                    else:
                        patterns_to_fuse.append([f'{name}.{i}', f'{name}.{i+1}'])
    
    try:
        if patterns_to_fuse:
            model = quant.fuse_modules(model, patterns_to_fuse)
            print(f"融合了 {len(patterns_to_fuse)} 个模块组")
        else:
            print("未找到可融合的模块组")
    except Exception as e:
        print(f"模块融合失败: {e}")
        print("继续执行，不进行模块融合")
    
    # 恢复原来的训练模式
    if was_training:
        model.train()
    
    return model


def check_model_quantizable(model: nn.Module) -> Dict[str, Any]:
    """检查模型是否适合量化
    
    Args:
        model: 要检查的模型
        
    Returns:
        包含检查结果的字典
    """
    results = {
        'total_modules': 0,
        'quantizable_modules': 0,
        'non_quantizable_modules': [],
        'warnings': []
    }
    
    quantizable_types = (
        nn.Conv1d, nn.Conv2d, nn.Linear,
        nn.ConvTranspose1d, nn.ConvTranspose2d
    )
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            results['total_modules'] += 1
            
            if isinstance(module, quantizable_types):
                results['quantizable_modules'] += 1
            else:
                results['non_quantizable_modules'].append((name, type(module).__name__))
                
                # 检查是否有需要特殊处理的层
                if isinstance(module, nn.MultiheadAttention):
                    results['warnings'].append(f"{name}: MultiheadAttention需要特殊处理")
                elif isinstance(module, (nn.GRU, nn.LSTM)):
                    results['warnings'].append(f"{name}: RNN层量化支持有限")
    
    results['quantization_ratio'] = (
        results['quantizable_modules'] / results['total_modules'] 
        if results['total_modules'] > 0 else 0
    )
    
    return results


def export_quantized_torchscript(model: nn.Module, 
                                example_input: torch.Tensor,
                                output_path: str) -> None:
    """导出量化模型为TorchScript格式
    
    Args:
        model: 量化后的模型
        example_input: 示例输入
        output_path: 输出路径
    """
    model.eval()
    
    # 使用TorchScript追踪
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
    
    # 优化模型
    traced_model = torch.jit.optimize_for_mobile(traced_model)
    
    # 保存模型
    traced_model.save(output_path)
    print(f"量化模型已导出到: {output_path}")
    
    # 计算模型大小
    import os
    model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"模型大小: {model_size:.2f} MB") 