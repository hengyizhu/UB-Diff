"""
改进的量化策略

解决UB-Diff模型量化效果不佳的问题
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Dict, Any, Optional, List
import warnings


class ImprovedQuantizationConfig:
    """改进的量化配置"""
    
    @staticmethod
    def get_aggressive_qconfig(backend: str = 'qnnpack') -> quant.QConfig:
        """获取更激进的量化配置
        
        Args:
            backend: 量化后端
            
        Returns:
            量化配置
        """
        if backend == 'qnnpack':
            # 针对ARM设备的优化配置
            activation = quant.FakeQuantize.with_args(
                observer=quant.MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=False
            )
            
            weight = quant.FakeQuantize.with_args(
                observer=quant.MovingAveragePerChannelMinMaxObserver,  # 使用per-channel
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,  # per-channel对称量化
                reduce_range=False
            )
        else:  # fbgemm
            activation = quant.FakeQuantize.with_args(
                observer=quant.MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=True  # x86需要reduce_range
            )
            
            weight = quant.FakeQuantize.with_args(
                observer=quant.MovingAveragePerChannelMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
                reduce_range=True
            )
        
        return quant.QConfig(activation=activation, weight=weight)


def analyze_model_quantizability(model: nn.Module) -> Dict[str, Any]:
    """深度分析模型的可量化性
    
    Args:
        model: 要分析的模型
        
    Returns:
        详细的分析结果
    """
    results = {
        'total_modules': 0,
        'quantizable_modules': 0,
        'conv1d_modules': 0,
        'conv2d_modules': 0,
        'linear_modules': 0,
        'non_quantizable_modules': [],
        'conv1d_details': [],
        'large_modules': [],
        'warnings': [],
        'recommendations': []
    }
    
    quantizable_types = (
        nn.Conv1d, nn.Conv2d, nn.Linear,
        nn.ConvTranspose1d, nn.ConvTranspose2d
    )
    
    total_params = 0
    quantizable_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            results['total_modules'] += 1
            
            # 计算参数数量
            module_params = sum(p.numel() for p in module.parameters())
            total_params += module_params
            
            if isinstance(module, quantizable_types):
                results['quantizable_modules'] += 1
                quantizable_params += module_params
                
                # 分类统计
                if isinstance(module, nn.Conv1d):
                    results['conv1d_modules'] += 1
                    results['conv1d_details'].append({
                        'name': name,
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size,
                        'params': module_params
                    })
                elif isinstance(module, nn.Conv2d):
                    results['conv2d_modules'] += 1
                elif isinstance(module, nn.Linear):
                    results['linear_modules'] += 1
                
                # 标记大模块
                if module_params > 1000000:  # 超过100万参数
                    results['large_modules'].append({
                        'name': name,
                        'type': type(module).__name__,
                        'params': module_params
                    })
            else:
                results['non_quantizable_modules'].append((name, type(module).__name__))
                
                # 特殊层的警告
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    results['warnings'].append(f"{name}: BatchNorm层应该与Conv层融合")
                elif isinstance(module, nn.LayerNorm):
                    results['warnings'].append(f"{name}: LayerNorm层不支持量化")
                elif isinstance(module, (nn.GELU, nn.SiLU)):
                    results['warnings'].append(f"{name}: 复杂激活函数可能影响量化效果")
    
    # 计算量化参数比例
    results['quantization_ratio'] = (
        results['quantizable_modules'] / results['total_modules'] 
        if results['total_modules'] > 0 else 0
    )
    
    results['param_quantization_ratio'] = (
        quantizable_params / total_params 
        if total_params > 0 else 0
    )
    
    results['total_params'] = total_params
    results['quantizable_params'] = quantizable_params
    
    # 生成建议
    if results['conv1d_modules'] > results['conv2d_modules']:
        results['recommendations'].append(
            "模型主要使用1D卷积，考虑转换为2D卷积以获得更好的量化支持"
        )
    
    if results['param_quantization_ratio'] < 0.5:
        results['recommendations'].append(
            "可量化参数比例较低，考虑重新设计模型架构"
        )
    
    if len(results['large_modules']) > 0:
        results['recommendations'].append(
            "存在大型模块，优先量化这些模块以获得最大收益"
        )
    
    return results


def convert_conv1d_to_conv2d_wrapper(conv1d: nn.Conv1d) -> nn.Module:
    """将Conv1d包装为可量化的形式
    
    Args:
        conv1d: 1D卷积层
        
    Returns:
        包装后的模块
    """
    class Conv1DWrapper(nn.Module):
        def __init__(self, conv1d_layer):
            super().__init__()
            # 创建等效的2D卷积
            self.conv2d = nn.Conv2d(
                in_channels=conv1d_layer.in_channels,
                out_channels=conv1d_layer.out_channels,
                kernel_size=(1, conv1d_layer.kernel_size[0]),
                stride=(1, conv1d_layer.stride[0]),
                padding=(0, conv1d_layer.padding[0]),
                bias=conv1d_layer.bias is not None
            )
            
            # 复制权重
            with torch.no_grad():
                # 重塑权重从 (out, in, k) 到 (out, in, 1, k)
                self.conv2d.weight.copy_(conv1d_layer.weight.unsqueeze(2))
                if conv1d_layer.bias is not None:
                    self.conv2d.bias.copy_(conv1d_layer.bias)
        
        def forward(self, x):
            # x: (B, C, L) -> (B, C, 1, L)
            if x.dim() == 3:
                x = x.unsqueeze(2)
            
            # 2D卷积
            x = self.conv2d(x)
            
            # (B, C, 1, L) -> (B, C, L)
            x = x.squeeze(2)
            return x
    
    return Conv1DWrapper(conv1d)


def apply_improved_quantization(model: nn.Module, 
                              backend: str = 'qnnpack',
                              convert_conv1d: bool = True) -> nn.Module:
    """应用改进的量化策略
    
    Args:
        model: 要量化的模型
        backend: 量化后端
        convert_conv1d: 是否转换1D卷积
        
    Returns:
        量化后的模型
    """
    print("应用改进的量化策略...")
    
    # 1. 分析模型
    analysis = analyze_model_quantizability(model)
    print(f"模型分析结果:")
    print(f"  总模块数: {analysis['total_modules']}")
    print(f"  可量化模块数: {analysis['quantizable_modules']}")
    print(f"  1D卷积数: {analysis['conv1d_modules']}")
    print(f"  2D卷积数: {analysis['conv2d_modules']}")
    print(f"  线性层数: {analysis['linear_modules']}")
    print(f"  参数量化比例: {analysis['param_quantization_ratio']:.2%}")
    
    # 2. 可选：转换1D卷积
    if convert_conv1d and analysis['conv1d_modules'] > 0:
        print(f"转换 {analysis['conv1d_modules']} 个1D卷积层...")
        model = _convert_conv1d_layers(model)
    
    # 3. 设置量化配置
    qconfig = ImprovedQuantizationConfig.get_aggressive_qconfig(backend)
    
    # 4. 智能设置量化配置
    _set_smart_qconfig(model, qconfig, analysis)
    
    # 5. 融合模块
    model = _fuse_modules_intelligently(model)
    
    # 6. 准备QAT
    model.train()
    quant.prepare_qat(model, inplace=True)
    
    print("改进的量化策略应用完成")
    return model


def _convert_conv1d_layers(model: nn.Module) -> nn.Module:
    """递归转换模型中的1D卷积层"""
    for name, child in model.named_children():
        if isinstance(child, nn.Conv1d):
            # 替换为包装器
            setattr(model, name, convert_conv1d_to_conv2d_wrapper(child))
        else:
            # 递归处理子模块
            _convert_conv1d_layers(child)
    return model


def _set_smart_qconfig(model: nn.Module, qconfig: quant.QConfig, analysis: Dict[str, Any]):
    """智能设置量化配置"""
    # 默认配置
    model.qconfig = qconfig
    
    # 为大型模块设置更精确的量化
    large_module_names = {item['name'] for item in analysis['large_modules']}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            # 不量化归一化层
            module.qconfig = None
        elif isinstance(module, (nn.GELU, nn.SiLU, nn.Tanh)):
            # 复杂激活函数使用更保守的量化
            module.qconfig = None
        elif name in large_module_names:
            # 大型模块使用精确量化
            module.qconfig = qconfig
        else:
            module.qconfig = qconfig


def _fuse_modules_intelligently(model: nn.Module) -> nn.Module:
    """智能融合模块"""
    # 保存当前训练模式
    was_training = model.training
    
    # 设置为eval模式进行融合
    model.eval()
    
    patterns_to_fuse = []
    
    # 查找可融合的模式
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            _find_fusable_patterns_in_sequential(module, name, patterns_to_fuse)
    
    # 执行融合
    if patterns_to_fuse:
        try:
            model = quant.fuse_modules(model, patterns_to_fuse)
            print(f"成功融合 {len(patterns_to_fuse)} 个模块组")
        except Exception as e:
            print(f"模块融合失败: {e}")
            warnings.warn("模块融合失败，继续执行量化")
    else:
        print("未找到可融合的模块组")
    
    # 恢复原来的训练模式
    if was_training:
        model.train()
    
    return model


def _find_fusable_patterns_in_sequential(sequential: nn.Sequential, 
                                       base_name: str, 
                                       patterns: List[List[str]]):
    """在Sequential模块中查找可融合的模式"""
    modules = list(sequential.children())
    
    i = 0
    while i < len(modules) - 1:
        current = modules[i]
        next_module = modules[i + 1]
        
        # Conv + BN 模式
        if isinstance(current, (nn.Conv1d, nn.Conv2d)) and \
           isinstance(next_module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            
            pattern = [f"{base_name}.{i}", f"{base_name}.{i+1}"]
            
            # 检查是否有ReLU
            if i + 2 < len(modules) and isinstance(modules[i + 2], (nn.ReLU, nn.LeakyReLU)):
                pattern.append(f"{base_name}.{i+2}")
                i += 3
            else:
                i += 2
            
            patterns.append(pattern)
        else:
            i += 1


def create_quantization_report(model: nn.Module, 
                             original_size_mb: float,
                             quantized_size_mb: float) -> str:
    """创建量化报告
    
    Args:
        model: 量化后的模型
        original_size_mb: 原始模型大小(MB)
        quantized_size_mb: 量化后模型大小(MB)
        
    Returns:
        量化报告字符串
    """
    analysis = analyze_model_quantizability(model)
    compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 0
    
    report = f"""
量化报告
========

模型统计:
- 总模块数: {analysis['total_modules']}
- 可量化模块数: {analysis['quantizable_modules']}
- 量化率: {analysis['quantization_ratio']:.2%}
- 参数量化比例: {analysis['param_quantization_ratio']:.2%}

模块分布:
- 1D卷积: {analysis['conv1d_modules']}
- 2D卷积: {analysis['conv2d_modules']}  
- 线性层: {analysis['linear_modules']}

大小对比:
- 原始大小: {original_size_mb:.2f} MB
- 量化后大小: {quantized_size_mb:.2f} MB
- 压缩比: {compression_ratio:.2f}x
- 大小减少: {((original_size_mb - quantized_size_mb) / original_size_mb * 100):.1f}%

建议:
"""
    
    for rec in analysis['recommendations']:
        report += f"- {rec}\n"
    
    if analysis['warnings']:
        report += "\n警告:\n"
        for warning in analysis['warnings']:
            report += f"- {warning}\n"
    
    return report 