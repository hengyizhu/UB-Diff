# UB-Diff 重构训练框架使用指南

## 概述

本文档介绍重构后的UB-Diff训练框架的使用方法。重构后的框架具有以下特点：

- **模块化设计**: 清晰的训练器、数据加载、生成器分离
- **类型安全**: 全面的类型注解
- **阶段化训练**: 三阶段训练流程
- **配置灵活**: 支持多种数据集和超参数配置
- **中文文档**: 完整的中文注释和文档

## 项目结构

```
model/
├── data/                    # 数据处理模块
│   ├── dataset.py          # 数据集加载
│   ├── transforms.py       # 数据变换
│   ├── dataset_config.json # 数据集配置
│   └── __init__.py
├── trainers/               # 训练器模块
│   ├── encoder_decoder_trainer.py  # 编解码器训练器
│   ├── finetune_trainer.py        # 微调训练器
│   ├── diffusion_trainer.py       # 扩散模型训练器
│   ├── utils.py                   # 训练工具函数
│   └── __init__.py
├── generation/             # 生成器模块
│   ├── generator.py        # UB-Diff生成器
│   ├── visualizer.py       # 可视化工具
│   └── __init__.py
├── components/             # 模型组件（已有）
├── ub_diff.py             # 主模型架构（已有）
└── architecture.py        # 旧架构（保留）

scripts/                   # 训练脚本
├── train_encoder_decoder.py    # 编解码器训练
├── finetune_seismic_decoder.py # 地震解码器微调
├── train_diffusion.py          # 扩散模型训练
└── generate_data.py            # 数据生成
```

## 训练流程

UB-Diff模型采用三阶段训练策略：

### 阶段1: 编码器-解码器训练

训练编码器和速度解码器，学习速度场的潜在表示。

```bash
python scripts/train_encoder_decoder.py \
    --train_data ./CurveFault-A/seismic_data \
    --train_label ./CurveFault-A/velocity_map \
    --dataset curvefault-a \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-4 \
    --val_every 2 \
    --output_path ./checkpoints/encoder_decoder \
    --device cuda:1 \
    --workers 16 \
    --use_wandb
```

### 阶段2: 地震解码器微调

基于预训练的编码器，微调地震解码器学习地震数据重构。

```bash
python scripts/finetune_seismic_decoder.py \
    --train_data ./CurveFault-A/seismic_data \
    --train_label ./CurveFault-A/velocity_map \
    --dataset curvefault-a \
    --checkpoint_path ./checkpoints/encoder_decoder/checkpoint_best.pth \
    --epochs 4 \
    --val_every 2 \
    --batch_size 64 \
    --lr 5e-5 \
    --output_path ./checkpoints/finetune \
    --device cuda:1 \
    --workers 16 \
    --use_wandb
```

### 阶段3: 扩散模型训练

冻结编解码器，训练扩散模型学习潜在空间分布。

```bash
python scripts/train_diffusion.py \
    --train_data ./CurveFault-A/seismic_data \
    --train_label ./CurveFault-A/velocity_map \
    --dataset curvefault-a \
    --checkpoint_path ./checkpoints/finetune/finetune_checkpoint_best.pth \
    --num_steps 5000 \
    --batch_size 16 \
    --learning_rate 8e-5 \
    --results_folder ./checkpoints/diffusion \
    --device cuda:1 \
    --use_wandb
```

## 数据生成

使用训练好的完整模型生成新数据：

```bash
python scripts/generate_data.py \
    --checkpoint_path ./checkpoints/diffusion/model-4.pt \
    --dataset curvefault-a \
    --num_samples 100 \
    --batch_size 16 \
    --output_dir ./generated_data \
    --visualize \
    --evaluate_quality \
    --real_data_path ./CurveFault-A/seismic_data \
    --real_label_path ./CurveFault-A/velocity_map \
    --device cuda:1
```

## 参数配置

### 数据集支持

框架支持以下数据集（在`model/data/dataset_config.json`中配置）：

- `flatvel-a`: 平层速度模型A
- `flatvel-b`: 平层速度模型B  
- `curvefault-a`: 弯曲断层模型A
- `curvefault-b`: 弯曲断层模型B
- `flatfault-a`: 平直断层模型A
- `flatfault-b`: 平直断层模型B

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `encoder_dim` | 512 | 编码器潜在维度 |
| `batch_size` | 64/16 | 批次大小（编解码器/扩散） |
| `learning_rate` | 1e-4/8e-5 | 学习率 |
| `time_steps` | 256 | 扩散时间步数 |
| `objective` | pred_v | 扩散目标函数 |
| `lambda_g1v` | 1.0 | L1损失权重 |
| `lambda_g2v` | 1.0 | L2损失权重 |

## 监控和日志

### Wandb集成

框架集成了Wandb用于实验记录：

```bash
# 启用wandb记录
--use_wandb --proj_name "UB-Diff-Experiment"
```

记录的指标包括：
- 训练/验证损失
- SSIM指数
- 学习率变化
- 梯度范数
- 生成样本

### 检查点管理

- 自动保存最佳模型
- 支持断点续训
- 完整的训练状态保存

## 数据处理

### 数据变换管道

框架提供了灵活的数据预处理：

```python
from model.data import create_standard_transforms, create_velocity_transforms

# 地震数据变换
seismic_transform = create_standard_transforms(
    data_min=0.0, data_max=10.0, k=1.0
)

# 速度场变换  
velocity_transform = create_velocity_transforms(
    label_min=1500.0, label_max=4500.0
)
```

### 数据增强

支持多种数据增强技术：
- 随机裁剪
- 水平翻转
- 噪声添加
- PCA降维

## 可视化和分析

### 生成结果可视化

```python
from model.generation import ModelVisualizer

visualizer = ModelVisualizer()

# 绘制速度场
visualizer.plot_velocity_field(velocity_data)

# 绘制地震数据
visualizer.plot_seismic_data(seismic_data)

# 对比真实vs生成
visualizer.plot_comparison(real_vel, gen_vel, real_seis, gen_seis)
```

### 训练指标可视化

```python
# 绘制训练曲线
metrics = {'loss': loss_history, 'ssim': ssim_history}
visualizer.plot_training_metrics(metrics)
```

## 性能优化

### 内存优化

- 支持数据预加载
- 梯度累积
- 混合精度训练（可选）

### 计算优化

- 多GPU支持（待实现）
- 数据并行加载
- 高效的扩散采样

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 启用梯度累积
   - 关闭数据预加载

2. **训练不收敛**
   - 检查学习率设置
   - 确认数据预处理正确
   - 调整损失函数权重

3. **生成质量差**
   - 增加训练步数
   - 调整扩散参数
   - 检查编解码器质量

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查模型参数
from model.trainers.utils import count_parameters
params = count_parameters(model)
print(f"可训练参数: {params['trainable_params']:,}")
```

## 扩展指南

### 添加新数据集

1. 在`dataset_config.json`中添加配置
2. 实现对应的数据加载逻辑
3. 调整数据变换参数

### 自定义训练器

```python
from model.trainers.utils import MetricLogger
from model.trainers.encoder_decoder_trainer import EncoderDecoderTrainer

class CustomTrainer(EncoderDecoderTrainer):
    def custom_loss(self, pred, target):
        # 自定义损失函数
        return custom_loss_value
```

### 新增模型组件

在`model/components/`中添加新的模型组件，遵循现有接口规范。

## 最佳实践

1. **数据管理**: 使用统一的数据格式和命名规范
2. **实验记录**: 始终使用wandb记录实验参数和结果
3. **代码版本**: 使用git记录代码变更
4. **资源监控**: 监控GPU使用率和内存占用
5. **定期备份**: 定期备份重要的检查点文件

## 联系支持

如有问题或建议，请查阅代码注释或提交issue。 