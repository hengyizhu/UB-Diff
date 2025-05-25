# UB-Diff 量化感知训练与树莓派部署

本项目实现了UB-Diff模型的量化感知训练(QAT)，用于将模型部署到树莓派4B等边缘设备上。

## 🎯 项目目标

- 通过量化感知训练减少模型大小（目标：减少75%）
- 保持生成质量的同时提高推理速度
- 导出适合树莓派部署的纯生成模型（不包含编码器）

## 📁 项目结构

```
qat_deployment/
├── models/                 # 量化模型定义
│   ├── quantized_ub_diff.py    # 量化的UB-Diff模型
│   └── quantization_utils.py   # 量化工具函数
├── trainers/              # QAT训练器
│   ├── qat_decoder_trainer.py  # 解码器QAT训练
│   └── qat_diffusion_trainer.py # 扩散模型QAT训练
├── scripts/               # 训练和导出脚本
│   ├── train_qat_decoders.py   # 训练量化解码器
│   ├── train_qat_diffusion.py  # 训练量化扩散模型
│   └── export_model.py         # 导出TorchScript模型
└── exported_models/       # 导出的模型文件
```

## 🚀 训练流程

### 1. 准备阶段

确保您已经有预训练的模型：
- 编码器-解码器模型
- 微调的地震解码器
- 训练好的扩散模型（位于 `./checkpoints/diffusion/model-4.pt`）

### 2. 阶段1：QAT解码器训练

分别训练速度解码器和地震解码器：

```bash

# 训练量化解码器（先速度后地震）
python qat_deployment/scripts/train_qat_decoders.py \
    --train_data ./CurveFault-A/seismic_data \
    --train_label ./CurveFault-A/velocity_map \
    --pretrained_path ./checkpoints/diffusion/model-4.pt \
    --dataset curvefault-a \
    --quantize_velocity \
    --quantize_seismic \
    --velocity_epochs 1 \
    --seismic_epochs 1 \
    --batch_size 64 \
    --device cuda:1
```

### 3. 阶段2：QAT扩散模型训练

使用量化的解码器训练扩散模型：

```bash
python qat_deployment/scripts/train_qat_diffusion.py \
    --train_data ./CurveFault-A/seismic_data \
    --train_label ./CurveFault-A/velocity_map \
    --pretrained_path ./checkpoints/diffusion/model-4.pt \
    --decoder_checkpoint ./checkpoints/qat_decoders/final_qat_decoders.pt \
    --dataset curvefault-a \
    --quantize_diffusion \
    --epochs 10 \
    --batch_size 16 \
    --device cuda:1
```

### 4. 模型导出

将QAT模型转换为量化模型并导出为TorchScript：

```bash
python qat_deployment/scripts/export_model.py \
    --checkpoint_path ./checkpoints/qat_diffusion/final_qat_model.pt \
    --output_dir ./qat_deployment/exported_models \
    --model_name ub_diff_rpi \
    --backend qnnpack \
    --optimize_for_mobile \
    --test_generation
```

## 📊 量化策略

### 量化配置

- **激活量化**: INT8 (per-tensor, affine)
- **权重量化**: INT8 (per-tensor, symmetric)
- **后端**: QNNPACK（适合ARM设备）

### 不量化的层

- BatchNorm层（会与Conv层融合）
- LayerNorm层
- 注意力机制中的某些操作

## 🔧 部署到树莓派

### 1. 环境准备

在树莓派上安装PyTorch：

```bash
# 安装依赖
sudo apt-get update
sudo apt-get install libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools

# 安装PyTorch（选择适合的版本）
pip3 install torch==1.13.0
```

### 2. 加载模型

```python
import torch
import numpy as np

# 加载量化模型
model = torch.jit.load('ub_diff_rpi.pt')
model.eval()

# 生成数据
with torch.no_grad():
    # 生成单个样本
    velocity, seismic = model(1)
    
    # 转换为numpy用于保存或可视化
    velocity_np = velocity.cpu().numpy()
    seismic_np = seismic.cpu().numpy()
```

### 3. 性能优化建议

1. **批次大小**: 使用batch_size=1以减少内存使用
2. **扩散步数**: 可以减少扩散步数以加快生成（牺牲一定质量）
3. **内存管理**: 及时释放不用的张量
4. **CPU亲和性**: 设置CPU亲和性以优化性能

## 📈 性能指标

### 模型大小对比

| 模型版本 | 大小 | 相对原始 |
|---------|------|---------|
| 原始FP32 | ~400MB | 100% |
| QAT INT8 | ~100MB | 25% |

### 推理速度（树莓派4B）

| 操作 | FP32时间 | INT8时间 | 加速比 |
|------|----------|----------|--------|
| 单次生成 | ~10s | ~2.5s | 4x |

## ⚠️ 注意事项

1. **数据标准化**: 确保使用与训练时相同的数据标准化参数
2. **随机种子**: 生成结果的多样性依赖于随机种子
3. **内存限制**: 树莓派内存有限，避免同时生成多个样本
4. **散热**: 长时间运行需要良好的散热

## 🛠️ 故障排除

### 常见问题

1. **内存不足**
   - 减少batch_size
   - 使用更少的扩散步数
   - 关闭其他应用

2. **量化精度损失**
   - 增加QAT训练轮数
   - 调整量化配置
   - 使用校准数据集

3. **推理速度慢**
   - 确认使用了QNNPACK后端
   - 检查是否正确加载了量化模型
   - 优化扩散步数

## 📚 参考资料

- [PyTorch量化文档](https://pytorch.org/docs/stable/quantization.html)
- [树莓派PyTorch安装](https://github.com/nmilosev/pytorch-arm-builds)
- [移动端优化指南](https://pytorch.org/mobile/home/)

---

如有问题，请查看主项目文档或提交Issue。 