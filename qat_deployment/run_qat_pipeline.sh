#!/bin/bash

# UB-Diff QAT训练和部署流程脚本 - 简化版本
# 参考full training脚本的风格，使用固定参数

set -e  # 遇到错误时退出

echo "=========================================="
echo "开始 UB-Diff QAT训练和部署流程"
echo "=========================================="

# 检查必要的目录是否存在
if [ ! -d "./CurveFault-A/seismic_data" ]; then
    echo "错误: 找不到训练数据目录 ./CurveFault-A/seismic_data"
    exit 1
fi

if [ ! -d "./CurveFault-A/velocity_map" ]; then
    echo "错误: 找不到标签数据目录 ./CurveFault-A/velocity_map"
    exit 1
fi

# 检查预训练模型是否存在
if [ ! -f "./checkpoints/diffusion/model-4.pt" ]; then
    echo "错误: 找不到预训练模型 ./checkpoints/diffusion/model-4.pt"
    exit 1
fi

# 创建必要的输出目录
mkdir -p ./checkpoints/qat_decoders
mkdir -p ./checkpoints/qat_diffusion
mkdir -p ./qat_deployment/exported_models

echo "=========================================="
echo "QAT阶段1: 训练QAT解码器"
echo "=========================================="

python qat_deployment/scripts/train_qat_decoders.py \
    --train_data "./CurveFault-A/seismic_data" \
    --train_label "./CurveFault-A/velocity_map" \
    --val_data "./CurveFault-A/seismic_data" \
    --val_label "./CurveFault-A/velocity_map" \
    --pretrained_path "./checkpoints/diffusion/model-4.pt" \
    --dataset "curvefault-a" \
    --quantize_velocity \
    --quantize_seismic \
    --velocity_epochs 50 \
    --seismic_epochs 50 \
    --batch_size 64 \
    --device cuda:1

if [ $? -ne 0 ]; then
    echo "错误: QAT解码器训练失败"
    exit 1
fi

echo "QAT解码器训练完成！"

echo "=========================================="
echo "QAT阶段2: 训练QAT扩散模型"
echo "=========================================="

python qat_deployment/scripts/final_fixed_train_qat_diffusion.py \
    --train_data "./CurveFault-A/seismic_data" \
    --train_label "./CurveFault-A/velocity_map" \
    --val_data "./CurveFault-A/seismic_data" \
    --val_label "./CurveFault-A/velocity_map" \
    --pretrained_path "./checkpoints/diffusion/model-4.pt" \
    --decoder_checkpoint "./checkpoints/qat_decoders/best_combined_qat_decoders.pt" \
    --dataset "curvefault-a" \
    --quantize_diffusion \
    --backend "qnnpack" \
    --convert_conv1d \
    --use_aggressive_quantization \
    --epochs 30 \
    --batch_size 64 \
    --lr 5e-5 \
    --device cuda:1 \
    --checkpoint_dir "./checkpoints/qat_diffusion"

if [ $? -ne 0 ]; then
    echo "错误: QAT扩散模型训练失败"
    exit 1
fi

echo "QAT扩散模型训练完成！"

echo "=========================================="
echo "QAT阶段3: 导出优化模型"
echo "=========================================="

python qat_deployment/scripts/export_model.py \
    --checkpoint_path "./checkpoints/qat_diffusion/best_qat_diffusion_final.pt" \
    --output_dir "./qat_deployment/exported_models" \
    --model_name "curvefault-a" \
    --backend "qnnpack" \
    --convert_conv1d \
    --force_quantization \
    --optimize_for_mobile

if [ $? -ne 0 ]; then
    echo "错误: 模型导出失败"
    exit 1
fi

echo "模型导出完成！"

echo "=========================================="
echo "🎉 QAT训练和部署流程完成！"
echo "=========================================="
echo "训练结果保存在以下目录："
echo "- QAT解码器: ./checkpoints/qat_decoders/"
echo "- QAT扩散模型: ./checkpoints/qat_diffusion/"
echo "- 导出模型: ./qat_deployment/exported_models/"
echo "==========================================" 
echo "下一步:"
echo "1. 将模型复制到树莓派: scp ./qat_deployment/exported_models/curvefault-a.pt pi@your-rpi:/path/to/model/"
echo "2. 在树莓派上测试模型: python test_model.py"
echo "3. 查看性能指标和使用说明: cat qat_deployment/README.md" 