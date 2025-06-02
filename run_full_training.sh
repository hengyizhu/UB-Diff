#!/bin/bash

# UB-Diff 完整训练流程脚本
# 包含主训练的三个阶段和QAT部署的三个步骤

set -e  # 遇到错误时退出

echo "=========================================="
echo "开始 UB-Diff 完整训练流程"
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

# 创建必要的输出目录
mkdir -p ./checkpoints/encoder_decoder
mkdir -p ./checkpoints/finetune
mkdir -p ./checkpoints/diffusion
mkdir -p ./checkpoints/qat_decoders
mkdir -p ./checkpoints/qat_diffusion

echo "=========================================="
echo "阶段1: 编码器-解码器训练"
echo "=========================================="

python scripts/train_encoder_decoder.py \
    --train_data ./CurveFault-A/seismic_data \
    --train_label ./CurveFault-A/velocity_map \
    --dataset curvefault-a \
    --epochs 300 \
    --batch_size 64 \
    --lr 5e-4 \
    --workers 16 \
    --preload_workers 16 \
    --device cuda:1 \
    --use_wandb

if [ $? -ne 0 ]; then
    echo "错误: 阶段1训练失败"
    exit 1
fi

echo "阶段1完成！"

echo "=========================================="
echo "阶段2: 地震解码器微调"
echo "=========================================="

python scripts/finetune_seismic_decoder.py \
    --train_data ./CurveFault-A/seismic_data \
    --train_label ./CurveFault-A/velocity_map \
    --dataset curvefault-a \
    --checkpoint_path ./checkpoints/encoder_decoder/checkpoint_epoch_220_best.pth \
    --epochs 300 \
    --val_every 10 \
    --batch_size 64 \
    --lr 5e-5 \
    --output_path ./checkpoints/finetune \
    --device cuda:1 \
    --preload_workers 16 \
    --workers 16 \
    --use_wandb

if [ $? -ne 0 ]; then
    echo "错误: 阶段2训练失败"
    exit 1
fi

echo "阶段2完成！"

echo "=========================================="
echo "阶段3: 扩散模型训练"
echo "=========================================="

python scripts/train_diffusion.py \
    --train_data ./CurveFault-A/seismic_data \
    --train_label ./CurveFault-A/velocity_map \
    --dataset curvefault-a \
    --checkpoint_path ./checkpoints/finetune/finetune_checkpoint_epoch_190_best.pth \
    --num_steps 150000 \
    --batch_size 16 \
    --workers 8 \
    --preload_workers 8 \
    --learning_rate 8e-5 \
    --results_folder ./checkpoints/diffusion \
    --device cuda:1 \
    --use_wandb

if [ $? -ne 0 ]; then
    echo "错误: 阶段3训练失败"
    exit 1
fi

echo "阶段3完成！"

echo "=========================================="
echo "QAT部署阶段1: 训练QAT解码器"
echo "=========================================="

python qat_deployment/scripts/train_qat_decoders.py \
    --train_data "./CurveFault-A/seismic_data" \
    --train_label "./CurveFault-A/velocity_map" \
    --val_data "./CurveFault-A/seismic_data" \
    --val_label "./CurveFault-A/velocity_map" \
    --pretrained_path "./checkpoints/diffusion/model-5.pt" \
    --dataset "curvefault-a" \
    --quantize_velocity \
    --quantize_seismic \
    --velocity_epochs 100 \
    --seismic_epochs 100 \
    --backend "qnnpack" \
    --batch_size 64 \
    --device cuda:1

if [ $? -ne 0 ]; then
    echo "错误: QAT解码器训练失败"
    exit 1
fi

echo "QAT解码器训练完成！"

echo "=========================================="
echo "QAT部署阶段2: 训练改进的QAT扩散模型"
echo "=========================================="

python qat_deployment/scripts/train_qat_diffusion.py \
    --train_data "./CurveFault-A/seismic_data" \
    --train_label "./CurveFault-A/velocity_map" \
    --val_data "./CurveFault-A/seismic_data" \
    --val_label "./CurveFault-A/velocity_map" \
    --pretrained_path "./checkpoints/diffusion/model-5.pt" \
    --decoder_checkpoint "./checkpoints/qat_decoders/final_qat_decoders.pt" \
    --dataset "curvefault-a" \
    --quantize_diffusion \
    --convert_conv1d \
    --use_aggressive_quantization \
    --backend "qnnpack" \
    --epochs 100 \
    --batch_size 16 \
    --device cuda:1

if [ $? -ne 0 ]; then
    echo "错误: QAT扩散模型训练失败"
    exit 1
fi

echo "QAT扩散模型训练完成！"

echo "=========================================="
echo "QAT部署阶段3: 导出优化模型"
echo "=========================================="

python qat_deployment/scripts/export_model.py \
    --checkpoint_path "./checkpoints/qat_diffusion/final_qat_model.pt" \
    --convert_conv1d \
    --force_quantization \
    --test_generation

if [ $? -ne 0 ]; then
    echo "错误: 模型导出失败"
    exit 1
fi

echo "模型导出完成！"

echo "=========================================="
echo "🎉 所有训练阶段完成！"
echo "=========================================="
echo "训练结果保存在以下目录："
echo "- 编码器-解码器: ./checkpoints/encoder_decoder/"
echo "- 微调模型: ./checkpoints/finetune/"
echo "- 扩散模型: ./checkpoints/diffusion/"
echo "- QAT解码器: ./checkpoints/qat_decoders/"
echo "- QAT扩散模型: ./checkpoints/qat_diffusion/"
echo "- 导出模型: ./exported_models/"
echo "==========================================" 