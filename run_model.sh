#!/bin/bash

# UB-Diff模型训练和生成流程脚本
# 使用方法：
# ./run_model.sh [选项]
# 选项：
#   --train-encdec    训练编码器和解码器
#   --finetune-dec    微调解码器
#   --train-diff      训练扩散模型
#   --generate        生成数据
#   --all-steps       执行所有步骤

# 默认参数设置
# 步骤控制参数
DO_TRAIN_ENCDEC=false           # 是否训练编码器解码器
DO_FINETUNE_DEC=false           # 是否微调解码器
DO_TRAIN_DIFF=false             # 是否训练扩散模型
DO_GENERATION=false             # 是否生成数据

DATASET="curvefault-a"          # 数据集
TRAIN_DATA="./seismic_data"     # 地震数据路径
TRAIN_LABEL="./velocity_map"    # 速度图路径
NUM_DATA=24000                  # 训练速度图大小
PAIRED_NUM=5000                 # 配对数据大小
LEARNING_RATE=5e-4              # 学习率
LEARNING_RATE_DIFF=8e-5         # 扩散学习率
LEARNING_RATE_DECAY=0.995       # 学习率衰减
BATCH_SIZE=64                   # 批量大小
EPOCH_BLOCK_TRAIN=20            # 训练每Block训练的轮数
NUM_BLOCK_TRAIN=2               # 训练的Block数
EPOCH_BLOCK_FINE=10             # 微调每Block训练的轮数
NUM_BLOCK_FINE=1                # 微调的Block数
TIME_STEP=256                   # 扩散模型时间步长
LATENT_DIM=128                  # 潜在维度
NUM_STEPS=15000                 # 扩散模型训练步数
SAVE_AND_SAMPLE_EVERY=3000      # 保存和采样频率
VAL_EVERY=20                    # 保存检查点的频率
NUM_SAMPLES=500                 # 生成样本数量
MODEL_FILE="./checkpoints"       # 扩散模型检查点

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --train-encdec)
      DO_TRAIN_ENCDEC=true
      shift
      ;;
    --finetune-dec)
      DO_FINETUNE_DEC=true
      shift
      ;;
    --train-diff)
      DO_TRAIN_DIFF=true
      shift
      ;;
    --generate)
      DO_GENERATION=true
      shift
      ;;
    --all-steps)
      DO_TRAIN_ENCDEC=true
      DO_FINETUNE_DEC=true
      DO_TRAIN_DIFF=true
      DO_GENERATION=true
      shift
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --train-data)
      TRAIN_DATA="$2"
      shift 2
      ;;
    --train-label)
      TRAIN_LABEL="$2"
      shift 2
      ;;
    --num-data)
      NUM_DATA="$2"
      shift 2
      ;;
    --paired-num)
      PAIRED_NUM="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --learning-rate-diff)
      LEARNING_RATE_DIFF="$2"
      shift 2
      ;;
    --learning-rate-decay)
      LEARNING_RATE_DECAY="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epoch-block-train)
      EPOCH_BLOCK_TRAIN="$2"
      shift 2
      ;;
    --num-block-train)
      NUM_BLOCK_TRAIN="$2"
      shift 2
      ;;
    --epoch-block-fine)
      EPOCH_BLOCK_FINE="$2"
      shift 2
      ;;
    --num-block-fine)
      NUM_BLOCK_FINE="$2"
      shift 2
      ;;
    --time-step)
      TIME_STEP="$2"
      shift 2
      ;;
    --latent-dim)
      LATENT_DIM="$2"
      shift 2
      ;;
    --num-steps)
      NUM_STEPS="$2"
      shift 2
      ;;
    --save-and-sample-every)
      SAVE_AND_SAMPLE_EVERY="$2"
      shift 2
      ;;
    --val-every)
      VAL_EVERY="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --model-file)
      MODEL_FILE="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 检查是否至少选择了一个步骤
if [[ "$DO_TRAIN_ENCDEC" == "false" && "$DO_FINETUNE_DEC" == "false" && "$DO_TRAIN_DIFF" == "false" && "$DO_GENERATION" == "false" ]]; then
    echo "错误：请至少选择一个执行步骤！"
    echo "可用的步骤选项："
    echo "  --train-encdec    训练编码器和解码器"
    echo "  --finetune-dec    微调解码器"
    echo "  --train-diff      训练扩散模型"
    echo "  --generate        生成数据"
    echo "  --all-steps       执行所有步骤"
    exit 1
fi

# 第一步：训练主要数据群体的编码器和解码器（速度图）
if [[ "$DO_TRAIN_ENCDEC" == "true" ]]; then
    echo "======================"
    echo "第一步：训练编码器和解码器（速度图）..."
    echo "======================"
    
    python model/train_EncDec.py --dataset $DATASET \
                          --train-data $TRAIN_DATA \
                          --train-label $TRAIN_LABEL \
                          --num_data $NUM_DATA \
                          --paired_num $PAIRED_NUM \
                          --epoch_block $EPOCH_BLOCK_TRAIN \
                          --num_block $NUM_BLOCK_TRAIN \
                          --val_every $VAL_EVERY \
                          --lr $LEARNING_RATE \
                          --lr-gamma $LEARNING_RATE_DECAY \
                          --batch-size $BATCH_SIZE
fi

# 第二步：基于良好的潜在表示训练少数数据群体的解码器（地震波形）
if [[ "$DO_FINETUNE_DEC" == "true" ]]; then
    echo "======================"
    echo "第二步：微调解码器（地震波形）..."
    echo "======================"
    python model/fine_tune_Dec_S.py --dataset $DATASET \
                             --train-data $TRAIN_DATA \
                             --train-label $TRAIN_LABEL \
                             --num_data $NUM_DATA \
                             --paired_num $PAIRED_NUM \
                             --epoch_block $EPOCH_BLOCK_FINE \
                             --num_block $NUM_BLOCK_FINE \
                             --val_every $VAL_EVERY \
                             --lr $LEARNING_RATE \
                             --lr-gamma $LEARNING_RATE_DECAY \
                             --batch-size $BATCH_SIZE 
fi

# 第三步：训练扩散模型
if [[ "$DO_TRAIN_DIFF" == "true" ]]; then
    echo "======================"
    echo "第三步：训练扩散模型..."
    echo "======================"
    python model/train_diff.py --dataset $DATASET \
                        --train-data $TRAIN_DATA \
                        --train-label $TRAIN_LABEL \
                        --num_data $NUM_DATA \
                        --time_steps $TIME_STEP \
                        --learning_rate $LEARNING_RATE_DIFF \
                        --num_steps $NUM_STEPS \
                        --save_and_sample_every $SAVE_AND_SAMPLE_EVERY \
                        --results_folder $MODEL_FILE \
                        --latent_dim $LATENT_DIM
fi

# 第四步：生成数据
if [[ "$DO_GENERATION" == "true" ]]; then
    echo "======================"
    echo "第四步：生成数据..."
    echo "======================"
    python model/generation.py --dataset $DATASET \
                        --train-data $TRAIN_DATA \
                        --train-label $TRAIN_LABEL \
                        --num_data $NUM_DATA \
                        --time_steps $TIME_STEP \
                        --model_file $MODEL_FILE

fi

echo "======================"
echo "所选步骤已完成！"
echo "======================" 