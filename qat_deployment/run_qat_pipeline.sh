#!/bin/bash

# QAT训练和部署流水线 - 改进版本
# 集成改进的量化策略，提升量化效果

set -e  # 遇到错误时退出

echo "🚀 开始改进的QAT训练和部署流水线"
echo "=================================="

# 配置参数
TRAIN_DATA="./CurveFault-A/seismic_data"
TRAIN_LABEL="./CurveFault-A/velocity_map"
VAL_DATA="./CurveFault-A/seismic_data"
VAL_LABEL="./CurveFault-A/velocity_map"
DATASET="curvefault-a"

# 模型路径
PRETRAINED_PATH="./checkpoints/diffusion/model-4.pt"

# 训练参数
BATCH_SIZE=64
EPOCHS_DECODER=1
EPOCHS_DIFFUSION=1  
LR=5e-5        # 降低学习率
DEVICE="cuda:1"

# 改进的量化参数
BACKEND="qnnpack"  # 适合ARM设备
CONVERT_CONV1D=true
USE_AGGRESSIVE_QUANTIZATION=true
QUANTIZATION_WARMUP_EPOCHS=5

# 输出目录
DECODER_CHECKPOINT_DIR="./checkpoints/qat_decoders"
DIFFUSION_CHECKPOINT_DIR="./checkpoints/qat_diffusion"
EXPORT_DIR="./qat_deployment/exported_models"

# 颜色输出函数
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_separator() {
    echo "=================================================================="
}

# 检查点目录
CHECKPOINTS_DIR="./checkpoints"
QAT_DECODERS_DIR="${CHECKPOINTS_DIR}/qat_decoders"
QAT_DIFFUSION_DIR="${CHECKPOINTS_DIR}/qat_diffusion"
EXPORTED_MODELS_DIR="./qat_deployment/exported_models"

# 解析命令行参数
show_help() {
    cat << EOF
UB-Diff QAT 自动训练脚本

用法: $0 [选项]

选项:
    -h, --help              显示此帮助信息
    --train-data PATH       训练数据路径 (默认: $TRAIN_DATA)
    --train-label PATH      训练标签路径 (默认: $TRAIN_LABEL)
    --pretrained-path PATH  预训练模型路径 (默认: $PRETRAINED_PATH)
    --dataset NAME          数据集名称 (默认: $DATASET)
    --device DEVICE         设备 (默认: $DEVICE)
    --velocity-epochs N     速度解码器训练轮数 (默认: $EPOCHS_DECODER)
    --seismic-epochs N      地震解码器训练轮数 (默认: $EPOCHS_DECODER)
    --diffusion-epochs N    扩散模型训练轮数 (默认: $EPOCHS_DIFFUSION)
    --decoder-batch-size N  解码器批次大小 (默认: $BATCH_SIZE)
    --diffusion-batch-size N 扩散模型批次大小 (默认: $BATCH_SIZE)
    --model-name NAME       导出模型名称 (默认: $DATASET)
    --skip-decoders         跳过解码器训练
    --skip-diffusion        跳过扩散模型训练
    --skip-export           跳过模型导出
    --resume-from STAGE     从指定阶段恢复 (decoders|diffusion|export)
    
示例:
    $0                      # 使用默认参数运行完整流程
    $0 --device cuda:0      # 在GPU 0上运行
    $0 --skip-decoders      # 跳过解码器训练
    $0 --resume-from diffusion  # 从扩散模型训练开始
EOF
}

SKIP_DECODERS=false
SKIP_DIFFUSION=false
SKIP_EXPORT=false
RESUME_FROM=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --train-data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        --train-label)
            TRAIN_LABEL="$2"
            shift 2
            ;;
        --pretrained)
            PRETRAINED_PATH="$2"
            shift 2
            ;;
        --pretrained-path)
            PRETRAINED_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --velocity-epochs)
            EPOCHS_DECODER="$2"
            shift 2
            ;;
        --seismic-epochs)
            EPOCHS_DIFFUSION="$2"
            shift 2
            ;;
        --diffusion-epochs)
            EPOCHS_DIFFUSION="$2"
            shift 2
            ;;
        --decoder-batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --diffusion-batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model-name)
            DATASET="$2"
            shift 2
            ;;
        --skip-decoders)
            SKIP_DECODERS=true
            shift
            ;;
        --skip-diffusion)
            SKIP_DIFFUSION=true
            shift
            ;;
        --skip-export)
            SKIP_EXPORT=true
            shift
            ;;
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 验证环境
check_environment() {
    log_info "检查环境..."
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        log_error "未找到Python"
        exit 1
    fi
    
    # 检查必要的文件
    if [[ ! -d "$TRAIN_DATA" ]]; then
        log_error "训练数据目录不存在: $TRAIN_DATA"
        exit 1
    fi
    
    if [[ ! -d "$TRAIN_LABEL" ]]; then
        log_error "训练标签目录不存在: $TRAIN_LABEL"
        exit 1
    fi
    
    if [[ ! -f "$PRETRAINED_PATH" ]]; then
        log_error "预训练模型不存在: $PRETRAINED_PATH"
        exit 1
    fi
    
    # 创建必要的目录
    mkdir -p "$QAT_DECODERS_DIR"
    mkdir -p "$QAT_DIFFUSION_DIR"
    mkdir -p "$EXPORTED_MODELS_DIR"
    
    log_success "环境检查完成"
}

# 显示配置
show_config() {
    print_separator
    log_info "QAT训练配置:"
    echo "训练数据路径: $TRAIN_DATA"
    echo "训练标签路径: $TRAIN_LABEL"
    echo "预训练模型: $PRETRAINED_PATH"
    echo "数据集: $DATASET"
    echo "设备: $DEVICE"
    echo "速度解码器轮数: $EPOCHS_DECODER"
    echo "地震解码器轮数: $EPOCHS_DIFFUSION"
    echo "解码器批次大小: $BATCH_SIZE"
    echo "扩散模型批次大小: $BATCH_SIZE"
    echo "导出模型名称: $DATASET"
    print_separator
}

# 训练QAT解码器
train_qat_decoders() {
    if [[ "$SKIP_DECODERS" == true ]]; then
        log_warning "跳过解码器训练"
        return 0
    fi
    
    if [[ "$RESUME_FROM" != "" && "$RESUME_FROM" != "decoders" ]]; then
        log_info "从$RESUME_FROM阶段恢复，跳过解码器训练"
        return 0
    fi
    
    print_separator
    log_info "开始训练QAT解码器..."
    
    local start_time=$(date +%s)
    
    python qat_deployment/scripts/train_qat_decoders.py \
        --train_data "$TRAIN_DATA" \
        --train_label "$TRAIN_LABEL" \
        --pretrained_path "$PRETRAINED_PATH" \
        --dataset "$DATASET" \
        --quantize_velocity \
        --quantize_seismic \
        --velocity_epochs $EPOCHS_DECODER \
        --seismic_epochs $EPOCHS_DIFFUSION \
        --batch_size $BATCH_SIZE \
        --device "$DEVICE"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $? -eq 0 ]]; then
        log_success "QAT解码器训练完成 (耗时: ${duration}秒)"
    else
        log_error "QAT解码器训练失败"
        exit 1
    fi
}

# 训练QAT扩散模型
train_qat_diffusion() {
    if [[ "$SKIP_DIFFUSION" == true ]]; then
        log_warning "跳过扩散模型训练"
        return 0
    fi
    
    if [[ "$RESUME_FROM" == "export" ]]; then
        log_info "从export阶段恢复，跳过扩散模型训练"
        return 0
    fi
    
    # 检查解码器检查点
    local decoder_checkpoint="${QAT_DECODERS_DIR}/final_qat_decoders.pt"
    if [[ ! -f "$decoder_checkpoint" ]]; then
        log_error "未找到解码器检查点: $decoder_checkpoint"
        log_error "请先运行解码器训练或提供正确的检查点路径"
        exit 1
    fi
    
    print_separator
    log_info "开始训练QAT扩散模型..."
    
    local start_time=$(date +%s)
    
    python qat_deployment/scripts/train_qat_diffusion.py \
        --train_data "$TRAIN_DATA" \
        --train_label "$TRAIN_LABEL" \
        --pretrained_path "$PRETRAINED_PATH" \
        --decoder_checkpoint "$decoder_checkpoint" \
        --dataset "$DATASET" \
        --quantize_diffusion \
        --backend "$BACKEND" \
        --convert_conv1d \
        --use_aggressive_quantization \
        --quantization_warmup_epochs $QUANTIZATION_WARMUP_EPOCHS \
        --epochs $EPOCHS_DIFFUSION \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --device "$DEVICE" \
        --checkpoint_dir "$DIFFUSION_CHECKPOINT_DIR"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $? -eq 0 ]]; then
        log_success "QAT扩散模型训练完成 (耗时: ${duration}秒)"
    else
        log_error "QAT扩散模型训练失败"
        exit 1
    fi
}

# 导出模型
export_model() {
    if [[ "$SKIP_EXPORT" == true ]]; then
        log_warning "跳过模型导出"
        return 0
    fi
    
    # 检查扩散模型检查点
    local diffusion_checkpoint="${QAT_DIFFUSION_DIR}/final_qat_model.pt"
    if [[ ! -f "$diffusion_checkpoint" ]]; then
        log_error "未找到扩散模型检查点: $diffusion_checkpoint"
        log_error "请先运行扩散模型训练"
        exit 1
    fi
    
    print_separator
    log_info "开始导出模型..."
    
    local start_time=$(date +%s)
    
    python qat_deployment/scripts/export_model.py \
        --checkpoint_path "$diffusion_checkpoint" \
        --output_dir "$EXPORTED_MODELS_DIR" \
        --model_name "$DATASET" \
        --backend "$BACKEND" \
        --convert_conv1d \
        --force_quantization \
        --optimize_for_mobile \
        --test_generation
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $? -eq 0 ]]; then
        log_success "模型导出完成 (耗时: ${duration}秒)"
        log_info "导出的模型位于: ${EXPORTED_MODELS_DIR}/${DATASET}.pt"
    else
        log_error "模型导出失败"
        exit 1
    fi
}

# 主函数
main() {
    local total_start_time=$(date +%s)
    
    log_info "开始UB-Diff QAT训练流程..."
    
    check_environment
    show_config
    
    # 执行训练阶段
    train_qat_decoders
    train_qat_diffusion
    export_model
    
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    
    print_separator
    log_success "QAT训练流程完成!"
    log_info "总耗时: ${total_duration}秒"
    log_info "导出的模型: ${EXPORTED_MODELS_DIR}/${DATASET}.pt"
    
    # 显示模型信息
    if [[ -f "${EXPORTED_MODELS_DIR}/${DATASET}.pt" ]]; then
        local model_size=$(du -h "${EXPORTED_MODELS_DIR}/${DATASET}.pt" | cut -f1)
        log_info "模型大小: $model_size"
    fi
    
    print_separator
    log_info "下一步:"
    echo "1. 将模型复制到树莓派: scp ${EXPORTED_MODELS_DIR}/${DATASET}.pt pi@your-rpi:/path/to/model/"
    echo "2. 在树莓派上测试模型: python test_model.py"
    echo "3. 查看性能指标和使用说明: cat qat_deployment/README.md"
}

# 执行主函数
main "$@" 