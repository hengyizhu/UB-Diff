"""
量化感知训练模型模块

包含用于树莓派部署的量化模型实现
"""

from .quantized_ub_diff import QuantizedUBDiff, QuantizedDecoder
from .quantization_utils import (
    prepare_qat_model, 
    convert_to_quantized,
    fuse_modules_for_qat,
    check_model_quantizable,
    export_quantized_torchscript,
    get_qat_qconfig
)

__all__ = [
    'QuantizedUBDiff',
    'QuantizedDecoder',
    'prepare_qat_model',
    'convert_to_quantized',
    'fuse_modules_for_qat',
    'check_model_quantizable',
    'export_quantized_torchscript',
    'get_qat_qconfig'
] 