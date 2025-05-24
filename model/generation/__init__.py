"""
数据生成模块

包含用于生成地震数据和速度场的各种生成器
"""

from .generator import UBDiffGenerator
from .visualizer import ModelVisualizer

__all__ = [
    'UBDiffGenerator',
    'ModelVisualizer'
] 