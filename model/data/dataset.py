"""
重构后的数据集加载模块

提供清晰、类型安全的数据集接口
"""

import os
import json
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict, Any, Union
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from functools import lru_cache

from . import transforms as T


class SeismicVelocityDataset(Dataset):
    """地震数据和速度场数据集
    
    支持灵活的数据加载和预处理，优化版本支持多进程预加载和内存映射
    """
    
    def __init__(self, 
                 seismic_folder: str,
                 velocity_folder: str,
                 seismic_transform: Optional[T.Compose] = None,
                 velocity_transform: Optional[T.Compose] = None,
                 sample_ratio: int = 1,
                 file_size: int = 500,
                 preload: bool = True,
                 fault_family: bool = True,
                 max_files: Optional[int] = None,
                 preload_workers: int = 4,
                 cache_size: int = 32,
                 use_memmap: bool = False):
        """
        Args:
            seismic_folder: 地震数据文件夹路径
            velocity_folder: 速度场数据文件夹路径
            seismic_transform: 地震数据变换
            velocity_transform: 速度场数据变换
            sample_ratio: 采样比率
            file_size: 每个文件包含的样本数
            preload: 是否预加载所有数据到内存
            fault_family: 是否为断层族数据集
            max_files: 最大文件数限制
            preload_workers: 预加载使用的进程数
            cache_size: LRU缓存大小（当不预加载时使用）
            use_memmap: 是否使用内存映射（仅对大文件有效）
        """
        self.seismic_folder = Path(seismic_folder)
        self.velocity_folder = Path(velocity_folder)
        self.sample_ratio = sample_ratio
        self.fault_family = fault_family
        self.file_size = file_size
        self.preload = preload
        self.preload_workers = min(preload_workers, mp.cpu_count())
        self.cache_size = cache_size
        self.use_memmap = use_memmap
        
        # 加载文件列表
        self.files = self._load_file_names()
        if max_files:
            self.files = self.files[:max_files]
            
        self.seismic_transform = seismic_transform
        self.velocity_transform = velocity_transform
        
        # 数据存储
        self.seismic_data_list = []
        self.velocity_data_list = []
        
        # 线程锁（用于多线程安全）
        self._lock = threading.Lock()
        
        # 预加载数据
        if self.preload:
            print(f"开始预加载数据，使用 {self.preload_workers} 个进程...")
            self._preload_data_parallel()
        else:
            # 设置LRU缓存
            self._load_sample_cached = lru_cache(maxsize=self.cache_size)(self._load_sample)

    def _load_file_names(self) -> List[str]:
        """加载文件名列表"""
        files = []
        for file_path in self.seismic_folder.iterdir():
            if file_path.is_file() and file_path.suffix == '.npy':
                files.append(file_path.name)
        return sorted(files)

    def _get_velocity_filename(self, seismic_filename: str) -> str:
        """根据地震数据文件名获取对应的速度场文件名"""
        if self.fault_family:
            return seismic_filename.replace('seis', 'vel')
        else:
            return seismic_filename.replace('data', 'model')

    def _load_sample_worker(self, seismic_file: str) -> Tuple[int, np.ndarray, Optional[np.ndarray]]:
        """多进程加载单个文件的数据"""
        try:
            # 加载地震数据
            seismic_path = self.seismic_folder / seismic_file
            if self.use_memmap:
                seismic_data = np.load(seismic_path, mmap_mode='r')[:, :, ::self.sample_ratio, :].astype(np.float32)
            else:
                seismic_data = np.load(seismic_path)[:, :, ::self.sample_ratio, :].astype(np.float32)
            
            # 加载速度场数据
            velocity_file = self._get_velocity_filename(seismic_file)
            velocity_path = self.velocity_folder / velocity_file
            
            if velocity_path.exists():
                if self.use_memmap:
                    velocity_data = np.load(velocity_path, mmap_mode='r').astype(np.float32)
                else:
                    velocity_data = np.load(velocity_path).astype(np.float32)
            else:
                velocity_data = None
                
            # 获取文件索引
            file_idx = self.files.index(seismic_file)
            return file_idx, seismic_data, velocity_data
            
        except Exception as e:
            print(f"加载文件 {seismic_file} 时出错: {e}")
            file_idx = self.files.index(seismic_file)
            return file_idx, None, None

    def _preload_data_parallel(self) -> None:
        """并行预加载所有数据到内存"""
        print(f"准备预加载 {len(self.files)} 个文件...")
        
        # 初始化数据列表
        self.seismic_data_list = [None] * len(self.files)
        self.velocity_data_list = [None] * len(self.files)
        
        # 使用线程池而不是进程池，避免pickle问题
        with ThreadPoolExecutor(max_workers=self.preload_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(self._load_sample_worker, file_name) 
                      for file_name in self.files]
            
            # 收集结果
            completed = 0
            for future in futures:
                try:
                    file_idx, seismic_data, velocity_data = future.result()
                    if seismic_data is not None:
                        self.seismic_data_list[file_idx] = seismic_data
                        self.velocity_data_list[file_idx] = velocity_data
                    completed += 1
                    
                    # 进度报告
                    if completed % 10 == 0 or completed == len(self.files):
                        print(f"预加载进度: {completed}/{len(self.files)}")
                        
                except Exception as e:
                    print(f"预加载任务失败: {e}")
                    
        print("数据预加载完成!")

    def _load_sample(self, seismic_file: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """加载单个文件的数据（用于非预加载模式）"""
        # 加载地震数据
        seismic_path = self.seismic_folder / seismic_file
        seismic_data = np.load(seismic_path)[:, :, ::self.sample_ratio, :].astype(np.float32)
        
        # 加载速度场数据
        velocity_file = self._get_velocity_filename(seismic_file)
        velocity_path = self.velocity_folder / velocity_file
        
        if velocity_path.exists():
            velocity_data = np.load(velocity_path).astype(np.float32)
        else:
            velocity_data = None
            
        return seismic_data, velocity_data

    def __len__(self) -> int:
        return len(self.files) * self.file_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_idx, sample_idx = divmod(idx, self.file_size)
        
        if self.preload:
            # 从预加载的数据中获取
            seismic = self.seismic_data_list[batch_idx][sample_idx]
            velocity = self.velocity_data_list[batch_idx][sample_idx] if \
                       self.velocity_data_list[batch_idx] is not None else None
        else:
            # 动态加载（使用缓存）
            file_name = self.files[batch_idx]
            seismic_data, velocity_data = self._load_sample_cached(file_name)
            seismic = seismic_data[sample_idx]
            velocity = velocity_data[sample_idx] if velocity_data is not None else None

        # 应用变换
        if self.seismic_transform:
            seismic = self.seismic_transform(seismic)
        
        if self.velocity_transform and velocity is not None:
            velocity = self.velocity_transform(velocity)
        
        # 处理缺失的velocity数据
        if velocity is None:
            velocity = torch.empty(0)
        
        return seismic, velocity


class DatasetConfig:
    """数据集配置管理器"""
    
    def __init__(self, config_path: str = 'model/data/dataset_config.json'):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载数据集配置"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"数据集配置文件不存在: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """获取指定数据集的配置信息"""
        if dataset_name not in self.config:
            raise KeyError(f"不支持的数据集: {dataset_name}")
        return self.config[dataset_name]

    def get_transforms(self, dataset_name: str, k: float = 1.0) -> Tuple[T.Compose, T.Compose]:
        """获取数据集对应的变换函数"""
        ctx = self.get_dataset_info(dataset_name)
        
        # 地震数据变换
        seismic_transform = T.Compose([
            T.LogTransform(k=k),
            T.MinMaxNormalize(
                T.log_transform(ctx['data_min'], k=k),
                T.log_transform(ctx['data_max'], k=k)
            )
        ])
        
        # 速度场变换
        velocity_transform = T.Compose([
            T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
        ])
        
        return seismic_transform, velocity_transform


def create_dataloaders(
    seismic_folder: str,
    velocity_folder: str,
    dataset_name: str,
    num_data: int,
    paired_num: int,
    batch_size: int = 64,
    num_workers: int = 4,
    train_split: float = 0.8,
    k: float = 1.0,
    preload: bool = True,
    preload_workers: int = 8,
    cache_size: int = 32,
    use_memmap: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """创建训练和测试数据加载器（优化版本）
    
    Args:
        seismic_folder: 地震数据文件夹
        velocity_folder: 速度场数据文件夹  
        dataset_name: 数据集名称
        num_data: 总数据量
        paired_num: 配对数据量
        batch_size: 批次大小
        num_workers: 数据加载线程数
        train_split: 训练集比例
        k: log变换参数
        preload: 是否预加载数据
        preload_workers: 预加载使用的线程数
        cache_size: LRU缓存大小（当不预加载时使用）
        use_memmap: 是否使用内存映射
        prefetch_factor: 预取因子
        persistent_workers: 是否使用持久worker
        
    Returns:
        (train_loader, test_loader, paired_loader, dataset_context)
    """
    # 获取数据集配置
    config = DatasetConfig()
    ctx = config.get_dataset_info(dataset_name)
    seismic_transform, velocity_transform = config.get_transforms(dataset_name, k)
    
    # 判断是否为断层族数据集
    fault_family = dataset_name in ['flatfault-a', 'curvefault-a', 'flatfault-b', 'curvefault-b']
    
    # 创建完整数据集
    full_dataset = SeismicVelocityDataset(
        seismic_folder=seismic_folder,
        velocity_folder=velocity_folder,
        seismic_transform=seismic_transform,
        velocity_transform=velocity_transform,
        fault_family=fault_family,
        preload=preload,
        preload_workers=preload_workers,
        cache_size=cache_size,
        use_memmap=use_memmap
    )
    
    # 数据集分割
    total_size = min(num_data, len(full_dataset))
    train_size = int(total_size * train_split)
    test_size = total_size - train_size
    
    # 创建子数据集
    from torch.utils.data import random_split, Subset
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, total_size))
    paired_indices = list(range(min(paired_num, total_size)))
    
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    paired_dataset = Subset(full_dataset, paired_indices)
    
    # DataLoader通用配置
    dataloader_kwargs = {
        'pin_memory': True,
        'drop_last': False,
        'persistent_workers': persistent_workers and num_workers > 0,
        'prefetch_factor': prefetch_factor if num_workers > 0 else 2,
    }
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        **dataloader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **dataloader_kwargs
    )
    
    paired_loader = DataLoader(
        paired_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        **dataloader_kwargs
    )
    
    print(f"数据集 {dataset_name} 加载完成:")
    print(f"  训练数据: {len(train_dataset)}")
    print(f"  测试数据: {len(test_dataset)}")
    print(f"  配对数据: {len(paired_dataset)}")
    print(f"  优化配置: 预加载={preload}, 预加载线程={preload_workers}, 内存映射={use_memmap}")
    
    return train_loader, test_loader, paired_loader, ctx


def create_diffusion_dataloader(
    seismic_folder: str,
    velocity_folder: str,
    dataset_name: str,
    num_data: int,
    batch_size: int = 16,
    num_workers: int = 4,
    k: float = 1.0,
    preload: bool = True,
    preload_workers: int = 8,
    cache_size: int = 32,
    use_memmap: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = True
) -> Tuple[DataLoader, Dict[str, Any]]:
    """创建扩散模型训练的数据加载器（优化版本）
    
    只需要速度场数据进行扩散训练
    """
    config = DatasetConfig()
    ctx = config.get_dataset_info(dataset_name)
    _, velocity_transform = config.get_transforms(dataset_name, k)
    
    fault_family = dataset_name in ['flatfault-a', 'curvefault-a', 'flatfault-b', 'curvefault-b']
    
    dataset = SeismicVelocityDataset(
        seismic_folder=seismic_folder,
        velocity_folder=velocity_folder,
        seismic_transform=None,  # 扩散训练不需要地震数据
        velocity_transform=velocity_transform,
        fault_family=fault_family,
        preload=preload,
        preload_workers=preload_workers,
        cache_size=cache_size,
        use_memmap=use_memmap
    )
    
    # 限制数据量
    from torch.utils.data import Subset
    indices = list(range(min(num_data, len(dataset))))
    subset = Subset(dataset, indices)
    
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2
    )
    
    print(f"扩散训练数据集加载完成: {len(subset)} 个样本")
    print(f"优化配置: 预加载={preload}, 预加载线程={preload_workers}, 内存映射={use_memmap}")
    
    return dataloader, ctx 