"""
重构后的数据变换模块

提供类型安全的数据预处理和增强功能
"""

import torch
import numpy as np
import random
from typing import Union, Tuple, Optional, Callable, Any
from sklearn.decomposition import PCA


# 类型别名
TensorLike = Union[torch.Tensor, np.ndarray]


def crop(vid: TensorLike, i: int, j: int, h: int, w: int) -> TensorLike:
    """裁剪操作"""
    return vid[..., i:(i + h), j:(j + w)]


def center_crop(vid: TensorLike, output_size: Tuple[int, int]) -> TensorLike:
    """中心裁剪"""
    h, w = vid.shape[-2:]
    th, tw = output_size

    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(vid, i, j, th, tw)


def hflip(vid: TensorLike) -> TensorLike:
    """水平翻转"""
    return vid.flip(dims=(-1,)) if isinstance(vid, torch.Tensor) else np.fliplr(vid)


def resize(vid: torch.Tensor, size: Union[int, Tuple[int, int]], 
           interpolation: str = 'bilinear') -> torch.Tensor:
    """调整大小"""
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)


def random_resize(vid: torch.Tensor, size: Union[int, Tuple[int, int]], 
                  random_factor: float, interpolation: str = 'bilinear') -> torch.Tensor:
    """随机调整大小"""
    scale = None
    r = 1 + random.random() * (random_factor - 1)
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:]) * r
        size = None
    else:
        size = tuple([int(elem * r) for elem in list(size)])
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)


def pad(vid: torch.Tensor, padding: Tuple[int, ...], fill: float = 0, 
        padding_mode: str = "constant") -> torch.Tensor:
    """填充操作"""
    return torch.nn.functional.pad(vid, padding, value=fill, mode=padding_mode)


def to_normalized_float_tensor(vid: np.ndarray) -> torch.Tensor:
    """转换为标准化的浮点张量"""
    return torch.from_numpy(vid).permute(3, 0, 1, 2).to(torch.float32) / 255


def normalize(vid: torch.Tensor, mean: Union[float, Tuple[float, ...]], 
              std: Union[float, Tuple[float, ...]]) -> torch.Tensor:
    """标准化"""
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std


def minmax_normalize(vid: TensorLike, vmin: float, vmax: float, scale: int = 2) -> TensorLike:
    """最小-最大标准化"""
    vid = vid - vmin
    vid = vid / (vmax - vmin)
    return (vid - 0.5) * 2 if scale == 2 else vid


def minmax_denormalize(vid: TensorLike, vmin: float, vmax: float, scale: int = 2) -> TensorLike:
    """最小-最大反标准化"""
    if scale == 2:
        vid = vid / 2 + 0.5
    return vid * (vmax - vmin) + vmin


def add_noise(data: np.ndarray, snr: float) -> np.ndarray:
    """添加噪声"""
    sig_avg_power_db = 10 * np.log10(np.mean(data ** 2))
    noise_avg_power_db = sig_avg_power_db - snr
    noise_avg_power = 10 ** (noise_avg_power_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_avg_power), data.shape)
    return data + noise


def log_transform(data: np.ndarray, k: float = 1, c: float = 0) -> np.ndarray:
    """对数变换"""
    return (np.log1p(np.abs(k * data) + c)) * np.sign(data)


def log_transform_tensor(data: torch.Tensor, k: float = 1, c: float = 0) -> torch.Tensor:
    """张量对数变换"""
    return (torch.log1p(torch.abs(k * data) + c)) * torch.sign(data)


def exp_transform(data: np.ndarray, k: float = 1, c: float = 0) -> np.ndarray:
    """指数变换（对数变换的逆）"""
    return (np.expm1(np.abs(data)) - c) * np.sign(data) / k


def tonumpy_denormalize(vid: torch.Tensor, vmin: float, vmax: float, 
                       exp: bool = True, k: float = 1, c: float = 0, 
                       scale: int = 2) -> np.ndarray:
    """转换为numpy并反标准化"""
    if exp:
        vmin = log_transform(vmin, k=k, c=c)
        vmax = log_transform(vmax, k=k, c=c)
    vid = minmax_denormalize(vid.cpu().numpy(), vmin, vmax, scale)
    return exp_transform(vid, k=k, c=c) if exp else vid


# 类接口
class RandomCrop:
    """随机裁剪变换"""
    
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    @staticmethod
    def get_params(vid: TensorLike, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """获取随机裁剪参数"""
        h, w = vid.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, vid: TensorLike) -> TensorLike:
        i, j, h, w = self.get_params(vid, self.size)
        return crop(vid, i, j, h, w)


class CenterCrop:
    """中心裁剪变换"""
    
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, vid: TensorLike) -> TensorLike:
        return center_crop(vid, self.size)


class Resize:
    """调整大小变换"""
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        return resize(vid, self.size)


class RandomResize:
    """随机调整大小变换"""
    
    def __init__(self, size: Union[int, Tuple[int, int]], random_factor: float = 1.25):
        self.size = size
        self.factor = random_factor

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        return random_resize(vid, self.size, self.factor)


class ToFloatTensorInZeroOne:
    """转换为[0,1]范围的浮点张量"""
    
    def __call__(self, vid: np.ndarray) -> torch.Tensor:
        return to_normalized_float_tensor(vid)


class Normalize:
    """标准化变换"""
    
    def __init__(self, mean: Union[float, Tuple[float, ...]], 
                 std: Union[float, Tuple[float, ...]]):
        self.mean = mean
        self.std = std

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        return normalize(vid, self.mean, self.std)


class MinMaxNormalize:
    """最小-最大标准化变换"""
    
    def __init__(self, datamin: float, datamax: float, scale: int = 2):
        self.datamin = datamin
        self.datamax = datamax
        self.scale = scale

    def __call__(self, vid: TensorLike) -> TensorLike:
        return minmax_normalize(vid, self.datamin, self.datamax, self.scale)


class RandomHorizontalFlip:
    """随机水平翻转"""
    
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, vid: TensorLike) -> TensorLike:
        if random.random() < self.p:
            return hflip(vid)
        return vid


class Pad:
    """填充变换"""
    
    def __init__(self, padding: Tuple[int, ...], fill: float = 0):
        self.padding = padding
        self.fill = fill

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        return pad(vid, self.padding, self.fill)


class TemporalDownsample:
    """时间下采样"""
    
    def __init__(self, rate: int = 1):
        self.rate = rate

    def __call__(self, vid: TensorLike) -> TensorLike:
        return vid[::self.rate]


class AddNoise:
    """添加噪声变换"""
    
    def __init__(self, snr: float = 10):
        self.snr = snr

    def __call__(self, vid: np.ndarray) -> np.ndarray:
        return add_noise(vid, self.snr)


class PCD:
    """主成分分析降维"""
    
    def __init__(self, n_comp: int = 8):
        self.n_comp = n_comp

    def __call__(self, data: np.ndarray) -> np.ndarray:
        b, c, h, w = data.shape
        data = data.reshape(b, c * h * w)
        
        pca = PCA(n_components=self.n_comp)
        transformed = pca.fit_transform(data)
        
        # 恢复形状
        inverse_transformed = pca.inverse_transform(transformed)
        return inverse_transformed.reshape(b, c, h, w)


class StackPCD:
    """堆叠PCA变换"""
    
    def __init__(self, n_comp: Tuple[int, int] = (32, 8)):
        self.n_comp = n_comp

    def __call__(self, data: np.ndarray) -> np.ndarray:
        b, c, h, w = data.shape
        
        # 第一次PCA
        data_flat = data.reshape(b, c * h * w)
        pca1 = PCA(n_components=self.n_comp[0])
        transformed1 = pca1.fit_transform(data_flat)
        
        # 第二次PCA
        pca2 = PCA(n_components=self.n_comp[1])
        transformed2 = pca2.fit_transform(transformed1)
        
        # 恢复
        inverse1 = pca2.inverse_transform(transformed2)
        inverse2 = pca1.inverse_transform(inverse1)
        
        return inverse2.reshape(b, c, h, w)


class LogTransform:
    """对数变换"""
    
    def __init__(self, k: float = 1, c: float = 0):
        self.k = k
        self.c = c

    def __call__(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(data, torch.Tensor):
            return log_transform_tensor(data, self.k, self.c)
        else:
            return log_transform(data, self.k, self.c)


class ToTensor:
    """转换为PyTorch张量"""
    
    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype

    def __call__(self, sample: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(sample).to(self.dtype)


class Compose:
    """组合多个变换"""
    
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


# 实用函数
def create_standard_transforms(data_min: float, data_max: float, 
                             k: float = 1.0, add_noise: bool = False, 
                             snr: float = 10.0) -> Compose:
    """创建标准的数据变换管道"""
    transforms_list = [
        ToTensor(),
        LogTransform(k=k),
        MinMaxNormalize(log_transform(data_min, k=k), log_transform(data_max, k=k))
    ]
    
    if add_noise:
        transforms_list.insert(-1, AddNoise(snr=snr))
    
    return Compose(transforms_list)


def create_velocity_transforms(label_min: float, label_max: float) -> Compose:
    """创建速度场变换管道"""
    return Compose([
        ToTensor(),
        MinMaxNormalize(label_min, label_max)
    ]) 