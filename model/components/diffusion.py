"""
扩散模型组件

包含1D U-Net和高斯扩散过程的实现
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any
from .networks import ResnetBlock, Upsample1D, Downsample1D
from .attention import LinearAttention, Attention


class ModelPrediction:
    """模型预测结果的包装类"""
    def __init__(self, pred_noise: torch.Tensor, pred_x_start: torch.Tensor):
        self.pred_noise = pred_noise
        self.pred_x_start = pred_x_start


def exists(x: Any) -> bool:
    """检查值是否存在"""
    return x is not None


def default_value(val: Any, d: Any) -> Any:
    """返回默认值"""
    return val if exists(val) else d


def identity(t: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """恒等函数"""
    return t


def normalize_to_neg_one_to_one(img: torch.Tensor) -> torch.Tensor:
    """标准化到[-1, 1]"""
    return img * 2 - 1


def unnormalize_to_zero_to_one(t: torch.Tensor) -> torch.Tensor:
    """反标准化到[0, 1]"""
    return (t + 1) * 0.5


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """随机或可学习的正弦位置编码"""
    
    def __init__(self, dim: int, is_random: bool = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # b -> b 1
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Residual(nn.Module):
    """残差连接"""
    
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class PreNorm(nn.Module):
    """预标准化"""
    
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x)


class Unet1D(nn.Module):
    """1D U-Net架构用于扩散模型"""
    
    def __init__(self, dim: int, init_dim: Optional[int] = None, 
                 out_dim: Optional[int] = None, dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
                 channels: int = 3, self_condition: bool = False,
                 resnet_block_groups: int = 8, learned_variance: bool = False,
                 learned_sinusoidal_cond: bool = False, 
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16):
        super().__init__()

        # 确定维度
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default_value(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = lambda dim_in, dim_out: ResnetBlock(
            dim_in, dim_out, groups=resnet_block_groups
        )

        # 时间嵌入
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 下采样层
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample1D(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        # 上采样层
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out),
                block_klass(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample1D(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default_value(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x: torch.Tensor, time: torch.Tensor, 
                x_self_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.self_condition:
            x_self_cond = default_value(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        # 下采样
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        # 中间层
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # 上采样
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """从张量a中按索引t提取值"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """线性beta调度"""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """余弦beta调度"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion1D(nn.Module):
    """1D高斯扩散过程"""
    
    def __init__(self, model: nn.Module, *, seq_length: int, betas: torch.Tensor,
                 sampling_timesteps: Optional[int] = None, objective: str = 'pred_noise',
                 ddim_sampling_eta: float = 0., auto_normalize: bool = False,
                 time_scale: int = 1, use_wandb: bool = False):
        super().__init__()
        
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        
        self.seq_length = seq_length
        self.objective = objective
        self.time_scale = time_scale
        self.use_wandb = use_wandb

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, \
            'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        betas = betas.float()
        self.num_timesteps = int(betas.shape[0])

        # 预计算alpha值
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # 注册缓冲区
        def register_buffer(name: str, val: torch.Tensor):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 用于q(x_t | x_{t-1})的计算
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # 用于后验q(x_{t-1} | x_t, x_0)的计算
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', 
                       torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', 
                       betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', 
                       (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # 采样相关
        self.sampling_timesteps = default_value(sampling_timesteps, timesteps)
        self.ddim_sampling_eta = ddim_sampling_eta

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, 
                                noise: torch.Tensor) -> torch.Tensor:
        """从噪声预测起始值"""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, 
                                x0: torch.Tensor) -> torch.Tensor:
        """从起始值预测噪声"""
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / 
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start: torch.Tensor, t: torch.Tensor, 
                  noise: torch.Tensor) -> torch.Tensor:
        """预测v参数化"""
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t: torch.Tensor, t: torch.Tensor, 
                            v: torch.Tensor) -> torch.Tensor:
        """从v预测起始值"""
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, 
                   t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算后验均值和方差"""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        return posterior_mean, posterior_variance

    def model_predictions(self, x: torch.Tensor, t: torch.Tensor, 
                         x_self_cond: Optional[torch.Tensor] = None,
                         clip_x_start: bool = False, 
                         rederive_pred_noise: bool = False) -> ModelPrediction:
        """模型预测"""
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, 
                 x_self_cond: Optional[torch.Tensor] = None,
                 clip_denoised: bool = True) -> torch.Tensor:
        """单步采样"""
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_predictions = self.model_predictions(x, batched_times, x_self_cond, clip_x_start=clip_denoised)
        pred_noise, x_start = model_predictions.pred_noise, model_predictions.pred_x_start

        if t == 0:
            return x_start

        posterior_mean, posterior_variance = self.q_posterior(x_start=x_start, x_t=x, t=batched_times)
        noise = torch.randn_like(x)
        return posterior_mean + (0.5 * posterior_variance.log()).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """完整采样循环"""
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            img = self.p_sample(img, t, self_cond)

        return img

    @torch.no_grad()
    def sample(self, batch_size: int = 16) -> torch.Tensor:
        """采样接口"""
        seq_length, channels = self.seq_length, self.channels
        return self.p_sample_loop((batch_size, channels, seq_length))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向加噪过程"""
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算扩散损失"""
        b, c, n = x_start.shape
        if noise is None:
            noise = torch.randn_like(x_start)

        # 前向过程
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 预测
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        model_out = self.model(x, t)

        loss = F.mse_loss(model_out, target, reduction='none')
        loss = loss.mean()

        return loss

    def forward(self, img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """前向传播"""
        b, c, n, device = *img.shape, img.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(img, t, *args, **kwargs)


# 添加partial函数的实现
def partial(func, **preset_kwargs):
    """简单的partial函数实现"""
    def wrapped(*args, **kwargs):
        return func(*args, **preset_kwargs, **kwargs)
    return wrapped


class GaussianDiffusion1DDefault(GaussianDiffusion1D):
    """默认配置的1D高斯扩散"""
    
    def __init__(self, model: nn.Module, seq_length: int, objective: str, 
                 betas: torch.Tensor, time_scale: int = 1, gamma: float = 0, 
                 use_wandb: bool = False):
        super().__init__(
            model, 
            seq_length=seq_length, 
            betas=betas,
            objective=objective, 
            time_scale=time_scale, 
            use_wandb=use_wandb
        ) 

# 在文件末尾添加导出
__all__ = [
    'Unet1D',
    'GaussianDiffusion1D', 
    'GaussianDiffusion1DDefault',
    'cosine_beta_schedule',
    'linear_beta_schedule',
    'ModelPrediction'
] 