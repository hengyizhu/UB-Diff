import math
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from einops import reduce
from einops.layers.torch import Rearrange
from typing import Optional, Tuple, List, Any

# 替换命名元组为简单类
class ModelPrediction:
    def __init__(self, pred_noise, pred_x_start):
        self.pred_noise = pred_noise
        self.pred_x_start = pred_x_start

# 辅助函数
def exists(x: Any) -> bool:
    return x is not None

# 使用普通函数替代lambda和callable检查
def default_value(val: Any, d: Any) -> Any:
    if exists(val):
        return val
    # 不再检查是否callable，直接使用固定值
    return d

def identity(t: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    return t

# 规范化函数
def normalize_to_neg_one_to_one(img: Tensor) -> Tensor:
    return img * 2 - 1

def unnormalize_to_zero_to_one(t: Tensor) -> Tensor:
    return (t + 1) * 0.5

# 小型辅助模块
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x

def Upsample(dim, dim_out = None):
    if dim_out is None:
        dim_out = dim
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, dim_out, 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    if dim_out is None:
        dim_out = dim
    return nn.Conv1d(dim, dim_out, 4, 2, 1)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# 正弦位置嵌入
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)
        self.rearrange_x_to_b1 = Rearrange('b -> b 1')
        self.rearrange_weights_to_1d = Rearrange('d -> 1 d')

    def forward(self, x):
        x = self.rearrange_x_to_b1(x)
        freqs = x * self.rearrange_weights_to_1d(self.weights) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# 构建块模块
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale: Tensor = scale_shift[0]
            shift: Tensor = scale_shift[1]
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.has_time_emb = exists(time_emb_dim)
        
        if self.has_time_emb:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 2)
            )

        self.rearrange_time_emb = Rearrange('b c -> b c 1')

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: Tensor, time_emb: Optional[Tensor] = None) -> Tensor:
        scale_shift_tuple: Optional[Tuple[Tensor, Tensor]] = None
        
        if self.has_time_emb and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = self.rearrange_time_emb(time_emb)
            chunk_output: List[Tensor] = torch.chunk(time_emb, 2, dim=1)
            scale_shift_tuple = (chunk_output[0], chunk_output[1])

        h = self.block1(x, scale_shift = scale_shift_tuple)
        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

        self.rearrange_to_heads_and_dim = Rearrange('b (h d) n -> b h d n', h=self.heads, d=self.dim_head)
        self.rearrange_from_heads_and_dim = Rearrange('b h d n -> b (h d) n', h=self.heads, d=self.dim_head)

    def forward(self, x):
        b, c, n = x.shape
        qkv_list: List[Tensor] = self.to_qkv(x).chunk(3, dim = 1)
        qkv_tuple: Tuple[Tensor, Tensor, Tensor] = (qkv_list[0], qkv_list[1], qkv_list[2])
        q = self.rearrange_to_heads_and_dim(qkv_tuple[0])
        k = self.rearrange_to_heads_and_dim(qkv_tuple[1])
        v = self.rearrange_to_heads_and_dim(qkv_tuple[2])

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = self.rearrange_from_heads_and_dim(out)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

        self.rearrange_to_heads_and_dim = Rearrange('b (h d) n -> b h d n', h=self.heads, d=self.dim_head)
        self.rearrange_from_heads_and_dim = Rearrange('b h n d -> b (h d) n', h=self.heads, d=self.dim_head)

    def forward(self, x):
        b, c, n = x.shape
        qkv_list: List[Tensor] = self.to_qkv(x).chunk(3, dim = 1)
        qkv_tuple: Tuple[Tensor, Tensor, Tensor] = (qkv_list[0], qkv_list[1], qkv_list[2])
        q = self.rearrange_to_heads_and_dim(qkv_tuple[0])
        k = self.rearrange_to_heads_and_dim(qkv_tuple[1])
        v = self.rearrange_to_heads_and_dim(qkv_tuple[2])

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = self.rearrange_from_heads_and_dim(out)
        return self.to_out(out)

# Unet1D模型
class Unet1D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # 确定维度
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        if init_dim is None:
            init_dim = dim
            
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim]
        for m in dim_mults:
            dims.append(dim * m)
            
        in_out = list(zip(dims[:-1], dims[1:]))

        # 时间嵌入
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
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

        # 层
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, groups = resnet_block_groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, groups = resnet_block_groups),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, groups = resnet_block_groups)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, groups = resnet_block_groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim, groups = resnet_block_groups),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim, groups = resnet_block_groups),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = out_dim if out_dim is not None else default_out_dim

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim = time_dim, groups = resnet_block_groups)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            # 处理self-conditioning输入
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for down_module_group in self.downs:
            block1 = down_module_group[0]
            block2 = down_module_group[1]
            attn = down_module_group[2]
            downsample = down_module_group[3]

            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for up_module_group in self.ups:
            block1 = up_module_group[0]
            block2 = up_module_group[1]
            attn = up_module_group[2]
            upsample = up_module_group[3]

            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# 辅助函数
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    余弦调度，如https://openreview.net/forum?id=-NEXDKk8gZ中所提出
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# 高斯扩散模型
class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        betas,
        sampling_timesteps = None,
        objective = 'pred_noise',
        ddim_sampling_eta = 0.,
        auto_normalize = False,
        time_scale = 1,
        use_wandb = False
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.time_scale = time_scale
        self.use_wandb = use_wandb
        self.seq_length = seq_length
        self.objective = objective

        # 检查目标
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        betas = betas.type(torch.float64)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # 采样相关参数
        self.sampling_timesteps = sampling_timesteps if sampling_timesteps is not None else timesteps
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # 将浮点64缓冲区注册为浮点32的辅助函数
        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 对q(x_t | x_{t-1})和其他的计算
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # 对后验q(x_{t-1} | x_t, x_0)的计算
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # 下面：log计算被剪裁，因为在扩散链开始时，后验方差为0
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # 计算损失权重
        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # 是否自动标准化
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    # 预测噪声和起始点之间的转换函数
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / 
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    # 后验分布
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 模型预测函数
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t * self.time_scale, x_self_cond)
        
        # 根据目标确定如何处理模型输出
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            
            if clip_x_start:
                x_start = torch.clamp(x_start, -1., 1.)
                
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            
            if clip_x_start:
                x_start = torch.clamp(x_start, -1., 1.)
                
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            
            if clip_x_start:
                x_start = torch.clamp(x_start, -1., 1.)
                
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    # 均值和方差
    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start = torch.clamp(x_start, -1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # 单步采样
    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        
        # t > 0时添加噪声，否则不添加
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    # 采样循环
    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        x_start = None

        # 逐步去噪
        for t in reversed(range(0, self.num_timesteps)):
            # 使用self-conditioning如果启用
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        img = self.unnormalize(img)
        return img

    # DDIM采样
    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True):
        batch, device = shape[0], self.betas.device
        total_timesteps, sampling_timesteps = self.num_timesteps, self.sampling_timesteps
        
        # 计算时间步
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            
            # 获取预测
            pred = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)
            pred_noise, x_start = pred.pred_noise, pred.pred_x_start

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = self.unnormalize(img)
        return img

    # 主采样函数
    @torch.no_grad()
    def sample(self, batch_size = 16):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, seq_length))

    # 添加噪声
    @torch.no_grad()
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # 损失计算
    def p_losses(self, x_start, t, noise = None):
        b, c, n = x_start.shape
        if noise is None:
            noise = torch.randn_like(x_start)

        # 噪声采样
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # 处理self-conditioning
        x_self_cond = None
        if self.self_condition and torch.rand(1)[0] < 0.5:  # 使用固定随机性
            with torch.no_grad():
                pred = self.model_predictions(x, t)
                x_self_cond = pred.pred_x_start
                x_self_cond = x_self_cond.detach()

        # 预测并计算梯度
        model_out = self.model(x, t*self.time_scale, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        
        return loss.mean()

    # 前向传播
    def forward(self, img, *args, **kwargs):
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

# 主函数，用于简单测试
if __name__ == '__main__':
    # 创建模型实例
    model = Unet1D(64, channels=1)
    # 创建测试输入
    test = torch.randn((2, 1, 256))
    time = torch.randint(0, 200, (2,)).long()
    # 运行模型
    out = model(test, time)
    print(f"输出形状: {out.shape}")
    
    # 测试TorchScript兼容性
    print("\n测试TorchScript兼容性...")
    try:
        # 跟踪
        traced_model = torch.jit.trace(model, (test, time))
        print("✓ 模型跟踪成功!")
        # 脚本化
        scripted_model = torch.jit.script(model)
        print("✓ 模型脚本化成功!")
    except Exception as e:
        print(f"✗ TorchScript转换失败: {e}") 