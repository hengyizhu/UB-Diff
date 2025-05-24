"""
注意力机制模块

包含各种注意力机制的实现，用于扩散模型和解码器
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .networks import RMSNorm


class LinearAttention(nn.Module):
    """线性注意力机制，计算复杂度为O(n)"""
    
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads
        
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, self.dim_head, n), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = out.contiguous().view(b, -1, n)
        return self.to_out(out)


class Attention(nn.Module):
    """标准自注意力机制"""
    
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads
        
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, self.dim_head, n), qkv)

        dots = torch.matmul(q.transpose(-1, -2), k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(v, attn.transpose(-1, -2))
        
        out = out.contiguous().view(b, -1, n)
        return self.to_out(out)


class PatchEmbed(nn.Module):
    """图像块嵌入层"""
    
    def __init__(self, img_size: int, patch_size: int, in_chans: int = 3, 
                 embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches_h = img_size[0] // patch_size
        num_patches_w = img_size[1] // patch_size
        self.n_patches = num_patches_h * num_patches_w
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, dim: int, n_heads: int = 8, qkv_bias: bool = True, 
                 attn_p: float = 0., proj_p: float = 0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_samples, n_tokens, dim = x.shape
        
        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, n_samples, n_heads, n_patches + 1, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        dp = (q @ k.transpose(-2, -1)) * self.scale
        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(1, 2)
        weighted_avg = weighted_avg.flatten(2)

        x = self.proj(weighted_avg)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, in_features: int, hidden_features: int, 
                 out_features: int, p: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, 
                 qkv_bias: bool = True, p: float = 0., attn_p: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadAttention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, 
            attn_p=attn_p, proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=hidden_features, 
            out_features=dim, 
            p=p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer模型"""
    
    def __init__(self, img_size: int = 384, patch_size: int = 16, 
                 in_chans: int = 3, n_classes: int = 1000, embed_dim: int = 768, 
                 depth: int = 12, n_heads: int = 12, mlp_ratio: float = 4., 
                 qkv_bias: bool = True, p: float = 0., attn_p: float = 0.):
        super().__init__()
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, p=p, attn_p=attn_p
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor, return_features: bool = True) -> torch.Tensor:
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        
        if return_features:
            return x
        
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x


def pair(t):
    """将单个值转换为对"""
    return t if isinstance(t, tuple) else (t, t) 