import torch
import torch.nn as nn
import torch.nn.functional as F


NORM_LAYERS = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln':nn.LayerNorm}

class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel = 3, stride = 1, padding = 1, norm = 'bn', dropout = None):
        super(ConvBlock, self).__init__()

        layers = [nn.Conv2d(in_chan, out_chan, kernel, stride, padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_chan))
        layers.append(nn.ReLU())

        if dropout:
            layers.append(nn.Dropout2d(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel = 2, stride = 2, padding = 0, output_padding = 0, norm = 'bn', dropout = None):
        super(DeconvBlock, self).__init__()

        layers = [nn.ConvTranspose2d(in_chan, out_chan, kernel_size=kernel,stride=stride,padding=padding,output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_chan))

        layers.append(nn.ReLU())

        if dropout:
            layers.append(nn.Dropout2d(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_chan, out_chan, kernel = 3, stride = 1, padding = 1, norm = 'bn'):
        super(ConvBlock_Tanh, self).__init__()

        layers = [nn.Conv2d(in_chan, out_chan, kernel, stride, padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_chan))
        layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)
    

class Encoder_v(nn.Module):
    def __init__(self, in_channels, dim1 = 32, dim2 = 64, dim3 = 128, dim4 = 256, dim5 = 512, checkpoint = None):
        # self.in_channels = in_channels
        super(Encoder_v, self).__init__()
        self.dim5 = dim5
        self.module_list1 = []
        self.module_list2 = []
        self.convBlock1_1 = ConvBlock(in_channels, dim1, kernel=1, stride=1, padding=0) # 70 * 70
        self.convBlock1_2 = ConvBlock(dim1, dim1, kernel=3, stride=1, padding=1)

        self.convBlock2_1 = ConvBlock(dim1, dim2, kernel=3, stride=2, padding=1) # 35 * 35
        self.convBlock2_2 = ConvBlock(dim2, dim2, kernel=1, stride=1, padding=0) # 35 * 35

        self.convBlock3_1 = ConvBlock(dim2, dim2, kernel=3, stride=1, padding=1) # 35* 35
        self.convBlock3_2 = ConvBlock(dim2, dim2, kernel=1, stride=1, padding=0)  # 35* 35

        self.convBlock4_1 = ConvBlock(dim2, dim3, kernel=3, stride=2, padding=1) # 18 * 18
        self.convBlock4_2 = ConvBlock(dim3, dim3, kernel=1, stride=1, padding=0) # 18 * 18

        self.convBlock5_1 = ConvBlock(dim3, dim3, kernel=3, stride=1, padding=0) # 16 * 16
        self.convBlock5_2 = ConvBlock(dim3, dim3, kernel=1, stride=1, padding=0) # 16 * 16

        self.module_list1.append(self.convBlock1_1)
        self.module_list1.append(self.convBlock1_2)
        self.module_list1.append(self.convBlock2_1)
        self.module_list1.append(self.convBlock2_2)
        self.module_list1.append(self.convBlock3_1)
        self.module_list1.append(self.convBlock3_2)
        self.module_list1.append(self.convBlock4_1)
        self.module_list1.append(self.convBlock4_2)
        self.module_list1.append(self.convBlock5_1)
        self.module_list1.append(self.convBlock5_2)

        self.convBlock6_1 = ConvBlock(dim3, dim4, kernel=3, stride=2, padding=1) # 8*8
        self.convBlock6_2 = ConvBlock(dim4, dim4, kernel=1, stride=1, padding=0) #8*8

        self.convBlock7_1 = ConvBlock(dim4, dim4, kernel=3, stride=2, padding=1) #4*4
        self.convBlock7_2 = ConvBlock(dim4, dim4, kernel=3, stride=2, padding=1) #2*2

        self.convBlock8 = ConvBlock_Tanh(dim4, dim5, kernel=3, stride=2, padding=1) # 1*1

        self.module_list2.append(self.convBlock6_1)
        self.module_list2.append(self.convBlock6_2)
        self.module_list2.append(self.convBlock7_1)
        self.module_list2.append(self.convBlock7_2)
        self.module_list2.append(self.convBlock8)

        self.module_list1 = nn.ModuleList(self.module_list1)
        self.module_list2 = nn.ModuleList(self.module_list2)

        if checkpoint:
            self.checkpoint = checkpoint



    def forward(self, x):
        for module in self.module_list1:
            x = module(x)
        for module in self.module_list2:
            x = module(x)
        return x

    def forward_1(self,x):
        for module in self.module_list1:
            x = module(x)
        return x

    def forward_2(self,x):
        x = x.view(x.shape[0], -1, 16, 16)
        for module in self.module_list2:
            x = module(x)
        return x
        

    def load_model(self):
        cpt = torch.load(self.checkpoint)

        loaded_module_list1_state_dict = cpt['module_list1']
        loaded_module_list2_state_dict = cpt['module_list2']

        self.module_list1.load_state_dict(loaded_module_list1_state_dict)
        self.module_list2.load_state_dict(loaded_module_list2_state_dict)

class Decoder_V(nn.Module):
    def __init__(self, out_channels, dim1 = 32, dim2 = 64, dim3 = 128, dim4 = 256, dim5 = 512):
        super(Decoder_V, self).__init__()

        self.deconvBlock1_1 = DeconvBlock(dim5, dim5, kernel=5)  # (None, 512, 5, 5)
        self.deconvBlock1_2 = ConvBlock(dim5, dim5, kernel=3, stride=1) #(None, 512, 5, 5)

        self.deconvBlock2_1 = DeconvBlock(dim5, dim4, kernel=4, stride=2, padding=1) #(None, 256, 10, 10)
        self.deconvBlock2_2 = ConvBlock(dim4, dim4, kernel=3, stride=1) #(None, 256, 10, 10)

        self.deconvBlock3_1 = DeconvBlock(dim4, dim3, kernel=4, stride=2, padding=1) #(None, 128, 20, 20)
        self.deconvBlock3_2 = ConvBlock(dim3, dim3, kernel=3, stride=1) #(None, 128, 20, 20)

        self.deconvBlock4_1 = DeconvBlock(dim3, dim2, kernel=4, stride=2, padding=1) #(None, 64, 40, 40)
        self.deconvBlock4_2 = ConvBlock(dim2, dim2, kernel=3, stride=1) #(None, 64, 40, 40)

        self.deconvBlock5_1 = DeconvBlock(dim2, dim1, kernel=4, stride=2, padding=1) #(None, 32, 80, 80)
        self.deconvBlock5_2 = ConvBlock(dim1, dim1, kernel=3, stride=1) #(None, 32, 80, 80)

        self.deconvBlock6 = ConvBlock_Tanh(dim1, out_channels, kernel=3, stride=1, padding=1) #(None, 32, 80, 80)

        self.blocks = nn.ModuleList()
        self.blocks.append(self.deconvBlock1_1)
        self.blocks.append(self.deconvBlock1_2)
        self.blocks.append(self.deconvBlock2_1)
        self.blocks.append(self.deconvBlock2_2)
        self.blocks.append(self.deconvBlock3_1)
        self.blocks.append(self.deconvBlock3_2)
        self.blocks.append(self.deconvBlock4_1)
        self.blocks.append(self.deconvBlock4_2)
        self.blocks.append(self.deconvBlock5_1)
        self.blocks.append(self.deconvBlock5_2)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = F.pad(x,[-5,-5,-5,-5], mode='constant', value=0.0)
        x = self.deconvBlock6(x)
        return x

class TDecoder_S(nn.Module):
    def __init__(self, dd, origin_h, origin_w, depth = 2, laten_dim=64, latent_h = 1, latent_w = 1):
        super(TDecoder_S, self).__init__()

        self.dd = dd
        self.dh = latent_h
        self.dw = latent_w

        self.origin_h = origin_h
        self.origin_w = origin_w

        self.laten_dim = laten_dim
        self.depth = depth

        if self.depth:
            self.decoder = VisionTransformer(img_size=(self.dh, self.dw), patch_size=1, in_chans=self.laten_dim * self.dd,
                               depth=self.depth, n_heads=8, mlp_ratio=self.dd, embed_dim=self.laten_dim * self.dd)
        self.mlp = nn.Linear(self.laten_dim * self.dd, self.origin_h * self.origin_w * self.dd)

    def forward(self, x):
        x = self.decoder(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        x = x.view(x.shape[0], self.dd, self.origin_h, self.origin_w)
        
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PatchEmbed(nn.Module):

    def __init__(self, img_size, patch_size, in_chans = 3, embed_dim = 768):
        super(PatchEmbed, self).__init__()

        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        self.n_patches = (image_height // patch_height) * (image_width // patch_width)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size= patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) #(n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2) #(n_samples, embed_dim, n_patches)
        x = x.transpose(1,2) #(_, n_patches, embed_dim)

        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads = 8, qkv_bias = True, attn_p = 0., proj_p = 0.):
        super(Attention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_sample, n_tokens, dim = x.shape

        qkv = self.qkv(x) # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(n_sample, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2,0,3,1,4) # (3, n_samples, n_heads, n_patches+1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1) #(n_samples, n_heads, head_dim, n_patches + 1)
        dp = (q @ k_t) * self.scale #(n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(-1) #(n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v #(n_samples, n_heads, n_patches+1, head_dim)
        weighted_avg = weighted_avg.transpose(1,2) #(n_samples, n_patches+1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) #(n_samples, n_patches + 1, dim)
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p = 0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x) #(n_samples, n_patches + 1, hidden_features)
        x = self.act(x) #(n_samples, n_patches + 1, hidden_features)
        x = self.drop(x) #(n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio = 4.0, qkv_bias = True, p = 0., attn_p = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias= qkv_bias, attn_p= attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size = 384, patch_size = 16, in_chans = 3, n_classes=1000, embed_dim = 768, depth = 12, n_heads = 12,
                mlp_ratio = 4., qkv_bias = True, p=0., attn_p=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim= embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+self.patch_embed.n_patches, embed_dim))

        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p = p, attn_p= attn_p)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)
        
    def forward(self, x, return_features: bool = True):
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1) #(n_samples, 1, embed_dim)
        x = torch.cat((cls_token,x), dim = 1) #(n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if return_features:
            x = x.permute(0, 2, 1)[:, :, 1:] # (n_samples, embed_dim, n_patches)
            return x
        else:
            cls_token = x[:, 0]
            return self.head(cls_token)

# 主函数，用于简单测试
if __name__ == '__main__':
    # 创建编码器实例
    encoder = Encoder_v(in_channels=3, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512)
    # 创建解码器实例
    decoder_v = Decoder_V(out_channels=3, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512)
    # 创建解码器实例
    decoder_s = TDecoder_S(dd=1, origin_h=70, origin_w=70, depth=12, laten_dim=512, latent_h=1, latent_w=1)
    
    # 创建测试输入 (batch_size, channels, height, width)
    test_input = torch.randn((2, 3, 70, 70))
    
    # 运行编码器
    encoded = encoder(test_input)
    print(f"编码器输出形状: {encoded.shape}")
    
    # 运行解码器
    decoded_v = decoder_v(encoded)
    print(f"解码器输出形状: {decoded_v.shape}")

    decoded_s = decoder_s(encoded)
    print(f"解码器输出形状: {decoded_s.shape}")
    
    # 测试TorchScript兼容性
    print("\n测试TorchScript兼容性...")
    try:
        # 尝试跟踪编码器
        traced_encoder = torch.jit.trace(encoder, test_input)
        print("✓ 编码器跟踪成功!")
        
        # 尝试脚本化编码器
        scripted_encoder = torch.jit.script(encoder)
        print("✓ 编码器脚本化成功!")
        
        # 尝试跟踪解码器
        traced_decoder_v = torch.jit.trace(decoder_v, encoded)
        print("✓ 解码器v跟踪成功!")
        
        # 尝试脚本化解码器
        scripted_decoder_v = torch.jit.script(decoder_v)
        print("✓ 解码器v脚本化成功!")

        # 尝试跟踪解码器
        traced_decoder_s = torch.jit.trace(decoder_s, encoded)
        print("✓ 解码器s跟踪成功!")
        
        # 尝试脚本化解码器
        scripted_decoder_s = torch.jit.script(decoder_s)
        print("✓ 解码器s脚本化成功!")
    except Exception as e:
        print(f"✗ TorchScript转换失败: {e}")