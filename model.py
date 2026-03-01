import torch
from torch import nn
from torch.nn import functional as F
import numbers
from einops import rearrange
from mlp import INR
from dtcwt_feat import DTCWTFeatureMS

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                                   groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class Conv1x1_SE(nn.Module):
    """
    先 1x1 做降维/投影，再用 SE 做通道重标定（稳定提分，几乎不改结构）
    """
    def __init__(self, in_ch, out_ch, bias=False, r=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

        mid = max(out_ch // r, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, mid, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, out_ch, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        return x * self.se(x)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, BasicConv=BasicConv):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = BasicConv(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, bias=bias,
                                relu=False, groups=hidden_features * 2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, BasicConv=BasicConv):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = BasicConv(dim * 3, dim * 3, kernel_size=3, stride=1, bias=bias, relu=False, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, BasicConv=BasicConv):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, BasicConv=BasicConv)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, BasicConv=BasicConv)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Fusion(nn.Module):
    def __init__(self, in_dim=32):
        super(Fusion, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        x_q = self.query_conv(x)
        y_k = self.key_conv(y)
        energy = x_q * y_k
        attention = self.sig(energy)
        attention_x = x * attention
        attention_y = y * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        y_gamma = self.gamma2(torch.cat((y, attention_y), dim=1))
        y_out = y * y_gamma[:, [0], :, :] + attention_y * y_gamma[:, [1], :, :]

        x_s = x_out + y_out

        return x_s


class MultiscaleNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 3, 3],
                 heads=[1, 2, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 use_dtcwt=True,
                 dtcwt_detach=True
                 ):
        super(MultiscaleNet, self).__init__()
        # ========= DTCWT feature injection (low+high) =========
        self.use_dtcwt = use_dtcwt
        self.dtcwt_detach = dtcwt_detach

        if self.use_dtcwt:
            # 输出 21 通道：low(3) + high_mag(18)
            self.dtcwt = DTCWTFeatureMS(J=3, use_log1p_high=True)
            fuse_in_ch = 3 + 21  # RGB(3) + DTCWT(21) = 24

            # 三个尺度各用一个 1x1 融合层（参数非常少，但更稳定）
       #     self.dtcwt = DTCWTFeatureMS(J=3, use_log1p_high=True)
            self.dtcwt_proj_max = nn.Conv2d(21, 3, 1, bias=True)
            self.dtcwt_proj_mid = nn.Conv2d(21, 3, 1, bias=True)
            self.dtcwt_proj_small = nn.Conv2d(21, 3, 1, bias=True)

            # 让一开始不注入（更稳）
            nn.init.zeros_(self.dtcwt_proj_max.weight);
            nn.init.zeros_(self.dtcwt_proj_max.bias)
            nn.init.zeros_(self.dtcwt_proj_mid.weight);
            nn.init.zeros_(self.dtcwt_proj_mid.bias)
            nn.init.zeros_(self.dtcwt_proj_small.weight);
            nn.init.zeros_(self.dtcwt_proj_small.bias)

            # 学习到才注入
            self.dtcwt_alpha_max = nn.Parameter(torch.tensor(0.01))
            self.dtcwt_alpha_mid = nn.Parameter(torch.tensor(0.01))
            self.dtcwt_alpha_small = nn.Parameter(torch.tensor(0.01))

            # ========= Rain-Gating from DTCWT high(18) =========
            self.dtcwt_gate_max = nn.Conv2d(18, 1, kernel_size=1, bias=True)
            self.dtcwt_gate_mid = nn.Conv2d(18, 1, kernel_size=1, bias=True)
            self.dtcwt_gate_small = nn.Conv2d(18, 1, kernel_size=1, bias=True)

            # 让初始 mask 偏小（别一上来全图都当雨）
            nn.init.zeros_(self.dtcwt_gate_max.weight);
            nn.init.constant_(self.dtcwt_gate_max.bias, -2.0)
            nn.init.zeros_(self.dtcwt_gate_mid.weight);
            nn.init.constant_(self.dtcwt_gate_mid.bias, -2.0)
            nn.init.zeros_(self.dtcwt_gate_small.weight);
            nn.init.constant_(self.dtcwt_gate_small.bias, -2.0)

            # 调制强度（从 0 开始最稳）
            self.gate_gamma_max = nn.Parameter(torch.tensor(0.0))
            self.gate_gamma_mid = nn.Parameter(torch.tensor(0.0))
            self.gate_gamma_small = nn.Parameter(torch.tensor(0.0))

            # 初始化成“几乎等于原 RGB”（identity-like），保证一开始不破坏原模型行为
       #     self._init_fuse_identity(self.dtcwt_fuse_max)
       #     self._init_fuse_identity(self.dtcwt_fuse_mid)
        #    self._init_fuse_identity(self.dtcwt_fuse_small)

        self.patch_embed_small = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1_small = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2_small = Downsample(dim)
        self.encoder_level2_small = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3_small = Downsample(int(dim * 2 ** 1))
        self.latent_small = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2_small = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2_small = Conv1x1_SE(int(dim * 2 ** 2), int(dim * 2 ** 1),  bias=bias)
        self.decoder_level2_small = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1_small = Upsample(int(dim * 2 ** 1))
        self.reduce_chan_level1_small = Conv1x1_SE(int(dim * 2 ** 1), int(dim * 1 ** 1),  bias=bias)
        self.decoder_level1_small = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output_small = nn.Conv2d(int(dim * 1 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.INR = INR(dim).cuda()

        self.patch_embed_mid = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1_mid1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.encoder_level1_mid2 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2_mid = Downsample(dim)
        self.down1_2_mid2 = Downsample(dim)
        self.encoder_level2_mid1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_mid2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3_mid = Downsample(int(dim * 2 ** 1))
        self.down2_3_mid2 = Downsample(int(dim * 2 ** 1))
        self.latent_mid1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.latent_mid2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2_mid = Upsample(int(dim * 2 ** 2))
        self.up3_2_mid2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2_mid1 = Conv1x1_SE(int(dim * 2 ** 2), int(dim * 2 ** 1), bias=bias)
        self.reduce_chan_level2_mid2 = Conv1x1_SE(int(dim * 2 ** 2), int(dim * 2 ** 1), bias=bias)
        self.decoder_level2_mid1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level2_mid2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1_mid = Upsample(int(dim * 2 ** 1))
        self.up2_1_mid2 = Upsample(int(dim * 2 ** 1))
        self.reduce_chan_level1_mid1 = Conv1x1_SE(int(dim * 2 ** 1), int(dim * 1 ** 1), bias=bias)
        self.reduce_chan_level1_mid2 = Conv1x1_SE(int(dim * 2 ** 1), int(dim * 1 ** 1), bias=bias)
        self.decoder_level1_mid1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1_mid2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output_mid = nn.Conv2d(int(dim * 1 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_mid_context = nn.Conv2d(int(dim * 1 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.INR2 = INR(dim).cuda()

        self.patch_embed_max = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1_max1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_max2 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_max3 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2_max = Downsample(dim)
        self.down1_2_max2 = Downsample(dim)
        self.down1_2_max3 = Downsample(dim)
        self.encoder_level2_max1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_max2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_max3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3_max = Downsample(int(dim * 2 ** 1))
        self.down2_3_max2 = Downsample(int(dim * 2 ** 1))
        self.down2_3_max3 = Downsample(int(dim * 2 ** 1))
        self.latent_max1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.latent_max2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.latent_max3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2_max = Upsample(int(dim * 2 ** 2))
        self.up3_2_max2 = Upsample(int(dim * 2 ** 2))
        self.up3_2_max3 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2_max1 =  Conv1x1_SE(int(dim * 2 ** 2), int(dim * 2 ** 1),  bias=bias)
        self.reduce_chan_level2_max2 =  Conv1x1_SE(int(dim * 2 ** 2), int(dim * 2 ** 1),  bias=bias)
        self.reduce_chan_level2_max3 =  Conv1x1_SE(int(dim * 2 ** 2), int(dim * 2 ** 1),  bias=bias)
        self.decoder_level2_max1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level2_max2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level2_max3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1_max = Upsample(int(dim * 2 ** 1))
        self.up2_1_max2 = Upsample(int(dim * 2 ** 1))
        self.up2_1_max3 = Upsample(int(dim * 2 ** 1))
        self.reduce_chan_level1_max1 =  Conv1x1_SE(int(dim * 2 ** 1), int(dim * 1 ** 1), bias=bias)
        self.reduce_chan_level1_max2 =  Conv1x1_SE(int(dim * 2 ** 1), int(dim * 1 ** 1),  bias=bias)
        self.reduce_chan_level1_max3 =  Conv1x1_SE(int(dim * 2 ** 1), int(dim * 1 ** 1),  bias=bias)
        self.decoder_level1_max1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1_max2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1_max3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output_max = nn.Conv2d(int(dim * 1 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_max_context1 = nn.Conv2d(int(dim * 1 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_max_context2 = nn.Conv2d(int(dim * 1 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.BF1 = Fusion(dim * 4)
        self.BF2 = Fusion(dim * 4)
        self.BF3 = Fusion(dim * 4)

        self.upsmall2mid1 = Upsample(int(dim * 4 ** 1))
        self.upsmall2mid2 = Upsample(int(dim * 2 ** 1))

        self.upmid2max1 = Upsample(int(dim * 4 ** 1))
        self.upmid2max2 = Upsample(int(dim * 2 ** 1))

    @staticmethod
    def _init_fuse_identity(conv1x1: nn.Conv2d):
        # 让输出初始≈输入RGB：y = x_rgb（忽略新增特征）
        nn.init.zeros_(conv1x1.weight)
        if conv1x1.bias is not None:
            nn.init.zeros_(conv1x1.bias)
        # 把前三个输入通道（RGB）拷贝到输出
        for c in range(3):
            conv1x1.weight.data[c, c, 0, 0] = 1.0


    def forward(self, inp_img):
        outputs = list()

        inp_img_max = inp_img
        inp_img_mid = F.interpolate(inp_img, scale_factor=0.5, mode='bilinear', align_corners=False)
        inp_img_small = F.interpolate(inp_img, scale_factor=0.25, mode='bilinear', align_corners=False)
        # ========= DTCWT inject (low+high) =========
        if self.use_dtcwt:
            feat_max, feat_mid, feat_small = self.dtcwt(inp_img_max, inp_img_mid, inp_img_small)
            # 注意：这里不要 detach（否则 dtcwt.post 学不到）

            inp_img_max = inp_img_max + self.dtcwt_alpha_max * self.dtcwt_proj_max(feat_max)
            inp_img_mid = inp_img_mid + self.dtcwt_alpha_mid * self.dtcwt_proj_mid(feat_mid)
            inp_img_small = inp_img_small + self.dtcwt_alpha_small * self.dtcwt_proj_small(feat_small)

        # ========= build rain masks from DTCWT high(18) =========
        high_max = feat_max[:, 3:, :, :].detach()  # [B,18,H,W]
        high_mid = feat_mid[:, 3:, :, :].detach()  # [B,18,H/2,W/2]
        high_small = feat_small[:, 3:, :, :].detach()  # [B,18,H/4,W/4]

        M_max = torch.sigmoid(self.dtcwt_gate_max(high_max))  # [B,1,H,W]
        M_mid = torch.sigmoid(self.dtcwt_gate_mid(high_mid))  # [B,1,H/2,W/2]
        M_small = torch.sigmoid(self.dtcwt_gate_small(high_small))  # [B,1,H/4,W/4]

        # 保存给训练脚本做 L_keep（推荐 detach 后用，防止“钻空子”）
        self._last_masks = (M_max, M_mid, M_small)

        inp_enc_level1_small = self.patch_embed_small(inp_img_small)
        g = torch.tanh(self.gate_gamma_small)  # 限幅更稳
        inp_enc_level1_small = inp_enc_level1_small * (1.0 + g * M_small)
        out_enc_level1_small = self.encoder_level1_small(inp_enc_level1_small)

        inp_enc_level2_small = self.down1_2_small(out_enc_level1_small)
        out_enc_level2_small = self.encoder_level2_small(inp_enc_level2_small)

        inp_enc_level4_small = self.down2_3_small(out_enc_level2_small)
        latent_small = self.latent_small(inp_enc_level4_small)
        latent_small_mid = self.upsmall2mid1(latent_small)
        latent_small_mid = self.upsmall2mid2(latent_small_mid)

        outputs.append(inp_img_small)
        INR = self.INR(latent_small_mid)
        inp_img_small_ = INR + inp_img_small
        outputs.append(inp_img_small_)

        inp_img_small_ = F.interpolate(inp_img_small_, scale_factor=2)

        mid_img = inp_img_mid + inp_img_small_

        inp_enc_level1_mid = self.patch_embed_mid(mid_img)
        g = torch.tanh(self.gate_gamma_mid)
        inp_enc_level1_mid = inp_enc_level1_mid * (1.0 + g * M_mid)
        out_enc_level1_mid = self.encoder_level1_mid1(inp_enc_level1_mid)

        inp_enc_level2_mid = self.down1_2_mid(out_enc_level1_mid)
        out_enc_level2_mid = self.encoder_level2_mid1(inp_enc_level2_mid)

        inp_enc_level4_mid = self.down2_3_mid(out_enc_level2_mid)
        latent_mid = self.latent_mid1(inp_enc_level4_mid)
        latent_mid_INR_max = self.upmid2max1(latent_mid)
        latent_mid_INR_max = self.upmid2max2(latent_mid_INR_max)

        outputs.append(mid_img / 2)
        INR2 = self.INR2(latent_mid_INR_max)
        mid_img_ = INR2 + mid_img
        outputs.append(mid_img_)

        mid_img_ = F.interpolate(mid_img_, scale_factor=2)

        max_img = inp_img_max + mid_img_

        inp_enc_level1_max = self.patch_embed_max(max_img)
        g = torch.tanh(self.gate_gamma_max)
        inp_enc_level1_max = inp_enc_level1_max * (1.0 + g * M_max)
        out_enc_level1_max = self.encoder_level1_max1(inp_enc_level1_max)

        inp_enc_level2_max = self.down1_2_max(out_enc_level1_max)
        out_enc_level2_max = self.encoder_level2_max1(inp_enc_level2_max)

        inp_enc_level4_max = self.down2_3_max(out_enc_level2_max)
        latent_max = self.latent_max1(inp_enc_level4_max)
        BFF_max_1 = latent_max

        inp_dec_level2_max = self.up3_2_max(latent_max)
        inp_dec_level2_max = torch.cat([inp_dec_level2_max, out_enc_level2_max], 1)
        inp_dec_level2_max = self.reduce_chan_level2_max1(inp_dec_level2_max)
        out_dec_level2_max = self.decoder_level2_max1(inp_dec_level2_max)

        inp_dec_level1_max = self.up2_1_max(out_dec_level2_max)
        inp_dec_level1_max = torch.cat([inp_dec_level1_max, out_enc_level1_max], 1)
        inp_dec_level1_max = self.reduce_chan_level1_max1(inp_dec_level1_max)
        out_dec_level1_max = self.decoder_level1_max1(inp_dec_level1_max)

        out_dec_level1_max = self.output_max_context1(out_dec_level1_max)
        out_enc_level1_max = self.encoder_level1_max2(out_dec_level1_max)

        inp_enc_level2_max = self.down1_2_max2(out_enc_level1_max)
        out_enc_level2_max = self.encoder_level2_max2(inp_enc_level2_max)

        inp_enc_level4_max = self.down2_3_max2(out_enc_level2_max)
        latent_max = self.latent_max2(inp_enc_level4_max)
        BFF_max_2 = latent_max

        inp_dec_level2_max = self.up3_2_max2(latent_max)
        inp_dec_level2_max = torch.cat([inp_dec_level2_max, out_enc_level2_max], 1)
        inp_dec_level2_max = self.reduce_chan_level2_max2(inp_dec_level2_max)
        out_dec_level2_max = self.decoder_level2_max2(inp_dec_level2_max)

        inp_dec_level1_max = self.up2_1_max2(out_dec_level2_max)
        inp_dec_level1_max = torch.cat([inp_dec_level1_max, out_enc_level1_max], 1)
        inp_dec_level1_max = self.reduce_chan_level1_max2(inp_dec_level1_max)
        out_dec_level1_max = self.decoder_level1_max2(inp_dec_level1_max)

        out_dec_level1_max = self.output_max_context2(out_dec_level1_max)
        out_enc_level1_max = self.encoder_level1_max3(out_dec_level1_max)

        inp_enc_level2_max = self.down1_2_max3(out_enc_level1_max)
        out_enc_level2_max = self.encoder_level2_max3(inp_enc_level2_max)

        inp_enc_level4_max = self.down2_3_max3(out_enc_level2_max)
        latent_max = self.latent_max3(inp_enc_level4_max)
        BFF_max_3 = latent_max

        BFF1 = self.BF1(BFF_max_1, BFF_max_2)
        BFF2 = self.BF2(BFF_max_2, BFF_max_3)

        BFF1 = F.interpolate(BFF1, scale_factor=0.5)
        BFF2 = F.interpolate(BFF2, scale_factor=0.5)

        inp_dec_level2_max = self.up3_2_max3(latent_max)

        BFF3_1 = latent_mid
        latent_mid = latent_mid + BFF1

        inp_dec_level2_mid = self.up3_2_mid(latent_mid)
        inp_dec_level2_mid = torch.cat([inp_dec_level2_mid, out_enc_level2_mid], 1)
        inp_dec_level2_mid = self.reduce_chan_level2_mid1(inp_dec_level2_mid)
        out_dec_level2_mid = self.decoder_level2_mid1(inp_dec_level2_mid)

        inp_dec_level1_mid = self.up2_1_mid(out_dec_level2_mid)
        inp_dec_level1_mid = torch.cat([inp_dec_level1_mid, out_enc_level1_mid], 1)
        inp_dec_level1_mid = self.reduce_chan_level1_mid1(inp_dec_level1_mid)
        out_dec_level1_mid = self.decoder_level1_mid1(inp_dec_level1_mid)

        out_dec_level1_mid = self.output_mid_context(out_dec_level1_mid)
        out_enc_level1_mid = self.encoder_level1_mid2(out_dec_level1_mid)

        inp_enc_level2_mid = self.down1_2_mid2(out_enc_level1_mid)
        out_enc_level2_mid = self.encoder_level2_mid2(inp_enc_level2_mid)

        inp_enc_level4_mid = self.down2_3_mid2(out_enc_level2_mid)
        latent_mid = self.latent_mid2(inp_enc_level4_mid)
        BFF3_2 = latent_mid
        BFF3 = self.BF3(BFF3_1, BFF3_2)
        BFF3 = F.interpolate(BFF3, scale_factor=0.5)

        latent_mid = latent_mid + BFF2

        inp_dec_level2_mid = self.up3_2_mid2(latent_mid)

        latent_small = latent_small + BFF3

        inp_dec_level2_small = self.up3_2_small(latent_small)
        inp_dec_level2_small = torch.cat([inp_dec_level2_small, out_enc_level2_small], 1)
        inp_dec_level2_small = self.reduce_chan_level2_small(inp_dec_level2_small)
        out_dec_level2_small = self.decoder_level2_small(inp_dec_level2_small)

        inp_dec_level1_small = self.up2_1_small(out_dec_level2_small)
        inp_dec_level1_small = torch.cat([inp_dec_level1_small, out_enc_level1_small], 1)
        inp_dec_level1_small = self.reduce_chan_level1_small(inp_dec_level1_small)
        out_dec_level1_small = self.decoder_level1_small(inp_dec_level1_small)

        small_2_mid = out_dec_level1_small

        out_dec_level1_small = self.output_small(out_dec_level1_small) + inp_img_small

        outputs.append(out_dec_level1_small)
        small = F.interpolate(out_dec_level1_small, scale_factor=2)

        inp_dec_level2_mid = torch.cat([inp_dec_level2_mid, out_enc_level2_mid], 1)
        inp_dec_level2_mid = self.reduce_chan_level2_mid2(inp_dec_level2_mid)
        out_dec_level2_mid = self.decoder_level2_mid2(inp_dec_level2_mid)

        inp_dec_level1_mid = self.up2_1_mid2(out_dec_level2_mid)
        inp_dec_level1_mid = torch.cat([inp_dec_level1_mid, out_enc_level1_mid], 1)
        inp_dec_level1_mid = self.reduce_chan_level1_mid2(inp_dec_level1_mid)
        out_dec_level1_mid = self.decoder_level1_mid2(inp_dec_level1_mid)

        small_2_mid = F.interpolate(small_2_mid, scale_factor=2)
        out_dec_level1_mid = out_dec_level1_mid + small_2_mid

        mid_2_max = out_dec_level1_mid

        out_dec_level1_mid = self.output_mid(out_dec_level1_mid) + inp_img_mid

        outputs.append(out_dec_level1_mid)
        mid = F.interpolate(out_dec_level1_mid, scale_factor=2)

        inp_dec_level2_max = torch.cat([inp_dec_level2_max, out_enc_level2_max], 1)
        inp_dec_level2_max = self.reduce_chan_level2_max3(inp_dec_level2_max)
        out_dec_level2_max = self.decoder_level2_max3(inp_dec_level2_max)

        inp_dec_level1_max = self.up2_1_max3(out_dec_level2_max)
        inp_dec_level1_max = torch.cat([inp_dec_level1_max, out_enc_level1_max], 1)
        inp_dec_level1_max = self.reduce_chan_level1_max2(inp_dec_level1_max)
        mid_2_max = F.interpolate(mid_2_max, scale_factor=2)
        out_dec_level1_max = self.decoder_level1_max3(inp_dec_level1_max) + mid_2_max

        out_dec_level1_max = self.output_max(out_dec_level1_max) + inp_img_max

        outputs.append(out_dec_level1_max)

        return outputs[::-1]
