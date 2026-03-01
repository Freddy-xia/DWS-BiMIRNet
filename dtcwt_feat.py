# dtcwt_feat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward


class DTCWTFeature(nn.Module):
    """
    输出：DTCWT 低频 + 高频幅值（J=1）
      - lowpass: 3ch
      - highpass magnitude: 3*6=18ch
    总计：21ch，最终输出 shape = [B, 21, H, W]
    """
    def __init__(self, J=1, biort='near_sym_b', qshift='qshift_b',
                 eps=1e-12, use_log1p_high=True):
        super().__init__()
        assert J == 1, "建议先用 J=1 跑通（你的网络本身已经多尺度了）"
        self.J = J
        self.xfm = DTCWTForward(J=J, biort=biort, qshift=qshift)
        self.eps = eps
        self.use_log1p_high = use_log1p_high
        self.post = nn.Sequential(
            nn.Conv2d(21, 21, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(21, 21, kernel_size=1, bias=True),
        )
        # 让一开始 post 输出为 0（配合 residual），不破坏原特征分布
        nn.init.zeros_(self.post[2].weight)
        nn.init.zeros_(self.post[2].bias)
    @staticmethod
    def _pad_to_multiple(x, m):
        # reflect pad so H,W divisible by m
        B, C, H, W = x.shape
        pad_h = (m - H % m) % m
        pad_w = (m - W % m) % m
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, 0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, (0, pad_w, 0, pad_h)

    @staticmethod
    def _unpad(x, pads):
        l, r, t, b = pads
        if r == 0 and b == 0:
            return x
        return x[..., :x.shape[-2]-b, :x.shape[-1]-r]

    def forward(self, x):
        # x: [B,3,H,W]
        # DTCWTForward 可能没有 parameters()，用 buffers 来判断 device
        buf = next(self.xfm.buffers(), None)
        if buf is None or buf.device != x.device:
            self.xfm = self.xfm.to(x.device)

        m = 2 ** self.J
        x_pad, pads = self._pad_to_multiple(x, m)

        with torch.no_grad():
            Yl, Yh = self.xfm(x_pad)

        # lowpass: [B,3,H/2,W/2] -> up to [B,3,H,W]
        low = F.interpolate(Yl, size=x_pad.shape[-2:], mode='bilinear', align_corners=False)

        # highpass: Yh[0]: [B,3,6,H/2,W/2,2] (real/imag)
        hp = Yh[0]
        mag = torch.sqrt(hp[..., 0] ** 2 + hp[..., 1] ** 2 + self.eps)  # [B,3,6,h,w]
        B, C, O, h, w = mag.shape
        high = mag.reshape(B, C * O, h, w)  # [B,18,h,w]
        high = F.interpolate(high, size=x_pad.shape[-2:], mode='bilinear', align_corners=False)

        if self.use_log1p_high:
            high = torch.log1p(high)  # 稳定训练（防止幅值过大）

        feat = torch.cat([low, high], dim=1)  # [B,21,H,W]
        feat = self._unpad(feat, pads)

        # ===== Scheme B: post-adapter (residual) =====
        feat = feat + self.post(feat)

        return feat

class DTCWTFeatureMS(nn.Module):
    """
    多尺度 DTCWT 特征（一次 J=3 分解，输出三尺度特征）：
      - feat_max:  [B,21,H,  W  ]  (low + upsample(Yh[0]))
      - feat_mid:  [B,21,H/2,W/2]  (low +      Yh[0])
      - feat_small:[B,21,H/4,W/4]  (low +      Yh[1])

    仍然是 low(3) + high_mag(18) = 21 通道，并带方案B的 post-adapter。
    """
    def __init__(self, J=3, biort='near_sym_b', qshift='qshift_b',
                 eps=1e-12, use_log1p_high=True):
        super().__init__()
        assert J >= 2, "small 至少需要用到 Yh[1]，所以 J>=2；推荐 J=3"
        self.J = J
        self.xfm = DTCWTForward(J=J, biort=biort, qshift=qshift)
        self.eps = eps
        self.use_log1p_high = use_log1p_high

        # ===== 方案B：post-adapter（1x1->GELU->1x1），residual =====
        self.post = nn.Sequential(
            nn.Conv2d(21, 21, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(21, 21, 1, bias=True),
        )
        nn.init.zeros_(self.post[2].weight)
        nn.init.zeros_(self.post[2].bias)

    @staticmethod
    def _pad_to_multiple(x, m):
        B, C, H, W = x.shape
        pad_h = (m - H % m) % m
        pad_w = (m - W % m) % m
        if pad_h == 0 and pad_w == 0:
            return x, (H, W)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, (H, W)  # 记录原始尺寸，后面 crop 回去

    def _ensure_device(self, x):
        buf = next(self.xfm.buffers(), None)
        if buf is None or buf.device != x.device:
            self.xfm = self.xfm.to(x.device)

    def _high_mag(self, yh):
        # yh: [B,3,6,h,w,2]
        mag = torch.sqrt(yh[..., 0] ** 2 + yh[..., 1] ** 2 + self.eps)  # [B,3,6,h,w]
        B, C, O, h, w = mag.shape
        high = mag.reshape(B, C * O, h, w)  # [B,18,h,w]
        if self.use_log1p_high:
            high = torch.log1p(high)
        return high

    def forward(self, x_max, x_mid=None, x_small=None):
        """
        x_max:  [B,3,H,W]
        x_mid:  [B,3,H/2,W/2]  (可传可不传；传了就严格对齐尺寸)
        x_small:[B,3,H/4,W/4]
        """
        self._ensure_device(x_max)

        m = 2 ** self.J
        x_pad, (H0, W0) = self._pad_to_multiple(x_max, m)

        # 只对 xfm no_grad：xfm 没参数，省显存；post 仍可训练
        with torch.no_grad():
            Yl, Yh = self.xfm(x_pad)

        # 低频：只有最深层 lowpass Yl（分辨率 H/2^J），上采样到各尺度
        low_max = F.interpolate(Yl, size=x_pad.shape[-2:], mode='bilinear', align_corners=False)
        low_max = low_max[..., :H0, :W0]  # crop 回原 max

        # 高频：
        high_mid_native = self._high_mag(Yh[0])  # [B,18,H/2,W/2] (在 pad 尺寸下)
        high_small_native = self._high_mag(Yh[1])  # [B,18,H/4,W/4]

        # 目标尺寸
        if x_mid is None:
            Hm, Wm = H0 // 2, W0 // 2
        else:
            Hm, Wm = x_mid.shape[-2], x_mid.shape[-1]

        if x_small is None:
            Hs, Ws = H0 // 4, W0 // 4
        else:
            Hs, Ws = x_small.shape[-2], x_small.shape[-1]

        # mid 特征
        low_mid = F.interpolate(Yl, size=(Hm, Wm), mode='bilinear', align_corners=False)
        high_mid = F.interpolate(high_mid_native, size=(Hm, Wm), mode='bilinear', align_corners=False)
        feat_mid = torch.cat([low_mid, high_mid], dim=1)  # [B,21,Hm,Wm]

        # small 特征
        low_small = F.interpolate(Yl, size=(Hs, Ws), mode='bilinear', align_corners=False)
        high_small = F.interpolate(high_small_native, size=(Hs, Ws), mode='bilinear', align_corners=False)
        feat_small = torch.cat([low_small, high_small], dim=1)  # [B,21,Hs,Ws]

        # max 特征：用 mid 的高频上采样回 max
        high_max = F.interpolate(high_mid_native, size=(H0, W0), mode='bilinear', align_corners=False)
        feat_max = torch.cat([low_max, high_max], dim=1)  # [B,21,H,W]

        # 方案B：post-adapter（对三个尺度都做 residual）
        feat_max = feat_max + self.post(feat_max)
        feat_mid = feat_mid + self.post(feat_mid)
        feat_small = feat_small + self.post(feat_small)

        return feat_max, feat_mid, feat_small

