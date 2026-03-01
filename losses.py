import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward

class DTCWTMagLoss(nn.Module):
    """
    DTCWT magnitude-domain loss:
    L = w_low * |Yl_pred - Yl_gt|_1  +  w_high * sum_j |mag(Yh_pred[j]) - mag(Yh_gt[j])|_1
    """
    def __init__(self, J=3, biort='near_sym_b', qshift='qshift_b',
                 w_low=0.2, w_high=1.0, eps=1e-12):
        super().__init__()
        self.J = J
        self.xfm = DTCWTForward(J=J, biort=biort, qshift=qshift)
        self.w_low = w_low
        self.w_high = w_high
        self.eps = eps

    @staticmethod
    def _pad_to_multiple(x, m=8):
        # reflect pad to make H,W divisible by m (m usually 2^J)
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
        if (r == 0) and (b == 0):
            return x
        return x[..., :x.shape[-2]-b, :x.shape[-1]-r]

    def forward(self, pred, gt):
        # Ensure module on same device (important if loss constructed on CPU)
        buf = next(self.xfm.buffers(), None)
        if buf is None or buf.device != pred.device:
            self.xfm = self.xfm.to(pred.device)

        # DTCWT 对尺寸有 2^J 的整除偏好；训练 patch=256 默认很适配，但这里做个兜底
        m = 2 ** self.J
        pred_p, pads = self._pad_to_multiple(pred, m=m)
        gt_p, _ = self._pad_to_multiple(gt, m=m)

        Yl_p, Yh_p = self.xfm(pred_p)
        Yl_g, Yh_g = self.xfm(gt_p)

        loss_low = F.l1_loss(Yl_p, Yl_g)

        loss_high = 0.0
        for j in range(self.J):
            # Yh[j]: [B, C, 6, H, W, 2]  (real/imag in last dim)
            hp = Yh_p[j]
            hg = Yh_g[j]
            mag_p = torch.sqrt(hp[..., 0] ** 2 + hp[..., 1] ** 2 + self.eps)
            mag_g = torch.sqrt(hg[..., 0] ** 2 + hg[..., 1] ** 2 + self.eps)
            loss_high = loss_high + F.l1_loss(mag_p, mag_g)

        return self.w_low * loss_low + self.w_high * loss_high

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x.to('cuda:0') - y.to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x.to('cuda:0')), self.laplacian_kernel(y.to('cuda:0')))
        return loss

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss



