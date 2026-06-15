"""Sam vgg."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN_BGR: tuple[float, float, float] = (103.939, 116.779, 123.68)


def _conv3x3(in_ch: int, out_ch: int, dilation: int = 1, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3,
                     padding=dilation, dilation=dilation, bias=bias)


def _conv1x1(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)


def _normalize_minmax(maps: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """(documentation omitted in the English release)."""
    min_per = torch.amin(maps, dim=(1, 2, 3), keepdim=True)
    maps = maps - min_per
    max_per = torch.amax(maps, dim=(1, 2, 3), keepdim=True)
    return maps / (max_per + eps)


class SamePadMaxPool2d(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, kernel_size: int = 2, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size - self.stride
        if pad <= 0:
            return F.max_pool2d(x, self.kernel_size, self.stride)
        pad_l = pad // 2
        pad_r = pad - pad_l
        x = F.pad(x, [pad_l, pad_r, pad_l, pad_r])
        return F.max_pool2d(x, self.kernel_size, self.stride)


class SAMVGGEncoder(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self) -> None:
        super().__init__()
        # block1
        self.conv1_1 = _conv3x3(3, 64)
        self.conv1_2 = _conv3x3(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        # block2
        self.conv2_1 = _conv3x3(64, 128)
        self.conv2_2 = _conv3x3(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        # block3
        self.conv3_1 = _conv3x3(128, 256)
        self.conv3_2 = _conv3x3(256, 256)
        self.conv3_3 = _conv3x3(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4_1 = _conv3x3(256, 512)
        self.conv4_2 = _conv3x3(512, 512)
        self.conv4_3 = _conv3x3(512, 512)
        self.pool4 = SamePadMaxPool2d(kernel_size=2, stride=1)

        self.conv5_1 = _conv3x3(512, 512, dilation=2)
        self.conv5_2 = _conv3x3(512, 512, dilation=2)
        self.conv5_3 = _conv3x3(512, 512, dilation=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)                # 1/8 spatial
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)                # 1/8 spatial (stride 1)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))      # 512ch, 1/8
        return x


class AttentiveConvLSTM(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, channels: int = 512, kernel_size: int = 3,
                 n_steps: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.n_steps = n_steps
        pad = kernel_size // 2

        # Attention path
        self.W_a = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.U_a = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.V_a = nn.Conv2d(channels, 1, kernel_size, padding=pad)

        # Gates: input / forget / output
        self.W_i = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.U_i = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.W_f = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.U_f = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.W_o = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.U_o = nn.Conv2d(channels, channels, kernel_size, padding=pad)

        # Candidate cell
        self.W_c = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.U_c = nn.Conv2d(channels, channels, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        h_t = torch.zeros(b, self.channels, h, w, device=x.device, dtype=x.dtype)
        c_t = torch.zeros(b, self.channels, h, w, device=x.device, dtype=x.dtype)

        for _ in range(self.n_steps):
            # Attention map
            e = self.V_a(torch.tanh(self.W_a(h_t) + self.U_a(x)))    # [B, 1, H, W]
            # softmax over spatial dims (H, W)
            e_flat = e.view(b, 1, -1)
            a_flat = F.softmax(e_flat, dim=-1)
            a = a_flat.view(b, 1, h, w)                              # [B, 1, H, W]
            x_tilde = x * a                                          # [B, C, H, W]

            i = torch.sigmoid(self.W_i(x_tilde) + self.U_i(h_t))
            f = torch.sigmoid(self.W_f(x_tilde) + self.U_f(h_t))
            o = torch.sigmoid(self.W_o(x_tilde) + self.U_o(h_t))
            c_hat = torch.tanh(self.W_c(x_tilde) + self.U_c(h_t))
            c_t = f * c_t + i * c_hat
            h_t = o * torch.tanh(c_t)

        return h_t


class LearningPrior(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, nb_gaussian: int = 16) -> None:
        super().__init__()
        self.nb_gaussian = nb_gaussian



        mu_x = torch.empty(nb_gaussian).uniform_(0.3, 0.7)
        mu_y = torch.empty(nb_gaussian).uniform_(0.3, 0.7)
        sigma_x = torch.empty(nb_gaussian).uniform_(0.05, 0.3)
        sigma_y = torch.empty(nb_gaussian).uniform_(0.05, 0.3)
        self.mu_x = nn.Parameter(mu_x)
        self.mu_y = nn.Parameter(mu_y)
        self.sigma_x = nn.Parameter(sigma_x)
        self.sigma_y = nn.Parameter(sigma_y)

    def forward(self, ref: torch.Tensor) -> torch.Tensor:
        """(documentation omitted in the English release)."""
        b, _, h, w = ref.shape
        device, dtype = ref.device, ref.dtype

        # clip
        mu_x = torch.clamp(self.mu_x, 0.25, 0.75)
        mu_y = torch.clamp(self.mu_y, 0.35, 0.65)
        sigma_x = torch.clamp(self.sigma_x, 0.1, 0.9)
        sigma_y = torch.clamp(self.sigma_y, 0.2, 0.8)


        ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")        # [H, W]

        # broadcast: [1, nb_g, H, W] - [1, nb_g, 1, 1] -> [1, nb_g, H, W]
        gx = grid_x.unsqueeze(0).unsqueeze(0)                          # [1, 1, H, W]
        gy = grid_y.unsqueeze(0).unsqueeze(0)
        mx = mu_x.view(1, -1, 1, 1)
        my = mu_y.view(1, -1, 1, 1)
        sx = sigma_x.view(1, -1, 1, 1)
        sy = sigma_y.view(1, -1, 1, 1)


        exponent = -((gx - mx) ** 2 / (2.0 * sx ** 2 + 1e-8)
                     + (gy - my) ** 2 / (2.0 * sy ** 2 + 1e-8))
        prior = torch.exp(exponent)                                    # [1, nb_g, H, W]


        return prior.expand(b, -1, -1, -1).contiguous()


class SAMVGG(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(
        self,
        n_lstm_steps: int = 4,
        nb_gaussian: int = 16,
        upsample_factor: int = 8,
        subtract_imagenet_mean: bool = True,
        pretrained_imagenet: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = SAMVGGEncoder()
        self.convlstm = AttentiveConvLSTM(channels=512, kernel_size=3,
                                          n_steps=n_lstm_steps)
        # Prior 1

        # padding = dilation * (kernel-1) / 2 = 4 * 4 / 2 = 8
        self.prior1 = LearningPrior(nb_gaussian=nb_gaussian)
        self.atrous1 = nn.Conv2d(512 + nb_gaussian, 512,
                                 kernel_size=5, padding=8, dilation=4)
        # Prior 2
        self.prior2 = LearningPrior(nb_gaussian=nb_gaussian)
        self.atrous2 = nn.Conv2d(512 + nb_gaussian, 512,
                                 kernel_size=5, padding=8, dilation=4)
        # Head
        self.conv_out = _conv1x1(512, 1)

        self.upsample_factor = upsample_factor
        self.subtract_imagenet_mean = subtract_imagenet_mean
        self.register_buffer(
            "mean_bgr",
            torch.tensor(IMAGENET_MEAN_BGR, dtype=torch.float32).view(1, 3, 1, 1),
        )

        if pretrained_imagenet:
            self._init_encoder_from_torchvision()

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    def _init_encoder_from_torchvision(self) -> None:
        """(documentation omitted in the English release)."""
        from torchvision.models import vgg16, VGG16_Weights
        src = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        src_features = list(src.features.children())

        conv_idx = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        targets = [
            self.encoder.conv1_1, self.encoder.conv1_2,
            self.encoder.conv2_1, self.encoder.conv2_2,
            self.encoder.conv3_1, self.encoder.conv3_2, self.encoder.conv3_3,
            self.encoder.conv4_1, self.encoder.conv4_2, self.encoder.conv4_3,
            self.encoder.conv5_1, self.encoder.conv5_2, self.encoder.conv5_3,
        ]
        for src_idx, dst in zip(conv_idx, targets):
            src_conv = src_features[src_idx]
            assert isinstance(src_conv, nn.Conv2d), f"vgg features[{src_idx}] is not Conv2d"
            # RGB→BGR flip (input channels)
            if src_conv.in_channels == 3 and dst.in_channels == 3:
                w = src_conv.weight.data[:, [2, 1, 0], :, :].clone()
            else:
                w = src_conv.weight.data.clone()
            dst.weight.data.copy_(w)
            if src_conv.bias is not None and dst.bias is not None:
                dst.bias.data.copy_(src_conv.bias.data)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_imagenet_mean:
            x = x - self.mean_bgr

        feat = self.encoder(x)                  # [B, 512, H/8, W/8]
        h = self.convlstm(feat)                 # [B, 512, H/8, W/8]

        # Prior 1: concat + atrous conv
        p1 = self.prior1(h)                     # [B, 16, H/8, W/8]
        h = torch.cat([h, p1], dim=1)
        h = F.relu(self.atrous1(h))             # [B, 512, H/8, W/8]

        # Prior 2: concat + atrous conv
        p2 = self.prior2(h)                     # [B, 16, H/8, W/8]
        h = torch.cat([h, p2], dim=1)
        h = F.relu(self.atrous2(h))             # [B, 512, H/8, W/8]


        out = self.conv_out(h)                  # [B, 1, H/8, W/8]

        out = F.interpolate(out, scale_factor=self.upsample_factor,
                            mode="bilinear", align_corners=True)

        out = _normalize_minmax(out)
        return out


__all__ = [
    "SAMVGG", "SAMVGGEncoder", "AttentiveConvLSTM", "LearningPrior",
    "IMAGENET_MEAN_BGR",
]
