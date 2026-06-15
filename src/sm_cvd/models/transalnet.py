"""Transalnet."""
from __future__ import annotations

import copy
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm, Linear, Softmax


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
ACT2FN = {"gelu": F.gelu, "relu": F.relu}


class _Attention(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.num_attention_heads = config["num_heads"]
        self.attention_head_size = int(config["hidden_size"] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config["hidden_size"], self.all_head_size)
        self.key = Linear(config["hidden_size"], self.all_head_size)
        self.value = Linear(config["hidden_size"], self.all_head_size)
        self.out = Linear(self.all_head_size, config["hidden_size"])
        self.attn_dropout = Dropout(config["attention_dropout_rate"])
        self.proj_dropout = Dropout(config["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        ctx = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        new_shape = ctx.size()[:-2] + (self.all_head_size,)
        ctx = ctx.view(*new_shape)
        return self.proj_dropout(self.out(ctx))


class _Mlp(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.fc1 = Linear(config["hidden_size"], config["mlp_dim"])
        self.fc2 = Linear(config["mlp_dim"], config["hidden_size"])
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config["dropout_rate"])
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class _Block(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.attention_norm = LayerNorm(config["hidden_size"], eps=1e-6)
        self.attn = _Attention(config)
        self.ffn_norm = LayerNorm(config["hidden_size"], eps=1e-6)
        self.ffn = _Mlp(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        return x + h


class _TransformerEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config["hidden_size"], eps=1e-6)
        for _ in range(config["num_layers"]):
            self.layer.append(copy.deepcopy(_Block(config)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for blk in self.layer:
            hidden_states = blk(hidden_states)
        return self.encoder_norm(hidden_states)


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
class TransEncoder(nn.Module):
    def __init__(self, in_channels: int, spatial_size: int, cfg: dict) -> None:
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=cfg["hidden_size"],
            kernel_size=1,
            stride=1,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, spatial_size, cfg["hidden_size"])
        )
        self.transformer_encoder = _TransformerEncoder(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)

        if x.shape[1] != self.position_embeddings.shape[1]:
            pe = self.position_embeddings.permute(0, 2, 1)
            pe = F.interpolate(pe, size=x.shape[1], mode="linear", align_corners=False)
            pe = pe.permute(0, 2, 1)
            x = x + pe
        else:
            x = x + self.position_embeddings

        x = self.transformer_encoder(x)
        B, n_patch, hidden = x.shape
        x = x.permute(0, 2, 1).contiguous().view(B, hidden, H, W)
        return x


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
_CFG1 = {
    "hidden_size": 768, "mlp_dim": 768 * 4, "num_heads": 12,
    "num_layers": 2, "attention_dropout_rate": 0.0, "dropout_rate": 0.0,
}
_CFG2 = {
    "hidden_size": 768, "mlp_dim": 768 * 4, "num_heads": 12,
    "num_layers": 2, "attention_dropout_rate": 0.0, "dropout_rate": 0.0,
}
_CFG3 = {
    "hidden_size": 512, "mlp_dim": 512 * 4, "num_heads": 8,
    "num_layers": 2, "attention_dropout_rate": 0.0, "dropout_rate": 0.0,
}







_SPATIAL_1 = 9 * 12
_SPATIAL_2 = 18 * 24
_SPATIAL_3 = 36 * 48


class _ResNet50Encoder(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, pretrained_imagenet: bool = True) -> None:
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_imagenet else None
        base = resnet50(weights=weights)
        layers = list(base.children())[:8]
        self.encoder = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in (5, 6, 7):
                outputs.append(x)
        return outputs


def _make_cfg_with_layers(base: dict, num_layers: int) -> dict:
    """(documentation omitted in the English release)."""
    cfg = dict(base)
    cfg["num_layers"] = int(num_layers)
    return cfg


class _Decoder(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, transformer_num_layers: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(768, 768, 3, 1, 1)
        self.conv2 = nn.Conv2d(768, 512, 3, 1, 1)
        self.conv3 = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128,  64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64,   32, 3, 1, 1)
        self.conv7 = nn.Conv2d(32,    1, 3, 1, 1)

        self.batchnorm1 = nn.BatchNorm2d(768)
        self.batchnorm2 = nn.BatchNorm2d(512)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.batchnorm5 = nn.BatchNorm2d(64)
        self.batchnorm6 = nn.BatchNorm2d(32)

        cfg1 = _make_cfg_with_layers(_CFG1, transformer_num_layers)
        cfg2 = _make_cfg_with_layers(_CFG2, transformer_num_layers)
        cfg3 = _make_cfg_with_layers(_CFG3, transformer_num_layers)
        self.TransEncoder1 = TransEncoder(2048, _SPATIAL_1, cfg1)
        self.TransEncoder2 = TransEncoder(1024, _SPATIAL_2, cfg2)
        self.TransEncoder3 = TransEncoder( 512, _SPATIAL_3, cfg3)

        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x3, x4, x5 = x

        x5 = self.TransEncoder1(x5)
        x5 = self.conv1(x5); x5 = self.batchnorm1(x5); x5 = self.relu(x5)
        x5 = self.upsample(x5)

        x4_a = self.TransEncoder2(x4)
        x4 = x5 * x4_a
        x4 = self.relu(x4)
        x4 = self.conv2(x4); x4 = self.batchnorm2(x4); x4 = self.relu(x4)
        x4 = self.upsample(x4)

        x3_a = self.TransEncoder3(x3)
        x3 = x4 * x3_a
        x3 = self.relu(x3)
        x3 = self.conv3(x3); x3 = self.batchnorm3(x3); x3 = self.relu(x3)
        x3 = self.upsample(x3)

        x2 = self.conv4(x3); x2 = self.batchnorm4(x2); x2 = self.relu(x2)
        x2 = self.upsample(x2)
        x2 = self.conv5(x2); x2 = self.batchnorm5(x2); x2 = self.relu(x2)

        x1 = self.upsample(x2)
        x1 = self.conv6(x1); x1 = self.batchnorm6(x1); x1 = self.relu(x1)
        x1 = self.conv7(x1)
        return self.sigmoid(x1)


class TranSalNet(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, pretrained_imagenet: bool = True,
                 transformer_num_layers: int = 2) -> None:
        super().__init__()
        self.encoder = _ResNet50Encoder(pretrained_imagenet=pretrained_imagenet)
        self.decoder = _Decoder(transformer_num_layers=transformer_num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.decoder(feats)


__all__ = ["TranSalNet", "TransEncoder"]
