"""
Reusable model components: GeM pooling, Attention Block, DistillHead,
ChannelAttention, FrequencySE.
"""

import torch
from torch import nn
from torch.nn import functional as F

from birdclef.utils import init_layer, init_bn


# ── GeM 1D Pooling ─────────────────────────────

class GeM1d(nn.Module):
    """
    Generalized Mean Pooling along time dimension with local kernel.

    p=1 → avg pooling, p→∞ → max pooling.
    p is learned during training.
    """

    def __init__(self, p: float = 3.0, kernel_size: int = 3, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = x.clamp(min=self.eps)
        return F.avg_pool1d(
            x.pow(self.p),
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        ).pow(1.0 / self.p)


# ── Attention Block (basic, effv2s / 10s) ──────

class AttBlock(nn.Module):
    """
    Basic attention-based SED head.
    clipwise = sum(softmax(att_logits) * frame_logits)

    Used by: effv2s, effv2s_10s branches.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.att = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (B, C, T)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        framewise = self.cla(x)
        clipwise = torch.sum(norm_att * framewise, dim=2)
        return clipwise, framewise


# ── AttBlock with BatchNorm (effb3) ────────────

class AttBlockBN(nn.Module):
    """
    Attention block with BatchNorm, used by effb3 branch.
    Also supports aux_pool for training-time auxiliary output.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.att = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        self.bn_att = nn.BatchNorm1d(out_features)
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        framewise = self.cla(x)
        clipwise = torch.sum(norm_att * framewise, dim=2)
        return clipwise, framewise


# ── AttBlock with aux_pool (HGNet) ─────────────

class AttBlockAux(nn.Module):
    """
    Enhanced attention block with auxiliary pooling for training.
    - Inference: returns (clipwise_att, None)
    - Training: returns (clipwise_att, clipwise_aux) where aux is topk or max

    Used by: hgnet branch.
    """

    def __init__(self, in_features: int, out_features: int,
                 aux_pool: str = 'topk', k: int = 3, r: float = 1.0):
        super().__init__()
        self.att = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        self.bn_att = nn.BatchNorm1d(out_features)
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

        self.aux_pool = aux_pool
        self.k = k
        self.r = r

    def forward(self, x):
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        framewise = self.cla(x)
        clipwise_att = torch.sum(norm_att * framewise, dim=2)

        if not self.training:
            return clipwise_att, None

        if self.aux_pool == 'topk':
            actual_k = min(self.k, framewise.size(-1))
            clipwise_aux = framewise.topk(actual_k, dim=2)[0].mean(dim=2)
        else:
            clipwise_aux = framewise.max(dim=2)[0]

        return clipwise_att, clipwise_aux


# ── Distill Head ───────────────────────────────

class DistillHead(nn.Module):
    """Project backbone feature map to Perch embedding space."""

    def __init__(self, backbone_dim: int, embed_dim: int = 1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
        init_layer(self.proj)

    def forward(self, feat_map):
        # feat_map: (B, C, H, W)
        gap = feat_map.mean(dim=[2, 3])  # (B, C)
        return self.proj(gap)


# ── Channel Attention (HGNet) ──────────────────

class ChannelAttention(nn.Module):
    """SE-like channel attention along feature dimension."""

    def __init__(self, dim: int = 2048, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, T)
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        out = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        return x * out


# ── Frequency SE (HGNet) ───────────────────────

class FrequencySE(nn.Module):
    """
    Squeeze-and-Excitation along frequency axis.
    Replaces FrequencyMasking in SpecAugment.
    """

    def __init__(self, n_mels: int = 128, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((n_mels, 1))
        self.fc = nn.Sequential(
            nn.Linear(n_mels, n_mels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_mels // reduction, n_mels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        y = self.avg_pool(x).squeeze(-1).squeeze(1)
        y = self.fc(y).view(b, 1, h, 1)
        return x * y.expand_as(x)
