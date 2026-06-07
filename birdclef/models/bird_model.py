"""
BirdModel: the main classification model with SED head.

Supports three architecture variants via flags:
  - use_hgnet=True: AttBlockAux, ChannelAttention, FrequencySE
  - use_effb3=True: AttBlockBN, optional detach_cls_branch
  - default (effv2s): basic AttBlock, simple forward
"""

from pathlib import Path

import torch
import timm
from torch import nn
from torch.nn import functional as F

from birdclef.utils import init_layer, init_bn
from birdclef.models.components import (
    GeM1d,
    AttBlock,
    AttBlockBN,
    AttBlockAux,
    DistillHead,
    ChannelAttention,
    FrequencySE,
)


class BirdModel(nn.Module):
    """
    BirdCLEF SED model with timm backbone + attention head + optional Perch distill head.

    Args:
        model_name: timm model name
        pretrained: use timm online pretrained weights
        pretrain_ckpt: path to custom pretrained checkpoint (None = skip)
        drop_path_rate, drop_rate: backbone stochastic depth / dropout
        num_classes: number of output classes (default 234)
        head_dropout: dropout rate in classification head
        n_mels: number of mel frequency bins
        use_hgnet: enable HGNet-specific modules (ChannelAttention, FrequencySE, AttBlockAux)
        use_effb3: enable effb3-specific modules (AttBlockBN, detach_cls_branch option)
        use_distill: create DistillHead for Perch embedding distillation
        embed_dim: Perch embedding dimension
        detach_cls_branch: stop gradient from distill head to cls branch (effb3 option)
    """

    def __init__(
        self,
        model_name,
        pretrained=True,
        pretrain_ckpt=None,
        drop_path_rate=0.0,
        drop_rate=0.2,
        num_classes=234,
        head_dropout=0.5,
        n_mels=128,
        use_hgnet=False,
        use_effb3=False,
        use_distill=True,
        embed_dim=1536,
        detach_cls_branch=False,
    ):
        super().__init__()

        self.use_hgnet = use_hgnet
        self.use_effb3 = use_effb3
        self.use_distill = use_distill
        self.detach_cls_branch = detach_cls_branch

        # --- BN0 ---
        self.bn0 = nn.BatchNorm2d(n_mels)
        init_bn(self.bn0)

        # --- Backbone ---
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            global_pool="",
            num_classes=0,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )

        # Load custom pretrain checkpoint (XLS pretrained weights for effv2s)
        if pretrain_ckpt is not None and pretrain_ckpt:
            ckpt_path = Path(pretrain_ckpt)
            if not ckpt_path.exists():
                print(f"[Pretrain] WARNING: checkpoint not found at {ckpt_path}, "
                      f"using timm pretrained={pretrained} instead")
            else:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                missing, unexpected = self.backbone.load_state_dict(ckpt, strict=False)
                print(f"[Pretrain] loaded from {ckpt_path}")
                print(f"[Pretrain] missing={len(missing)}, unexpected={len(unexpected)}")

        # --- Detect backbone output dim ---
        with torch.no_grad():
            self.backbone_dim = self.backbone(torch.randn(1, 1, n_mels, n_mels)).shape[1]

        # --- GeM pooling ---
        self.gem = GeM1d(p=3.0, kernel_size=3)

        # --- FC + AttBlock ---
        self.fc1 = nn.Linear(self.backbone_dim, self.backbone_dim, bias=True)
        init_layer(self.fc1)

        if use_hgnet:
            self.att_block = AttBlockAux(self.backbone_dim, num_classes)
        elif use_effb3:
            self.att_block = AttBlockBN(self.backbone_dim, num_classes)
        else:
            self.att_block = AttBlock(self.backbone_dim, num_classes)

        self.dropout = nn.Dropout(head_dropout)

        # --- Distill head ---
        if use_distill:
            self.distill_head = DistillHead(self.backbone_dim, embed_dim)

        # --- HGNet-specific modules (with random fork for deterministic init) ---
        if use_hgnet:
            with torch.random.fork_rng():
                torch.manual_seed(999)
                self.channel_att = ChannelAttention()
                self.freq_se = FrequencySE(n_mels=n_mels, reduction=16)

    def forward(self, x, return_distill=False):
        # --- BN0 + freq SE ---
        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)

        if self.use_hgnet:
            x = self.freq_se(x)

        feat_map = self.backbone(x)  # (B, C, H, W)

        # --- Distill embedding ---
        distill_emb = None
        if return_distill and self.use_distill:
            distill_emb = self.distill_head(feat_map)

        # --- Detach for cls branch (effb3 option) ---
        if self.use_distill and self.training and self.detach_cls_branch:
            feat = feat_map.detach()
        else:
            feat = feat_map

        # --- Feature processing ---
        feat = feat.mean(dim=2)  # (B, C, W)

        if self.use_hgnet:
            feat = self.channel_att(feat)

        feat = self.gem(feat)
        feat = self.dropout(feat)
        feat = feat.transpose(1, 2)
        feat = F.relu_(self.fc1(feat))
        feat = feat.transpose(1, 2)
        feat = self.dropout(feat)

        # --- Attention head ---
        clipwise, framewise = self.att_block(feat)

        if return_distill:
            return clipwise, framewise, distill_emb
        return clipwise, framewise
