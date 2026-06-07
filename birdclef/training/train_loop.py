"""
Training and validation loops.

Supports:
  - AMP (automatic mixed precision)
  - Perch embedding distillation
  - Self-distillation with online pseudo-labels (stage 2/3)
  - EMA (effb3)
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from birdclef.utils import compute_distill_loss


def train_one_epoch(
    model, loader, lms_transform, mixup, optimizer, scheduler, scaler,
    device, perch=None, cfg=None,
    spec_aug_fn=None, spec_aug_torch=None,
    teacher=None,
    ema=None,
    loss_weights=(0.5, 0.5),
    alpha_distill=0.1,
    distill_loss_type="cosine",
    sd_alpha=0.5, sd_threshold=0.4, sd_power=2.0,
):
    """
    Train for one epoch.

    Args:
        model: BirdModel
        loader: DataLoader
        lms_transform: LogMel transform module
        mixup: MixUp module (None to disable)
        optimizer, scheduler, scaler: training utilities
        device: torch device
        perch: PerchTeacher for embedding distillation (None to disable)
        cfg: CFG object (for backward compat; not directly used)
        spec_aug_fn: albumentations-based specaug function
        spec_aug_torch: torchaudio SpecAugment module (HGNet variant)
        teacher: teacher model for self-distillation (None to disable)
        ema: ModelEMA instance (None to disable)
        loss_weights: (clip_weight, frame_weight) for BCE loss
        alpha_distill: weight for Perch distillation loss
        distill_loss_type: "cosine", "mse", or "norm_mse"
        sd_alpha, sd_threshold, sd_power: self-distillation params

    Returns:
        tuple of (total_loss, cls_loss, dist_loss, clip_loss, frame_loss) averages
    """
    model.train()
    lms_transform.train()
    bce = nn.BCEWithLogitsLoss()

    total_loss = total_cls = total_dist = total_clip = total_frame = 0.0
    n_batches = 0
    use_distill = perch is not None
    use_pseudo = teacher is not None

    pbar = tqdm(loader, desc="Train", leave=False)

    for wave, label in pbar:
        # Perch embedding (before GPU transfer for ONNX efficiency)
        if use_distill:
            with torch.no_grad():
                perch_emb = perch.embed(wave).to(device, non_blocking=True)
        else:
            perch_emb = None

        wave = wave.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        lms = lms_transform(wave)

        # --- Self-distillation pseudo-labels ---
        if use_pseudo:
            with torch.no_grad():
                pseudo = torch.sigmoid(teacher(lms)[0])
                pseudo = pseudo * (pseudo > sd_threshold) + pseudo ** sd_power
                pseudo = torch.clamp(pseudo, 0.0, 1.0)
                label = sd_alpha * pseudo + (1 - sd_alpha) * label

        # --- MixUp ---
        if mixup is not None:
            lms, label, lam, idx = mixup(lms, label)
            target_emb = (
                lam[:, None] * perch_emb + (1 - lam[:, None]) * perch_emb[idx]
            ) if use_distill else None
        else:
            target_emb = perch_emb if use_distill else None

        # --- SpecAugment (after MixUp) ---
        if spec_aug_fn is not None:
            from birdclef.augmentations import apply_spec_aug
            lms = apply_spec_aug(lms, spec_aug_fn)

        if spec_aug_torch is not None:
            lms = spec_aug_torch(lms)

        # --- Forward ---
        with torch.autocast(
            device_type=device.type,
            enabled=(scaler.is_enabled() and device.type == "cuda")
        ):
            if use_distill:
                clipwise, framewise, distill_emb = model(lms, return_distill=True)
            else:
                clipwise, framewise = model(lms, return_distill=False)
                distill_emb = None

            # Classification loss
            clip_w, frame_w = loss_weights
            loss_clip = bce(clipwise, label)

            # Handle framewise: for HGNet aux_pool mode (framewise may be None)
            if framewise is not None:
                if framewise.dim() == 2:
                    # HGNet aux_pool mode: framewise is already pooled to (B, n_classes)
                    loss_frame = bce(framewise, label)
                else:
                    # Standard: (B, n_classes, T) -> max_pool over time
                    loss_frame = bce(framewise.max(dim=2)[0], label)
                cls_loss = clip_w * loss_clip + frame_w * loss_frame
            else:
                loss_frame = torch.tensor(0.0, device=device)
                cls_loss = loss_clip

            # Distillation loss
            if use_distill:
                dist_loss = compute_distill_loss(distill_emb, target_emb, loss_type=distill_loss_type)
                loss = cls_loss + alpha_distill * dist_loss
            else:
                dist_loss = torch.tensor(0.0, device=device)
                loss = cls_loss

        # --- Backward ---
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_dist += dist_loss.item()
        total_clip += loss_clip.item()
        total_frame += loss_frame.item()
        n_batches += 1

        pbar.set_postfix(
            loss=f"{total_loss / n_batches:.4f}",
            cls=f"{total_cls / n_batches:.4f}",
            dist=f"{total_dist / n_batches:.4f}",
        )

    n = max(n_batches, 1)
    return (total_loss / n, total_cls / n, total_dist / n,
            total_clip / n, total_frame / n)


@torch.no_grad()
def validate(model, loader, lms_transform, scaler, device):
    """
    Run validation.

    Returns:
        (logits_np, labels_np) as numpy arrays
    """
    model.eval()
    lms_transform.eval()
    all_logits, all_labels = [], []

    for wave, label in tqdm(loader, desc="Val", leave=False):
        wave = wave.to(device, non_blocking=True)
        lms = lms_transform(wave)

        with torch.autocast(
            device_type=device.type,
            enabled=(scaler.is_enabled() and device.type == "cuda")
        ):
            clipwise, _ = model(lms)

        all_logits.append(clipwise.cpu())
        all_labels.append(label)

    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()
