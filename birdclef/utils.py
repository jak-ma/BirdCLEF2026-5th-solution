"""
Shared utilities: random seed, grad scaler, weight init, distill loss,
output path resolution, teacher auto-detection.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def set_random_seed(seed: int, deterministic: bool = True):
    """Set random seed for reproducibility across numpy, torch, and cuda."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def make_grad_scaler(device, enabled=True):
    """Create a GradScaler compatible with both old and new PyTorch APIs."""
    enabled = bool(enabled and device.type == "cuda")
    try:
        return torch.GradScaler(device.type, enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def init_layer(layer):
    """Xavier uniform init for Linear/Conv layers."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Init BatchNorm: bias=0, weight=1."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def compute_distill_loss(pred, target, loss_type="cosine"):
    """
    Distillation loss between student and teacher embeddings.

    Args:
        pred: student embedding (B, D)
        target: teacher embedding (B, D), will be detached
        loss_type: "cosine", "mse", or "norm_mse"

    Returns:
        scalar loss
    """
    target = target.detach()

    if loss_type == "cosine":
        pred = F.normalize(pred.float(), dim=1)
        target = F.normalize(target.float(), dim=1)
        return 1.0 - (pred * target).sum(dim=1).mean()

    elif loss_type == "mse":
        return F.mse_loss(pred.float(), target.float())

    elif loss_type == "norm_mse":
        pred = F.normalize(pred.float(), dim=1)
        target = F.normalize(target.float(), dim=1)
        return F.mse_loss(pred, target)

    else:
        raise ValueError(f"Unknown distill loss type: {loss_type}")


# ── Output path helpers ─────────────────────────

OUTPUT_ROOT = Path("outputs")


def get_stage1_output_dir(model: str, seed: int) -> Path:
    """Return the expected output directory for a stage 1 run."""
    return OUTPUT_ROOT / f"{model}_stage1_seed{seed}"


def get_stage2_output_dir(model: str, seed: int, sd_round: int = 1) -> Path:
    """Return the expected output directory for a stage 2 self-distill round."""
    base = OUTPUT_ROOT / f"{model}_stage2_seed{seed}"
    return base / f"sd_r{sd_round}"


def get_stage3_output_dir(model: str, seed: int, k: int = 2, full_random: bool = True) -> Path:
    """Return the expected output directory for a stage 3 run."""
    suffix = "_random" if full_random else "_fixed"
    return OUTPUT_ROOT / f"{model}_stage3_K{k}{suffix}_seed{seed}"


def get_10s_output_dir(model: str, seed: int) -> Path:
    """Return the expected output directory for a 10s run."""
    return OUTPUT_ROOT / f"{model}_10s_seed{seed}"


def resolve_teacher_for_stage2(model: str, seed: int) -> Path:
    """
    Auto-detect the Stage 1 teacher checkpoint for Stage 2.

    Looks for: outputs/{model}_stage1_seed{seed}/best_model_fold0.pt
    If the current stage2 already has previous rounds, returns the latest round's checkpoint.

    Returns Path if found, otherwise prints help and returns None.
    """
    stage1_ckpt = get_stage1_output_dir(model, seed) / "best_model_fold0.pt"
    if stage1_ckpt.exists():
        return stage1_ckpt

    # Also check if stage2 already has previous rounds
    stage2_base = OUTPUT_ROOT / f"{model}_stage2_seed{seed}"
    if stage2_base.exists():
        existing_rounds = sorted([
            d for d in stage2_base.iterdir()
            if d.is_dir() and d.name.startswith("sd_r")
        ], key=lambda d: int(d.name.replace("sd_r", "")))
        if existing_rounds:
            latest = existing_rounds[-1] / "best_model_fold0.pt"
            if latest.exists():
                return latest

    print(f"\n{'='*60}")
    print(f"  ERROR: Stage 1 checkpoint not found!")
    print(f"  Expected: {stage1_ckpt}")
    print(f"  Please run Stage 1 first:")
    print(f"    python scripts/{model}_stage1.py")
    print(f"{'='*60}\n")
    return None


def resolve_teacher_for_stage3(model: str, seed: int) -> Path:
    """
    Auto-detect the Stage 2 teacher checkpoint for Stage 3.

    Looks for the latest self-distillation round in:
    outputs/{model}_stage2_seed{seed}/sd_r{N}/best_model_fold0.pt

    If nothing found, falls back to Stage 1 checkpoint.

    Returns Path if found, otherwise prints help and returns None.
    """
    stage2_base = OUTPUT_ROOT / f"{model}_stage2_seed{seed}"

    if stage2_base.exists():
        existing_rounds = sorted([
            d for d in stage2_base.iterdir()
            if d.is_dir() and d.name.startswith("sd_r")
        ], key=lambda d: int(d.name.replace("sd_r", "")))
        for rdir in reversed(existing_rounds):
            ckpt = rdir / "best_model_fold0.pt"
            if ckpt.exists():
                return ckpt

    # Fallback: try stage1
    stage1_ckpt = get_stage1_output_dir(model, seed) / "best_model_fold0.pt"
    if stage1_ckpt.exists():
        return stage1_ckpt

    print(f"\n{'='*60}")
    print(f"  ERROR: Stage 2 checkpoint not found!")
    print(f"  Expected location: {stage2_base}/sd_r{{N}}/best_model_fold0.pt")
    print(f"  Please run Stage 2 first:")
    print(f"    python scripts/{model}_stage2.py")
    print(f"{'='*60}\n")
    return None


def resolve_teacher_for_b3_stage2(seed: int) -> Path:
    """
    Auto-detect the Eff-B3 Stage 1 teacher checkpoint for Stage 2.

    Looks for: outputs/effb3_stage1_seed{seed}/best_model_fold0.pt
    """
    stage1_ckpt = get_stage1_output_dir("effb3", seed) / "best_model_fold0.pt"
    if stage1_ckpt.exists():
        return stage1_ckpt

    print(f"\n{'='*60}")
    print(f"  ERROR: Eff-B3 Stage 1 checkpoint not found!")
    print(f"  Expected: {stage1_ckpt}")
    print(f"  Please run Stage 1 first:")
    print(f"    python scripts/effb3_stage1.py")
    print(f"{'='*60}\n")
    return None
