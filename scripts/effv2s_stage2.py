#!/usr/bin/env python
"""
BirdCLEF 2026 — EfficientNetV2-S + Self-Distillation (Stage 2)
========================================================================
+ Online pseudo-labeling using Stage 1 model as teacher
+ Multi-round self-distillation: run repeatedly for sd_r1 → sd_r2 → ...
+ Teacher auto-detected from Stage 1 (or previous SD round) output

LB: 0.928
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from birdclef.utils import set_random_seed, resolve_teacher_for_stage2, get_stage2_output_dir
from birdclef.data.utils import build_combined_df, assign_folds
from birdclef.training.train_fold import train_fold


# ── Config ─────────────────────────────────────
class CFG:
    seed = 1086
    n_folds = 4
    n_classes = 234
    max_epoch = 30
    warmup_epoch = 5
    batch_size = 64
    lr = 4e-4
    weight_decay = 1e-4
    use_amp = True
    num_workers = 16

    model_name = "tf_efficientnetv2_s.in21k"
    pretrained = False  # Uses custom XLS pretrain below
    pretrain_ckpt = "effv2s_10s/pretrained/tf_efficientnetv2_s_in21k_pretrain_from_bigXCV2Ext_swa.ckpt"
    drop_path_rate = 0.15
    drop_rate = 0.2
    head_dropout = 0.35

    sr = 32_000
    duration = 5

    mel_spectrogram_params = dict(
        sample_rate=32_000, n_fft=2048, hop_length=512,
        f_min=20, f_max=16_000, n_mels=128, normalized=True,
    )
    top_db = 80.0

    mixup_alpha = 1.0
    mixup_theta = 0.8
    rating_threshold = 2.5

    filt_aug_db_range = (-6, 6)
    filt_aug_n_band = (3, 6)
    filt_aug_min_bw = 6

    aug_flip_p = 0.5
    aug_dropout_frac = (0.375, 0.375)
    aug_dropout_holes = 1
    aug_dropout_p = 0.7

    perch_embed_dim = 1536
    perch_hf_repo = "justinchuby/Perch-onnx"
    perch_hf_file = "perch_v2_no_dft.onnx"
    alpha_distill = 0.1

    data_dir = Path("birdc")

    # Self-distillation params (teacher_path set automatically, or override here)
    teacher_path = None  # Auto-detect; set manually if needed
    sd_alpha = 0.5
    sd_threshold = 0.4
    sd_power = 2.0


# ── Main ───────────────────────────────────────
if __name__ == "__main__":
    cfg = CFG()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_random_seed(cfg.seed)

    # Auto-detect teacher
    if cfg.teacher_path is None:
        cfg.teacher_path = resolve_teacher_for_stage2("effv2s", cfg.seed)
    if cfg.teacher_path is None:
        sys.exit(1)  # Error message already printed

    # Determine SD round
    teacher_dir = Path(cfg.teacher_path).parent
    teacher_parent_name = teacher_dir.parent.name if teacher_dir.name.startswith("sd_r") else teacher_dir.name
    if "sd_r" in teacher_dir.name:
        sd_round = int(teacher_dir.name.replace("sd_r", "")) + 1
    elif "stage1" in teacher_parent_name or "stage1" in str(cfg.teacher_path):
        sd_round = 1
    else:
        sd_round = 1

    output_dir = get_stage2_output_dir("effv2s", cfg.seed, sd_round)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Self-Distillation Round {sd_round}")
    print(f"Teacher: {cfg.teacher_path}")
    print(f"Output:  {output_dir}")
    print(f"Alpha:   {cfg.sd_alpha}, Threshold: {cfg.sd_threshold}, Power: {cfg.sd_power}")
    print(f"{'='*50}\n")

    taxonomy = pd.read_csv(cfg.data_dir / "taxonomy.csv")
    df, classes, labels_arr = build_combined_df(cfg.data_dir, taxonomy)
    print(f"Total: {len(df)} (focal={(df['source']=='focal').sum()}, sc={(df['source']=='soundscape').sum()})")

    df = assign_folds(df, labels_arr, cfg.n_folds, cfg.seed)

    train_fold(df, labels_arr, fold_id=0, cfg=cfg, device=device, output_dir=output_dir)
