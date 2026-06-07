#!/usr/bin/env python
"""
BirdCLEF 2026 — EfficientNetV2-S + SED (AttBlock) 5s — Stage 1
========================================================================
+ Dual loss (0.5 clip + 0.5 frame) + Perch v2 distillation
+ FilterAugment + GeM1d + MixUp + Rating filter
+ CoarseDropout + HorizontalFlip (spectrogram augmentation)
+ Mel params aligned to vladimirsydor 2024 XC pretrain (128 mels, no resize)
+ XLS pretrain (models_2025 V2 Ext SWA)

preprocess LB: 0.925
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from birdclef.utils import set_random_seed, get_stage1_output_dir
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


# ── Main ───────────────────────────────────────
if __name__ == "__main__":
    cfg = CFG()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = get_stage1_output_dir("effv2s", cfg.seed)

    print(f"device: {device}")
    print(f"output: {output_dir}")
    set_random_seed(cfg.seed)

    taxonomy = pd.read_csv(cfg.data_dir / "taxonomy.csv")
    df, classes, labels_arr = build_combined_df(cfg.data_dir, taxonomy)
    print(f"Total: {len(df)} (focal={(df['source']=='focal').sum()}, sc={(df['source']=='soundscape').sum()})")

    df = assign_folds(df, labels_arr, cfg.n_folds, cfg.seed)
    for f in range(cfg.n_folds):
        ft = ((df["source"] == "focal") & (df["fold"] != f)).sum()
        fv = ((df["source"] == "focal") & (df["fold"] == f)).sum()
        sc = (df["source"] == "soundscape").sum()
        print(f"  fold {f}: train={ft+sc} (focal={ft}, sc={sc}), val={fv}")

    train_fold(df, labels_arr, fold_id=0, cfg=cfg, device=device, output_dir=output_dir)
