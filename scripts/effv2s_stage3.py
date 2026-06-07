#!/usr/bin/env python
"""
BirdCLEF 2026 — EfficientNetV2-S Stage 3: Unlabeled Soundscapes + Full Random Crop
=====================================================================================
+ Online pseudo-labeling with Stage 2 teacher (auto-detected)
+ Unlabeled soundscapes with full-file random cropping
+ LB: ~0.935
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from birdclef.utils import set_random_seed, resolve_teacher_for_stage3, get_stage3_output_dir
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

    # Self-distillation params (teacher_path auto-detected from Stage 2)
    teacher_path = None
    sd_alpha = 0.5
    sd_threshold = 0.4
    sd_power = 2.0

    # Unlabeled soundscapes
    use_unlabeled_sc = True
    unlabeled_chunks_per_file = 2
    unlabeled_use_full_random = True


# ── Main ───────────────────────────────────────
if __name__ == "__main__":
    cfg = CFG()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_random_seed(cfg.seed)

    # Auto-detect teacher from Stage 2
    if cfg.teacher_path is None:
        cfg.teacher_path = resolve_teacher_for_stage3("effv2s", cfg.seed)
    if cfg.teacher_path is None:
        sys.exit(1)

    output_dir = get_stage3_output_dir("effv2s", cfg.seed, cfg.unlabeled_chunks_per_file, cfg.unlabeled_use_full_random)

    print("\n" + "=" * 60)
    print("  Stage 3: Online Pseudo-Labeling with Unlabeled Soundscapes")
    print("=" * 60)
    print(f"  Teacher:                    {cfg.teacher_path}")
    print(f"  Output:                     {output_dir}")
    print(f"  use_unlabeled_sc:           {cfg.use_unlabeled_sc}")
    print(f"  unlabeled_chunks_per_file:  {cfg.unlabeled_chunks_per_file}")
    print(f"  unlabeled_use_full_random:  {cfg.unlabeled_use_full_random}")
    print("=" * 60 + "\n")

    taxonomy = pd.read_csv(cfg.data_dir / "taxonomy.csv")
    df, classes, labels_arr = build_combined_df(
        cfg.data_dir, taxonomy,
        use_unlabeled_sc=cfg.use_unlabeled_sc,
        unlabeled_chunks_per_file=cfg.unlabeled_chunks_per_file,
    )

    n_focal = (df['source'] == 'focal').sum()
    n_sc = (df['source'] == 'soundscape').sum()
    print(f"\nTotal: {len(df)} (focal={n_focal}, soundscape={n_sc})")

    df = assign_folds(df, labels_arr, cfg.n_folds, cfg.seed)

    train_fold(df, labels_arr, fold_id=0, cfg=cfg, device=device, output_dir=output_dir)
