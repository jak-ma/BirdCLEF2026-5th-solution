#!/usr/bin/env python
"""
BirdCLEF 2026 — EfficientNet-B3 Stage 2: Self-Distillation + Unlabeled SC
============================================================================
+ Online ImageNet pretrained weights (timm auto-download)
+ Online pseudo-labeling (Stage 1 EMA model as teacher, auto-detected)
+ Unlabeled soundscapes with full-random cropping
+ EMA continued

LB: 0.920
"""

import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from birdclef.utils import set_random_seed, resolve_teacher_for_b3_stage2, get_stage1_output_dir
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
    lr = 1e-3
    weight_decay = 1e-4
    use_amp = True
    num_workers = 16

    model_name = "tf_efficientnet_b3.ns_jft_in1k"
    pretrained = True  # Online ImageNet weights via timm
    drop_path_rate = 0.05
    drop_rate = 0.1
    head_dropout = 0.2

    sr = 32_000
    duration = 5

    mel_spectrogram_params = dict(
        sample_rate=32_000, n_fft=3072, hop_length=420,
        f_min=0, f_max=16_000, n_mels=384, normalized=True,
    )
    lms_shape = (384, 384)
    top_db = 80.0

    mixup_alpha = 1.0
    mixup_theta = 0.8
    rating_threshold = 2.5

    use_filt_aug = True
    filt_aug_db_range = (-6, 6)
    filt_aug_n_band = (3, 6)
    filt_aug_min_bw = 6

    use_spec_aug = False
    time_mask_param = 40
    freq_mask_param = 20
    n_specaug = 2

    use_perch_distill = True
    perch_embed_dim = 1536
    perch_hf_repo = "justinchuby/Perch-onnx"
    perch_hf_file = "perch_v2_no_dft.onnx"
    alpha_distill = 0.1
    distill_loss_type = "cosine"
    detach_cls_branch = False

    ema_decay = 0.999

    # Self-distillation (teacher auto-detected from Stage 1)
    teacher_path = None
    sd_alpha = 0.5
    sd_threshold = 0.4
    sd_power = 2.0

    # Unlabeled soundscapes
    use_unlabeled_sc = True
    unlabeled_chunks_per_file = 2
    unlabeled_use_full_random = True

    data_dir = Path("birdc")


# ── Main ───────────────────────────────────────
if __name__ == "__main__":
    cfg = CFG()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_random_seed(cfg.seed)

    # Auto-detect teacher from Stage 1
    if cfg.teacher_path is None:
        cfg.teacher_path = resolve_teacher_for_b3_stage2(cfg.seed)
    if cfg.teacher_path is None:
        sys.exit(1)

    suffix = "_random" if cfg.unlabeled_use_full_random else "_fixed"
    output_dir = get_stage1_output_dir("effb3", cfg.seed).parent / f"effb3_stage2_K{cfg.unlabeled_chunks_per_file}{suffix}_seed{cfg.seed}"

    print("\n" + "=" * 60)
    print("  B3 Stage 2: Self-Distillation + Unlabeled SC")
    print("=" * 60)
    print(f"  Teacher:  {cfg.teacher_path}")
    print(f"  Output:   {output_dir}")
    print("=" * 60 + "\n")

    taxonomy = pd.read_csv(cfg.data_dir / "taxonomy.csv")
    df, classes, labels_arr = build_combined_df(
        cfg.data_dir, taxonomy,
        use_unlabeled_sc=cfg.use_unlabeled_sc,
        unlabeled_chunks_per_file=cfg.unlabeled_chunks_per_file,
    )
    print(f"Total: {len(df)} (focal={(df['source']=='focal').sum()}, "
          f"soundscape={(df['source']=='soundscape').sum()})")

    df = assign_folds(df, labels_arr, cfg.n_folds, cfg.seed)

    train_fold(df, labels_arr, fold_id=0, cfg=cfg, device=device, output_dir=output_dir)
