#!/usr/bin/env python
"""
BirdCLEF 2026 — EfficientNet-B3 + SED Head — Stage 1
========================================================================
+ 384 mel bins + Resize(384, 384)
+ Online ImageNet pretrained weights (timm auto-download)
+ EMA (decay=0.999)
+ Perch distillation (cosine, sub_batched)
+ FilterAugment, SpecAugment disabled
+ OneCycleLR

LB: 0.918
"""

import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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

    data_dir = Path("birdc")


# ── Main ───────────────────────────────────────
if __name__ == "__main__":
    cfg = CFG()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = get_stage1_output_dir("effb3", cfg.seed)

    print(f"device: {device}")
    print(f"output: {output_dir}")
    print(f"pretrained: timm online ImageNet (model={cfg.model_name})")
    set_random_seed(cfg.seed)

    taxonomy = pd.read_csv(cfg.data_dir / "taxonomy.csv")
    df, classes, labels_arr = build_combined_df(cfg.data_dir, taxonomy)
    print(f"Total: {len(df)} (focal={(df['source']=='focal').sum()}, "
          f"soundscape={(df['source']=='soundscape').sum()})")

    df = assign_folds(df, labels_arr, cfg.n_folds, cfg.seed)
    for f in range(cfg.n_folds):
        ft = ((df["source"] == "focal") & (df["fold"] != f)).sum()
        fv = ((df["source"] == "focal") & (df["fold"] == f)).sum()
        sc = (df["source"] == "soundscape").sum()
        print(f"  fold {f}: train={ft+sc} (focal={ft}, sc={sc}), val={fv} (focal only)")

    train_fold(df, labels_arr, fold_id=0, cfg=cfg, device=device, output_dir=output_dir)
