#!/usr/bin/env python
"""
BirdCLEF 2026 — HGNetV2-B0 + SED Head (AttBlock + AuxPool) — Stage 1
=========================================================================
+ Online ImageNet (SSLD) pretrained weights via timm
+ ChannelAttention + FrequencySE
+ AttBlock with aux_pool (topk) — dual training output
+ Warmup + Cosine annealing scheduler
+ Rare species oversampling (min_samples=66)
+ Waveform noisy augmentation
+ TimeMasking-only SpecAugment (FreqMasking replaced by FrequencySE)
+ Perch v2 distillation

LB: ~0.925+
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
    seed = 520
    n_folds = 4
    n_classes = 234
    max_epoch = 35
    warmup_epoch = 5
    batch_size = 64
    lr = 8e-4
    weight_decay = 6e-4
    use_amp = True
    num_workers = 16

    model_name = "hgnetv2_b0.ssld_stage2_ft_in1k"
    pretrained = True  # Online ImageNet (SSLD) weights via timm
    drop_path_rate = 0.15
    drop_rate = 0.2
    head_dropout = 0.45

    sr = 32_000
    duration = 5

    n_mels = 128
    mel_spectrogram_params = dict(
        sample_rate=32_000, n_fft=2048, hop_length=512,
        f_min=20, f_max=16_000, n_mels=n_mels, normalized=True,
    )
    lms_shape = (n_mels, 313)
    top_db = 80.0

    mixup_alpha = 1.0
    mixup_theta = 0.8

    rating_threshold = 2.5
    data_dir = Path("birdc")

    perch_embed_dim = 1536
    perch_hf_repo = "justinchuby/Perch-onnx"
    perch_hf_file = "perch_v2_no_dft.onnx"
    alpha_distill = 0.1

    # Upsampling
    min_samples_per_class = 66
    add_ss_3x = False

    # Filter aug
    use_filt_aug = True
    filt_aug_db_range = (-6, 6)
    filt_aug_n_band = (3, 6)
    filt_aug_min_bw = 6

    # Waveform noisy aug
    use_noisy_aug = True
    AUG_PROB = 0.5
    AUG_GAIN_DB_RANGE = (-6.0, 6.0)
    AUG_NOISE_SNR_DB_RANGE = (3.0, 18.0)

    # HGNet-specific loss weights
    loss_weights = (0.6, 0.4)


# ── Main ───────────────────────────────────────
if __name__ == "__main__":
    cfg = CFG()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = get_stage1_output_dir("hgnet", cfg.seed)

    print(f"device: {device}")
    print(f"output: {output_dir}")
    print(f"pretrained: timm online ImageNet (SSLD)")

    print("Preparing dataframe")
    taxonomy = pd.read_csv(cfg.data_dir / "taxonomy.csv")
    df, classes, labels_arr = build_combined_df(
        cfg.data_dir, taxonomy,
        min_samples_per_class=cfg.min_samples_per_class,
    )
    print(f"Total: {len(df)} (focal={(df['source']=='focal').sum()}, "
          f"soundscape={(df['source']=='soundscape').sum()})")

    print("Assigning folds (focal only)...")
    df = assign_folds(df, labels_arr, cfg.n_folds, cfg.seed)

    for f in range(cfg.n_folds):
        ft = ((df["source"] == "focal") & (df["fold"] != f)).sum()
        fv = ((df["source"] == "focal") & (df["fold"] == f)).sum()
        sc = (df["source"] == "soundscape").sum()
        print(f"  fold {f}: train={ft+sc} (focal={ft}, sc={sc}), val={fv} (focal only)")

    train_fold(df, labels_arr, fold_id=0, cfg=cfg, device=device, output_dir=output_dir)
