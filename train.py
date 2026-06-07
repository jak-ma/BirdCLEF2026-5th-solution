#!/usr/bin/env python
"""
BirdCLEF 2026 — Unified training entry point.

Usage:
    python train.py --model effv2s --stage 1  --seed 1086
    python train.py --model hgnet               --seed 520
    python train.py --model effb3  --stage 2
    python train.py --model effv2s --stage 3
    python train.py --model effv2s --stage 10s

Auto-detects teacher checkpoint for Stage 2/3 from previous stage outputs.
"""

import argparse
import sys
from pathlib import Path

# Ensure the package root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch

from birdclef.utils import (
    set_random_seed,
    get_stage1_output_dir, get_stage2_output_dir, get_stage3_output_dir, get_10s_output_dir,
    resolve_teacher_for_stage2, resolve_teacher_for_stage3, resolve_teacher_for_b3_stage2,
)
from birdclef.data.utils import build_combined_df, assign_folds
from birdclef.training.train_fold import train_fold


# ── Config builder ──────────────────────────────

def build_cfg(model: str, stage: str, seed: int, teacher_path: str = None):
    """Build CFG object for the given model/stage/seed combination."""
    cfg = type('CFG', (), {})()

    # ── Common defaults ──
    cfg.seed = seed
    cfg.n_folds = 4
    cfg.n_classes = 234
    cfg.sr = 32_000
    cfg.top_db = 80.0
    cfg.mixup_alpha = 1.0
    cfg.mixup_theta = 0.8
    cfg.rating_threshold = 2.5
    cfg.data_dir = Path("birdc")
    cfg.perch_embed_dim = 1536
    cfg.perch_hf_repo = "justinchuby/Perch-onnx"
    cfg.perch_hf_file = "perch_v2_no_dft.onnx"
    cfg.use_amp = True
    cfg.num_workers = 16
    cfg.filt_aug_db_range = (-6, 6)
    cfg.filt_aug_n_band = (3, 6)
    cfg.filt_aug_min_bw = 6

    # ── Model-specific ──
    if model == "effv2s":
        cfg.model_name = "tf_efficientnetv2_s.in21k"
        cfg.pretrained = False
        cfg.pretrain_ckpt = "effv2s_10s/pretrained/tf_efficientnetv2_s_in21k_pretrain_from_bigXCV2Ext_swa.ckpt"
        cfg.drop_path_rate = 0.15
        cfg.drop_rate = 0.2
        cfg.head_dropout = 0.35
        cfg.duration = 5
        cfg.batch_size = 64
        cfg.lr = 4e-4
        cfg.weight_decay = 1e-4
        cfg.mel_spectrogram_params = dict(
            sample_rate=32_000, n_fft=2048, hop_length=512,
            f_min=20, f_max=16_000, n_mels=128, normalized=True,
        )
        cfg.alpha_distill = 0.1
        cfg.max_epoch = 30
        cfg.warmup_epoch = 5
        cfg.aug_flip_p = 0.5
        cfg.aug_dropout_frac = (0.375, 0.375)
        cfg.aug_dropout_p = 0.7

        if stage == "10s":
            cfg.duration = 10
            cfg.aug_dropout_frac = (0.2, 0.1)
            cfg.aug_dropout_holes = (1, 3)
            cfg.perch_input_sec = 5
        else:
            cfg.aug_dropout_holes = 1

        # Stage 2/3: self-distillation
        if stage in ("2", "3"):
            cfg.sd_alpha = 0.5
            cfg.sd_threshold = 0.4
            cfg.sd_power = 2.0
            if stage == "3":
                cfg.use_unlabeled_sc = True
                cfg.unlabeled_chunks_per_file = 2
                cfg.unlabeled_use_full_random = True

    elif model == "effb3":
        cfg.model_name = "tf_efficientnet_b3.ns_jft_in1k"
        cfg.pretrained = True  # Online ImageNet via timm
        cfg.drop_path_rate = 0.05
        cfg.drop_rate = 0.1
        cfg.head_dropout = 0.2
        cfg.duration = 5
        cfg.batch_size = 64
        cfg.lr = 1e-3
        cfg.weight_decay = 1e-4
        cfg.mel_spectrogram_params = dict(
            sample_rate=32_000, n_fft=3072, hop_length=420,
            f_min=0, f_max=16_000, n_mels=384, normalized=True,
        )
        cfg.lms_shape = (384, 384)
        cfg.use_filt_aug = True
        cfg.use_spec_aug = False
        cfg.use_perch_distill = True
        cfg.alpha_distill = 0.1
        cfg.distill_loss_type = "cosine"
        cfg.detach_cls_branch = False
        cfg.ema_decay = 0.999
        cfg.max_epoch = 30
        cfg.warmup_epoch = 5

        if stage == "2":
            cfg.sd_alpha = 0.5
            cfg.sd_threshold = 0.4
            cfg.sd_power = 2.0
            cfg.use_unlabeled_sc = True
            cfg.unlabeled_chunks_per_file = 2
            cfg.unlabeled_use_full_random = True

    elif model == "hgnet":
        cfg.model_name = "hgnetv2_b0.ssld_stage2_ft_in1k"
        cfg.pretrained = True  # Online ImageNet via timm
        cfg.drop_path_rate = 0.15
        cfg.drop_rate = 0.2
        cfg.head_dropout = 0.45
        cfg.duration = 5
        cfg.batch_size = 64
        cfg.lr = 8e-4
        cfg.weight_decay = 6e-4
        cfg.n_mels = 128
        cfg.mel_spectrogram_params = dict(
            sample_rate=32_000, n_fft=2048, hop_length=512,
            f_min=20, f_max=16_000, n_mels=128, normalized=True,
        )
        cfg.lms_shape = (128, 313)
        cfg.alpha_distill = 0.1
        cfg.max_epoch = 35
        cfg.warmup_epoch = 5
        cfg.use_filt_aug = True
        cfg.use_noisy_aug = True
        cfg.AUG_PROB = 0.5
        cfg.AUG_GAIN_DB_RANGE = (-6.0, 6.0)
        cfg.AUG_NOISE_SNR_DB_RANGE = (3.0, 18.0)
        cfg.min_samples_per_class = 66
        cfg.loss_weights = (0.6, 0.4)
        cfg.add_ss_3x = False

    else:
        raise ValueError(f"Unknown model: {model}")

    cfg.teacher_path = teacher_path
    return cfg


# ── Main ───────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Training")
    parser.add_argument("--model", type=str, required=True,
                        choices=["effv2s", "effb3", "hgnet"],
                        help="Model architecture")
    parser.add_argument("--stage", type=str, default="1",
                        choices=["1", "2", "3", "10s"],
                        help="Training stage")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: model-specific)")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold ID to train")
    parser.add_argument("--teacher_path", type=str, default=None,
                        help="Teacher checkpoint path (auto-detected if not set)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-generated if not set)")
    parser.add_argument("--data_dir", type=str, default="birdc",
                        help="Data root directory")
    args = parser.parse_args()

    # Set default seeds
    if args.seed is None:
        args.seed = 520 if args.model == "hgnet" else 1086

    # Build config
    cfg = build_cfg(args.model, args.stage, args.seed, args.teacher_path)
    cfg.data_dir = Path(args.data_dir)

    # Auto-detect teacher for Stage 2/3
    if cfg.teacher_path is None and args.stage in ("2", "3"):
        if args.model == "effb3":
            cfg.teacher_path = resolve_teacher_for_b3_stage2(args.seed)
        elif args.stage == "3":
            cfg.teacher_path = resolve_teacher_for_stage3(args.model, args.seed)
        else:
            cfg.teacher_path = resolve_teacher_for_stage2(args.model, args.seed)
        if cfg.teacher_path is None:
            sys.exit(1)

    # Auto-determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if args.stage == "1":
            output_dir = get_stage1_output_dir(args.model, args.seed)
        elif args.stage == "2":
            # For stage 2, determine round number from teacher path
            if cfg.teacher_path:
                teacher_dir = Path(cfg.teacher_path).parent
                try:
                    if "sd_r" in teacher_dir.name:
                        sd_round = int(teacher_dir.name.replace("sd_r", "")) + 1
                    elif "stage1" in str(cfg.teacher_path):
                        sd_round = 1
                    else:
                        sd_round = 1
                except ValueError:
                    sd_round = 1
            else:
                sd_round = 1
            output_dir = get_stage2_output_dir(args.model, args.seed, sd_round)
        elif args.stage == "3":
            k = getattr(cfg, 'unlabeled_chunks_per_file', 2)
            fr = getattr(cfg, 'unlabeled_use_full_random', True)
            output_dir = get_stage3_output_dir(args.model, args.seed, k, fr)
        else:  # 10s
            output_dir = get_10s_output_dir(args.model, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"model: {args.model}, stage: {args.stage}, seed: {cfg.seed}")
    print(f"output: {output_dir}")
    if cfg.teacher_path:
        print(f"teacher: {cfg.teacher_path}")

    set_random_seed(cfg.seed)

    # Build data
    taxonomy = pd.read_csv(cfg.data_dir / "taxonomy.csv")
    min_samples = getattr(cfg, 'min_samples_per_class', 0)
    use_unlabeled = getattr(cfg, 'use_unlabeled_sc', False)
    unl_chunks = getattr(cfg, 'unlabeled_chunks_per_file', 2)

    df, classes, labels_arr = build_combined_df(
        cfg.data_dir, taxonomy,
        use_unlabeled_sc=use_unlabeled,
        unlabeled_chunks_per_file=unl_chunks,
        min_samples_per_class=min_samples,
    )
    print(f"Total: {len(df)} (focal={(df['source']=='focal').sum()}, "
          f"soundscape={(df['source']=='soundscape').sum()})")

    df = assign_folds(df, labels_arr, cfg.n_folds, cfg.seed)
    for f in range(cfg.n_folds):
        ft = ((df["source"] == "focal") & (df["fold"] != f)).sum()
        fv = ((df["source"] == "focal") & (df["fold"] == f)).sum()
        sc = (df["source"] == "soundscape").sum()
        print(f"  fold {f}: train={ft+sc} (focal={ft}, sc={sc}), val={fv}")

    train_fold(df, labels_arr, fold_id=args.fold, cfg=cfg, device=device, output_dir=output_dir)
