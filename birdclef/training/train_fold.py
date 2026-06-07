"""
train_fold: single-fold training orchestration.

Handles all the variant-specific setup:
  - effv2s: OneCycleLR, albumentations specaug, no EMA
  - effv2s stage2/3: teacher loading, self-distillation
  - effb3: OneCycleLR, EMA, Perch sub-batched
  - hgnet: Warmup+Cosine scheduler, torchaudio SpecAugment, gradient clipping
"""

import gc
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from huggingface_hub import hf_hub_download

from birdclef.utils import set_random_seed, make_grad_scaler
from birdclef.models.bird_model import BirdModel
from birdclef.models.ema import ModelEMA
from birdclef.training.train_loop import train_one_epoch, validate


def train_fold(
    df, labels_arr, fold_id, cfg, device, output_dir,
):
    """
    Train a single cross-validation fold.

    cfg must contain all the parameters needed (see individual branch CFG classes).
    The function auto-detects which training mode to use based on cfg attributes:
      - hasattr(cfg, 'ema_decay'): use EMA (effb3)
      - cfg.teacher_path is not None: load teacher for self-distillation
      - 'use_hgnet' in cfg: HGNet-specific settings
      - 'pretrain_ckpt' in cfg: load pretrained backbone

    Args:
        df: combined DataFrame with 'fold' column
        labels_arr: (N, n_classes) label matrix
        fold_id: which fold to use as validation
        cfg: CFG object
        device: torch device
        output_dir: path to save checkpoints and logs
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    set_random_seed(cfg.seed)

    # --- Detect training modes ---
    is_hgnet = hasattr(cfg, 'model_name') and 'hgnet' in cfg.model_name.lower()
    has_ema = hasattr(cfg, 'ema_decay')
    has_teacher = hasattr(cfg, 'teacher_path') and cfg.teacher_path is not None
    has_pretrain = hasattr(cfg, 'pretrain_ckpt') and cfg.pretrain_ckpt
    is_10s = cfg.duration == 10

    # --- Build loaders ---
    focal_trn = (df["source"] == "focal") & (df["fold"] != fold_id)
    focal_val = (df["source"] == "focal") & (df["fold"] == fold_id)
    trn_mask = focal_trn | (df["source"] == "soundscape")

    # Print training set composition
    n_focal = focal_trn.sum()
    n_sc_total = (df["source"] == "soundscape").sum()
    n_unl = ((df["source"] == "soundscape") & (df["label_list"].apply(len) == 0)).sum()
    n_labeled_sc = n_sc_total - n_unl
    print(f"\n[Fold {fold_id} training set composition]")
    print(f"  focal:        {n_focal:>7d}")
    if n_labeled_sc > 0 or n_unl > 0:
        print(f"  labeled_sc:   {n_labeled_sc:>7d}")
        print(f"  unlabeled_sc: {n_unl:>7d}")
    print(f"  total train:  {n_focal + n_sc_total:>7d}")
    print(f"  val (focal):  {focal_val.sum():>7d}")

    # --- Build Dataset ---
    from birdclef.data.dataset import BirdDataset

    ds_kwargs = dict(
        sr=cfg.sr, duration=cfg.duration, data_dir=cfg.data_dir,
    )
    if is_hgnet:
        ds_kwargs.update(
            use_noisy_aug=cfg.use_noisy_aug,
            noisy_aug_prob=cfg.AUG_PROB,
            noisy_gain_db_range=cfg.AUG_GAIN_DB_RANGE,
            noisy_snr_db_range=cfg.AUG_NOISE_SNR_DB_RANGE,
        )
    if is_10s:
        ds_kwargs['use_10s_soundscape'] = True
    if hasattr(cfg, 'unlabeled_use_full_random'):
        ds_kwargs['unlabeled_use_full_random'] = cfg.unlabeled_use_full_random

    trn_loader = DataLoader(
        BirdDataset(df[trn_mask], labels_arr[trn_mask.values], is_train=True, **ds_kwargs),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        BirdDataset(df[focal_val], labels_arr[focal_val.values], is_train=False, **ds_kwargs),
        batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    print(f"train batches: {len(trn_loader)}, val batches: {len(val_loader)}")

    # --- LogMel transform ---
    is_10s_sc = hasattr(cfg, 'mel_spectrogram_params') and cfg.mel_spectrogram_params.get('n_mels') == 128 and cfg.duration == 10

    if hasattr(cfg, 'lms_shape'):
        # effb3 / hgnet: Resize-based transform
        from birdclef.data.transforms import LogMelSpectrogramTransform
        lms_transform = LogMelSpectrogramTransform(
            cfg.mel_spectrogram_params, cfg.top_db, cfg.lms_shape,
            use_filt_aug=getattr(cfg, 'use_filt_aug', True),
            filt_aug_db_range=cfg.filt_aug_db_range,
            filt_aug_n_band=cfg.filt_aug_n_band,
            filt_aug_min_bw=cfg.filt_aug_min_bw,
            use_spec_aug=getattr(cfg, 'use_spec_aug', False),
            time_mask_param=getattr(cfg, 'time_mask_param', 40),
            freq_mask_param=getattr(cfg, 'freq_mask_param', 20),
            n_specaug=getattr(cfg, 'n_specaug', 2),
        ).to(device)
    else:
        # effv2s: No-resize transform
        from birdclef.data.transforms import LogMelTransform
        lms_transform = LogMelTransform(
            cfg.mel_spectrogram_params, cfg.top_db,
            filt_aug_db_range=cfg.filt_aug_db_range,
            filt_aug_n_band=cfg.filt_aug_n_band,
            filt_aug_min_bw=cfg.filt_aug_min_bw,
        ).to(device)

    # --- MixUp ---
    from birdclef.augmentations import MixUp
    mixup = MixUp(alpha=getattr(cfg, 'mixup_alpha', 1.0),
                   theta=getattr(cfg, 'mixup_theta', 0.8))

    # --- SpecAugment ---
    spec_aug_fn = None
    spec_aug_torch = None
    if hasattr(cfg, 'aug_flip_p') and hasattr(cfg, 'aug_dropout_p'):
        # Albumentations-based specaug (effv2s)
        from birdclef.augmentations import build_spec_aug
        spec_aug_fn = build_spec_aug(cfg)
    elif is_hgnet:
        # Torchaudio SpecAugment (HGNet)
        from birdclef.augmentations import SpecAugment
        spec_aug_torch = SpecAugment().to(device)

    # --- Model ---
    n_mels_val = getattr(cfg, 'n_mels', cfg.mel_spectrogram_params["n_mels"])
    pretrain_ckpt = getattr(cfg, 'pretrain_ckpt', None)

    model = BirdModel(
        model_name=cfg.model_name,
        pretrained=cfg.pretrained,
        pretrain_ckpt=pretrain_ckpt,
        drop_path_rate=cfg.drop_path_rate,
        drop_rate=cfg.drop_rate,
        num_classes=cfg.n_classes,
        head_dropout=cfg.head_dropout,
        n_mels=n_mels_val,
        use_hgnet=is_hgnet,
        use_effb3=has_ema,
        use_distill=getattr(cfg, 'use_perch_distill', True),
        embed_dim=getattr(cfg, 'perch_embed_dim', 1536),
        detach_cls_branch=getattr(cfg, 'detach_cls_branch', False),
    ).to(device)

    print(f"backbone_dim: {model.backbone_dim}, GeM p: {model.gem.p.item():.2f}")
    if is_hgnet:
        print(f"att_block aux_pool: '{model.att_block.aux_pool}'")

    # --- Sanity check mel shape ---
    with torch.no_grad():
        dummy = torch.randn(2, cfg.sr * cfg.duration).to(device)
        mel_out = lms_transform(dummy)
        print(f"[Sanity] mel output shape: {mel_out.shape}")
        del dummy, mel_out

    # --- EMA ---
    ema = None
    if has_ema:
        ema = ModelEMA(model, decay=cfg.ema_decay)

    # --- Teacher (self-distillation) ---
    teacher = None
    if has_teacher:
        if not Path(cfg.teacher_path).exists():
            raise FileNotFoundError(f"[Teacher] {cfg.teacher_path} not found!")

        teacher = BirdModel(
            model_name=cfg.model_name,
            pretrained=False,
            pretrain_ckpt=None,
            drop_path_rate=cfg.drop_path_rate,
            drop_rate=cfg.drop_rate,
            num_classes=cfg.n_classes,
            head_dropout=cfg.head_dropout,
            n_mels=n_mels_val,
            use_hgnet=is_hgnet,
            use_effb3=has_ema,
            use_distill=getattr(cfg, 'use_perch_distill', True),
            embed_dim=getattr(cfg, 'perch_embed_dim', 1536),
            detach_cls_branch=getattr(cfg, 'detach_cls_branch', False),
        ).to(device)

        state = torch.load(cfg.teacher_path, map_location=device)
        missing, unexpected = teacher.load_state_dict(state, strict=False)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"[Teacher] loaded from {cfg.teacher_path}")
        print(f"[Teacher] missing={len(missing)}, unexpected={len(unexpected)}")

    # --- Perch ONNX teacher ---
    perch = None
    use_perch_distill = getattr(cfg, 'use_perch_distill', hasattr(cfg, 'perch_hf_repo'))
    if use_perch_distill:
        onnx_path = hf_hub_download(repo_id=cfg.perch_hf_repo, filename=cfg.perch_hf_file)
        from birdclef.perch_teacher import PerchTeacher
        perch_kwargs = dict(
            onnx_path=onnx_path,
            prefer_cuda=torch.cuda.is_available(),
            sr=cfg.sr, duration=cfg.duration,
            perch_input_sec=getattr(cfg, 'perch_input_sec', cfg.duration),
            perch_embed_dim=cfg.perch_embed_dim,
        )
        if has_ema:
            perch_kwargs['sub_batch_size'] = 16
        perch = PerchTeacher(**perch_kwargs)

    # --- Optimizer ---
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # --- Scheduler ---
    if is_hgnet:
        warmup_steps = cfg.warmup_epoch * len(trn_loader)
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                          total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer,
                                   T_max=cfg.max_epoch * len(trn_loader) - warmup_steps,
                                   eta_min=8e-5)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                 milestones=[warmup_steps])
    else:
        scheduler = OneCycleLR(
            optimizer, max_lr=cfg.lr, epochs=cfg.max_epoch,
            steps_per_epoch=len(trn_loader),
            pct_start=cfg.warmup_epoch / cfg.max_epoch,
            div_factor=25, final_div_factor=4.0,
        )

    scaler = make_grad_scaler(device, enabled=cfg.use_amp)

    # --- Training loop ---
    best_score = 0.0
    best_state = None
    results = []

    loss_weights = getattr(cfg, 'loss_weights', (0.5, 0.5))
    sd_alpha = getattr(cfg, 'sd_alpha', 0.5)
    sd_threshold = getattr(cfg, 'sd_threshold', 0.4)
    sd_power = getattr(cfg, 'sd_power', 2.0)
    distill_loss_type = getattr(cfg, 'distill_loss_type', 'cosine')
    grad_clip_norm = getattr(cfg, 'grad_clip_norm', None)  # HGNet seed2

    for epoch in range(cfg.max_epoch):
        t0 = time()

        # Warmup: no mixup
        cur_mixup = mixup if epoch >= cfg.warmup_epoch else None

        if is_hgnet:
            cur_spec_aug_torch = spec_aug_torch if epoch > cfg.warmup_epoch else None
            cur_spec_aug_fn = None
        else:
            cur_spec_aug_torch = None
            cur_spec_aug_fn = spec_aug_fn

        lms_transform.train()
        losses = train_one_epoch(
            model=model, loader=trn_loader, lms_transform=lms_transform,
            mixup=cur_mixup, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, device=device, perch=perch,
            spec_aug_fn=cur_spec_aug_fn, spec_aug_torch=cur_spec_aug_torch,
            teacher=teacher, ema=ema,
            loss_weights=loss_weights,
            alpha_distill=getattr(cfg, 'alpha_distill', 0.1),
            distill_loss_type=distill_loss_type,
            sd_alpha=sd_alpha, sd_threshold=sd_threshold, sd_power=sd_power,
        )

        # --- Gradient clipping after scaler.step (HGNet seed2) ---
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        # --- Validate ---
        lms_transform.eval()

        if ema is not None:
            # Validate with EMA weights
            real_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.shadow)

        logits, labels = validate(
            model=model, loader=val_loader,
            lms_transform=lms_transform, scaler=scaler, device=device,
        )

        if ema is not None:
            model.load_state_dict(real_state)

        val_loss = F.binary_cross_entropy_with_logits(
            torch.from_numpy(logits), torch.from_numpy(labels)
        ).item()
        mask = labels.sum(axis=0) > 0
        val_auc = roc_auc_score(labels[:, mask], logits[:, mask], average="macro") if mask.sum() > 0 else 0.0

        elapsed = time() - t0
        lr_now = scheduler.get_last_lr()[0]
        gem_p = model.gem.p.item()

        results.append([epoch, lr_now, *losses[:5], val_loss, val_auc, gem_p, elapsed])

        loss_str = f"loss={losses[0]:.4f}" if len(losses) >= 1 else ""
        cls_str = f"cls={losses[1]:.4f}" if len(losses) >= 2 else ""
        dist_str = f"dist={losses[2]:.4f}" if len(losses) >= 3 else ""
        print(
            f"[fold {fold_id} | ep {epoch:2d}] lr={lr_now:.6f} "
            f"{loss_str} {cls_str} {dist_str} "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.5f} "
            f"gem_p={gem_p:.3f} {elapsed:.0f}s"
        )

        if val_auc > best_score:
            best_score = val_auc
            if ema is not None:
                best_state = ema.state_dict()
            else:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, output_dir / f"best_model_fold{fold_id}.pt")
            print(f"  -> new best: {best_score:.5f}")

        # HGNet: save last 2 epochs
        if is_hgnet and epoch >= cfg.max_epoch - 2:
            torch.save(model.state_dict(), output_dir / f"last_model_fold{fold_id}_ep{epoch}.pt")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Save results ---
    cols = ["epoch", "lr", "loss", "cls_loss", "dist_loss", "clip_loss", "frame_loss",
            "val_loss", "val_auc", "gem_p", "elapsed"]
    pd.DataFrame(results, columns=cols[:len(results[0])]).to_csv(
        output_dir / f"result_fold{fold_id}.csv", index=False)
    print(f"[fold {fold_id}] best val_auc = {best_score:.5f}\n")
    return best_score
