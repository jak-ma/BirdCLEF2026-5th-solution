"""
Audio & spectrogram augmentations: FilterAugment, MixUp, SpecAugment,
CoarseDropout + HorizontalFlip (albumentations), waveform noisy aug.
"""

import random

import numpy as np
import torch
import torchaudio.transforms as T
from torch import nn

# Optional: albumentations (only needed for specaug on effv2s-style models)
try:
    import albumentations as A
    _HAS_ALBUMENTATIONS = True
except ImportError:
    _HAS_ALBUMENTATIONS = False


# ── FilterAugment ──────────────────────────────

def filt_aug(features, db_range=(-6, 6), n_band=(3, 6), min_bw=6):
    """
    Apply per-band random gain in dB on the mel frequency axis.

    Args:
        features: (B, n_freq, n_time) mel spectrogram
        db_range: (min_db, max_db) gain range per band
        n_band: (min_bands, max_bands) number of frequency bands
        min_bw: minimum bandwidth per band (in mel bins)

    Returns:
        augmented features, same shape as input
    """
    B, n_freq, _ = features.shape
    n_bands = torch.randint(n_band[0], n_band[1], (1,)).item()
    if n_bands <= 1:
        return features

    while n_freq - n_bands * min_bw + 1 < 0:
        min_bw -= 1

    bndry = torch.sort(
        torch.randint(0, n_freq - n_bands * min_bw + 1, (n_bands - 1,))
    )[0] + torch.arange(1, n_bands) * min_bw
    bndry = torch.cat([torch.tensor([0]), bndry, torch.tensor([n_freq])])

    factors = torch.rand((B, n_bands + 1), device=features.device) * (db_range[1] - db_range[0]) + db_range[0]
    freq_filt = torch.ones((B, n_freq, 1), device=features.device)

    for i in range(n_bands):
        l, r = int(bndry[i].item()), int(bndry[i + 1].item())
        for j in range(B):
            freq_filt[j, l:r, :] = torch.linspace(
                factors[j, i].item(), factors[j, i + 1].item(), r - l,
                device=features.device,
            ).unsqueeze(-1)

    return features * (10 ** (freq_filt / 10))


# ── MixUp ──────────────────────────────────────

class MixUp(nn.Module):
    """Waveform-level mixup with union labels above theta threshold."""

    def __init__(self, alpha: float = 1.0, theta: float = 1.0):
        super().__init__()
        self.beta_dist = torch.distributions.Beta(alpha, alpha)
        self.theta = theta

    def forward(self, lms, label):
        """
        Args:
            lms: (B, C, H, W) mel spectrograms
            label: (B, n_classes) multi-label targets

        Returns:
            lms_mixed, label_mixed, lam, shuffle_idx
        """
        b = lms.shape[0]
        lam = self.beta_dist.sample((b,)).to(lms.device)
        lam = torch.maximum(lam, 1 - lam).float()
        idx = torch.randperm(b, device=lms.device)

        lms = lam[:, None, None, None] * lms + (1 - lam[:, None, None, None]) * lms[idx]
        label = lam[:, None] * label + (1 - lam[:, None]) * label[idx]
        label = label.clamp(max=1.0)
        label[label >= self.theta] = 1.0
        return lms, label, lam, idx


# ── Albumentations-based SpecAugment ────────────

def build_spec_aug(cfg):
    """
    Build albumentations pipeline for CoarseDropout + HorizontalFlip.

    cfg must have:
        mel_spectrogram_params["n_mels"], sr, duration,
        aug_flip_p, aug_dropout_frac, aug_dropout_holes, aug_dropout_p

    For 10s version, cfg may also have aug_dropout_holes as a range instead of fixed.
    """
    if not _HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for build_spec_aug. Install with: pip install albumentations")

    n_mels = cfg.mel_spectrogram_params["n_mels"]
    hop = cfg.mel_spectrogram_params.get("hop_length", 512)
    n_time = int(cfg.sr * cfg.duration / hop) + 1
    max_h = int(n_mels * cfg.aug_dropout_frac[0])
    max_w = int(n_time * cfg.aug_dropout_frac[1])

    # Support both fixed max_holes and range (10s variant)
    num_holes = cfg.aug_dropout_holes
    if isinstance(num_holes, int):
        return A.Compose([
            A.HorizontalFlip(p=cfg.aug_flip_p),
            A.CoarseDropout(
                max_height=max_h,
                max_width=max_w,
                max_holes=num_holes,
                fill_value=0.0,
                p=cfg.aug_dropout_p,
            ),
        ])
    else:
        # num_holes is a tuple (min, max) for range-based (10s variant uses num_holes_range)
        return A.Compose([
            A.HorizontalFlip(p=cfg.aug_flip_p),
            A.CoarseDropout(
                num_holes_range=(1, num_holes),
                hole_height_range=(1, max_h),
                hole_width_range=(1, max_w),
                fill=0.0,
                p=cfg.aug_dropout_p,
            ),
        ])


def apply_spec_aug(lms, aug_fn):
    """
    Apply albumentations augmentation to a batch of spectrograms.

    Args:
        lms: (B, 1, H, W) tensor (on GPU or CPU)
        aug_fn: albumentations Compose function

    Returns:
        (B, 1, H, W) tensor, same device/dtype as input
    """
    device = lms.device
    dtype = lms.dtype
    B = lms.shape[0]

    imgs = lms.squeeze(1).cpu().numpy()
    out = np.empty_like(imgs)
    for i in range(B):
        augmented = aug_fn(image=imgs[i])["image"]
        out[i] = augmented

    return torch.from_numpy(out).unsqueeze(1).to(device=device, dtype=dtype)


# ── Torchaudio SpecAugment (HGNet variant) ─────

class SpecAugment(nn.Module):
    """
    TimeMasking-only SpecAugment (FrequencyMasking replaced by FrequencySE).

    Used by HGNet branch.
    """

    def __init__(self, freq_mask_param=6, time_mask_param=10,
                 n_freq_masks=2, n_time_masks=3):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, x):
        # Frequency masking is disabled (replaced by FrequencySE)
        # for _ in range(self.n_freq_masks):
        #     x = self.freq_mask(x)

        for _ in range(self.n_time_masks):
            x = self.time_mask(x)
        return x


# ── Waveform Noisy Augmentation (HGNet variant) ─

def apply_noisy_aug(w, aug_prob=0.5, gain_db_range=(-6.0, 6.0), noise_snr_db_range=(3.0, 18.0)):
    """
    Simple waveform augmentation: gain jitter + noise injection.

    Args:
        w: numpy array (1D waveform)
        aug_prob: probability of applying each augmentation
        gain_db_range: (min, max) gain in dB
        noise_snr_db_range: (min, max) SNR in dB for noise injection

    Returns:
        augmented waveform (numpy array, same shape)
    """
    if np.random.random() < aug_prob:
        w = w * (10 ** (np.random.uniform(*gain_db_range) / 20))
    if np.random.random() < aug_prob:
        sp = (w ** 2).mean()
        if sp > 1e-10:
            w = w + np.random.randn(*w.shape).astype(w.dtype) * np.sqrt(
                sp / (10 ** (np.random.uniform(*noise_snr_db_range) / 10)))
    return w
