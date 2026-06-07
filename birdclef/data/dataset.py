"""
BirdDataset: loads focal clips and soundscape segments.

Supports four variants:
  1. Basic (effv2s): focal + soundscape with start_sec/end_sec, internal crop
  2. 10s (effv2s_10s): soundscape expands around center segment for 10s context
  3. HGNet: adds waveform noisy augmentation
  4. Stage 2/3 (effv2s + effb3): unlabeled soundscapes with full_random cropping
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import Dataset


class BirdDataset(Dataset):
    """
    Dataset for BirdCLEF 2026 training.

    Args:
        df: DataFrame with columns [audio_id, filename, primary_label, source,
            start_sec, end_sec, label_list]
        labels: (N, n_classes) multi-label array
        is_train: training mode (enables random cropping)
        sr: audio sample rate (default 32000)
        duration: target segment duration in seconds (default 5)
        data_dir: path to data root (containing train_audio/ and train_soundscapes/)
        use_10s_soundscape: 10s mode: expand around center 5s segment with ±2.5s context
        use_noisy_aug: enable waveform gain/noise augmentation (HGNet)
        noisy_aug_prob: probability per augmentation type
        noisy_gain_db_range: gain jitter range in dB
        noisy_snr_db_range: noise SNR range in dB
        unlabeled_use_full_random: for unlabeled SC, use full file random crop
    """

    def __init__(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        is_train: bool = True,
        sr: int = 32_000,
        duration: int = 5,
        data_dir: Path = Path("birdc"),
        use_10s_soundscape: bool = False,
        use_noisy_aug: bool = False,
        noisy_aug_prob: float = 0.5,
        noisy_gain_db_range: tuple = (-6.0, 6.0),
        noisy_snr_db_range: tuple = (3.0, 18.0),
        unlabeled_use_full_random: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.is_train = is_train
        self.sr = sr
        self.duration = duration
        self.audio_len = sr * duration
        self.data_dir = data_dir

        self.use_10s_soundscape = use_10s_soundscape
        self.use_noisy_aug = use_noisy_aug
        self.noisy_aug_prob = noisy_aug_prob
        self.noisy_gain_db_range = noisy_gain_db_range
        self.noisy_snr_db_range = noisy_snr_db_range
        self.unlabeled_use_full_random = unlabeled_use_full_random

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.labels[idx]

        if row["source"] == "focal":
            wav = self._load_focal(row)
        else:
            # Soundscape: detect unlabeled (label all zeros) for full_random mode
            is_unlabeled_sc = (
                self.unlabeled_use_full_random
                and self.is_train
                and label.sum() == 0
            )
            wav = self._load_soundscape(row, full_random=is_unlabeled_sc)

        # Optional waveform noisy augmentation (HGNet)
        if self.is_train and self.use_noisy_aug:
            wav_np = wav.numpy()
            wav_np = self._apply_noisy_aug(wav_np)
            wav = torch.from_numpy(wav_np)

        return wav, label.astype(np.float32)

    def _apply_noisy_aug(self, w):
        """Gain jitter + noise injection (HGNet style)."""
        if np.random.random() < self.noisy_aug_prob:
            w = w * (10 ** (np.random.uniform(*self.noisy_gain_db_range) / 20))
        if np.random.random() < self.noisy_aug_prob:
            sp = (w ** 2).mean()
            if sp > 1e-10:
                w = w + np.random.randn(*w.shape).astype(w.dtype) * np.sqrt(
                    sp / (10 ** (np.random.uniform(*self.noisy_snr_db_range) / 10)))
        return w

    def _load_focal(self, row) -> torch.Tensor:
        path = str(self.data_dir / "train_audio" / row["filename"])
        wav, _ = torchaudio.load(path)
        return self._crop(wav[0])

    def _load_soundscape(self, row, full_random=False) -> torch.Tensor:
        path = str(self.data_dir / "train_soundscapes" / row["filename"])
        wav, _ = torchaudio.load(path)
        wav = wav[0]

        if full_random:
            # Full file: _crop will randomly select a segment
            return self._crop(wav)

        if self.use_10s_soundscape:
            # 10s mode: original 5s segment centered, expanded with ±2.5s context
            center_start = int(row["start_sec"] * self.sr)
            center_end = int(row["end_sec"] * self.sr)
            ctx = int(2.5 * self.sr)
            start = max(0, center_start - ctx)
            end = min(wav.shape[0], center_end + ctx)
            wav = wav[start:end]
        else:
            # Standard mode: precise segment
            start_sample = int(row["start_sec"] * self.sr)
            end_sample = int(row["end_sec"] * self.sr)
            wav = wav[start_sample:end_sample]

        return self._crop(wav)

    def _crop(self, wav):
        if wav.shape[0] < self.audio_len:
            return F.pad(wav, (0, self.audio_len - wav.shape[0]))
        if self.is_train:
            start = random.randint(0, wav.shape[0] - self.audio_len)
            return wav[start:start + self.audio_len]
        return wav[:self.audio_len]
