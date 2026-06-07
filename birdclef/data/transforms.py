"""
LogMel transforms: two variants.

  1. LogMelTransform (effv2s / effv2s_10s):
     - No resize, directly uses MelSpectrogram output shape
     - FilterAugment applied in transform

  2. LogMelSpectrogramTransform (effb3 / hgnet):
     - Resize to fixed shape via torchvision Resize
     - Optional SpecAugment (TimeMasking + FrequencyMasking)
     - Optional FilterAugment
"""

import torch
import torchaudio
from torch import nn
from torchvision.transforms import v2 as tvt_v2

from birdclef.augmentations import filt_aug


class LogMelTransform(nn.Module):
    """
    Mel spectrogram transform WITHOUT resize (effv2s style).
    FilterAugment is applied inline during training.

    Args:
        mel_params: dict for torchaudio.transforms.MelSpectrogram
        top_db: amplitude-to-dB top_db parameter
        filt_aug_db_range: FilterAugment dB range (None = no filt_aug)
        filt_aug_n_band: FilterAugment band range
        filt_aug_min_bw: FilterAugment min bandwidth
    """

    def __init__(self, mel_params, top_db=80.0,
                 filt_aug_db_range=(-6, 6), filt_aug_n_band=(3, 6), filt_aug_min_bw=6):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(**mel_params)
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)
        self.top_db = top_db
        self.filt_aug_db_range = filt_aug_db_range
        self.filt_aug_n_band = filt_aug_n_band
        self.filt_aug_min_bw = filt_aug_min_bw

    def forward(self, wave):
        with torch.no_grad():
            lms = self.db(self.mel(wave))

        if self.training and self.filt_aug_db_range is not None:
            lms = filt_aug(lms, self.filt_aug_db_range,
                           self.filt_aug_n_band, self.filt_aug_min_bw)

        lms = torch.clamp((lms + self.top_db) / self.top_db, 0.0, 1.0)
        return lms[:, None, :, :]


class LogMelSpectrogramTransform(nn.Module):
    """
    Mel spectrogram transform WITH resize to fixed shape (effb3 / hgnet style).
    Optional FilterAugment and SpecAugment (TimeMasking + FrequencyMasking).

    Args:
        mel_params: dict for torchaudio.transforms.MelSpectrogram
        top_db: amplitude-to-dB top_db parameter
        lms_shape: (H, W) target size for Resize
        use_filt_aug: whether to apply FilterAugment
        filt_aug_db_range, filt_aug_n_band, filt_aug_min_bw: FilterAugment params
        use_spec_aug: whether to apply TimeMasking + FrequencyMasking
        time_mask_param, freq_mask_param, n_specaug: SpecAugment params
    """

    def __init__(
        self,
        mel_params,
        top_db=80.0,
        lms_shape=(128, 313),
        use_filt_aug=True,
        filt_aug_db_range=(-6, 6),
        filt_aug_n_band=(3, 6),
        filt_aug_min_bw=6,
        use_spec_aug=True,
        time_mask_param=40,
        freq_mask_param=20,
        n_specaug=2,
    ):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**mel_params)
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)
        self.resize = tvt_v2.Resize(size=lms_shape)
        self.top_db = top_db

        self.use_filt_aug = use_filt_aug
        self.filt_aug_db_range = filt_aug_db_range
        self.filt_aug_n_band = filt_aug_n_band
        self.filt_aug_min_bw = filt_aug_min_bw

        self.use_spec_aug = use_spec_aug
        self.n_specaug = n_specaug
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)

    def forward(self, wave):
        with torch.no_grad():
            mel = self.mel_transform(wave)
            lms = self.db(mel)

        if self.training and self.use_filt_aug:
            lms = filt_aug(
                lms,
                db_range=self.filt_aug_db_range,
                n_band=self.filt_aug_n_band,
                min_bw=self.filt_aug_min_bw,
            )

        lms = self.resize(lms)
        lms = torch.clamp((lms + self.top_db) / self.top_db, 0.0, 1.0)
        lms = lms[:, None, :, :]

        if self.training and self.use_spec_aug:
            for _ in range(self.n_specaug):
                lms = self.freq_mask(self.time_mask(lms))

        return lms
