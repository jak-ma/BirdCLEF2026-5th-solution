"""
Frozen Perch v2 ONNX teacher for embedding distillation.

Supports:
  - Standard 5s input (effv2s, effb3, hgnet)
  - 10s input with center cropping to 5s (effv2s_10s)
  - Sub-batch inference for memory efficiency (effb3)
"""

import numpy as np
import torch
from torch.nn import functional as F
import onnxruntime as ort


class PerchTeacher:
    """
    Frozen Perch v2 via ONNX runtime.

    Args:
        onnx_path: path to the Perch ONNX model file
        prefer_cuda: use CUDAExecutionProvider if available
        sr: sample rate (default 32000)
        duration: target duration in seconds (default 5)
        perch_input_sec: Perch model's expected input length in seconds
            (default 5; set to 5 for 10s models that need center cropping)
        perch_embed_dim: expected embedding dimension (default 1536)
        sub_batch_size: sub-batch size for ONNX inference (None = no sub-batching)
    """

    def __init__(
        self,
        onnx_path,
        prefer_cuda=True,
        sr=32_000,
        duration=5,
        perch_input_sec=5,
        perch_embed_dim=1536,
        sub_batch_size=None,
    ):
        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if prefer_cuda and "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        self.embed_idx = next(
            (i for i, o in enumerate(self.session.get_outputs())
             if o.shape and o.shape[-1] == perch_embed_dim),
            1,
        )
        self.sr = sr
        self.duration = duration
        self.perch_input_sec = perch_input_sec
        self.perch_len = sr * perch_input_sec
        self.target_len = sr * duration
        self.sub_batch_size = sub_batch_size

        print(f"[Perch] providers={self.session.get_providers()} embed_idx={self.embed_idx}")
        if self.perch_input_sec != self.duration:
            print(f"[Perch] model input={self.perch_input_sec}s, target duration={self.duration}s → center crop")
        if sub_batch_size is not None:
            print(f"[Perch] sub_batch_size={sub_batch_size}")

    @torch.no_grad()
    def embed(self, wave):
        """
        Extract Perch embeddings from waveform.

        Args:
            wave: (B,) or (B, 1) or (B, T) waveform tensor

        Returns:
            (B, D) embedding tensor
        """
        if wave.ndim == 3:
            wave = wave.squeeze(1)

        wave = wave.float()

        # If input is longer than perch expects, center crop
        if wave.shape[1] > self.perch_len:
            start = (wave.shape[1] - self.perch_len) // 2
            wave = wave[:, start:start + self.perch_len]
        elif wave.shape[1] < self.perch_len:
            wave = F.pad(wave, (0, self.perch_len - wave.shape[1]))

        wav_np = np.ascontiguousarray(wave.cpu().numpy().astype(np.float32))

        # Sub-batched inference (for memory efficiency on large batch sizes)
        if self.sub_batch_size is not None and self.sub_batch_size < wav_np.shape[0]:
            all_outs = []
            for i in range(0, wav_np.shape[0], self.sub_batch_size):
                wav_sub = wav_np[i: i + self.sub_batch_size]
                outs = self.session.run(None, {self.input_name: wav_sub})
                all_outs.append(torch.from_numpy(outs[self.embed_idx]))
            return torch.cat(all_outs, dim=0).float()

        outs = self.session.run(None, {self.input_name: wav_np})
        return torch.from_numpy(outs[self.embed_idx]).float()
