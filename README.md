# BirdCLEF+ 2026 — 5th Place Solution

This repository contains the **CNN training code** for my 5th place finish in the [BirdCLEF+ 2026 Kaggle competition](https://www.kaggle.com/competitions/birdclef-2026). The final ensemble also includes two public models (ProtoSSM and Distilled SED) whose inference notebooks live under `notebooks/` — this repo covers the full CNN training pipeline that built the core of my submission. 

Kaggle write-up: [5th Place Solution: Diversity and Bug - Both Are All You Need](https://www.kaggle.com/competitions/birdclef-2026/writeups/5th-place-solution-both-are-all-you-need)

---

## Acknowledgments

First, thank you to the sponsor and Kaggle for this wonderful event. The high-quality data and fair environment made this competition challenging and rewarding.

I'm deeply grateful to all participants who shared insights, notebooks, and discussions. The collaborative spirit of this community is truly inspiring — I learned so much from you, and this solution wouldn't exist without those exchanges.

This is my first gold medal, and I'm still overwhelmed. Reaching the top took real luck — fortunate validation splits, lucky ensemble choices, and timing that worked in my favour. Sometimes things just click, and this was one of those rare moments.

Special thanks to:

- [@ttahara](https://www.kaggle.com/ttahara) for the [HGNetV2-B0 public baseline](https://www.kaggle.com/code/ttahara/birdclef-2026-hgnetv2-b0-baseline-training) that all my CNN models are built upon
- The 2025 BirdCLEF [2nd place](https://www.kaggle.com/competitions/birdclef-2025/writeups/volodymyr-vialactea-2nd-place-journey-down-the-rab) and [5th place](https://www.kaggle.com/competitions/birdclef-2025/writeups/noir-5th-place-solution-self-distillation-is-all-y) solutions, which heavily influenced my ensemble and pseudo-labeling strategies
- [@tuckerarrants](https://www.kaggle.com/tuckerarrants) for the [Distilled SED Baseline](https://www.kaggle.com/competitions/birdclef-2026/discussion/694479)
- Everyone who shared public ProtoSSM models

---

## About the Competition

BirdCLEF+ 2026 challenged participants to identify which of **234 bird species** are present in short audio clips extracted from continuous field recordings. The data was collected from the Pantanal wetlands in South America using passive acoustic monitors that run 24/7, producing 1-minute soundscape files that must be processed into 5-second analysis windows.

The core difficulty lies in the extreme class imbalance — some species appear in hundreds of training clips, others in fewer than 10 — and the fact that most soundscape recordings contain overlapping vocalizations from multiple species, ambient noise, and varying recording conditions.

The evaluation metric was **macro-averaged ROC-AUC**, with any species that has zero true positives in the test set excluded from the average. This means that even a handful of rare species with poor predictions can significantly drag down the final score.

---

## Repository Layout

```
├── train.py                         ← Unified CLI entry point
├── requirements.txt
│
├── birdclef/                        ← Core library
│   ├── utils.py                     ← Seed, grad scaler, weight init, distill loss,
│   │                                   output path resolution, teacher auto-detection
│   ├── augmentations.py             ← FilterAugment, MixUp, SpecAugment
│   ├── perch_teacher.py             ← ONNX-based Perch v2 teacher
│   │
│   ├── models/
│   │   ├── components.py            ← GeM1d, AttBlock variants, DistillHead, CA, FreqSE
│   │   ├── bird_model.py            ← Unified BirdModel (effv2s / hgnet / effb3)
│   │   └── ema.py                   ← Exponential Moving Average
│   │
│   ├── data/
│   │   ├── dataset.py               ← BirdDataset (focal, soundscape, 10s, unlabeled)
│   │   ├── transforms.py            ← LogMelTransform / LogMelSpectrogramTransform
│   │   └── utils.py                 ← Label parsing, fold assignment, data building
│   │
│   └── training/
│       ├── train_loop.py            ← Single-epoch training + validation
│       └── train_fold.py            ← Full fold training orchestrator
│
├── scripts/                         ← Individual experiment scripts
│   ├── effv2s_stage1.py             ← EffV2-S, supervised baseline
│   ├── effv2s_stage2.py             ← EffV2-S, self-distillation (multi-round)
│   ├── effv2s_stage3.py             ← EffV2-S, unlabeled soundscapes
│   ├── effv2s_10s.py                ← EffV2-S, 10-second windows
│   ├── effb3_stage1.py              ← Eff-B3, EMA training
│   ├── effb3_stage2.py              ← Eff-B3, self-distillation + unlabeled SC
│   └── hgnet_stage1.py             ← HGNetV2-B0, rare-species oversampling
│
└── notebooks/                       ← Model inference notebooks (ensemble)
```

---

## Data Handling

I simply concatenated competition **focal + soundscapes** data. All labeled soundscapes are used for training; validation is split from focal data only (MultilabelStratifiedKFold by audio_id).

### Augmentations

| Augmentation            | Description                                                                                                                                                                 |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MixUp**         | Audio-level mix with max-lambda strategy (λ = max(β, 1−β) where β ~ Beta(α, α)), ensuring the original sample dominates (λ ≥ 0.5). Union labels above θ=0.8.      |
| **FilterAugment** | Random piecewise-linear gain across frequency bands, simulating spectral coloration. ⚠️ See the "bug" section below — my implementation is unconventional but effective. |
| **Wave-noisy**    | Gain jitter + additive noise + random shift (HGNet only).                                                                                                                   |
| **SpecAugment**   | Time masking only; frequency masking replaced by FrequencySE (HGNet) or CoarseDropout + HorizontalFlip via albumentations (effv2s).                                         |

---

## Model Families

All CNN models are based on [@ttahara](https://www.kaggle.com/ttahara)'s public baseline with the original SED head replaced by an attention-based design. Every model uses Perch v2 embedding distillation.

|      Backbone      |         Pretrain         | lms_shape | Duration | Upsampling | Self-Distilled | Pseudo |   Seeds   |
| :-----------------: | :----------------------: | :-------: | :------: | :--------: | :------------: | :----: | :--------: |
|    `hgnetv2b0`    |     ImageNet (timm)     | 128, 313 |    5s    |     ✓     |       ✗       |   ✗   | 520 / 3407 |
| `efficientnetv2s` | XLS (vladimirsydor 2025) | 128, 313 |    5s    |     ✗     |     2-iter     | 1-iter | 1086 / 42 |
| `efficientnetv2s` | XLS (vladimirsydor 2025) | 128, 626 |   10s   |     ✗     |       ✗       |   ✗   |  1086 / 1  |
| `efficientnetb3` |     ImageNet (timm)     | 384, 384 |    5s    |     ✗     |       ✗       | 1-iter | 1086 / 888 |

For each single model, aside from hyperparameter tuning, the greatest gains came from: **Distillation** (~50%), **data augmentations** (~30%), **model architecture** (~20%).

### Single Model Improvement Trajectory

|    Improvement    | Public Score |
| :---------------: | :-----------: |
|     Baseline     |  0.88–0.895  |
|    + Data Aug    | 0.895–0.905 |
| + Perch-Distilled | 0.905–0.915 |
|   + Pretrained   | 0.915–0.925 |
| + Self-Distilled | 0.925–0.928 |
|  + Pseudo iter 1  | 0.928–0.935 |
|       + TTA       | +0.005–0.008 |
|  Post-Processing  | +0.005–0.01 |

---

## The "Bug" — Unconventional FilterAugment

In data augmentation, the conventional FrequencyFilterAug applies `dB + noise` (addition in the dB domain). My approach applies `dB × linear_gain` (multiplication in the dB domain). From a physical audio processing perspective, multiplying Log-Mel completely destroys the physical meaning — a quiet frequency band at −80dB, multiplied by 2, becomes −160dB, which is physically absurd. Yet this "crazy scaling" delivered a consistent improvement of **0.01+** on my models.

```python
# The "bug" that worked: multiplicative gain in dB domain
def freq_filt_aug(features, db_range=(-6, 6), n_band=(3, 6), min_bw=6):
    # ... boundary generation ...
    return features * (10 ** (freq_filt / 10))
    #                    ^^^^^^^^^^^^^^^^^^^^
    # Instead of: features + linear_gain  (conventional)
    # This does:   features × 10^(dB/10)  (multiplicative in linear, insane in dB)
```

See `birdclef/augmentations.py::filt_aug()` for the implementation. Enabling this is the default for all models.

---

## Training Pipeline

### Stage 1 — Supervised

All models trained on labeled data: focal recordings + the 59 labeled soundscape files. Perch v2 embedding distillation throughout.

### Stage 2 — Self-Distillation

A frozen copy of Stage 1 acts as a teacher. The student is trained on blended labels: `label = α × teacher_prediction + (1−α) × ground_truth`. Predictions below τ are compressed via power transform (`p̂^power`). Multi-round: run the script repeatedly (sd_r1 → sd_r2 → …).

### Stage 3 — Unlabeled Soundscapes

The competition provides 10,000+ unlabeled soundscapes. The teacher generates soft labels for random 5s crops, and the student learns from them alongside labeled data. Unlabeled samples use full-file random cropping.

### 10-Second Variant

Doubling the input window to 10s; soundscape segments expanded ±2.5s around the original 5s annotation; Perch distillation uses center crop to align with the fixed 5s ONNX input.

---

## Post-Processing

- **TTA**: Dual-window inference (normal + 2.5s shift).
- **Smoothing**: Neighboring frame smoothing with window `[0.1, 0.8, 0.1]` (inspired by [2025 5th place](https://www.kaggle.com/competitions/birdclef-2025/writeups/noir-5th-place-solution-self-distillation-is-all-y)).
- **File Peak Scale**: Scales predictions within each audio file by the mean of top-k window probabilities (inspired by [2025 2nd place](https://www.kaggle.com/competitions/birdclef-2025/writeups/volodymyr-vialactea-2nd-place-journey-down-the-rab)).

---

## Final Ensemble

- **CNN ensemble**: `hgnetv2-b0` + `efficientnetv2-s` + `efficientnet-b3` with SED Head (clipwise only).
- **Sequential model**: ProtoSSM (public, notebooks/)
- **Distilled SED**: Public baseline ([discussion](https://www.kaggle.com/competitions/birdclef-2026/discussion/694479))

```
Final = 0.6 × (0.25·hgnet + 0.40·effv2_5s + 0.25·effv2_10s + 0.10·effb3)
      + 0.4 × (0.6·proto + 0.4·sed)
```

**Diversity Is All You Need**: I minimized Mel parameters as much as possible and optimized single-model inference speed to ensemble more models. One key lesson: applying identical post-processing to all models hurts LB — in my final submission, different post-processing was applied at different positions.

---

## Setup

```bash
# 1. Install PyTorch (adjust CUDA version for your GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install other dependencies
pip install -r requirements.txt
```

### Data

Place competition data under `./birdc/` (or set `--data_dir`):

```
birdc/
├── taxonomy.csv
├── train.csv
├── train_audio/                   ← focal recordings (.ogg)
├── train_soundscapes/             ← 1-minute soundscape files (.ogg)
└── train_soundscapes_labels.csv
```

### Weights

#### Pretrained Weights

- **EffV2-S models** require the [XLS pretrained backbone](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models?select=models_2025):

  ```
  effv2s_10s/pretrained/tf_efficientnetv2_s_in21k_pretrain_from_bigXCV2Ext_swa.ckpt
  ```

  If missing, a warning is printed and the model falls back to ImageNet init.
  
- **Eff-B3 and HGNet** use `pretrained=True` — weights are downloaded automatically by timm on first run (internet required).

#### Final CNN Weights

- All models's weights in link:  [birdclef2026-5th-final-cnn-models](https://www.kaggle.com/datasets/jakkma/birdclef2026-5th-final-cnn-models)

---

## Usage

### Step-by-Step Pipeline

All outputs are saved under `./outputs/`. Each stage auto-detects the required teacher checkpoint from the previous stage. **No parameter changes needed.**

```bash
# 1. Stage 1 — Supervised baselines
python scripts/effv2s_stage1.py      # → outputs/effv2s_stage1_seed1086/
python scripts/effb3_stage1.py       # → outputs/effb3_stage1_seed1086/
python scripts/hgnet_stage1.py       # → outputs/hgnet_stage1_seed520/
python scripts/effv2s_10s.py         # → outputs/effv2s_10s_seed1086/

# 2. Stage 2 — Self-distillation (auto-detects Stage 1 output)
python scripts/effv2s_stage2.py      # → outputs/effv2s_stage2_seed1086/sd_r1/
python scripts/effv2s_stage2.py      # ← run again → sd_r2/
python scripts/effb3_stage2.py       # → outputs/effb3_stage2_*_seed1086/

# 3. Stage 3 — Unlabeled soundscapes (auto-detects Stage 2 output)
python scripts/effv2s_stage3.py      # → outputs/effv2s_stage3_K2_random_seed1086/
```

If a required checkpoint is missing, the script prints a clear error message:

```
ERROR: Stage 1 checkpoint not found!
Expected: outputs/effv2s_stage1_seed1086/best_model_fold0.pt
Please run Stage 1 first:
  python scripts/effv2s_stage1.py
```

### Unified CLI

```bash
python train.py --model effv2s --stage 1  --seed 1086
python train.py --model effv2s --stage 2  --seed 1086
python train.py --model effv2s --stage 3  --seed 1086
python train.py --model effb3  --stage 1  --seed 1086
python train.py --model effb3  --stage 2  --seed 1086
python train.py --model hgnet             --seed 520
python train.py --model effv2s --stage 10s --seed 1086
```

---

## Outputs

```txt
outputs/
├── effv2s_stage1_seed1086/           best_model_fold0.pt, result_fold0.csv
├── effv2s_stage2_seed1086/
│   ├── sd_r1/                        best_model_fold0.pt, result_fold0.csv
│   └── sd_r2/                        ...
├── effv2s_stage3_K2_random_seed1086/ best_model_fold0.pt, result_fold0.csv
├── effv2s_10s_seed1086/
├── effb3_stage1_seed1086/
├── effb3_stage2_K2_random_seed1086/
└── hgnet_stage1_seed520/
```

---

## Model Architecture

```
Audio waveform (32kHz, T samples)
  │
  ├─→ MelSpectrogram → AmplitudeToDB
  │     → FilterAugment (multiplicative dB gain, training only)
  │     → [Resize to fixed shape] (b3/hgnet only)
  │     → Clamp + normalize to [0, 1]
  │
  ├─→ BatchNorm2d (over frequency axis)
  ├─→ [FrequencySE] (hgnet only)
  ├─→ timm backbone (global_pool='', 1-channel input)
  │     → feature map: (B, C, H, W)
  │
  ├─→ Classification branch:
  │     mean(H) → [ChannelAttention] → GeM1d → Dropout → FC → ReLU → Dropout
  │     → AttBlock: attention-weighted sum over time
  │       ├─ clipwise logits: (B, 234)  ← training loss
  │       └─ framewise logits: (B, 234, T) ← auxiliary supervision
  │
  └─→ Distillation branch:
        mean(H, W) → Linear(C → 1536) ← trained to match Perch embeddings
```

**Loss:**

```
L = w_clip × BCE(clipwise, y) + w_frame × BCE(pool(framewise), y)
    + 0.1 × (1 − cos(student_emb, perch_emb))
```

EffV2-S/B3: `(w_clip, w_frame) = (0.5, 0.5)`; HGNet: `(0.6, 0.4)`.

---

## License

MIT License
