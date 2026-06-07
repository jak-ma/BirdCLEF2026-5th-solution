"""
Data preparation: parse labels, build combined DataFrame, assign folds,
build unlabeled soundscape DataFrame.
"""

import ast
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def parse_soundscape_labels(data_dir: Path) -> pd.DataFrame:
    """
    Parse train_soundscapes_labels.csv into structured format.

    Returns DataFrame with columns:
        filename, start, end, label_list, start_sec, end_sec, primary_label, source
    """
    raw = pd.read_csv(data_dir / "train_soundscapes_labels.csv")

    sc = (
        raw.groupby(["filename", "start", "end"])["primary_label"]
        .apply(
            lambda s: sorted({
                lbl.strip()
                for x in s if pd.notna(x)
                for lbl in str(x).split(";")
                if lbl.strip()
            })
        )
        .reset_index(name="label_list")
    )

    sc["start_sec"] = pd.to_timedelta(sc["start"]).dt.total_seconds().astype(int)
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["primary_label"] = sc["label_list"].apply(lambda x: x[0] if x else None)
    sc["source"] = "soundscape"

    return sc


def build_combined_df(
    data_dir: Path,
    taxonomy: pd.DataFrame,
    rating_threshold: float = 2.5,
    use_unlabeled_sc: bool = False,
    unlabeled_chunks_per_file: int = 2,
    min_samples_per_class: int = 0,
) -> tuple:
    """
    Build the combined training DataFrame with focal + soundscape (+ unlabeled).

    Args:
        data_dir: root data directory
        taxonomy: taxonomy DataFrame (primary_label column)
        rating_threshold: minimum rating for focal clips (XC collection)
        use_unlabeled_sc: include unlabeled soundscapes
        unlabeled_chunks_per_file: K chunks per unlabeled file
        min_samples_per_class: if >0, oversample rare species (HGNet)

    Returns:
        (df, labels_list, labels_arr)
    """
    labels_list = taxonomy["primary_label"].values.tolist()
    label2idx = {l: i for i, l in enumerate(labels_list)}

    # --- Focal data ---
    train_csv = pd.read_csv(data_dir / "train.csv")
    before = len(train_csv)
    train_csv = train_csv[
        (train_csv["rating"] >= rating_threshold)
        | (train_csv["collection"] == "iNat")
    ].reset_index(drop=True)
    print(f"Rating filter ({rating_threshold}): "
          f"{before} → {len(train_csv)} (removed {before - len(train_csv)})")

    focal = train_csv[["filename", "primary_label", "secondary_labels"]].copy()
    focal["source"] = "focal"
    focal["start_sec"] = 0
    focal["end_sec"] = 0
    focal["label_list"] = [
        [pl] + ast.literal_eval(sls)
        for pl, sls in focal[["primary_label", "secondary_labels"]].values
    ]
    focal["audio_id"] = focal["filename"].str.replace(".ogg", "", regex=False)

    # --- Labeled soundscapes ---
    ss_path = data_dir / "train_soundscapes_labels.csv"
    if ss_path.exists():
        sc = parse_soundscape_labels(data_dir)
        sc["audio_id"] = sc["filename"].str.replace(".wav", "", regex=False)
        sc["audio_id"] = sc["audio_id"].str.replace(".ogg", "", regex=False)
    else:
        sc = pd.DataFrame(columns=focal.columns)

    # --- Unlabeled soundscapes ---
    if use_unlabeled_sc:
        labeled_filenames = set(sc["filename"].values) if not sc.empty else set()
        unl_df = _build_unlabeled_sc_df(data_dir, labeled_filenames, unlabeled_chunks_per_file)
    else:
        unl_df = pd.DataFrame()

    # --- Merge ---
    keep_cols = ["audio_id", "filename", "primary_label", "source",
                 "start_sec", "end_sec", "label_list"]

    dfs = [focal[keep_cols], sc[keep_cols]]
    if len(unl_df) > 0:
        dfs.append(unl_df[keep_cols])

    df = pd.concat(dfs, ignore_index=True)

    # --- Label array ---
    labels_arr = np.zeros((len(df), len(labels_list)), dtype=np.float32)
    for i, ll in enumerate(df["label_list"].values):
        for l in ll:
            if l in label2idx:
                labels_arr[i, label2idx[l]] = 1.0

    # --- Oversampling for rare species (HGNet) ---
    if min_samples_per_class > 0:
        print('[Execute Upsampling]')
        focal_mask = df["source"] == "focal"
        counts = df.loc[focal_mask, "primary_label"].value_counts()
        rare_species = counts[counts < min_samples_per_class].index.tolist()

        extra_dfs = []
        extra_labels = []

        for sp in rare_species:
            sp_mask = focal_mask & (df["primary_label"] == sp)
            sp_df = df[sp_mask]
            sp_labels = labels_arr[sp_mask.values]

            n_copies = int(np.ceil(min_samples_per_class / len(sp_df))) - 1
            for _ in range(max(0, n_copies)):
                extra_dfs.append(sp_df.copy())
                extra_labels.append(sp_labels.copy())

        if extra_dfs:
            n_before = len(df)
            df = pd.concat([df] + extra_dfs, ignore_index=True)
            labels_arr = np.concatenate([labels_arr] + extra_labels, axis=0)
            print(f"[Upsample] {len(rare_species)} rare species: "
                  f"{n_before} -> {len(df)} samples (+{len(df) - n_before} copies)")
    else:
        print("[No execute upsampling]")

    return df, labels_list, labels_arr


def _build_unlabeled_sc_df(data_dir: Path, labeled_filenames: set, K: int) -> pd.DataFrame:
    """
    Build DataFrame for unlabeled soundscapes.

    Args:
        data_dir: root data directory
        labeled_filenames: set of soundscape filenames that are already labeled
        K: number of chunks per file

    Returns:
        DataFrame with source="soundscape", label_list=[]
    """
    soundscapes_dir = data_dir / "train_soundscapes"
    if not soundscapes_dir.exists():
        print(f"[Unlabeled] {soundscapes_dir} does not exist, skipping")
        return pd.DataFrame()

    all_sc_files = sorted(soundscapes_dir.glob("*.ogg"))
    if not all_sc_files:
        all_sc_files = sorted(soundscapes_dir.glob("*.wav"))

    unlabeled_files = [f for f in all_sc_files if f.name not in labeled_filenames]
    print(f"[Unlabeled] Found {len(all_sc_files)} soundscape files, "
          f"{len(unlabeled_files)} unlabeled")

    unl_rows = []
    for fp in unlabeled_files:
        for k in range(K):
            unl_rows.append({
                "audio_id": fp.stem,
                "filename": fp.name,
                "primary_label": None,
                "source": "soundscape",
                "start_sec": k * 5,
                "end_sec": k * 5 + 5,
                "label_list": [],
            })

    unl_df = pd.DataFrame(unl_rows)
    print(f"[Unlabeled] Added {len(unl_df)} chunks (K={K})")
    return unl_df


def assign_folds(df: pd.DataFrame, labels_arr: np.ndarray,
                 n_folds: int, seed: int) -> pd.DataFrame:
    """
    Assign cross-validation folds using MultilabelStratifiedKFold.

    Only focal samples are split into folds; all soundscapes go to training.

    Folds are assigned by audio_id (group-level) to prevent leakage.
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    df = df.copy()
    df["fold"] = -1
    focal_mask = df["source"] == "focal"

    groups = df.loc[focal_mask, "audio_id"].values
    unique_groups = np.unique(groups)
    group_map = {g: i for i, g in enumerate(unique_groups)}

    group_labels = np.zeros((len(unique_groups), labels_arr.shape[1]), dtype=np.float32)
    for i, g in enumerate(unique_groups):
        group_labels[i] = labels_arr[focal_mask.values][groups == g].max(axis=0)

    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    group_folds = np.full(len(unique_groups), -1, dtype=int)
    for fold_id, (_, val_idx) in enumerate(mskf.split(group_labels, group_labels)):
        group_folds[val_idx] = fold_id

    df.loc[focal_mask, "fold"] = [group_folds[group_map[g]] for g in groups]
    return df


def build_loaders(df, labels_arr, fold_id, cfg, BirdDataset):
    """
    Build train and validation DataLoaders.

    Train: all focal not in fold + all soundscapes (including unlabeled)
    Val: focal in fold only
    """
    focal_trn = (df["source"] == "focal") & (df["fold"] != fold_id)
    focal_val = (df["source"] == "focal") & (df["fold"] == fold_id)
    sc_trn = df["source"] == "soundscape"

    trn_mask = focal_trn | sc_trn
    val_mask = focal_val

    trn_ds = BirdDataset(df[trn_mask], labels_arr[trn_mask.values], is_train=True)
    val_ds = BirdDataset(df[val_mask], labels_arr[val_mask.values], is_train=False)

    from torch.utils.data import DataLoader

    trn_loader = DataLoader(
        trn_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    return trn_loader, val_loader
