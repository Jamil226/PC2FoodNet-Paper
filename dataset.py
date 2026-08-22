"""
dataset.py — Turkish Food Dataset loader for PC2FoodNet.

Supports three loading modes:
  1. build_full_dataloader()    — entire dataset for training (no holdout)
  2. build_cv_fold()            — single 80:20 stratified fold
  3. build_kfold_splits()       — all k stratified folds for k-fold CV
  4. build_dataloaders()        — legacy train/val/test split (kept for compat)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Nutrition prior builder
# ---------------------------------------------------------------------------

def _parse_float(value: str | float | int) -> float:
    """Strip unit strings like '420 kcal' and return a float."""
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^\d.\-]", "", str(value))
    return float(cleaned) if cleaned else 0.0


def build_nutrition_tensors(
    json_path: str | Path,
    class_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parse porsiyon_nutrition_data.json and return:
      densities   (num_classes,)  g/mL  — approximate from macronutrient ratios
      kcal_per_g  (num_classes,)  kcal/g

    Density approximation:
      fat ~ 0.90 g/mL, protein + carb + fibre ~ 1.30 g/mL → macro-weighted mean
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lookup: dict[str, dict] = {rec["yemek"]: rec for rec in data}

    densities: list[float] = []
    kcal_per_g: list[float] = []

    for name in class_names:
        if name not in lookup:
            print(f"[dataset] WARNING: '{name}' not in nutrition JSON — using fallback.")
            densities.append(1.05)
            kcal_per_g.append(2.50)
            continue

        rec = lookup[name]
        kalori       = _parse_float(rec.get("kalori",       250.0))
        protein      = _parse_float(rec.get("protein",        5.0))
        karbonhidrat = _parse_float(rec.get("karbonhidrat",  20.0))
        yag          = _parse_float(rec.get("yağ",            5.0))
        lif          = _parse_float(rec.get("lif",            2.0))

        kpg = kalori / 100.0
        kcal_per_g.append(max(kpg, 0.1))

        total_macro = protein + karbonhidrat + yag + lif + 1e-8
        density = (yag * 0.90 + (protein + karbonhidrat + lif) * 1.30) / total_macro
        densities.append(float(density))

    return (
        torch.tensor(densities,  dtype=torch.float32),
        torch.tensor(kcal_per_g, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_transforms(split: str = "train", img_size: int = 224) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # val / test / inference
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ---------------------------------------------------------------------------
# Core Dataset
# ---------------------------------------------------------------------------

class TurkishFoodDataset(Dataset):
    """
    ImageFolder-style dataset for Turkish food images.

    Structure: data_dir/<ClassName>/<image.jpg|png|webp|...>

    Returns (image_tensor, label_index).
    transform can be swapped per-subset via SubsetWithTransform.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        data_dir: str | Path,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
        img_size: int = 224,
        json_path: Optional[str | Path] = None,
    ):
        self.data_dir  = Path(data_dir)
        self.transform = transform or get_transforms(split, img_size)

        self.class_names: list[str] = sorted([
            p.name for p in self.data_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ])
        self.class_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }

        self.densities = None
        self.kcal_per_g = None
        if json_path and Path(json_path).exists():
            d, k = build_nutrition_tensors(json_path, self.class_names)
            self.densities = d
            self.kcal_per_g = k

        self.samples: list[tuple[Path, int]] = []
        for cls_name in self.class_names:
            cls_dir = self.data_dir / cls_name
            idx = self.class_to_idx[cls_name]
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((img_path, idx))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.data_dir}. "
                "Expected: data_dir/<ClassName>/<image.jpg>"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, float, float, float]:
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        gt_wgt, gt_vol, gt_nrg = 0.0, 0.0, 0.0
        if self.densities is not None and self.kcal_per_g is not None:
            density = float(self.densities[label])
            kcal_per_g = float(self.kcal_per_g[label])
            gt_wgt = 200.0  # Canonical serving
            gt_vol = gt_wgt / max(density, 1e-3)
            gt_nrg = gt_wgt * kcal_per_g

        return img_tensor, label, gt_vol, gt_wgt, gt_nrg

    @property
    def labels(self) -> list[int]:
        """All integer labels — used for stratified splitting."""
        return [s[1] for s in self.samples]


# ---------------------------------------------------------------------------
# Subset with independent transform
# ---------------------------------------------------------------------------

class SubsetWithTransform(Dataset):
    """Wrap a list of indices into the parent dataset with its own transform."""

    def __init__(self, parent: TurkishFoodDataset, indices: list[int],
                 transform: transforms.Compose):
        self.parent    = parent
        self.indices   = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, float, float, float]:
        img_path, label = self.parent.samples[self.indices[i]]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        gt_wgt, gt_vol, gt_nrg = 0.0, 0.0, 0.0
        if self.parent.densities is not None and self.parent.kcal_per_g is not None:
            density = float(self.parent.densities[label])
            kcal_per_g = float(self.parent.kcal_per_g[label])
            gt_wgt = 200.0
            gt_vol = gt_wgt / max(density, 1e-3)
            gt_nrg = gt_wgt * kcal_per_g

        return img_tensor, label, gt_vol, gt_wgt, gt_nrg


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def _make_loader(ds: Dataset, batch_size: int, shuffle: bool,
                 num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=shuffle,
    )


# ---------------------------------------------------------------------------
# Mode 1 — Full dataset loader (all images, no holdout)
# ---------------------------------------------------------------------------

def build_full_dataloader(
    data_dir: str | Path,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 8,
) -> tuple[DataLoader, list[str]]:
    """
    Use the **entire** dataset for training. No validation split.
    Intended as the first training pass before cross-validation.

    Returns: (train_loader, class_names)
    """
    ds = TurkishFoodDataset(data_dir, split="train", img_size=img_size)
    loader = _make_loader(ds, batch_size, shuffle=True, num_workers=num_workers)
    print(f"[dataset] Full-dataset mode: {len(ds):,} images · {len(ds.class_names)} classes")
    return loader, ds.class_names


# ---------------------------------------------------------------------------
# Mode 2 — Single 80:20 stratified fold
# ---------------------------------------------------------------------------

def build_cv_fold(
    data_dir: str | Path,
    fold: int = 0,
    n_folds: int = 5,
    val_ratio: float = 0.20,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 8,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    80:20 stratified split (or k-fold if n_folds > 1 and fold index provided).

    When n_folds=1, a simple stratified 80:20 train_test_split is used.
    When n_folds>1, StratifiedKFold is used and `fold` selects the fold.

    Returns: (train_loader, val_loader, class_names)
    """
    json_path = Path(data_dir) / "porsiyon_nutrition_data.json"
    base = TurkishFoodDataset(data_dir, split="train", img_size=img_size, json_path=json_path)
    labels = np.array(base.labels)
    indices = np.arange(len(base))

    if n_folds == 1:
        # Simple 80:20 stratified split
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, stratify=labels, random_state=seed
        )
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(indices, labels))
        train_idx, val_idx = splits[fold]

    train_tf = get_transforms("train", img_size)
    val_tf   = get_transforms("val",   img_size)

    train_ds = SubsetWithTransform(base, list(train_idx), train_tf)
    val_ds   = SubsetWithTransform(base, list(val_idx),   val_tf)

    train_loader = _make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = _make_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers)

    fold_label = f"fold {fold+1}/{n_folds}" if n_folds > 1 else "80:20 split"
    print(f"[dataset] CV ({fold_label}): "
          f"train={len(train_ds):,} | val={len(val_ds):,} | classes={len(base.class_names)}")

    return train_loader, val_loader, base.class_names


# ---------------------------------------------------------------------------
# Mode 3 — All k folds (returns a list of (train_loader, val_loader) pairs)
# ---------------------------------------------------------------------------

def build_kfold_splits(
    data_dir: str | Path,
    n_folds: int = 5,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 8,
    seed: int = 42,
) -> tuple[list[tuple[DataLoader, DataLoader]], list[str]]:
    """
    Build all k stratified folds at once.

    Returns: ([(train_dl, val_dl), ...], class_names)  — length == n_folds
    """
    json_path = Path(data_dir) / "porsiyon_nutrition_data.json"
    base = TurkishFoodDataset(data_dir, split="train", img_size=img_size, json_path=json_path)
    labels  = np.array(base.labels)
    indices = np.arange(len(base))
    train_tf = get_transforms("train", img_size)
    val_tf   = get_transforms("val",   img_size)

    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds  = []
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        train_ds = SubsetWithTransform(base, list(train_idx), train_tf)
        val_ds   = SubsetWithTransform(base, list(val_idx),   val_tf)
        folds.append((
            _make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers),
            _make_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers),
        ))
        print(f"[dataset]   fold {fold_i+1}/{n_folds}: "
              f"train={len(train_ds):,} | val={len(val_ds):,}")

    return folds, base.class_names


# ---------------------------------------------------------------------------
# Mode 4 — Legacy train/val/test split (backward compat)
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str | Path,
    val_split: float = 0.15,
    test_split: float = 0.05,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 8,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Legacy: stratified train / val / test split. Returns 4-tuple."""
    base    = TurkishFoodDataset(data_dir, split="train", img_size=img_size)
    labels  = np.array(base.labels)
    indices = np.arange(len(base))
    train_tf = get_transforms("train", img_size)
    val_tf   = get_transforms("val",   img_size)

    # First cut off test set
    trainval_idx, test_idx = train_test_split(
        indices, test_size=test_split, stratify=labels, random_state=seed
    )
    # Then split trainval into train / val
    rel_val = val_split / (1.0 - test_split)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=rel_val,
        stratify=labels[trainval_idx], random_state=seed
    )

    train_ds = SubsetWithTransform(base, list(train_idx), train_tf)
    val_ds   = SubsetWithTransform(base, list(val_idx),   val_tf)
    test_ds  = SubsetWithTransform(base, list(test_idx),  val_tf)

    return (
        _make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers),
        _make_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers),
        _make_loader(test_ds,  batch_size, shuffle=False, num_workers=num_workers),
        base.class_names,
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "turkish-food"

    print("\n=== Mode 1: Full dataset ===")
    dl, classes = build_full_dataloader(data_dir, batch_size=8, num_workers=2)
    imgs, lbs = next(iter(dl))
    print(f"  batch: {imgs.shape}  labels: {lbs.tolist()}")

    print("\n=== Mode 2: 80:20 single fold ===")
    tr, vl, _ = build_cv_fold(data_dir, n_folds=1, batch_size=8, num_workers=2)
    print(f"  train batches: {len(tr)} | val batches: {len(vl)}")

    print("\n=== Mode 3: 5-fold CV ===")
    folds, _ = build_kfold_splits(data_dir, n_folds=5, batch_size=8, num_workers=2)
    print(f"  folds built: {len(folds)}")

    json_path = Path(data_dir) / "porsiyon_nutrition_data.json"
    densities, kcal_per_g = build_nutrition_tensors(json_path, classes)
    print(f"\nDensities  min={densities.min():.3f} max={densities.max():.3f}")
    print(f"kcal_per_g min={kcal_per_g.min():.3f} max={kcal_per_g.max():.3f}")
