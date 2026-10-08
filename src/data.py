"""Datasets, transforms, samplers and loaders for the tri-branch model (v2).

Label-space note (v1 → v2 fix)
------------------------------
The v1 loader concatenated RAF-DB (7 **basic** labels) with the synthetic
set (11 **compound** labels) and trained one 11-way head over the mixed
label space — basic label ``3`` ("Happy") collided with compound label
``3`` ("Sadly Angry"). v2 separates the spaces:

* **Phase 1 (SupCon)** trains on synthetic compound faces (labels 0–10)
  plus, optionally, RAF-DB train faces whose basic labels are offset by
  ``+NUM_CLASSES`` (→ 11–17) via :class:`RemappedLabelsDataset`, so
  contrastive positives never mix the two taxonomies.
* **Phase 2 (Focal)** trains the 11-way head on synthetic compound faces
  only.
* **Validation / test** use compound labels: the synthetic set built from
  the FER-2013 ``test`` pools (disjoint source faces → no leakage). The
  RAF-DB test split (basic labels) is exposed separately as an auxiliary
  transfer check, exactly as the v1 dataset-wise evaluation does.

All datasets accept ``return_landmarks=True`` to emit ``(image, landmarks,
label)`` triples for the geometry branch; landmarks come from the
pre-augmentation image, so tri-branch training transforms exclude
geometric ops (see :func:`get_train_transforms_tribranch`).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from src.landmarks import LandmarkExtractor
from src.splicing import SyntheticCompoundDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Transforms ───────────────────────────────────────────────────────────
def get_train_transforms(image_size: int = 224):
    """Legacy v1 training transforms (incl. geometric ops).

    Only for basic-emotion / ablation training where landmarks are unused.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.GaussNoise(p=1.0),
            A.MotionBlur(p=1.0),
        ], p=0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.2, hue=0.1, p=0.5),
        A.CoarseDropout(num_holes_range=(1, 8),
                        hole_height_range=(8, 16),
                        hole_width_range=(8, 16), p=0.3),
        A.Affine(translate_percent=0.1, scale=(0.85, 1.15),
                 rotate=(-15, 15), p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_train_transforms_tribranch(image_size: int = 224):
    """Photometric-only training transforms for tri-branch training.

    No flip / rotate / affine / dropout cutout: Branch-3 landmarks are
    extracted from the pre-augmentation image, so geometric augmentation
    would silently misalign them.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ── Real-face datasets (with optional landmark output) ───────────────────
class _LandmarkMixin:
    return_landmarks: bool = True
    extractor: Optional[LandmarkExtractor] = None

    def _get_extractor(self) -> LandmarkExtractor:
        # Lazy per-process construction → safe under forked DataLoader workers.
        if self.extractor is None:
            self.extractor = LandmarkExtractor()
        return self.extractor

    def _landmarks_for(self, image_rgb: np.ndarray) -> torch.Tensor:
        lm_flat, _ = self._get_extractor().extract_flat(image_rgb)
        return torch.from_numpy(lm_flat).float()

    def _pack(self, image, image_rgb, label):
        label_t = torch.tensor(label, dtype=torch.long)
        if self.return_landmarks:
            return image, self._landmarks_for(image_rgb), label_t
        return image, label_t


class RAFDBDataset(Dataset, _LandmarkMixin):
    FOLDER_TO_IDX = {
        "anger": 0, "angry": 0,
        "disgust": 1, "disgusted": 1,
        "fear": 2, "fearful": 2, "afraid": 2,
        "happiness": 3, "happy": 3,
        "sadness": 4, "sad": 4,
        "surprise": 5, "surprised": 5,
        "neutral": 6,
        "1": 5, "2": 2, "3": 1, "4": 3, "5": 4, "6": 0, "7": 6,
    }
    SPLIT_ALIASES = {
        "train": ["train", "Train", "TRAIN", "training"],
        "test": ["test", "Test", "TEST", "testing", "valid", "val", "validation"],
    }

    def __init__(self, split="train", transform=None, raf_db_dir=None,
                 return_landmarks=True, extractor=None):
        self.transform = transform
        self.return_landmarks = return_landmarks
        self.extractor = extractor
        self.data_dir = self._resolve_split_dir(split, raf_db_dir)
        self.samples = []
        for folder in sorted(os.listdir(self.data_dir)):
            label_idx = self.FOLDER_TO_IDX.get(folder.lower())
            if label_idx is None:
                continue
            class_dir = os.path.join(self.data_dir, folder)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), label_idx))

    def _resolve_split_dir(self, split, raf_db_dir):
        from src.config import cfg as _default_cfg
        base = raf_db_dir or _default_cfg.RAF_DB_DIR
        if not os.path.isdir(base):
            raise FileNotFoundError(f"RAF_DB_DIR does not exist: {base}")
        for alias in self.SPLIT_ALIASES.get(split, [split]):
            candidate = os.path.join(base, alias)
            if os.path.isdir(candidate):
                return candidate
        raise FileNotFoundError(
            f"None of {self.SPLIT_ALIASES.get(split, [split])} found under {base}. "
            f"Contents of {base}: {os.listdir(base)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = self.transform(image=rgb)["image"] if self.transform else rgb
        return self._pack(img, rgb, label)


class FER2013Dataset(Dataset):
    """Raw basic-emotion pool (no landmarks needed — only a synthesis source)."""

    def __init__(self, split="train", transform=None, fer2013_dir=None, basic_classes=None):
        from src.config import cfg as _default_cfg
        base = fer2013_dir or _default_cfg.FER2013_DIR
        classes = basic_classes or _default_cfg.BASIC_CLASSES
        self.transform = transform
        self.data_dir = os.path.join(base, split)
        self.samples = []
        for label_idx, cls in enumerate(classes):
            class_dir = os.path.join(self.data_dir, cls.lower())
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.long)


class JAFFEDataset(Dataset, _LandmarkMixin):
    EMOTION_MAP = {"HA": 3, "SA": 4, "AN": 0, "FE": 2, "DI": 1, "SU": 5, "NE": 6}

    def __init__(self, transform=None, jaffe_dir=None,
                 return_landmarks=True, extractor=None):
        from src.config import cfg as _default_cfg
        base = jaffe_dir or _default_cfg.JAFFE_DIR
        self.transform = transform
        self.return_landmarks = return_landmarks
        self.extractor = extractor
        self.samples = []
        for root, _, files in os.walk(base):
            for fname in files:
                if not fname.lower().endswith((".tiff", ".tif", ".jpg", ".png")):
                    continue
                code = fname.split(".")[1][:2].upper()
                if code not in self.EMOTION_MAP:
                    continue
                self.samples.append((os.path.join(root, fname), self.EMOTION_MAP[code]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            rgb = np.array(Image.open(path).convert("RGB"))
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = self.transform(image=rgb)["image"] if self.transform else rgb
        return self._pack(out, rgb, label)


class RemappedLabelsDataset(Dataset):
    """Wrap a dataset, shifting every label by ``offset`` (Phase-1 use)."""

    def __init__(self, dataset: Dataset, offset: int):
        self.dataset = dataset
        self.offset = offset
        self.samples = getattr(dataset, "samples", None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        *rest, label = item
        return (*rest, label + self.offset)


class FER2013Pool:
    """Minimal FER-2013 pool for dataset-wise evaluation (v1-compatible)."""

    def __init__(self, split="train"):
        ds = FER2013Dataset(split=split, transform=None)
        self.by_class = {i: [] for i in range(7)}
        for path, label in ds.samples:
            self.by_class[label].append(path)


# ── Sampler / loaders ────────────────────────────────────────────────────
def _iter_labels(dataset) -> List[int]:
    if isinstance(dataset, RemappedLabelsDataset):
        return [label + dataset.offset for label in _iter_labels(dataset.dataset)]
    if isinstance(dataset, ConcatDataset):
        labels: List[int] = []
        for ds in dataset.datasets:
            labels += _iter_labels(ds)
        return labels
    if isinstance(dataset, SyntheticCompoundDataset):
        return [s[2] for s in dataset.samples]
    if hasattr(dataset, "samples") and dataset.samples is not None:
        return [s[1] for s in dataset.samples]
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):  # Subset
        return [dataset.dataset.samples[i][1] for i in dataset.indices]
    raise RuntimeError(f"Cannot infer labels for {type(dataset).__name__}.")


def make_weighted_sampler(dataset) -> WeightedRandomSampler:
    label_list = _iter_labels(dataset)
    if len(label_list) == 0:
        raise RuntimeError(
            "make_weighted_sampler: dataset is empty. "
            "Check RAF_DB_DIR / FER2013_DIR paths in Config.")
    counts = np.bincount(label_list, minlength=max(label_list) + 1).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    sample_weights = torch.tensor([weights[l] for l in label_list], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def class_counts(dataset) -> torch.Tensor:
    """Per-class counts (used for ``FocalLoss(alpha='balanced')``)."""
    labels = _iter_labels(dataset)
    return torch.bincount(torch.tensor(labels), minlength=max(labels) + 1)


def get_dataloaders(cfg=None) -> Dict[str, DataLoader]:
    """Build the two-phase loader dict.

    Returns ``{"train_contrast", "train_focal", "val", "rafdb_test"}``:

    * ``train_contrast`` — synthetic-train (compound 0–10) + optionally
      RAF-DB train (basic labels offset to 11–17). Phase-1 SupCon.
    * ``train_focal`` — synthetic-train only (compound 0–10). Phase-2 Focal.
    * ``val`` — synthetic-val from FER-2013 ``test`` pools (compound 0–10).
    * ``rafdb_test`` — RAF-DB test (basic labels; auxiliary transfer check).
    """
    from src.config import cfg as _default_cfg
    cfg = cfg or _default_cfg
    train_tf = get_train_transforms_tribranch(cfg.IMAGE_SIZE)
    val_tf = get_val_transforms(cfg.IMAGE_SIZE)

    fer_train = FER2013Dataset(split="train", transform=None)
    fer_test = FER2013Dataset(split="test", transform=None)
    if len(fer_train) == 0:
        raise RuntimeError(f"No FER-2013 train images under {fer_train.data_dir}.")
    n_syn = cfg.SYNTHETIC_PER_CLASS * cfg.NUM_CLASSES
    synthetic_train = SyntheticCompoundDataset(
        fer_train, num_synthetic=n_syn, compound_to_basic=cfg.COMPOUND_TO_BASIC,
        neutral_idx=cfg.NEUTRAL_IDX, transform=train_tf, mode="poisson",
        return_landmarks=True, image_size=cfg.IMAGE_SIZE)

    if len(fer_test) > 0:
        synthetic_val = SyntheticCompoundDataset(
            fer_test, num_synthetic=max(cfg.NUM_CLASSES * 50, n_syn // 10),
            compound_to_basic=cfg.COMPOUND_TO_BASIC, neutral_idx=cfg.NEUTRAL_IDX,
            transform=val_tf, mode="poisson", return_landmarks=True,
            image_size=cfg.IMAGE_SIZE)
    else:  # fallback: stratified hold-out from the train pools (may leak identities)
        print("  WARNING: FER-2013 test split empty — holding out synthetic val "
              "from train pools (source faces may overlap train).")
        rng = np.random.RandomState(cfg.SEED)
        idx = rng.permutation(len(synthetic_train.samples))
        cut = max(cfg.NUM_CLASSES, int(0.15 * len(idx)))
        val_samples = [synthetic_train.samples[i] for i in idx[:cut]]
        synthetic_train.samples = [synthetic_train.samples[i] for i in idx[cut:]]
        synthetic_val = SyntheticCompoundDataset(
            fer_train, num_synthetic=0, compound_to_basic=cfg.COMPOUND_TO_BASIC,
            neutral_idx=cfg.NEUTRAL_IDX, transform=val_tf, mode="poisson",
            return_landmarks=True, image_size=cfg.IMAGE_SIZE)
        synthetic_val.samples = val_samples

    train_focal = synthetic_train
    if cfg.PHASE1_USE_RAFDB:
        rafdb_train = RAFDBDataset(split="train", transform=train_tf,
                                   return_landmarks=True)
        rafdb_offset = RemappedLabelsDataset(rafdb_train, offset=cfg.NUM_CLASSES)
        train_contrast = ConcatDataset([synthetic_train, rafdb_offset])
        print(f"  RAF-DB train (offset labels): {len(rafdb_train)}")
    else:
        train_contrast = synthetic_train

    rafdb_test = RAFDBDataset(split="test", transform=val_tf, return_landmarks=True)

    print(f"  Synthetic train : {len(synthetic_train)}")
    print(f"  Synthetic val   : {len(synthetic_val)}")
    print(f"  RAF-DB test     : {len(rafdb_test)}")

    dl_kwargs = dict(num_workers=cfg.NUM_WORKERS, pin_memory=True)
    loaders = {
        "train_contrast": DataLoader(
            train_contrast, batch_size=cfg.BATCH_SIZE,
            sampler=make_weighted_sampler(train_contrast), **dl_kwargs),
        "train_focal": DataLoader(
            train_focal, batch_size=cfg.BATCH_SIZE,
            sampler=make_weighted_sampler(train_focal), **dl_kwargs),
        "val": DataLoader(synthetic_val, batch_size=cfg.BATCH_SIZE,
                          shuffle=False, **dl_kwargs),
        "rafdb_test": DataLoader(rafdb_test, batch_size=cfg.BATCH_SIZE,
                                 shuffle=False, **dl_kwargs),
    }
    return loaders
