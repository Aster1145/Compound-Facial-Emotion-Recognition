"""Landmark-guided Poisson-blending synthesis (v2 dataset synthesis).

Replaces the v1 rigid alpha-blend / horizontal-splice logic. For every
synthetic compound sample::

    Emotion A face ──► eyes + eyebrows mask ──┐
                                              ├─► cv2.seamlessClone ──► compound face
    Emotion B face ──► mouth + jaw mask ──────┘        (onto a neutral base face)

Masks are precise convex-hull regions from MediaPipe FaceMesh landmarks
(:mod:`src.landmarks`), and :func:`cv2.seamlessClone` (Poisson blending)
stitches them onto the neutral base with no seam artifacts.

Degraded paths (documented, never a horizontal splice):
* a region whose source face yields no landmarks is skipped (base kept);
* if *both* sources fail, a soft full-face alpha mix is used as a last
  resort so the sample is still usable.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.landmarks import (
    EYE_REGION_LEFT,
    EYE_REGION_RIGHT,
    MOUTH_JAW_REGION,
    LandmarkExtractor,
    landmarks_to_pixel,
    mask_from_indices,
)


class LandmarkPoissonSplicer:
    """Stitches Emotion-A eyes and Emotion-B mouth onto a neutral base face."""

    REGIONS_FROM_A = (EYE_REGION_LEFT, EYE_REGION_RIGHT)
    REGIONS_FROM_B = (MOUTH_JAW_REGION,)

    def __init__(
        self,
        image_size: int = 224,
        extractor: Optional[LandmarkExtractor] = None,
        dilate_px: int = 3,
        feather_px: int = 7,
        clone_flags: int = cv2.NORMAL_CLONE,
    ):
        self.image_size = image_size
        self.extractor = extractor or LandmarkExtractor()
        self.dilate_px = dilate_px
        self.feather_px = feather_px
        self.clone_flags = clone_flags

    # -- helpers ------------------------------------------------------------
    def _prep(self, img_rgb: np.ndarray) -> np.ndarray:
        if img_rgb.shape[:2] != (self.image_size, self.image_size):
            img_rgb = cv2.resize(
                img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
            )
        return img_rgb

    def _clone_region(
        self,
        canvas_rgb: np.ndarray,
        src_rgb: np.ndarray,
        src_px: Optional[np.ndarray],
        indices,
    ) -> Tuple[np.ndarray, bool]:
        """Poisson-clone one landmark region from ``src`` onto ``canvas``."""
        if src_px is None:
            return canvas_rgb, False
        h, w = canvas_rgb.shape[:2]
        mask, center = mask_from_indices(
            src_px, indices, (h, w),
            dilate_px=self.dilate_px, feather_px=self.feather_px,
        )
        if mask.sum() == 0:
            return canvas_rgb, False
        try:
            out = cv2.seamlessClone(src_rgb, canvas_rgb, mask, center, self.clone_flags)
        except cv2.error:
            return canvas_rgb, False
        return out, True

    # -- main API -----------------------------------------------------------
    def splice(
        self,
        imgA_rgb: np.ndarray,
        imgB_rgb: np.ndarray,
        base_rgb: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Build one synthetic compound face.

        Args:
            imgA_rgb: Emotion-A source (donates eyes/eyebrows).
            imgB_rgb: Emotion-B source (donates mouth/jaw).
            base_rgb: neutral base face (defaults to the Emotion-B image).

        Returns:
            (spliced_rgb, info): blended image + per-region success flags.
        """
        imgA = self._prep(imgA_rgb)
        imgB = self._prep(imgB_rgb)
        canvas = self._prep(base_rgb) if base_rgb is not None else imgB.copy()

        lmA, okA = self.extractor.extract(imgA)
        lmB, okB = self.extractor.extract(imgB)
        h, w = canvas.shape[:2]
        pxA = landmarks_to_pixel(lmA, w, h) if okA else None
        pxB = landmarks_to_pixel(lmB, w, h) if okB else None

        info = {"eyes_A_ok": False, "mouth_B_ok": False, "fallback": None}
        if pxA is not None:
            for _, _, idx in self.REGIONS_FROM_A:
                canvas, done = self._clone_region(canvas, imgA, pxA, idx)
                info["eyes_A_ok"] = info["eyes_A_ok"] or done
        if pxB is not None:
            for _, _, idx in self.REGIONS_FROM_B:
                canvas, done = self._clone_region(canvas, imgB, pxB, idx)
                info["mouth_B_ok"] = info["mouth_B_ok"] or done

        if not info["eyes_A_ok"] and not info["mouth_B_ok"]:
            # Last resort: soft full-face alpha mix (NOT a horizontal splice).
            alpha = float(np.random.uniform(0.45, 0.55))
            canvas = (
                alpha * imgA.astype(np.float32) + (1 - alpha) * imgB.astype(np.float32)
            ).astype(np.uint8)
            info["fallback"] = "alpha_mix"
        return canvas, info


class SyntheticCompoundDataset(Dataset):
    """Compound-emotion dataset synthesized by landmark Poisson blending.

    Each item splices Emotion A (eyes/eyebrows) + Emotion B (mouth/jaw) from
    :attr:`COMPOUND_TO_BASIC` onto a neutral base face. ``__getitem__``
    returns ``(image, landmarks, label)`` with landmarks as a flat ``(1404,)``
    float tensor extracted from the *pre-augmentation* splice — hence the
    tri-branch training transforms exclude geometric ops (see
    :func:`src.data.get_train_transforms_tribranch`) so landmarks stay aligned.

    Args:
        fer_dataset: basic-emotion pool exposing ``.samples`` (path, label).
            Must contain neutral faces (basic idx 6) for base images.
        num_synthetic: total samples (≈ ``num_synthetic // 11`` per class).
        compound_to_basic: compound → (emotion_A, emotion_B) mapping.
        neutral_idx: basic index used for neutral base faces.
        transform: albumentations transform applied to the spliced image.
        mode: ``"poisson"`` (default) or ``"alpha"`` (legacy v1 blend, kept
            for ablations).
        return_landmarks: if False, return legacy ``(image, label)`` pairs.
        extractor / splicer: shared instances (constructed lazily if omitted).
        image_size: splice working resolution.
    """

    def __init__(
        self,
        fer_dataset,
        num_synthetic: int = 8800,
        compound_to_basic=None,
        neutral_idx: int = 6,
        transform=None,
        mode: str = "poisson",
        return_landmarks: bool = True,
        extractor: Optional[LandmarkExtractor] = None,
        splicer: Optional[LandmarkPoissonSplicer] = None,
        image_size: int = 224,
    ):
        if mode not in ("poisson", "alpha"):
            raise ValueError(f"mode must be 'poisson' or 'alpha', got {mode!r}.")
        self.transform = transform
        self.mode = mode
        self.return_landmarks = return_landmarks
        self.image_size = image_size
        self.neutral_idx = neutral_idx
        self.compound_to_basic = dict(compound_to_basic or {})
        self.by_class = {i: [] for i in range(7)}
        for path, label in fer_dataset.samples:
            self.by_class[int(label)].append(path)
        if not self.by_class[neutral_idx]:
            raise RuntimeError(
                "SyntheticCompoundDataset needs neutral base faces (basic idx "
                f"{neutral_idx}) but the pool has none."
            )
        self.extractor = extractor  # lazy: built on first use (worker-safe)
        self.splicer = splicer
        self.samples: List[tuple] = self._generate(num_synthetic)

    # -- sample planning ------------------------------------------------------
    def _generate(self, n: int):
        samples = []
        n_compounds = len(self.compound_to_basic)
        per_class = n // max(n_compounds, 1)
        for compound_idx, (e1, e2) in self.compound_to_basic.items():
            pool1 = self.by_class.get(e1, [])
            pool2 = self.by_class.get(e2, [])
            poolN = self.by_class.get(self.neutral_idx, [])
            if not pool1 or not pool2 or not poolN:
                continue
            for _ in range(per_class):
                p1 = pool1[np.random.randint(len(pool1))]
                p2 = pool2[np.random.randint(len(pool2))]
                pN = poolN[np.random.randint(len(poolN))]
                samples.append((p1, p2, compound_idx, pN))
        return samples

    def _lazy_modules(self):
        # Built on first __getitem__ call so forked DataLoader workers each
        # create their own MediaPipe instance *after* forking.
        if self.extractor is None:
            self.extractor = LandmarkExtractor()
        if self.splicer is None:
            self.splicer = LandmarkPoissonSplicer(
                image_size=self.image_size, extractor=self.extractor
            )
        return self.extractor, self.splicer

    # -- synthesis --------------------------------------------------------------
    def _splice(self, pathA: str, pathB: str, pathBase: str) -> np.ndarray:
        """Synthesize one compound face from two source + one base image.

        ``"poisson"`` mode: landmark masks (eyes/eyebrows from A, mouth/jaw
        from B) are Poisson-blended onto the neutral base via
        :func:`cv2.seamlessClone` — no horizontal seam. ``"alpha"`` mode
        reproduces the legacy v1 alpha blend for ablations.
        """
        imgA = cv2.cvtColor(cv2.imread(pathA), cv2.COLOR_BGR2RGB)
        imgB = cv2.cvtColor(cv2.imread(pathB), cv2.COLOR_BGR2RGB)
        base = cv2.cvtColor(cv2.imread(pathBase), cv2.COLOR_BGR2RGB)
        if self.mode == "alpha":
            imgA = cv2.resize(imgA, (self.image_size, self.image_size))
            imgB = cv2.resize(imgB, (self.image_size, self.image_size))
            alpha = float(np.random.uniform(0.45, 0.55))
            return (
                alpha * imgA.astype(np.float32) + (1 - alpha) * imgB.astype(np.float32)
            ).astype(np.uint8)
        _, splicer = self._lazy_modules()
        spliced, _ = splicer.splice(imgA, imgB, base)
        return spliced

    # -- dataset protocol -------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p1, p2, label, pN = self.samples[idx]
        spliced = self._splice(p1, p2, pN)  # RGB uint8, pre-augmentation

        if self.return_landmarks:
            extractor, _ = self._lazy_modules()
            lm_flat, _ = extractor.extract_flat(spliced)
            landmarks = torch.from_numpy(lm_flat).float()  # (1404,)
        else:
            landmarks = None

        if self.transform:
            image = self.transform(image=spliced)["image"]
        else:
            image = torch.from_numpy(spliced).permute(2, 0, 1).float() / 255.0

        label_t = torch.tensor(label, dtype=torch.long)
        if self.return_landmarks:
            return image, landmarks, label_t
        return image, label_t
