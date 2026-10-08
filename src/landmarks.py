"""MediaPipe FaceMesh landmark extraction + facial-region index sets.

* :class:`LandmarkExtractor` turns an RGB face image into the normalized
  468 × 3 ``(x, y, z)`` tensor consumed by Branch 3 (:class:`GeometryMLP`).
  ``x``/``y`` are in ``[0, 1]``; ``z`` is MediaPipe's relative depth.
  Detection failures yield an all-zeros tensor (never an exception) so
  training loops stay robust.
* The ``*_IDX`` sets below drive :mod:`src.splicing`: eye/eyebrow regions
  are sourced from Emotion A and the mouth/jaw region from Emotion B.

MediaPipe indices follow the canonical 468-point FaceMesh topology
(``refine_landmarks=False`` so exactly 468 points are returned).
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None


# ── Canonical FaceMesh region indices ────────────────────────────────────
LEFT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
                388, 387, 386, 385, 384, 398]
RIGHT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
                 157, 158, 159, 160, 161, 246]
LEFT_BROW_IDX = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_BROW_IDX = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
LIPS_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321,
            405, 314, 17, 84, 181, 91, 146, 78, 191, 80, 81, 82, 13, 312,
            311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
# Lower face-oval band (chin/jaw) combined with the lips for the mouth/jaw mask.
JAW_IDX = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
           162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251,
           389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377]

# Splice regions: (name, source emotion, landmark indices)
EYE_REGION_LEFT = ("eye_brow_left", "A", LEFT_EYE_IDX + LEFT_BROW_IDX)
EYE_REGION_RIGHT = ("eye_brow_right", "A", RIGHT_EYE_IDX + RIGHT_BROW_IDX)
MOUTH_JAW_REGION = ("mouth_jaw", "B", LIPS_IDX + JAW_IDX)


def landmarks_to_pixel(lm_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized ``(468, 3)`` landmarks to integer pixel coords."""
    px = np.stack([lm_norm[:, 0] * width, lm_norm[:, 1] * height], axis=1)
    return np.clip(px, 0, [width - 1, height - 1]).astype(np.int32)


def mask_from_indices(
    lm_px: np.ndarray,
    indices,
    shape: Tuple[int, int],
    dilate_px: int = 3,
    feather_px: int = 7,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Binary mask + centroid for a landmark region (convex-hull fill).

    Returns:
        mask: ``uint8`` ``(H, W)`` mask (255 inside the region).
        center: ``(x, y)`` centroid for :func:`cv2.seamlessClone`.
    """
    h, w = shape[:2]
    pts = lm_px[np.asarray(indices, dtype=np.intp)]
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
        mask = cv2.dilate(mask, k)
    if feather_px > 1:
        ksize = feather_px + (1 - feather_px % 2)  # odd kernel
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    ys, xs = np.nonzero(mask)
    center = (int(xs.mean()), int(ys.mean())) if len(xs) else (w // 2, h // 2)
    return mask, center


class LandmarkExtractor:
    """Lazy, worker-safe MediaPipe FaceMesh wrapper.

    The underlying ``FaceMesh`` object is created lazily on first use (per
    process), which keeps it safe under forked :class:`DataLoader` workers.
    An optional on-disk ``.npy`` cache (keyed by file bytes) avoids
    re-running detection every epoch.

    Args:
        min_detection_confidence: MediaPipe detection threshold.
        cache_dir: directory for cached landmarks (``""``/``None`` disables).
    """

    N_LANDMARKS = 468

    def __init__(self, min_detection_confidence: float = 0.5, cache_dir: str = ""):
        if mp is None:
            raise ImportError(
                "mediapipe is required for landmark extraction — "
                "`pip install mediapipe`. "
                "(See `mediapipe.solutions.face_mesh`.)"
            )
        self.min_detection_confidence = min_detection_confidence
        self.cache_dir = cache_dir or ""
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self._facemesh = None  # lazy per-process instance

    # -- internals ---------------------------------------------------------
    def _get_facemesh(self):
        if self._facemesh is None:
            self._facemesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,  # exactly 468 landmarks
                min_detection_confidence=self.min_detection_confidence,
            )
        return self._facemesh

    def _cache_path(self, key: bytes) -> Optional[str]:
        if not self.cache_dir:
            return None
        digest = hashlib.sha1(key).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.npy")

    @staticmethod
    def _zeros() -> np.ndarray:
        return np.zeros((LandmarkExtractor.N_LANDMARKS, 3), dtype=np.float32)

    # -- public API ---------------------------------------------------------
    def extract(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Extract ``(468, 3)`` normalized landmarks from an RGB image.

        Returns:
            (landmarks, ok): ``ok=False`` (with zeros) when no face is found.
        """
        results = self._get_facemesh().process(image_rgb)
        if not results.multi_face_landmarks:
            return self._zeros(), False
        face = results.multi_face_landmarks[0]
        lm = np.array([[p.x, p.y, p.z] for p in face.landmark[: self.N_LANDMARKS]],
                      dtype=np.float32)
        if lm.shape[0] < self.N_LANDMARKS:  # defensive; should not happen
            pad = np.zeros((self.N_LANDMARKS - lm.shape[0], 3), dtype=np.float32)
            lm = np.concatenate([lm, pad], axis=0)
        return lm, True

    def extract_flat(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, bool]:
        lm, ok = self.extract(image_rgb)
        return lm.reshape(-1), ok

    def from_file(self, path: str) -> Tuple[np.ndarray, bool]:
        """Extract landmarks for an image file, using the disk cache if set."""
        cache_path = None
        if self.cache_dir:
            with open(path, "rb") as f:
                cache_path = self._cache_path(f.read())
            if cache_path and os.path.exists(cache_path):
                return np.load(cache_path), bool(np.load(cache_path).any())
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if img is None:
            return self._zeros(), False
        lm, ok = self.extract(img)
        if cache_path:
            np.save(cache_path, lm)
        return lm, ok
