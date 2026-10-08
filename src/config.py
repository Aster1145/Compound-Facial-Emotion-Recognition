"""Central configuration for the Dual-Domain Tri-Branch model (v2).

_v1 → v2 changes_
* ``VIT_MODEL_NAME`` is kept only as an offline fallback; the global branch
  now defaults to the Hugging Face facial-expression ViT (``HF_FACE_VIT_ID``).
* ``LSTM_*`` fields are gone — the ALSTM branch was removed entirely and
  replaced by the facial-geometry MLP (see :mod:`src.branches`).
* New sections: local backbone selection (+ optional VGGFace2 weights),
  landmark geometry, Poisson-blending synthesis, and the two-phase
  curriculum (SupCon → Focal) hyper-parameters.
* Dataset roots keep the original Kaggle defaults but can now be overridden
  with environment variables so the same code runs outside Kaggle.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


class Config:
    # ── Dataset paths (env-overridable; Kaggle defaults from v1) ──────────
    RAF_DB_DIR = os.environ.get(
        "RAF_DB_DIR", "/kaggle/input/datasets/prajwalm2213/raf-db/RAF-DB"
    )
    FER2013_DIR = os.environ.get(
        "FER2013_DIR", "/kaggle/input/datasets/prajwalm2213/fer-2013"
    )
    JAFFE_DIR = os.environ.get(
        "JAFFE_DIR", "/kaggle/input/datasets/prajwalm2213/dataset2/jaffe/jaffe"
    )
    CKP_DIR = os.environ.get(
        "CKP_DIR", "/kaggle/input/datasets/shuvoalok/ck-dataset"
    )
    AFFECTNET_DIR = os.environ.get(
        "AFFECTNET_DIR", "/kaggle/input/datasets/mstjebashazida/affectnet"
    )

    # ── Output ────────────────────────────────────────────────────────────
    CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/kaggle/working/checkpoints")
    LOG_DIR = os.environ.get("LOG_DIR", "/kaggle/working/logs")
    EVAL_DIR = os.environ.get("EVAL_DIR", "/kaggle/working/eval_results")
    LANDMARK_CACHE_DIR = os.environ.get("LANDMARK_CACHE_DIR", "")  # "" = no cache

    # ── Model ─────────────────────────────────────────────────────────────
    IMAGE_SIZE = 224
    NUM_CLASSES = 11          # compound emotions only
    BASIC_CLASSES_N = 7       # basic emotions (label-offset pseudo-classes)
    EMBED_DIM = 512
    DROPOUT = 0.3

    # Branch 1 — local textures: timm CNN backbone.
    LOCAL_BACKBONE = os.environ.get("LOCAL_BACKBONE", "efficientnet_b0")  # or "resnet50"
    # Optional VGGFace2 weights for Branch 1: local .pth/.bin path, http(s) URL,
    # or "<hub-repo>/<file>" on Hugging Face. None → timm ImageNet weights.
    VGGFACE2_WEIGHTS = os.environ.get("VGGFACE2_WEIGHTS") or None

    # Branch 2 — global context: facial-expression ViT from Hugging Face.
    HF_FACE_VIT_ID = os.environ.get("HF_FACE_VIT_ID", "trpakov/vit-face-expression")
    VIT_MODEL_NAME = "vit_base_patch16_224"  # offline fallback for Branch 2

    # Branch 3 — facial geometry MLP over 468 × (x, y, z) landmarks.
    NUM_LANDMARKS = 468
    LANDMARK_DIM = 3
    GEOMETRY_IN = NUM_LANDMARKS * LANDMARK_DIM  # 1404
    GEOMETRY_HIDDEN = 1024

    # ── Landmark-guided Poisson-blending synthesis ────────────────────────
    SYNTHETIC_PER_CLASS = int(os.environ.get("SYNTHETIC_PER_CLASS", "800"))
    MEDIAPIPE_CONFIDENCE = 0.5
    SPLICE_DILATE_PX = 3
    SPLICE_FEATHER_PX = 7

    # ── Training ──────────────────────────────────────────────────────────
    BATCH_SIZE = 32           # ≥64 recommended for Phase-1 SupCon if VRAM allows
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    SEED = 42
    NUM_WORKERS = 4

    # Phase 1 — latent-space clustering with Supervised Contrastive Loss.
    PHASE1_EPOCHS = 40
    PHASE1_LR = 3e-4
    SUPCON_TEMPERATURE = 0.07
    PHASE1_KNN_K = 5
    PHASE1_USE_RAFDB = True   # include RAF-DB train (offset labels) for Phase 1

    # Phase 2 — classification with Focal Loss.
    PHASE2_EPOCHS = 25
    PHASE2_LR = 1e-4
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = None        # None | "balanced" | per-class Tensor

    # ── Compound class definitions ────────────────────────────────────────
    COMPOUND_CLASSES = [
        "Happily Surprised",   "Happily Disgusted",  "Sadly Fearful",
        "Sadly Angry",         "Sadly Surprised",    "Sadly Disgusted",
        "Fearfully Angry",     "Fearfully Surprised", "Angrily Surprised",
        "Angrily Disgusted",   "Disgustedly Surprised",
    ]
    BASIC_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    # compound_idx -> (basic_emotion_A, basic_emotion_B); indices into BASIC_CLASSES.
    # Emotion A supplies eyes/eyebrows, Emotion B supplies mouth/jaw.
    COMPOUND_TO_BASIC = {
        0: (3, 5), 1: (3, 1), 2: (4, 2),  3: (4, 0), 4: (4, 5),
        5: (4, 1), 6: (2, 0), 7: (2, 5),  8: (0, 5), 9: (0, 1), 10: (1, 5),
    }
    NEUTRAL_IDX = 6  # basic index of the neutral base face used for splicing

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_config(**overrides) -> Config:
    """Instantiate :class:`Config`, applying any keyword overrides."""
    cfg = Config()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Unknown Config field: {key}")
        setattr(cfg, key, value)
    for d in (cfg.CHECKPOINT_DIR, cfg.LOG_DIR, cfg.EVAL_DIR):
        os.makedirs(d, exist_ok=True)
    if getattr(cfg, "LANDMARK_CACHE_DIR", ""):
        os.makedirs(cfg.LANDMARK_CACHE_DIR, exist_ok=True)
    return cfg


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


cfg = make_config()
set_seed(cfg.SEED)
