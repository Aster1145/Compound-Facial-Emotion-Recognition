"""Compound Facial Emotion Recognition — Dual-Domain Tri-Branch (v2).

Refactored codebase. Public API::

    from src.config   import Config
    from src.model    import CompoundEmotionModel, build_tribranch_model
    from src.data     import get_dataloaders
    from src.train    import run_stage, train_model_two_phase
    from src.splicing import LandmarkPoissonSplicer, SyntheticCompoundDataset
    from src.losses   import SupervisedContrastiveLoss, FocalLoss

Module map
----------
config   — central hyper-parameters & dataset paths.
attention— retained CBAM blocks (Channel / Spatial attention), unchanged.
branches — Branch 1 local textures (timm CNN + CBAM), Branch 2 global
             context (HF facial-expression ViT), Branch 3 facial geometry MLP.
fusion   — AttentionFusion gate producing dynamic α, β, γ weights.
model    — CompoundEmotionModel (dual-domain tri-branch) + factory.
landmarks— MediaPipe FaceMesh landmark extraction + region index sets.
splicing — landmark-guided Poisson-blending synthesis + _splice().
data     — datasets, tri-branch transforms, loaders, samplers.
losses   — Supervised Contrastive Loss + Focal Loss.
train    — two-phase curriculum (run_stage rewrite + orchestrator).
eval     — metrics for the tri-branch model.
"""

from src.config import Config
from src.model import CompoundEmotionModel, build_tribranch_model
from src.losses import SupervisedContrastiveLoss, FocalLoss

__all__ = [
    "Config",
    "CompoundEmotionModel",
    "build_tribranch_model",
    "SupervisedContrastiveLoss",
    "FocalLoss",
]

__version__ = "2.0.0"
