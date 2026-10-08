# Compound Facial Emotion Recognition — Dual-Domain Tri-Branch (v2)

> **v2 refactor:** the notebook-only v1 codebase (SA-CNN + ALSTM + ViT, rigid
> splicing, single-loss 3-stage curriculum) is preserved below for the 30%
> review record. All new work lives in the **`src/` package** + the
> **`train_tribranch.py`** CLI, implementing a domain-specific tri-branch
> model with landmark-guided Poisson-blending synthesis and a two-phase
> SupCon → Focal training curriculum.

## 🧠 v2 Architecture (Dual-Domain Tri-Branch)

```
face image (224×224) ─┬─► Branch 1 LOCAL TEXTURES ──────────────────► 512-D
                      │   efficientnet_b0 / resnet50 (timm, VGGFace2
                      │   weights optional) + retained CBAM on the
                      │   final-stage feature map → 512-D projection
                      │
                      ├─► Branch 2 GLOBAL CONTEXT ──────────────────► 512-D
                      │   trpakov/vit-face-expression (HF transformers,
                      │   AffectNet-trained); head removed, [CLS] → 512-D
                      │
landmarks (468×3) ────┴─► Branch 3 FACIAL GEOMETRY (replaces ALSTM) ─► 512-D
                          3-layer MLP: 1404 → 1024 → 1024 → 512
                                              │
                              [ AttentionFusion gate → α, β, γ ]
                                              │
                                   fused embedding (512-D)
                                              │
                               Phase 1: SupCon clustering (head frozen)
                               Phase 2: Focal-loss head (γ=2.0, branches frozen)
                                              │
                                    11 compound classes
```

| v1 component | v2 replacement |
|---|---|
| Custom SA-CNN | `timm` `efficientnet_b0`/`resnet50` (+ optional VGGFace2 weights) with retained CBAM |
| `vit_base_patch16_224` (ImageNet) | `trpakov/vit-face-expression` (AffectNet facial-expression ViT, `[CLS]` → 512-D) |
| `ALSTM` | **Removed.** `GeometryMLP` over 468×3 MediaPipe landmarks |
| Rigid alpha-blend / horizontal splice | Landmark masks (eyes/eyebrows from A, mouth/jaw from B) + `cv2.seamlessClone` onto a neutral base |
| `CrossEntropyLoss` 3-stage curriculum | Phase 1: SupCon on fused embeddings (head frozen) → Phase 2: Focal Loss γ=2.0 (extractors frozen) |
| Mixed basic/compound label space | Disjoint spaces: RAF-DB offset to 11–17 in Phase 1; compound-only head/val |

## 📂 v2 Codebase Map

```
src/
├── config.py     # hyper-parameters, paths, compound taxonomy, phase settings
├── attention.py  # retained Channel/Spatial/CBAM blocks (unchanged)
├── branches.py   # LocalTextureEncoder / GlobalContextEncoder / GeometryMLP
├── fusion.py     # AttentionFusion gate (dynamic α, β, γ)
├── model.py      # CompoundEmotionModel + build_tribranch_model
├── landmarks.py  # MediaPipe FaceMesh extraction + region index sets
├── splicing.py   # LandmarkPoissonSplicer + SyntheticCompoundDataset._splice()
├── data.py       # datasets, tri-branch transforms, loaders, samplers
├── losses.py     # SupervisedContrastiveLoss + FocalLoss
├── train.py      # rewritten run_stage (contrastive/classification) + orchestrator
└── eval.py       # compound-label metrics for triple batches
train_tribranch.py  # CLI: python train_tribranch.py --batch-size 64 ...
requirements.txt    # v2 dependencies
```

## 🚀 v2 Quickstart (Kaggle)

```bash
pip install -r requirements.txt
python train_tribranch.py --batch-size 64 --epochs1 40 --epochs2 25

# ResNet-50 branch with VGGFace2 warm-start:
python train_tribranch.py --local-backbone resnet50 \
    --vggface2-weights /kaggle/input/vggface2/resnet50_ft.pth
```

Library usage:

```python
from src.config import make_config
from src.data import get_dataloaders
from src.model import build_tribranch_model
from src.train import train_model_two_phase
from src.eval import evaluate_tribranch

cfg = make_config(BATCH_SIZE=64)
loaders = get_dataloaders(cfg)          # train_contrast / train_focal / val / rafdb_test
model = build_tribranch_model(cfg)
summary, ckpt, log = train_model_two_phase(model, loaders, cfg)
print(evaluate_tribranch(model, loaders["val"], cfg)["f1_macro"])
```

Notes:
- **VGGFace2 weights** for Branch 1 accept a local `.pth`/`.bin` file, an
  `http(s)` URL, or a `<repo>/<file>` Hugging Face id (loaded
  `strict=False`). Without it, timm ImageNet weights are used.
- **Offline fallback:** if the HF facial-expression ViT is unreachable,
  Branch 2 falls back to a timm ViT with a warning.
- **Landmark alignment:** tri-branch training transforms are photometric-only
  (no flip/rotate/affine), because Branch-3 landmarks are extracted from the
  pre-augmentation image.
- **Validation labels:** val/test use compound labels (synthetic val from
  disjoint FER-2013 test pools); RAF-DB test (basic labels) is kept as an
  auxiliary transfer check.

---

# Compound Facial Emotion Recognition using Morphological Splicing & Multi-Branch Attention-Transformer Networks (v1 — 30% review record)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Project Status: 30% Complete](https://img.shields.io/badge/Status-30%25%20Completed%20(Phase%201)-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced deep learning framework designed to categorize **11 fine-grained Compound Emotion Classes** (e.g., *Happily Surprised*, *Fearfully Angry*, *Sadly Surprised*) by combining Spatial Attention CNNs (SA-CNN), Attention LSTMs (ALSTM), and Vision Transformers (ViT) with a dynamic **Generic Attention Fusion Module**.

---

## 📌 Major Project Status — Phase 1 Review (30% Completion)

This repository contains the complete codebase, benchmark suite, notebook evaluations, and **30% Progress Presentation Deck** for the College Major Project Phase 1 Evaluation.

### Key Milestones Completed (Phase 1):
- [x] **Problem Formulation & Literature Survey:** Defined 11 compound emotion taxonomy beyond traditional 7 basic emotions.
- [x] **Multi-Branch Architecture Design:** Constructed SA-CNN, ALSTM, ViT-Base, and Dynamic Gated Softmax Attention Fusion module in PyTorch.
- [x] **Preprocessing & Augmentation:** Implemented dataset loaders (RAF-DB, AffectNet, CK+, JAFFE) and Morphological Splicing data enrichment.
- [x] **Benchmark Suite (6 Models):** Evaluated `ConvNeXt_ViT`, `SACNN_ViT`, `EfficientNetV2_ViT`, `ProposedModel`, `ALSTM_ViT`, and `SACNN_ALSTM`.
- [x] **Presentation Deck Generated:** Provided [`Compound_Facial_Emotion_Recognition_30Percent_Presentation.pptx`](./Compound_Facial_Emotion_Recognition_30Percent_Presentation.pptx) for review defense.

---

## 📊 Empirical Benchmarking & Results

All models were evaluated on the **RAF-DB test set (3,068 images)** using a 3-Stage Curriculum Learning pipeline (Backbone Frozen $\rightarrow$ Partial Unfreeze $\rightarrow$ Full Unfreeze):

| Architecture Name | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Params (M) | FLOPs (G) | Inference Speed (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ConvNeXt + ViT** | **87.97%** | **82.89%** | **78.74%** | **80.43%** | 114.9M | 42.6G | 51.0 FPS |
| **SA-CNN + ViT** | **87.61%** | 81.15% | **78.89%** | 79.81% | 103.1M | 44.3G | 53.1 FPS |
| **EfficientNetV2 + ViT** | 87.35% | 82.45% | 77.74% | 79.65% | 107.6M | 39.1G | 40.9 FPS |
| **Proposed Hybrid (Splicing+SACNN+ALSTM+ViT)** | 86.86% | 80.92% | 77.18% | 78.71% | 106.4M | 45.5G | 43.4 FPS |
| **ALSTM + ViT** | 86.86% | 81.90% | 77.36% | 79.16% | 89.9M | 34.9G | 53.4 FPS |
| **SA-CNN + ALSTM** *(Lightweight Edge)* | 80.67% | 63.24% | 62.03% | 62.38% | **20.1M** | **11.8G** | **79.8 FPS** |

---

## 🏗️ System Architecture Pipeline

```
[ Input Face (224x224) ]
           │
  (Morphological Splicing)
           │
 ┌─────────┼─────────┐
 │         │         │
 ▼         ▼         ▼
SA-CNN   ALSTM      ViT
 (512)   (512)     (512)
 │         │         │
 └─────────┼─────────┘
           │
   [ Dynamic Gated ]
   [ Softmax Fusion]
           │
 [ MLP Classifier Head ]
           │
   [ 11 Compound Classes ]
```

---

## 📂 Repository Contents

```
.
├── Compound_Facial_Emotion_Recognition_30Percent_Presentation.pptx  # 14-slide PPT presentation for 30% review
├── build_full_presentation.py                                       # Script to generate PPTX deck
├── compound-results-fixed.ipynb                                     # Main evaluation & benchmarking notebook
├── compound-results (7).ipynb                                       # Raw experimental notebook & plots
├── compound-results (4).ipynb                                       # Stage 1-3 checkpoints log
└── README.md                                                        # Project documentation
```

---

## 🚀 Presentation & Review Deck

The 30% Completion PowerPoint file [`Compound_Facial_Emotion_Recognition_30Percent_Presentation.pptx`](./Compound_Facial_Emotion_Recognition_30Percent_Presentation.pptx) is formatted for faculty panel defense. It contains:
1. Introduction & Motivation (Basic vs. Compound Emotions)
2. Problem Statement & Key Research Objectives
3. Literature Survey & Identified Technical Gaps
4. Proposed System Methodology & Architecture Diagrams
5. Datasets (RAF-DB, AffectNet, CK+, JAFFE) & Hyperparameters
6. Phase 1 Milestone Progress (30% Status Table)
7. Empirical Results & Performance Comparison Table
8. Computational Efficiency Trade-Offs (Accuracy vs. FPS)
9. Error Analysis & Confusion Matrix Breakdown (*Fearfully Angry* vs *Sadly Surprised*)
10. Remaining Roadmap (70% Work Plan: Focal Loss, INT8 Quantization, Real-time Web App)
11. Project Timeline & Gantt Chart Schedule

---

## 🗓️ Remaining Work Roadmap (Phases 2 & 3)

- **Phase 2 (30% $\rightarrow$ 70%):** Focal Loss implementation for fine-grained compound emotion pairs, hyperparameter optimization, and INT8 model quantization (ONNX / TensorRT).
- **Phase 3 (70% $\rightarrow$ 100%):** Real-time webcam Web Application GUI, live emotion logging, final thesis report, and research paper publication.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
