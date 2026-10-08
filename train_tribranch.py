"""CLI entry point — two-phase training of the Dual-Domain Tri-Branch model.

Example (Kaggle)::

    pip install -r requirements.txt
    python train_tribranch.py --batch-size 64 --epochs1 40 --epochs2 25

Local paths can be passed explicitly or via env vars (``RAF_DB_DIR``,
``FER2013_DIR``, ``CHECKPOINT_DIR``, ...). See :mod:`src.config`.
"""
import argparse
import json
import os

from src.config import make_config, set_seed
from src.data import get_dataloaders
from src.eval import evaluate_tribranch
from src.model import build_tribranch_model
from src.train import train_model_two_phase


def parse_args():
    p = argparse.ArgumentParser(description="Train the tri-branch compound-emotion model.")
    p.add_argument("--raf-db-dir", default=None)
    p.add_argument("--fer2013-dir", default=None)
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--log-dir", default="./logs")
    p.add_argument("--eval-dir", default="./eval_results")
    p.add_argument("--local-backbone", default="efficientnet_b0",
                   choices=["efficientnet_b0", "resnet50"])
    p.add_argument("--vggface2-weights", default=None,
                   help="Local path / URL / <repo>/<file> for Branch-1 VGGFace2 weights.")
    p.add_argument("--hf-face-vit", default="trpakov/vit-face-expression")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs1", type=int, default=40, help="Phase-1 (SupCon) epochs.")
    p.add_argument("--epochs2", type=int, default=25, help="Phase-2 (Focal) epochs.")
    p.add_argument("--lr1", type=float, default=3e-4)
    p.add_argument("--lr2", type=float, default=1e-4)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--supcon-temp", type=float, default=0.07)
    p.add_argument("--synthetic-per-class", type=int, default=800)
    p.add_argument("--no-rafdb-phase1", action="store_true",
                   help="Exclude RAF-DB from Phase-1 contrastive training.")
    p.add_argument("--resume-phase", type=int, default=1, choices=[1, 2])
    p.add_argument("--ckpt-prefix", default="TriBranch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    overrides = dict(
        CHECKPOINT_DIR=args.checkpoint_dir, LOG_DIR=args.log_dir, EVAL_DIR=args.eval_dir,
        LOCAL_BACKBONE=args.local_backbone, VGGFACE2_WEIGHTS=args.vggface2_weights,
        HF_FACE_VIT_ID=args.hf_face_vit, BATCH_SIZE=args.batch_size,
        PHASE1_EPOCHS=args.epochs1, PHASE2_EPOCHS=args.epochs2,
        PHASE1_LR=args.lr1, PHASE2_LR=args.lr2,
        FOCAL_GAMMA=args.focal_gamma, SUPCON_TEMPERATURE=args.supcon_temp,
        SYNTHETIC_PER_CLASS=args.synthetic_per_class,
        PHASE1_USE_RAFDB=not args.no_rafdb_phase1,
        SEED=args.seed, NUM_WORKERS=args.num_workers)
    if args.raf_db_dir:
        overrides["RAF_DB_DIR"] = args.raf_db_dir
    if args.fer2013_dir:
        overrides["FER2013_DIR"] = args.fer2013_dir
    cfg = make_config(**overrides)
    set_seed(cfg.SEED)

    print(f"Device          : {cfg.DEVICE}")
    print(f"Local backbone  : {cfg.LOCAL_BACKBONE} "
          f"(VGGFace2: {cfg.VGGFACE2_WEIGHTS or 'timm pretrained'})")
    print(f"Global ViT      : {cfg.HF_FACE_VIT_ID}")
    print(f"RAF-DB exists   : {os.path.exists(cfg.RAF_DB_DIR)}")
    print(f"FER-2013 exists : {os.path.exists(cfg.FER2013_DIR)}")

    loaders = get_dataloaders(cfg)
    model = build_tribranch_model(cfg)
    print(f"Total params    : {sum(p.numel() for p in model.parameters()):,}")

    summary, ckpt_path, _ = train_model_two_phase(
        model, loaders, cfg, ckpt_prefix=args.ckpt_prefix,
        resume_phase=args.resume_phase)

    print("\nFinal evaluation on compound val split:")
    results = evaluate_tribranch(model, loaders["val"], cfg)
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                "f1_weighted", "auc_macro"):
        print(f"  {key:16s}: {results[key]:.2f}")
    print(f"  fusion α,β,γ  : {[f'{w:.3f}' for w in results['fusion_weights_mean']]}")
    eval_path = os.path.join(cfg.EVAL_DIR, f"{args.ckpt_prefix}_eval.json")
    with open(eval_path, "w") as f:
        json.dump({**results, "train_summary": summary}, f, indent=2, default=str)
    print(f"  Metrics -> {eval_path}")


if __name__ == "__main__":
    main()
