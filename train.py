"""
train.py — PC2FoodNet 5-fold stratified cross-validation trainer.

Data leakage prevention strategy
──────────────────────────────────
1. StratifiedKFold splits indices BEFORE any image is loaded or transformed.
2. Training augmentation is applied ONLY to the training-fold indices via
   SubsetWithTransform; validation indices receive only resize+crop+normalise.
3. Nutrition priors (density, kcal/g) come from the domain-knowledge JSON,
   not from training-image statistics — so no leakage across folds.
4. Each fold trains a completely fresh model (no parameter sharing).
5. Normalisation uses fixed ImageNet statistics (not computed from any fold).
6. Checkpoints from fold N are never used to initialise fold N+1
   (unless --warmstart_from is explicitly supplied).

Usage
──────
  # 5-fold CV (default)
  python train.py --epochs 50 --batch_size 32

  # Warm-start every fold from a pretrained checkpoint
  python train.py --epochs 30 --warmstart_from runs/full/best.pt

  # Eval-only on a saved fold checkpoint
  python train.py --eval_only --checkpoint runs/cv/fold_1/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import build_kfold_splits, build_nutrition_tensors
from model import PC2FoodNet
from results_manager import ResultsManager


# ────────────────────────────────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────────────────────────────────

def gaussian_nll(pred: torch.Tensor, logvar: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logvar + ((pred - target) ** 2) / (logvar.exp() + 1e-8)).mean()


class PC2Loss(nn.Module):
    """
    phys_warmup_epochs: number of epochs during which the physics consistency
    term is linearly ramped from 0 → λ_phys.  This prevents the untrained
    volume/physics heads from dominating gradients at the start of training.
    """
    def __init__(self, λ_cls=1.0, λ_vol=0.5, λ_wgt=1.0,
                 λ_nrg=1.0, λ_phys=0.2, phys_warmup_epochs: int = 10):
        super().__init__()
        self.λ = dict(cls=λ_cls, vol=λ_vol, wgt=λ_wgt, nrg=λ_nrg, phys=λ_phys)
        self.phys_warmup = phys_warmup_epochs
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, out, labels, epoch: int = 999,
                gt_volume=None, gt_weight=None, gt_energy=None):
        loss_cls = self.ce(out["logits"], labels)
        total    = self.λ["cls"] * loss_cls
        details  = {"cls": loss_cls.item()}

        if gt_weight is not None:
            lw = gaussian_nll(out["weight"], out["logvar_weight"], gt_weight)
            total += self.λ["wgt"] * lw;  details["wgt"] = lw.item()
        if gt_energy is not None:
            le = gaussian_nll(out["energy"], out["logvar_energy"], gt_energy)
            total += self.λ["nrg"] * le;  details["nrg"] = le.item()
        if gt_volume is not None:
            lv = gaussian_nll(out["volume"], out["logvar_volume"], gt_volume)
            total += self.λ["vol"] * lv;  details["vol"] = lv.item()

        # Ramp physics consistency term gradually — avoids exploding loss at init
        phys_scale = min(1.0, epoch / max(self.phys_warmup, 1))
        phys = nn.functional.smooth_l1_loss(
            out["weight"].clamp(0, 5000),
            out["physics_weight"].detach().clamp(0, 5000)
        )
        total += self.λ["phys"] * phys_scale * phys
        details["phys"]  = phys.item()
        details["phys_scale"] = phys_scale
        details["total"] = total.item()
        return total, details


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self):    self.val = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.sum += val * n; self.count += n; self.val = val
    @property
    def avg(self): return self.sum / max(self.count, 1)


def topk(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    return (logits.topk(k, 1).indices == labels.unsqueeze(1)).any(1).float().mean().item() * 100


# ────────────────────────────────────────────────────────────────────────────
# One epoch
# ────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model, loader: DataLoader, criterion: PC2Loss,
    optimizer: AdamW | None, device: torch.device,
    scaler: torch.amp.GradScaler, split: str = "train",
    epoch: int = 999,
) -> dict[str, float]:
    is_train = split == "train"
    model.train(is_train)
    torch.set_grad_enabled(is_train)
    meters = {k: AverageMeter() for k in [
        "total", "cls", "phys", "wgt", "vol", "nrg",
        "acc1", "acc5", "rmse_vol", "rmse_wgt", "rmse_nrg"
    ]}

    for imgs, labels, gt_vol, gt_wgt, gt_nrg in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        gt_vol = gt_vol.to(device, non_blocking=True)
        gt_wgt = gt_wgt.to(device, non_blocking=True)
        gt_nrg = gt_nrg.to(device, non_blocking=True)
        bs     = imgs.size(0)

        with torch.amp.autocast("cuda"):
            out           = model(imgs)
            loss, details = criterion(out, labels, epoch=epoch,
                                      gt_volume=gt_vol, gt_weight=gt_wgt, gt_energy=gt_nrg)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        meters["total"].update(details["total"], bs)
        meters["cls"].update(details.get("cls", 0), bs)
        meters["phys"].update(details.get("phys", 0), bs)
        meters["wgt"].update(details.get("wgt", 0), bs)
        meters["vol"].update(details.get("vol", 0), bs)
        meters["nrg"].update(details.get("nrg", 0), bs)

        meters["acc1"].update(topk(out["logits"], labels, 1), bs)
        meters["acc5"].update(topk(out["logits"], labels, 5), bs)

        # RMSE
        with torch.no_grad():
            rmse_vol = torch.sqrt(torch.nn.functional.mse_loss(out["volume"], gt_vol))
            rmse_wgt = torch.sqrt(torch.nn.functional.mse_loss(out["weight"], gt_wgt))
            rmse_nrg = torch.sqrt(torch.nn.functional.mse_loss(out["energy"], gt_nrg))
            meters["rmse_vol"].update(rmse_vol.item(), bs)
            meters["rmse_wgt"].update(rmse_wgt.item(), bs)
            meters["rmse_nrg"].update(rmse_nrg.item(), bs)

    torch.set_grad_enabled(True)
    return {k: m.avg for k, m in meters.items()}


# ────────────────────────────────────────────────────────────────────────────
# Model / optimiser factories
# ────────────────────────────────────────────────────────────────────────────

def make_model(args, densities, kcal_per_g, num_classes, device) -> PC2FoodNet:
    """Always returns a freshly initialised model (no parameter sharing)."""
    m = PC2FoodNet(
        num_classes    = num_classes,
        densities      = densities,
        kcal_per_g     = kcal_per_g,
        pretrained     = True,
        projection_dim = args.projection_dim,
        dropout        = args.dropout,
        residual_bound = args.residual_bound,
    ).to(device)
    m.initialize_regression_biases(250.0, 200.0, 350.0)
    return m


def make_scheduler(optimizer, args):
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer,
                               T_max=max(args.epochs - args.warmup_epochs, 1),
                               eta_min=1e-6)
    return SequentialLR(optimizer, [warmup, cosine],
                        milestones=[args.warmup_epochs])


# ────────────────────────────────────────────────────────────────────────────
# Train one fold
# ────────────────────────────────────────────────────────────────────────────

def train_fold(
    *,
    fold: int,
    model: PC2FoodNet,
    train_dl: DataLoader,
    val_dl: DataLoader,
    args,
    device: torch.device,
    fold_dir: Path,
    rm: ResultsManager,
) -> dict[str, float]:
    fold_dir.mkdir(parents=True, exist_ok=True)

    criterion = PC2Loss()
    scaler    = torch.amp.GradScaler("cuda")
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args)
    writer    = SummaryWriter(fold_dir / "tb")

    best_acc: float   = 0.0
    best_metrics: dict = {}

    print(f"\n{'─'*65}")
    print(f"  FOLD {fold+1}/{args.n_folds}  |  "
          f"train={len(train_dl.dataset):,}  val={len(val_dl.dataset):,}")
    print(f"{'─'*65}")

    for epoch in range(args.epochs):
        t0 = time.time()

        train_m = run_epoch(model, train_dl, criterion, optimizer,
                            device, scaler, "train", epoch=epoch)
        val_m   = run_epoch(model, val_dl,   criterion, None,
                            device, scaler, "val",   epoch=epoch)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        is_best = val_m["acc1"] > best_acc
        if is_best:
            best_acc     = val_m["acc1"]
            best_metrics = dict(val_m)

        # ── Record ──────────────────────────────────────────────────────
        rm.record_epoch(fold, epoch, "train", train_m)
        rm.record_epoch(fold, epoch, "val",   val_m)

        # TensorBoard
        for k, v in {**{f"train/{k}": mv for k, mv in train_m.items()},
                      **{f"val/{k}":   mv for k, mv in val_m.items()}}.items():
            writer.add_scalar(k, v, epoch + 1)
        writer.add_scalar("lr", lr_now, epoch + 1)

        # Console
        star = "  ★" if is_best else ""
        print(
            f"  [{epoch+1:>3}/{args.epochs}] {elapsed:.0f}s  lr={lr_now:.2e}  "
            f"train loss={train_m['total']:.4f} acc@1={train_m['acc1']:.2f}%  "
            f"val loss={val_m['total']:.4f} acc@1={val_m['acc1']:.2f}%  "
            f"acc@5={val_m['acc5']:.2f}%{star}"
        )

        # Checkpoint
        ckpt = {
            "fold": fold, "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "args": vars(args),
        }
        torch.save(ckpt, fold_dir / "last.pt")
        if is_best:
            torch.save(ckpt, fold_dir / "best.pt")

    writer.close()
    return best_metrics


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("PC2FoodNet 5-fold CV Trainer")

    # Data
    p.add_argument("--data_dir",     default="turkish-food")
    p.add_argument("--img_size",     type=int,   default=224)
    p.add_argument("--num_workers",  type=int,   default=8)
    p.add_argument("--seed",         type=int,   default=42)

    # CV
    p.add_argument("--n_folds", type=int, default=4,
                   help="Number of stratified CV folds (80:20 each)")
    p.add_argument("--start_fold",   type=int,   default=1,
                   help="Fold to start from (1-indexed)")

    # Training
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs",type=int,   default=5)

    # Model
    p.add_argument("--projection_dim", type=int,   default=512)
    p.add_argument("--dropout",        type=float, default=0.30)
    p.add_argument("--residual_bound", type=float, default=0.20)

    # Misc
    p.add_argument("--output_dir",      default="runs/cv")
    p.add_argument("--results_dir",     default="results/cv")
    p.add_argument("--warmstart_from",  default=None,
                   help="Checkpoint to warm-start EVERY fold from")
    p.add_argument("--checkpoint",      default=None,
                   help="Single fold checkpoint (for --eval_only)")
    p.add_argument("--eval_only",       action="store_true")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()

    # ── GPU ────────────────────────────────────────────────────────────────
    assert torch.cuda.is_available(), "GPU required — CPU training is disabled."
    device = torch.device("cuda")
    print(f"[✓] GPU     : {torch.cuda.get_device_name(0)}")
    print(f"[✓] Mode    : {args.n_folds}-fold stratified CV  (80 % train / 20 % val)")
    print(f"[✓] Leakage : StratifiedKFold on indices · per-subset transforms · "
          "fresh model per fold · fixed ImageNet normalisation")

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    output_dir  = Path(args.output_dir)
    results_dir = Path(args.results_dir)

    # ── Nutrition priors ────────────────────────────────────────────────────
    json_path = Path(args.data_dir) / "porsiyon_nutrition_data.json"

    # ── Build ALL folds (indices fixed before any image is read) ───────────
    print(f"\n[*] Building {args.n_folds} stratified folds ...")
    folds, class_names = build_kfold_splits(
        args.data_dir,
        n_folds     = args.n_folds,
        batch_size  = args.batch_size,
        img_size    = args.img_size,
        num_workers = args.num_workers,
        seed        = args.seed,
    )
    num_classes = len(class_names)
    densities, kcal_per_g = build_nutrition_tensors(json_path, class_names)

    # Save class map once
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2)
    )

    # ── Eval-only mode ─────────────────────────────────────────────────────
    if args.eval_only:
        assert args.checkpoint, "--checkpoint is required for --eval_only"
        ckpt  = torch.load(args.checkpoint, map_location=device)
        model = make_model(args, densities, kcal_per_g, num_classes, device)
        model.load_state_dict(ckpt["model"])
        fold_i = ckpt.get("fold", 0)
        _, val_dl = folds[fold_i]
        criterion = PC2Loss()
        scaler    = torch.amp.GradScaler("cuda")
        m = run_epoch(model, val_dl, criterion, None, device, scaler, "val")
        print(f"\nEval  Acc@1={m['acc1']:.2f}%  Acc@5={m['acc5']:.2f}%  "
              f"Loss={m['total']:.4f}")
        return

    # ── Cross-validation ───────────────────────────────────────────────────
    rm            = ResultsManager(results_dir, n_folds=args.n_folds)
    fold_results:  list[dict] = []
    total_params:  int = 0

    for fold_i, (train_dl, val_dl) in enumerate(folds):
        fold_dir = output_dir / f"fold_{fold_i + 1}"

        if (fold_i + 1) < args.start_fold:
            print(f"Skipping Fold {fold_i + 1} (already completed)")
            import csv
            history_file = results_dir / f"fold_{fold_i + 1}" / "train_history.csv"
            if history_file.exists():
                with open(history_file) as f:
                    for row in csv.DictReader(f):
                        split = row.pop("split")
                        for k, v in row.items():
                            row[k] = float(v) if "." in v or "e" in v.lower() else (int(v) if v.isdigit() else v)
                        rm.history[fold_i][split].append(row)
                val_rows = rm.history[fold_i]["val"]
                if val_rows:
                    best = max(val_rows, key=lambda x: x["acc1"])
                    fold_results.append({k: v for k, v in best.items() if k != "epoch"})
            continue

        # Fresh model — NO parameter sharing between folds
        model = make_model(args, densities, kcal_per_g, num_classes, device)
        if total_params == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n[model] Parameters: {total_params:,}")

        # Optional warm-start from external checkpoint (e.g. full-dataset run)
        if args.warmstart_from:
            ws_ckpt = torch.load(args.warmstart_from, map_location=device)
            model.load_state_dict(ws_ckpt["model"])
            print(f"  [fold {fold_i+1}] warm-started from {args.warmstart_from}")

        # Train
        best_m = train_fold(
            fold     = fold_i,
            model    = model,
            train_dl = train_dl,
            val_dl   = val_dl,
            args     = args,
            device   = device,
            fold_dir = fold_dir,
            rm       = rm,
        )
        fold_results.append(best_m)

        # Per-fold results artifacts
        rm.save_fold_artifacts(fold_i, model, val_dl, class_names, device)

        print(f"\n  ✔ Fold {fold_i+1} best → "
              f"Acc@1={best_m['acc1']:.2f}%  "
              f"Acc@5={best_m['acc5']:.2f}%  "
              f"Loss={best_m['total']:.4f}")

    # ── Aggregate results ───────────────────────────────────────────────────
    rm.finalise(fold_results)

    mean_acc1 = np.mean([r["acc1"] for r in fold_results])
    std_acc1  = np.std( [r["acc1"] for r in fold_results])
    mean_acc5 = np.mean([r["acc5"] for r in fold_results])
    std_acc5  = np.std( [r["acc5"] for r in fold_results])
    mean_loss = np.mean([r["total"] for r in fold_results])
    std_loss  = np.std( [r["total"] for r in fold_results])

    print(f"\n{'═'*65}")
    print(f"  {args.n_folds}-Fold CV Final Results  (80:20 stratified split)")
    print(f"{'═'*65}")
    print(f"  Acc@1  : {mean_acc1:.2f} ± {std_acc1:.2f} %")
    print(f"  Acc@5  : {mean_acc5:.2f} ± {std_acc5:.2f} %")
    print(f"  Loss   : {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"  Results: {results_dir.resolve()}")
    print(f"{'═'*65}\n")

    # Save final summary JSON
    summary = {
        "n_folds": args.n_folds,
        "split": "stratified 80:20 per fold",
        "mean_acc1": mean_acc1, "std_acc1": std_acc1,
        "mean_acc5": mean_acc5, "std_acc5": std_acc5,
        "mean_loss": mean_loss, "std_loss": std_loss,
        "fold_results": fold_results,
    }
    (results_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[✓] cv_summary.json saved.")


if __name__ == "__main__":
    main()
