"""
results_manager.py — Centralised logging, plotting, and CSV export for PC2FoodNet.

Produces inside results/<run_name>/:
  ├── fold_{k}/
  │   ├── train_history.csv       per-epoch metrics
  │   ├── curves.png              loss + accuracy curves
  │   └── confusion_matrix.png    val-set confusion matrix
  ├── aggregate.csv               mean ± std across all folds
  ├── fold_summary.csv            one row per fold (best val metrics)
  ├── acc_boxplot.png             Acc@1 distribution across folds
  └── loss_boxplot.png            Loss distribution across folds
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support)

# Set modern Seaborn theme
sns.set_theme(
    context="paper",
    style="whitegrid",
    palette="deep",
    font="sans-serif",
    font_scale=1.1,
    rc={
        "axes.edgecolor": "#333333",
        "axes.labelweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#fdfdfd",
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
    }
)


# ---------------------------------------------------------------------------
# Colour palette (consistent across all figures)
# ---------------------------------------------------------------------------
_FOLD_COLOURS = [
    "#4361EE", # vibrant blue
    "#F72585", # vibrant pink
    "#4CC9F0", # vibrant cyan
    "#7209B7", # deep purple
    "#3A0CA3", # royal blue
]


class ResultsManager:
    """
    Collects per-epoch metrics for every fold, then serialises results.

    Usage:
        rm = ResultsManager(results_dir=Path("results/cv"), n_folds=5)
        for fold, (train_dl, val_dl) in enumerate(folds):
            for epoch in range(epochs):
                rm.record_epoch(fold, epoch, "train", train_metrics)
                rm.record_epoch(fold, epoch, "val",   val_metrics)
            rm.save_fold_artifacts(fold, model, val_dl, class_names, device)
        rm.finalise(fold_results)
    """

    def __init__(self, results_dir: Path | str, n_folds: int):
        self.root    = Path(results_dir)
        self.n_folds = n_folds
        self.root.mkdir(parents=True, exist_ok=True)

        # history[fold][split] = list of metric dicts (one per epoch)
        self.history: dict[int, dict[str, list[dict]]] = {
            f: {"train": [], "val": []} for f in range(n_folds)
        }
        # fold-level precision/recall/F1 scalars (filled by _save_clf_metrics)
        self.fold_clf_metrics: dict[int, dict[str, float]] = {}

    # ── Per-epoch recording ────────────────────────────────────────────────

    def record_epoch(self, fold: int, epoch: int,
                     split: str, metrics: dict[str, float]) -> None:
        entry = {"epoch": epoch + 1, **metrics}
        self.history[fold][split].append(entry)

    # ── Per-fold artifacts ─────────────────────────────────────────────────

    def save_fold_artifacts(
        self,
        fold: int,
        model: torch.nn.Module,
        val_loader,
        class_names: list[str],
        device: torch.device,
    ) -> None:
        fold_dir = self.root / f"fold_{fold + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        self._save_history_csv(fold, fold_dir)
        self._save_curves(fold, fold_dir)
        self._save_confusion_matrix(fold, model, val_loader,
                                    class_names, device, fold_dir)

    def _save_history_csv(self, fold: int, fold_dir: Path) -> None:
        path = fold_dir / "train_history.csv"
        all_rows: list[dict] = []
        for split in ("train", "val"):
            for row in self.history[fold][split]:
                all_rows.append({"split": split, **row})

        if not all_rows:
            return
        fieldnames = list(all_rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    def _save_curves(self, fold: int, fold_dir: Path) -> None:
        train_h = self.history[fold]["train"]
        val_h   = self.history[fold]["val"]
        if not train_h:
            return

        epochs = [r["epoch"] for r in train_h]
        col    = _FOLD_COLOURS[fold % len(_FOLD_COLOURS)]

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.suptitle(f"Fold {fold + 1} — Training Curves", fontsize=13, fontweight="bold")

        # Loss
        axes[0].plot(epochs, [r["total"] for r in train_h],
                     color=col, label="Train", linewidth=2)
        if val_h:
            axes[0].plot(epochs, [r["total"] for r in val_h],
                         color=col, linestyle="--", label="Val", linewidth=2)
        axes[0].set_title("Total Loss"); axes[0].set_xlabel("Epoch")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        # Acc@1
        axes[1].plot(epochs, [r["acc1"] for r in train_h],
                     color=col, label="Train", linewidth=2)
        if val_h:
            axes[1].plot(epochs, [r["acc1"] for r in val_h],
                         color=col, linestyle="--", label="Val", linewidth=2)
        axes[1].set_title("Accuracy @1 (%)"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(0, 100); axes[1].legend(); axes[1].grid(alpha=0.3)

        axes[2].plot(epochs, [r["acc5"] for r in train_h],
                     color=col, label="Train", linewidth=2)
        if val_h:
            axes[2].plot(epochs, [r["acc5"] for r in val_h],
                         color=col, linestyle="--", label="Val", linewidth=2)
        axes[2].set_title("Accuracy @5 (%)"); axes[2].set_xlabel("Epoch")
        axes[2].set_ylim(0, 100); axes[2].legend(); axes[2].grid(alpha=0.3)

        # RMSE
        if "rmse_wgt" in train_h[0]:
            axes[3].plot(epochs, [r["rmse_wgt"] for r in train_h],
                         color=col, label="Train Wgt RMSE", linewidth=2)
            if val_h:
                axes[3].plot(epochs, [r["rmse_wgt"] for r in val_h],
                             color=col, linestyle="--", label="Val Wgt RMSE", linewidth=2)
            axes[3].set_title("Weight RMSE (g)"); axes[3].set_xlabel("Epoch")
            axes[3].legend(); axes[3].grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(fold_dir / "curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _run_inference(self, model, val_loader, device, fold_dir: Optional[Path] = None):
        """Single inference pass — returns (all_preds, all_labels) and saves val_predictions.npz."""
        model.eval()
        all_preds, all_labels = [], []
        all_probs = []
        vol_preds, vol_gts = [], []
        wgt_preds, wgt_gts = [], []
        nrg_preds, nrg_gts = [], []
        vol_logvars, wgt_logvars, nrg_logvars = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                imgs, labels = batch[0], batch[1]
                gt_v = batch[2] if len(batch) > 2 else torch.zeros_like(labels).float()
                gt_w = batch[3] if len(batch) > 3 else torch.zeros_like(labels).float()
                gt_e = batch[4] if len(batch) > 4 else torch.zeros_like(labels).float()

                imgs = imgs.to(device, non_blocking=True)
                device_type = "cuda" if device.type == "cuda" else "cpu"
                with torch.amp.autocast(device_type=device_type, enabled=(device.type == "cuda")):
                    out = model(imgs)

                preds = out["logits"].argmax(1).cpu().tolist()
                probs = out["probs"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
                all_probs.append(probs)

                vol_preds.extend(out["volume"].cpu().tolist())
                vol_gts.extend(gt_v.tolist())
                wgt_preds.extend(out["weight"].cpu().tolist())
                wgt_gts.extend(gt_w.tolist())
                nrg_preds.extend(out["energy"].cpu().tolist())
                nrg_gts.extend(gt_e.tolist())

                vol_logvars.extend(out["logvar_volume"].cpu().tolist())
                wgt_logvars.extend(out["logvar_weight"].cpu().tolist())
                nrg_logvars.extend(out["logvar_energy"].cpu().tolist())

        if fold_dir is not None:
            all_probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
            np.savez_compressed(
                fold_dir / "val_predictions.npz",
                preds=np.array(all_preds, dtype=np.int64),
                labels=np.array(all_labels, dtype=np.int64),
                probs=all_probs_arr,
                volume_pred=np.array(vol_preds, dtype=np.float32),
                volume_gt=np.array(vol_gts, dtype=np.float32),
                weight_pred=np.array(wgt_preds, dtype=np.float32),
                weight_gt=np.array(wgt_gts, dtype=np.float32),
                energy_pred=np.array(nrg_preds, dtype=np.float32),
                energy_gt=np.array(nrg_gts, dtype=np.float32),
                logvar_volume=np.array(vol_logvars, dtype=np.float32),
                logvar_weight=np.array(wgt_logvars, dtype=np.float32),
                logvar_energy=np.array(nrg_logvars, dtype=np.float32),
            )

        return all_preds, all_labels

    def _save_confusion_matrix(
        self,
        fold: int,
        model: torch.nn.Module,
        val_loader,
        class_names: list[str],
        device: torch.device,
        fold_dir: Path,
    ) -> None:
        all_preds, all_labels = self._run_inference(model, val_loader, device, fold_dir=fold_dir)

        cm = confusion_matrix(all_labels, all_preds,
                              labels=list(range(len(class_names))))
        cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100

        n = len(class_names)
        figsize = max(12, n * 0.45)
        fig, ax = plt.subplots(figsize=(figsize, figsize))

        # Create custom string annotations: hide 0, show >1 as int, >0 as <1
        annot = np.empty_like(cm_pct, dtype=object)
        for i in range(n):
            for j in range(n):
                val = cm_pct[i, j]
                if val == 0:
                    annot[i, j] = ""
                elif val < 1:
                    annot[i, j] = "<1"
                else:
                    annot[i, j] = f"{val:.0f}"

        sns.heatmap(cm_pct, ax=ax, annot=annot, fmt="",
                    cmap="mako_r", linewidths=0.5, linecolor="white",
                    cbar_kws={"shrink": 0.8, "label": "% of True Class"},
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=100)

        ax.set_title(f"Confusion Matrix — Fold {fold + 1}",
                     fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold", labelpad=15)
        ax.set_ylabel("True Class", fontsize=12, fontweight="bold", labelpad=15)

        # Rotate ticks for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

        plt.tight_layout()
        fig.savefig(fold_dir / "confusion_matrix.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        # ── Classification report (precision / recall / F1) ────────────────
        self._save_clf_metrics(fold, all_preds, all_labels, class_names, fold_dir)

    def _save_clf_metrics(
        self,
        fold: int,
        all_preds: list[int],
        all_labels: list[int],
        class_names: list[str],
        fold_dir: Path,
    ) -> None:
        """Compute precision, recall, F1 per class + macro/weighted averages.
        Saves classification_report.csv and per_class_f1.png.
        """
        labels_idx = list(range(len(class_names)))

        # Per-class arrays
        prec, rec, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, labels=labels_idx, zero_division=0
        )

        # Macro and weighted aggregates
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        # ── CSV ──────────────────────────────────────────────────────────────
        csv_path = fold_dir / "classification_report.csv"
        with open(csv_path, "w", newline="") as f:
            import csv as _csv
            writer = _csv.DictWriter(
                f, fieldnames=["class", "precision", "recall", "f1", "support"]
            )
            writer.writeheader()
            for i, name in enumerate(class_names):
                writer.writerow({
                    "class": name,
                    "precision": round(float(prec[i]), 4),
                    "recall":    round(float(rec[i]),  4),
                    "f1":        round(float(f1[i]),   4),
                    "support":   int(support[i]),
                })
            # Aggregate rows
            for avg_name, p, r, f in [
                ("macro avg",    prec_macro, rec_macro, f1_macro),
                ("weighted avg", prec_wt,    rec_wt,    f1_wt),
            ]:
                writer.writerow({
                    "class": avg_name,
                    "precision": round(float(p), 4),
                    "recall":    round(float(r), 4),
                    "f1":        round(float(f), 4),
                    "support":   len(all_labels),
                })

        # ── Per-class F1 bar chart ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 0.45), 5))
        x = np.arange(len(class_names))
        bars = ax.bar(x, f1 * 100, color=_FOLD_COLOURS[fold % len(_FOLD_COLOURS)],
                      alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axhline(f1_macro * 100, color="black", linewidth=1.5,
                   linestyle="--", label=f"Macro F1 = {f1_macro*100:.1f}%")
        ax.axhline(f1_wt * 100, color="crimson", linewidth=1.5,
                   linestyle=":", label=f"Weighted F1 = {f1_wt*100:.1f}%")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=90, fontsize=8)
        ax.set_ylabel("F1 Score (%)", fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_title(f"Per-Class F1 — Fold {fold + 1}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(fold_dir / "per_class_f1.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        # ── Precision-Recall bar chart (macro view) ───────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(max(12, len(class_names) * 0.45), 5))
        col = _FOLD_COLOURS[fold % len(_FOLD_COLOURS)]
        for ax_, arr, title in [
            (axes[0], prec * 100, "Precision (%)"),
            (axes[1], rec  * 100, "Recall (%)"),
        ]:
            ax_.bar(x, arr, color=col, alpha=0.8, edgecolor="white", linewidth=0.5)
            ax_.set_xticks(x)
            ax_.set_xticklabels(class_names, rotation=90, fontsize=8)
            ax_.set_ylabel(title, fontsize=10)
            ax_.set_ylim(0, 105)
            ax_.set_title(f"{title} per Class — Fold {fold + 1}",
                          fontsize=10, fontweight="bold")
            ax_.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(fold_dir / "precision_recall.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        print(f"[results] fold {fold+1} — "
              f"macro P={prec_macro*100:.2f}%  R={rec_macro*100:.2f}%  "
              f"F1={f1_macro*100:.2f}%  "
              f"(weighted F1={f1_wt*100:.2f}%)")

        # Store for aggregate use
        self.fold_clf_metrics[fold] = {
            "precision_macro":    float(prec_macro),
            "recall_macro":       float(rec_macro),
            "f1_macro":           float(f1_macro),
            "precision_weighted": float(prec_wt),
            "recall_weighted":    float(rec_wt),
            "f1_weighted":        float(f1_wt),
        }

    # ── Aggregate artifacts (call after all folds) ─────────────────────────

    def finalise(self, fold_results: list[dict[str, float]]) -> None:
        """Generate aggregate CSV, fold-summary CSV, box plots, and PR/F1 aggregate."""
        # Merge classification metrics into fold_results for unified CSVs
        merged = []
        for i, fr in enumerate(fold_results):
            row = dict(fr)
            row.update(self.fold_clf_metrics.get(i, {}))
            merged.append(row)

        self._save_fold_summary_csv(merged)
        self._save_aggregate_csv(merged)
        self._save_aggregate_curves()
        self._save_boxplots(merged)
        
        # New fancy plot types
        self._save_prf_aggregate_plot(merged)
        self._save_radar_plot(merged)
        self._save_heatmap_plot(merged)
        self._save_pointplot(merged)

    def _save_fold_summary_csv(self, fold_results: list[dict]) -> None:
        if not fold_results:
            return
        path = self.root / "fold_summary.csv"
        
        # Gather all unique keys across all folds
        all_keys = set()
        for r in fold_results:
            all_keys.update(r.keys())
        fieldnames = ["fold"] + sorted(list(all_keys))
        
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, row in enumerate(fold_results, start=1):
                writer.writerow({"fold": i, **row})
        print(f"[results] fold_summary.csv → {path}")

    def _save_aggregate_csv(self, fold_results: list[dict]) -> None:
        if not fold_results:
            return
        path = self.root / "aggregate.csv"
        
        # Gather all unique keys across all folds
        all_keys = set()
        for r in fold_results:
            all_keys.update(r.keys())
        metrics = sorted(list(all_keys))
        
        rows = []
        for m in metrics:
            vals = [r[m] for r in fold_results if m in r]
            rows.append({
                "metric": m,
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals)),
                "min":    float(np.min(vals)),
                "max":    float(np.max(vals)),
            })
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "min", "max"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[results] aggregate.csv     → {path}")

    def _save_aggregate_curves(self) -> None:
        """Overlay val Acc@1 and val Loss across all folds on one figure."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("All Folds — Validation Metrics", fontsize=13, fontweight="bold")

        for fold in range(self.n_folds):
            val_h = self.history[fold]["val"]
            if not val_h:
                continue
            epochs = [r["epoch"] for r in val_h]
            col    = _FOLD_COLOURS[fold % len(_FOLD_COLOURS)]
            label  = f"Fold {fold + 1}"

            axes[0].plot(epochs, [r["acc1"]  for r in val_h],
                         color=col, label=label, linewidth=1.8)
            axes[1].plot(epochs, [r["total"] for r in val_h],
                         color=col, label=label, linewidth=1.8)

        # Mean curve
        min_len = min(
            (len(self.history[f]["val"]) for f in range(self.n_folds)
             if self.history[f]["val"]), default=0
        )
        active_folds = [f for f in range(self.n_folds) if self.history[f]["val"]]
        if min_len > 0 and len(active_folds) > 1:
            mean_acc  = np.mean([[r["acc1"]  for r in self.history[f]["val"][:min_len]]
                                  for f in active_folds], axis=0)
            mean_loss = np.mean([[r["total"] for r in self.history[f]["val"][:min_len]]
                                  for f in active_folds], axis=0)
            xs = list(range(1, min_len + 1))
            axes[0].plot(xs, mean_acc,  color="black", linewidth=2.5,
                         linestyle="--", label="Mean", zorder=5)
            axes[1].plot(xs, mean_loss, color="black", linewidth=2.5,
                         linestyle="--", label="Mean", zorder=5)

        axes[0].set_title("Val Acc@1 (%)"); axes[0].set_xlabel("Epoch")
        axes[0].set_ylim(0, 100); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].set_title("Val Loss");      axes[1].set_xlabel("Epoch")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.root / "all_folds_val.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[results] all_folds_val.png → {self.root / 'all_folds_val.png'}")

    def _save_boxplots(self, fold_results: list[dict]) -> None:
        metrics_to_plot = [
            ("acc1",  "Val Acc@1 (%)",  "acc_violin.png"),
            ("acc5",  "Val Acc@5 (%)",  "acc5_violin.png"),
            ("total", "Val Loss",        "loss_violin.png"),
        ]
        import pandas as pd
        for key, ylabel, fname in metrics_to_plot:
            vals = [r[key] for r in fold_results if key in r]
            if not vals:
                continue
            
            fig, ax = plt.subplots(figsize=(5, 5))
            df = pd.DataFrame({ylabel: vals, "Group": [f"{self.n_folds}-Fold CV"] * len(vals)})
            
            # Violin + Swarm overlay
            sns.violinplot(data=df, x="Group", y=ylabel, ax=ax, color="#4361EE", inner="quartile", alpha=0.6)
            sns.swarmplot(data=df, x="Group", y=ylabel, ax=ax, color="black", size=7, alpha=0.8)
            
            ax.set_xlabel("")
            ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
            ax.set_title(f"{ylabel} Distribution ({self.n_folds} folds)", fontsize=12, fontweight="bold", pad=15)
            ax.grid(axis="y", alpha=0.3)
            
            m, s = np.mean(vals), np.std(vals)
            ax.text(0.4, m, f"  {m:.2f} ± {s:.2f}", va="center", fontsize=10, fontweight="bold", color="#333333", transform=ax.get_yaxis_transform())
            
            plt.tight_layout()
            fig.savefig(self.root / fname, dpi=200, bbox_inches="tight")
            plt.close(fig)
        print(f"[results] violin plots saved → {self.root}")

    def _save_prf_aggregate_plot(self, fold_results: list[dict]) -> None:
        """Grouped bar: Precision / Recall / F1 (macro) per fold + mean line."""
        keys = ["precision_macro", "recall_macro", "f1_macro",
                "precision_weighted", "recall_weighted", "f1_weighted"]
        if not any(k in fold_results[0] for k in keys):
            return

        import pandas as pd
        
        # ── Macro P/R/F1 per fold ─────────────────────────────────────────
        n = len(fold_results)
        
        # Build DataFrame for Seaborn
        data = []
        for i, r in enumerate(fold_results):
            if "precision_macro" in r:
                data.append({"Fold": f"Fold {i+1}", "Metric": "Precision", "Score": r["precision_macro"] * 100})
                data.append({"Fold": f"Fold {i+1}", "Metric": "Recall",    "Score": r["recall_macro"] * 100})
                data.append({"Fold": f"Fold {i+1}", "Metric": "F1",        "Score": r["f1_macro"] * 100})
        
        if not data:
            return
            
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(max(8, n * 1.8), 5))
        sns.barplot(data=df, x="Fold", y="Score", hue="Metric", ax=ax,
                    palette=["#4361EE", "#F72585", "#4CC9F0"], edgecolor="none", alpha=0.9)
                    
        # Annotate bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", padding=3, fontsize=8, color="#333333")

        # Mean reference lines
        for metric, color in zip(["Precision", "Recall", "F1"], ["#4361EE", "#F72585", "#4CC9F0"]):
            mean_val = df[df["Metric"] == metric]["Score"].mean()
            ax.axhline(mean_val, color=color, linestyle="--", linewidth=1.5, alpha=0.5)

        ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_xlabel("", fontsize=12)
        ax.set_ylim(80, 103)
        ax.set_title(f"Macro Precision / Recall / F1 per Fold ({self.n_folds}-Fold CV)",
                     fontsize=14, fontweight="bold", pad=15)
                     
        # Clean legend
        sns.move_legend(ax, "lower right", title=None, frameon=True, 
                        facecolor="white", edgecolor="#eeeeee", fontsize=10)
                        
        plt.tight_layout()
        fig.savefig(self.root / "prf_per_fold.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[results] prf_per_fold.png → {self.root}")

    def _save_radar_plot(self, fold_results: list[dict]) -> None:
        import pandas as pd
        from math import pi
        
        if not fold_results or "precision_macro" not in fold_results[0]: return
        
        categories = ["Macro Precision", "Macro Recall", "Macro F1", "Wgt Precision", "Wgt Recall", "Wgt F1"]
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        plt.xticks(angles[:-1], categories, color='grey', size=11, fontweight='bold')
        ax.set_rlabel_position(0)
        plt.yticks([85, 90, 95, 100], ["85", "90", "95", "100"], color="grey", size=8)
        plt.ylim(80, 100)
        
        colors = ["#4361EE", "#F72585", "#4CC9F0", "#FFB703"]
        for i, r in enumerate(fold_results):
            values = [
                r["precision_macro"]*100, r["recall_macro"]*100, r["f1_macro"]*100,
                r["precision_weighted"]*100, r["recall_weighted"]*100, r["f1_weighted"]*100
            ]
            values += values[:1]
            c = colors[i % len(colors)]
            ax.plot(angles, values, color=c, linewidth=2, linestyle='solid', label=f"Fold {i+1}")
            ax.fill(angles, values, color=c, alpha=0.1)
            
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        plt.title("Fold Performance Radar", size=14, fontweight='bold', y=1.1)
        fig.savefig(self.root / "prf_radar_fancy.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _save_heatmap_plot(self, fold_results: list[dict]) -> None:
        import pandas as pd
        if not fold_results or "precision_macro" not in fold_results[0]: return
        
        data = []
        for i, r in enumerate(fold_results):
            data.append({
                "P (Macro)": r["precision_macro"]*100, "R (Macro)": r["recall_macro"]*100, "F1 (Macro)": r["f1_macro"]*100,
                "P (Wgt)": r["precision_weighted"]*100, "R (Wgt)": r["recall_weighted"]*100, "F1 (Wgt)": r["f1_weighted"]*100
            })
        df = pd.DataFrame(data, index=[f"Fold {i+1}" for i in range(len(fold_results))])
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(df, annot=True, fmt=".2f", cmap="mako", linewidths=.5, ax=ax, vmin=85, vmax=98)
        plt.title("Metrics Heatmap (Scores in %)", fontsize=14, fontweight='bold', pad=15)
        fig.savefig(self.root / "prf_heatmap_fancy.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _save_pointplot(self, fold_results: list[dict]) -> None:
        import pandas as pd
        if not fold_results or "precision_macro" not in fold_results[0]: return
        
        data = []
        for i, r in enumerate(fold_results):
            data.append({"Fold": f"Fold {i+1}", "Metric": "Precision", "Score": r["precision_macro"]*100})
            data.append({"Fold": f"Fold {i+1}", "Metric": "Recall", "Score": r["recall_macro"]*100})
            data.append({"Fold": f"Fold {i+1}", "Metric": "F1", "Score": r["f1_macro"]*100})
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.pointplot(data=df, x="Metric", y="Score", hue="Fold", markers=["o", "s", "D", "v"], linestyles="-", ax=ax, palette="deep")
        plt.title("Pointplot Comparison (Macro)", fontsize=14, fontweight='bold', pad=15)
        plt.ylim(85, 100)
        plt.grid(alpha=0.3)
        fig.savefig(self.root / "prf_pointplot_fancy.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ── Weighted F1 violin ───────────────────────────────────────────
        wf1_vals = [r.get("f1_weighted", 0) * 100 for r in fold_results if r.get("f1_weighted", 0) > 0]
        if wf1_vals:
            fig, ax = plt.subplots(figsize=(5, 5))
            df = pd.DataFrame({"Weighted F1 (%)": wf1_vals, "Group": [f"{self.n_folds}-Fold CV"] * len(wf1_vals)})
            
            sns.violinplot(data=df, x="Group", y="Weighted F1 (%)", ax=ax, color="#4CC9F0", inner="quartile", alpha=0.6)
            sns.swarmplot(data=df, x="Group", y="Weighted F1 (%)", ax=ax, color="black", size=7, alpha=0.8)
            
            ax.set_xlabel("")
            ax.set_ylabel("Weighted F1 (%)", fontsize=11, fontweight="bold")
            ax.set_title("Weighted F1 Distribution", fontsize=12, fontweight="bold", pad=15)
            ax.grid(axis="y", alpha=0.3)
            
            m, s = np.mean(wf1_vals), np.std(wf1_vals)
            ax.text(0.4, m, f"  {m:.2f} ± {s:.2f}", va="center", fontsize=10, fontweight="bold", color="#333333", transform=ax.get_yaxis_transform())
            
            plt.tight_layout()
            fig.savefig(self.root / "f1_weighted_violin.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

