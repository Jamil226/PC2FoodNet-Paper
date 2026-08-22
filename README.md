# PC2FoodNet — Complete Experiment Documentation

> **For inclusion in research paper.**
> Describes every step from raw images to final cross-validated results.

---

## Abstract

We present **PC2FoodNet**, a physics-constrained, confidence-calibrated multi-task neural network for Turkish food recognition and nutritional estimation from a single RGB image. The model jointly predicts food class, volume, weight, and energy (kcal) by combining a data-driven deep learning backbone with learnable soft-physics priors derived from per-class macronutrient data. We evaluate on a 40-class Turkish food image dataset (~22,000 images) using 4-fold stratified cross-validation with strict data leakage prevention.

---

## 1. Dataset

| Property | Detail |
|----------|--------|
| **Name** | Turkish Food Dataset |
| **Classes** | 40 Turkish dishes |
| **Total images** | 22,070 |
| **Image format** | JPEG / PNG / WebP |
| **Source** | `turkish-food/` directory |
| **Nutrition priors** | `turkish-food/porsiyon_nutrition_data.json` |
| **Nutrition fields** | calories (kcal/portion), protein (g), carbohydrate (g), fat (g), fibre (g) |

### 1.1 Class List (40 dishes)

| # | Class | # | Class |
|---|-------|---|-------|
| 0 | Adana Kebap | 20 | Lahmacun |
| 1 | Aşure | 21 | Mantı |
| 2 | Baklava | 22 | Menemen |
| 3 | Beyaz Ekmek | 23 | Mercimek Çorbası |
| 4 | Börek | 24 | Mısır Ekmeği |
| 5 | Bulgur Pilavı | 25 | Nohut |
| 6 | Çiğ Köfte | 26 | Pastane Poğaçası |
| 7 | Esmer Ekmek | 27 | Patates Kızartması |
| 8 | Ezogelin Çorbası | 28 | Pide |
| 9 | Gözleme | 29 | Pirinç Pilavı |
| 10 | Hamburger | 30 | Revani |
| 11 | Haşlanmış Yumurta | 31 | Şakşuka |
| 12 | İskender | 32 | Şekerpare |
| 13 | Kadayıf | 33 | Sütlaç |
| 14 | Karnıyarık | 34 | Tarhana Çorbası |
| 15 | Kısır | 35 | Tavuk |
| 16 | Köfte | 36 | Tavuk Döner |
| 17 | Kumpir | 37 | Tulumba Tatlısı |
| 18 | Kuru Fasulye | 38 | Yaprak Sarma |
| 19 | Künefe | 39 | Yulaf Ekmeği |

### 1.2 Nutrition Prior Derivation

From the JSON file, two per-class scalar priors are computed:

**kcal per gram:**
```
kcal_per_g[c] = kalori[c] / 100
```

**Approximate density (g/mL)** — weighted by macronutrient ratios:
```
density[c] = (fat * 0.90 + (protein + carbs + fibre) * 1.30) / total_macro
```

These priors are **domain knowledge** (not derived from training images) and are therefore identical for every fold — no leakage.

---

## 2. Data Preprocessing Pipeline

```
Raw image (any size, RGB)
        │
        ▼  ──── TRAINING SPLIT ONLY ────────────────────────────────────────
  RandomResizedCrop(224, scale=(0.6, 1.0))   random crop + resize
  RandomHorizontalFlip()                      p=0.5
  ColorJitter(brightness=0.3, contrast=0.3,
              saturation=0.3, hue=0.05)       colour augmentation
  RandomRotation(±15°)                        spatial augmentation
        │
        ▼  ──── ALL SPLITS ─────────────────────────────────────────────────
  ToTensor()                                  uint8 [0,255] → float [0,1]
  Normalize(mean=(0.485,0.456,0.406),
            std =(0.229,0.224,0.225))         ImageNet statistics (fixed)
        │
        ▼
  Tensor (3 × 224 × 224)
```

> **Leakage note**: Augmentation is applied **only** to training-fold images.
> Validation images receive only resize + centre-crop + normalise.
> Normalisation statistics are **fixed** (ImageNet), not computed from any fold.

---

## 3. Cross-Validation Protocol

### 3.1 Split Strategy

| Property | Value |
|----------|-------|
| Method | `sklearn.model_selection.StratifiedKFold` |
| Folds (k) | 5 |
| Train per fold | ~80 % (~17,656 images) |
| Val per fold | ~20 % (~4,414 images) |
| Stratification | Class label (ensures each class appears proportionally in every fold) |
| Random seed | 42 (fixed for reproducibility) |

### 3.2 Data Leakage Prevention — Step by Step

```
Step 1. Load file paths and labels only — NO images read yet.
        │
Step 2. StratifiedKFold.split(indices, labels) 
        → produces train_indices[], val_indices[]
        │  ← NO image data involved at this point
        │
Step 3. SubsetWithTransform(parent, train_indices, train_transform)
        SubsetWithTransform(parent, val_indices,   val_transform)
        │  ← transforms are DIFFERENT per subset
        │  ← images loaded only during __getitem__, never cross-shared
        │
Step 4. Fresh PC2FoodNet() initialised from ImageNet pretrained weights.
        No parameters carried over from any previous fold.
        │
Step 5. Training loop uses ONLY train_indices DataLoader.
        Validation loop uses ONLY val_indices DataLoader.
        │
Step 6. Results (best_acc, metrics) logged to ResultsManager.
        → per-fold CSV, curves.png, confusion_matrix.png
        │
        └─ Repeat for folds 2, 3, 4, 5 with completely fresh model.
```

### 3.3 Fold Distribution

Each fold is stratified — every class contributes approximately:

```
n_train_per_class ≈ (22070 × 0.80) / 40 ≈ 441 images
n_val_per_class   ≈ (22070 × 0.20) / 40 ≈ 110 images
```

---

## 4. Model Architecture — PC2FoodNet

### 4.1 Backbone

| Component | Specification |
|-----------|---------------|
| Architecture | EfficientNet-V2-S |
| Pre-training | ImageNet-1K (`EfficientNet_V2_S_Weights.DEFAULT`) |
| Feature dim | 1,280-D |
| Modification | Original classifier replaced with `nn.Identity()` |
| Fine-tuning | All backbone parameters are trainable |
| Parameters | ~20.3M |

### 4.2 Projection Head

```
Linear(1280 → 512) → LayerNorm(512) → GELU → Dropout(0.30)
```

### 4.3 Task Heads

| Head | Output | Architecture |
|------|--------|-------------|
| Classifier | 40 logits | Linear(512 → 40) |
| Volume head | (mean, log-var) | MLP(512 → 256 → 2) |
| Weight direct head | (mean, log-var) | MLP(512 → 256 → 2) |
| Energy direct head | (mean, log-var) | MLP(512 → 256 → 2) |
| Weight residual head | scalar | MLP(512 → 256 → 1) |
| Energy residual head | scalar | MLP(512 → 256 → 1) |
| Weight gate head | scalar | MLP(512 → 256 → 1) |
| Energy gate head | scalar | MLP(512 → 256 → 1) |

*MLP = Linear → GELU → Dropout(0.30) → Linear*

### 4.4 Physics-Constrained Inference Chain

```
Step 1 — Classification
  logits  = Classifier(features)          # (B, 40)
  probs   = softmax(logits)               # (B, 40)

Step 2 — Class-probability weighted priors
  E[density]    = probs ⊗ densities       # (B,)  dot product
  E[kcal_per_g] = probs ⊗ kcal_per_g    # (B,)

Step 3 — Volume estimation
  volume = _positive(volume_head.mean)    # always > 0

Step 4 — Physics weight path
  physics_weight = volume × E[density]
                 × exp(0.20 × tanh(weight_residual))   # bounded ±20%

Step 5 — Gated weight blend
  gate_w  = sigmoid(weight_gate_head)
  weight  = gate_w × direct_weight + (1 − gate_w) × physics_weight

Step 6 — Physics energy path
  physics_energy = weight × E[kcal_per_g]
                 × exp(0.20 × tanh(energy_residual))   # bounded ±20%

Step 7 — Gated energy blend
  gate_e  = sigmoid(energy_gate_head)
  energy  = gate_e × direct_energy + (1 − gate_e) × physics_energy
```

### 4.5 Positivity Activation

All continuous outputs are guaranteed strictly positive via:
```
_positive(x) = expm1(softplus(x).clamp(max=10)) + 1e-4
```

This is smooth, differentiable everywhere, and bounded away from zero.

### 4.6 Uncertainty Quantification

Each continuous output (volume, weight, energy) produces a **log-variance** alongside its mean:
```
std = exp(0.5 × logvar),   logvar ∈ [−6, 6]
```
This enables heteroscedastic uncertainty estimation at inference time.

---

## 5. Training Protocol

### 5.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs per fold | 50 |
| Batch size | 32 |
| Optimiser | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| LR schedule | Linear warmup (5 epochs) → Cosine annealing (45 epochs) |
| LR at end | 1 × 10⁻⁶ |
| Gradient clipping | max norm = 1.0 |
| Mixed precision | `torch.cuda.amp.autocast` (FP16 forward + FP32 accumulation) |

### 5.2 Loss Function

```
L_total = λ_cls × L_CE
        + λ_wgt × L_NLL(weight)
        + λ_nrg × L_NLL(energy)
        + λ_vol × L_NLL(volume)
        + λ_phys × L_L1(weight_gated, weight_physics.detach())
```

| Term | Weight | Description |
|------|--------|-------------|
| `L_CE` | λ_cls = 1.0 | CrossEntropy with label smoothing ε = 0.1 |
| `L_NLL(weight)` | λ_wgt = 1.0 | Gaussian NLL over (mean, logvar) output |
| `L_NLL(energy)` | λ_nrg = 1.0 | Gaussian NLL over (mean, logvar) output |
| `L_NLL(volume)` | λ_vol = 0.5 | Gaussian NLL over (mean, logvar) output |
| `L_phys` | λ_phys = 0.2 | L1 consistency between gated and physics weight |

**Gaussian NLL:**
```
L_NLL(μ, log σ², y) = 0.5 × (log σ² + (μ − y)² / σ²)
```

> **Note**: Volume and weight ground truth labels are not available in the current dataset (images only). `L_NLL(volume)`, `L_NLL(weight)`, and `L_NLL(energy)` are activated only when such annotations are provided. In the current experiments, only `L_CE` and `L_phys` are active.

### 5.3 Bias Warm-Start

Regression head output biases are initialised so the network's initial predictions match dataset-level medians:

```
bias = inverse_positive(median)
     = softplus⁻¹(log1p(median − 1e−4))

Targets:  volume = 250 mL,  weight = 200 g,  energy = 350 kcal
Log-var bias initialised to −1.0  (std ≈ 0.61, small initial uncertainty)
```

### 5.4 Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 5000 Ada Generation (32 GB VRAM) |
| CUDA | 12.4 |
| PyTorch | ≥ 2.4.0 |
| Mixed precision | AMP (FP16) |

---

## 6. Evaluation Protocol

### 6.1 Metrics

| Metric | Formula | Applicable to |
|--------|---------|---------------|
| **Acc@1** | top-1 correct / total | Classification |
| **Acc@5** | top-5 correct / total | Classification |
| **Total Loss** | L_total (see §5.2) | All heads |
| **Physics consistency** | L1(weight, physics_weight) | Physics path |

### 6.2 Reporting

- Per-fold: best validation Acc@1, Acc@5, Total Loss
- Aggregate: mean ± standard deviation across all 5 folds

---

## 7. Results Artefacts

All results are saved automatically to `results/cv/`:

```
results/cv/
├── fold_1/
│   ├── train_history.csv       ← per-epoch train & val metrics
│   ├── curves.png              ← loss + Acc@1 + Acc@5 vs epoch
│   └── confusion_matrix.png    ← normalised confusion matrix (val set)
├── fold_2/ ... fold_5/         ← same structure
├── all_folds_val.png           ← val Acc@1 & Loss overlaid for all folds
├── acc_boxplot.png             ← Acc@1 distribution across folds
├── acc5_boxplot.png            ← Acc@5 distribution across folds
├── loss_boxplot.png            ← Loss distribution across folds
├── fold_summary.csv            ← one row per fold (best val metrics)
├── aggregate.csv               ← mean, std, min, max per metric
└── cv_summary.json             ← machine-readable full summary
```

### 7.1 Expected Results Table (to fill after training)

| Fold | Val Acc@1 (%) | Val Acc@5 (%) | Val Loss |
|------|--------------|--------------|---------|
| 1 | — | — | — |
| 2 | — | — | — |
| 3 | — | — | — |
| 4 | — | — | — |
| 5 | — | — | — |
| **Mean ± Std** | **— ± —** | **— ± —** | **— ± —** |

---

## 8. How to Reproduce

### Step 1 — Environment

```bash
cd Rashed-Model/
pixi install
pixi run gpu-check      # confirm NVIDIA GPU detected
```

### Step 2 — Dataset sanity check

```bash
pixi run dataset-check
# Expected output:
#   Classes: 40  |  Fold splits: 17,656 train / 4,414 val per fold
```

### Step 3 — Run 4-fold cross-validation

```bash
pixi run train-cv
# Equivalent to:
# python train.py --n_folds 5 --epochs 50 --batch_size 32
```

### Step 4 — Inspect results

```bash
ls results/cv/
# Opens TensorBoard:
tensorboard --logdir runs/cv
```

### Step 5 — Key output files

| File | Purpose |
|------|---------|
| `results/cv/fold_summary.csv` | Best val metrics per fold |
| `results/cv/aggregate.csv` | Mean ± std table |
| `results/cv/all_folds_val.png` | Convergence curves figure |
| `results/cv/acc_boxplot.png` | Accuracy box plot |
| `runs/cv/fold_*/best.pt` | Best model checkpoint per fold |

---

## 9. Novelty Summary

| Contribution | Description |
|---|---|
| **Physics-gated blending** | A learned sigmoid gate per image decides how much to trust physics vs. data-driven estimates. Differentiable and per-sample adaptive. |
| **Soft-expected priors** | `probs ⊗ densities` propagates the full classification uncertainty into the physics chain — not just the top-1 class. |
| **Bounded exponential residual** | `exp(0.20 × tanh(r))` constrains physics corrections to ±20%, preventing model collapse while allowing calibration. |
| **Heteroscedastic regression** | (mean, log-variance) outputs for all three continuous quantities enable calibrated uncertainty at inference. |
| **Turkish food focus** | First known model with physics-informed nutritional estimation specifically targeting 40 Turkish dishes with per-class domain priors. |

---

## 10. File Reference

| File                                        | Role                                               |
| ---------------------------------------------| ----------------------------------------------------|
| `model.py`                                  | `PC2FoodNet` architecture                          |
| `dataset.py`                                | Dataset loader, CV splits, nutrition prior builder |
| `train.py`                                  | 4-fold CV training loop                            |
| `results_manager.py`                        | Curve plots, confusion matrices, CSV export        |
| `pixi.toml`                                 | Reproducible environment definition                |
| `README.md`                                 | Project overview and quick-start guide             |
| `experiment.md`                             | This document                                      |
| `turkish-food/`                             | 40-class image dataset                             |
| `turkish-food/porsiyon_nutrition_data.json` | Per-class nutrition priors                         |
| `results/cv/`                               | All training artefacts (generated at runtime)      |
| `runs/cv/`                                  | Model checkpoints + TensorBoard logs               |
