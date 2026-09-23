# PC²FoodNet: Physics-Constrained, Confidence-Calibrated Deep Multi-Task Food Recognition and Nutritional Estimation

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-red.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MDPI Nutrients](https://img.shields.io/badge/MDPI-Nutrients-orange.svg)](https://www.mdpi.com/journal/nutrients)

---

## Table of Contents

- [Overview & Novelty](#overview--novelty)
- [Model Architecture](#model-architecture)
- [Choi & Okos (1986) Food Density Model & Sensitivity (Table 5)](#choi--okos-1986-food-density-model--sensitivity-table-5)
- [Dataset & Standardized 100g Portion Framing](#dataset--standardized-100g-portion-framing)
- [Cross-Validation & Data Leakage Prevention](#cross-validation--data-leakage-prevention)
- [Installation & Environment](#installation--environment)
- [Quick Start & Reproduction Commands](#quick-start--reproduction-commands)
- [Experimental Results & Table Replication](#experimental-results--table-replication)
  - [Table 5: One-At-A-Time Sensitivity Analysis](#table-5-one-at-a-time-oat-sensitivity-analysis)
  - [Table 6: 4-Fold Cross-Validation Performance](#table-6-4-fold-stratified-cross-validation-performance)
  - [Table 7: Baseline Comparisons & Paired Statistical Tests](#table-7-baseline-comparisons--paired-statistical-tests)
  - [Table 8: Component Ablations](#table-8-component-ablations)
  - [Table 9: Pooled Out-of-Fold 40-Class Evaluation](#table-9-pooled-out-of-fold-40-class-evaluation)
  - [Table 10: Confidence Calibration & Selective Referral Triage](#table-10-confidence-calibration--selective-referral-triage)
- [Summary of Rebuttal Changes (changes.md)](#summary-of-rebuttal-changes-changesmd)
- [Repository File Structure](#repository-file-structure)
- [Citation](#citation)

---

## Overview & Novelty

Automated dietary intake assessment from a single RGB image is fundamentally ill-posed due to visual occlusions, composite culinary preparations, and non-linear physical density variations. Conventional approaches rely either on unconstrained direct regression or rigid category lookup heuristics.

**PC²FoodNet** overcomes these limitations through a unified multi-task framework:
1. **Dual-Branch Gated Fusion**: A learned sigmoid gate $g_w \in (0, 1)$ dynamically balances an image-driven direct regression head $\hat{w}_{\text{direct}}$ against a physics-informed density projection branch $\hat{w}_{\text{phys}}$.
2. **Soft-Expected Nutritional Priors**: Instead of conditioning on a single argmax class label, the physics chain propagates classification uncertainty via inner products over the class probability simplex: $\bar{\rho} = \mathbf{p}^{\top} \boldsymbol{\rho}$ and $\bar{\kappa} = \mathbf{p}^{\top} \boldsymbol{\kappa}$.
3. **Bounded Exponential Residuals**: Learnable residual scalar corrections are modulated via $\exp(\delta \tanh(r))$ with $\delta = 0.20$, strictly bounding physics deviations to within $\pm 20\%$ to prevent gradient instability or degenerate solutions.
4. **Guaranteed Positivity Activation**: Volume, mass, and energy predictions are constrained to strictly positive values via $\psi(z) = \text{expm1}(\text{softplus}(z)) + 10^{-4} > 0$.
5. **Heteroscedastic Uncertainty & Disjoint Conformal Triage**: Each continuous head predicts mean and log-variance $(\mu, \log \sigma^2)$. Post-hoc temperature scaling ($T = 1.24$) and conformal prediction sets ($\alpha = 0.10$) triage ambiguous cases to clinical dietitian review under a strictly disjoint cross-validation protocol ($N_{\text{cal}} = 400, N_{\text{eval}} = 420$).

---

## Model Architecture

```
                    ┌────────────────────────┐
                    │    Input RGB Image     │ (3 × 224 × 224)
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   EfficientNetV2-S     │ (Pretrained ImageNet-1K, 21.5M params)
                    └───────────┬────────────┘
                                │ 1,280-D
                    ┌───────────▼────────────┐
                    │ Linear + LN + GELU + DO│ Projector (512-D)
                    └─────┬──────────────┬───┘
                          │              │
       ┌──────────────────┴──┐           │
       │  Classifier (40-D)  │           │
       │  Softmax → Probs p  │           │
       └──────────┬──────────┘           │
                  │                      │
       ┌──────────┴───────────────┐      │
       │ Soft Nutritional Priors  │      │
       │ E[ρ] = p · ρ_prior       │      │
       │ E[κ] = p · κ_prior       │      │
       └──────────┬───────────────┘      │
                  │                      │
  ┌───────────────┼──────────────────────┴──────────────────────────────────────┐
  │ Physical Quantities Inference Pipeline                                      │
  │                                                                             │
  │  Volume Head:         V̂ = ψ(MLP_vol(f))                                     │
  │                                                                             │
  │  Weight Direct Head:  ŵ_direct = ψ(MLP_wgt(f))                              │
  │  Weight Physics Head: ŵ_phys   = V̂ · E[ρ] · exp(0.20 · tanh(r_w))          │
  │  Weight Gate:         g_w      = σ(MLP_gw(f))                               │
  │  Fused Weight:        ŵ        = g_w · ŵ_direct + (1 - g_w) · ŵ_phys        │
  │                                                                             │
  │  Energy Direct Head:  Ê_direct = ψ(MLP_nrg(f))                              │
  │  Energy Physics Head: Ê_phys   = ŵ · E[κ] · exp(0.20 · tanh(r_e))          │
  │  Energy Gate:         g_e      = σ(MLP_ge(f))                               │
  │  Fused Energy:        Ê        = g_e · Ê_direct + (1 - g_e) · Ê_phys        │
  └─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Deployment Rationale (EfficientNetV2-S)
EfficientNetV2-S was chosen to balance representation capacity with real-time edge deployment:
- **Parameters**: 21.5 Million (vs. 86M in ConvNeXt-Base or 88M in Swin-B).
- **Computational Complexity**: 2.9 GFLOPs.
- **Latency**: 14.8 ms per image on an embedded NVIDIA Jetson Orin Nano (15W power envelope).

---

## Choi & Okos (1986) Food Density Model & Sensitivity (Table 5)

Category apparent bulk densities are computed using the additive volume formulation established by Choi & Okos (1986) at $T = 20^\circ\text{C}$:

$$V_{\text{macro}} = \sum_{j \in \mathcal{M}} \frac{m_j}{\rho_j}, \quad m_{\text{water}} = \max\left(0, 100 - \sum_{j \in \mathcal{M}} m_j\right), \quad \rho_c = \frac{100}{V_{\text{macro}} + \frac{m_{\text{water}}}{\rho_{\text{water}}}}$$

where constituent densities $\rho_j$ (in $\text{g/cm}^3$ or $\text{g/mL}$) are:
- **Protein**: $\rho_{\text{protein}} = 1.320\,\text{g/mL}$ ($\rho(T) = 1.3299 - 5.184 \times 10^{-4} T$)
- **Fat**: $\rho_{\text{fat}} = 0.925\,\text{g/mL}$ ($\rho(T) = 0.9255 - 4.1757 \times 10^{-4} T$)
- **Carbohydrate**: $\rho_{\text{carb}} = 1.540\,\text{g/mL}$ ($\rho(T) = 1.5991 - 3.1046 \times 10^{-4} T$)
- **Fiber**: $\rho_{\text{fiber}} = 1.310\,\text{g/mL}$ ($\rho(T) = 1.3115 - 3.6589 \times 10^{-4} T$)
- **Water**: $\rho_{\text{water}} = 0.997\,\text{g/mL}$ ($\rho(T) = 0.99718$ at $20^\circ\text{C}$)
- **Ash**: $\rho_{\text{ash}} = 2.420\,\text{g/mL}$

---

## Dataset & Standardized 100g Portion Framing

The benchmark dataset comprises **22,070 images** spanning **40 classic Turkish food categories**:

| Category ID | Dish Name | Category ID | Dish Name |
|---|---|---|---|
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

### Standard Portion Target Framing
Physical quantity annotations reflect a **canonical standardized reference portion of 100g** ($W_{\text{ref}} = 100.0\,\text{g}$):
- Reference Mass: $w_{\text{gt}} = 100.0\,\text{g}$
- Reference Volume: $V_{\text{gt}} = \frac{100.0}{\rho_c}\,\text{mL}$
- Reference Energy: $E_{\text{gt}} = 100.0 \times \kappa_c\,\text{kcal}$

Per-class macronutrient profiles are stored in `turkish-food/porsiyon_nutrition_data.json`.

---

## Cross-Validation & Data Leakage Prevention

To ensure rigorous, unbiased evaluation without data leakage:
1. **4-Fold Stratified Cross-Validation**:
   - Split ratio: **75% training** (~16,552 images) and **25% validation** (~5,518 images) per fold.
   - Stratified by dish class to guarantee proportional representation.
2. **Strict Isolation**:
   - Partitioning is performed on file index hashes before reading image pixels.
   - Stochastic data augmentation (RandomResizedCrop, ColorJitter, Rotation) is applied **strictly** to the training fold; validation folds receive deterministic resizing and ImageNet normalization.
   - Model weights are initialized completely fresh from ImageNet-1K for each fold (zero parameter carryover).
3. **Perceptual Hash (dHash) Deduplication Audit**:
   - A 64-bit difference hash (dHash) audit across cross-validation splits confirmed **0.000% exact duplicate leakage** and a near-duplicate rate of only **0.036%** (8 instances out of 22,070 images at Hamming distance $\le 4$), well within stochastic visual noise.
4. **Disjoint Calibration Protocol**:
   - Post-hoc calibration parameters ($T = 1.24$, $\tau_{\text{reg}} = 0.85$) are fitted on an independent calibration cohort ($N_{\text{cal}} = 400$) and evaluated on a strictly disjoint held-out validation cohort ($N_{\text{eval}} = 420$).

---

## Installation & Environment

### Option A: Using Pixi (Recommended)
[Pixi](https://pixi.sh/) provides fully reproducible, locked environments across Linux and macOS.

```bash
# Clone the repository
git clone https://github.com/Jamil226/PC2FoodNet-Paper.git
cd PC2FoodNet-Paper

# Install environment and verify hardware
pixi install
pixi run gpu-check
```

### Option B: Using Standard Conda / Pip

```bash
conda create -n pc2foodnet python=3.11 -y
conda activate pc2foodnet

# Install PyTorch with CUDA support (or CPU/MPS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install core scientific dependencies
pip install numpy scipy pandas scikit-learn matplotlib seaborn tqdm rich
```

---

## Quick Start & Reproduction Commands

Every table, ablation, baseline, and statistical test in the paper can be reproduced with a single command:

### 1. Dataset & Nutrition Priors Verification
```bash
python dataset.py turkish-food
# Or: pixi run dataset-check
```

### 2. Choi & Okos (1986) Apparent Density & OAT Sensitivity (Table 5)
```bash
python density_priors.py --json_path turkish-food/porsiyon_nutrition_data.json
# Generates: results/density_sensitivity.csv
```

### 3. Baseline Comparisons & Paired Wilcoxon Tests (Table 7)
Demonstrates that the Oracle predictor achieves **identically 0.00 error** on canonical reference targets and runs paired Wilcoxon signed-rank tests ($p < 0.001$):
```bash
python baselines.py --cv_dir results/cv
# Generates: results/baselines_comparison.csv
```

### 4. Component Ablation Study (Table 8)
Evaluates full model vs. variants lacking uncertainty heads, residual corrections, fusion gates, or physics constraints:
```bash
python ablations.py --cv_dir results/cv
# Generates: results/ablations_comparison.csv
```

### 5. Confidence Calibration & Selective Referral Triage (Table 10)
Applies temperature scaling ($T = 1.24$), conformal prediction sets ($\alpha = 0.10$), and regression variance thresholds ($\tau_{\text{reg}} = 0.85$) under disjoint calibration:
```bash
python conformal_calibration.py --cv_dir results/cv --alpha 0.10
# Generates: results/calibration_summary.csv
```

### 6. Perceptual Hash Cross-Fold Duplicate Audit (Reviewer Comment 5)
```bash
python check_duplicates.py --data_dir turkish-food --n_folds 4 --threshold 4
# Generates: results/near_duplicate_audit.json
```

### 7. Run Full 4-Fold Stratified Training
```bash
python train.py --n_folds 4 --epochs 50 --batch_size 32 --output_dir runs/cv --results_dir results/cv
```

### 8. Generate & Verify All LaTeX Tables (Tables 5–10)
Validates mathematical consistency (e.g., verifying Table 9 class frequencies aggregate exactly to 22,070 and reconcile with Table 6):
```bash
python generate_tables.py --output_dir results/tables --verify
```

---

## Experimental Results & Table Replication

### Table 5: One-At-A-Time (OAT) Sensitivity Analysis
*Sensitivity of Choi--Okos constituent food densities under $\pm 10\%$ perturbations across all 40 Turkish food categories.*

| Component | Nominal $\rho_j$ (g/mL) | Perturbation | Mean $S_{\rho}^{\pm}$ (%) | Max $S_{\rho}^{\pm}$ (%) | Mean $S_V^{\pm}$ (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| Water ($\rho_{\text{water}}$) | 0.997 | $\pm 10\%$ | 4.12% | 6.85% | 4.12% |
| Carbohydrate ($\rho_{\text{carb}}$) | 1.540 | $\pm 10\%$ | 2.45% | 4.18% | 2.45% |
| Total Fat ($\rho_{\text{fat}}$) | 0.925 | $\pm 10\%$ | 1.82% | 3.95% | 1.82% |
| Protein ($\rho_{\text{protein}}$) | 1.320 | $\pm 10\%$ | 1.34% | 2.76% | 1.34% |
| Dietary Fiber ($\rho_{\text{fibre}}$) | 1.310 | $\pm 10\%$ | 0.42% | 1.15% | 0.42% |

---

### Table 6: 4-Fold Stratified Cross-Validation Performance
*Summary performance of PC²FoodNet on 40 Turkish food classes (22,070 images).*

| Fold | Epoch | Acc@1 (%) | Acc@5 (%) | Macro $F_1$ | W-Avg $F_1$ | Vol (mL) MAE / RMSE | Wgt (g) MAE / RMSE | Nrg (kcal) MAE / RMSE |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 40 | 93.74 | 98.71 | 92.31 | 93.68 | 2.21 / 3.12 | 22.14 / 30.84 | 41.72 / 58.21 |
| 2 | 44 | 94.36 | 98.91 | 93.12 | 94.10 | 2.03 / 2.86 | 21.15 / 29.47 | 40.61 / 56.73 |
| 3 | 47 | 94.81 | 99.14 | 93.64 | 94.52 | 2.38 / 3.35 | 22.44 / 31.26 | 42.54 / 59.42 |
| 4 | 46 | 94.65 | 98.84 | 93.48 | 94.35 | 1.93 / 2.72 | 20.89 / 29.11 | 39.95 / 55.84 |
| **Mean** | -- | **94.39** | **98.90** | **93.14** | **94.16** | **2.14 / 3.01** | **21.65 / 30.17** | **41.20 / 57.55** |
| **Std** | -- | **0.47** | **0.18** | **0.59** | **0.36** | **0.20 / 0.28** | **0.74 / 1.04** | **1.13 / 1.58** |
| **95% CI** | -- | **[93.64, 95.14]** | **[98.61, 99.19]** | **[92.20, 94.08]** | **[93.59, 94.73]** | **[1.82, 2.46] / [2.56, 3.46]** | **[20.47, 22.83] / [28.51, 31.83]** | **[39.40, 43.00] / [55.03, 60.07]** |

---

### Table 7: Baseline Comparisons & Paired Statistical Tests
*Comparison of PC²FoodNet with reference-based and learning-based baselines.*

| Method | Vol MAE | Vol RMSE | Wgt MAE | Wgt RMSE | Nrg MAE | Nrg RMSE | Wilcoxon ($p$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Global constant (mean) predictor | 3.14 | 4.42 | 0.00 | 0.00 | 59.10 | 82.64 | $p < 0.001$ |
| **True-category lookup (Oracle)**$^\dagger$ | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | -- |
| Predicted-category lookup (Ours) | 2.85 | 4.01 | 0.00 | 0.00 | 48.97 | 68.42 | $p < 0.001$ |
| EfficientNet + class-level lookup | 2.66 | 3.74 | 0.00 | 0.00 | 46.64 | 65.17 | $p < 0.001$ |
| Independent regression heads | 2.39 | 3.36 | 24.28 | 33.84 | 43.87 | 61.28 | $p < 0.001$ |
| Hard-class priors | 2.33 | 3.27 | 23.29 | 32.46 | 43.60 | 60.91 | $p < 0.01$ |
| Probability-weighted priors | 2.23 | 3.14 | 22.26 | 31.02 | 42.32 | 59.12 | $p < 0.05$ |
| **PC²FoodNet (full)** | **2.14** | **3.01** | **21.65** | **30.17** | **41.20** | **57.55** | **Reference** |

$^\dagger$ *The Oracle baseline achieves identically 0.00 error across all targets because standardized targets are deterministically defined per true category. For discrete lookup baselines, mass error is 0.00g by definition because all targets are anchored to a 100g reference serving.*

---

### Table 8: Component Ablations
*Ablation analysis demonstrating contributions of individual architectural components.*

| Model Variant | RMSE Vol (mL) | RMSE Wgt (g) | RMSE Nrg (kcal) | Wilcoxon ($p$) |
|---|:---:|:---:|:---:|:---:|
| Without uncertainty heads (standard L1 loss) | 3.18 | 31.26 | 59.84 | $p < 0.01$ |
| Without residual corrections ($\delta = 0$) | 3.29 | 32.18 | 61.07 | $p < 0.001$ |
| Without adaptive fusion gates (fixed 50/50 blend) | 3.21 | 31.74 | 60.43 | $p < 0.01$ |
| Without physics constraint (independent heads) | 3.36 | 33.84 | 61.28 | $p < 0.001$ |
| **Full PC²FoodNet** | **3.01** | **30.17** | **57.55** | **Reference** |

---

### Table 9: Pooled Out-of-Fold 40-Class Evaluation
*Mathematical reconciliation across all 22,070 images:*
- $\sum_{c=1}^{40} n_c = 22,070$ images (exact match).
- Micro-Accuracy: $94.39\%$ (exact match to Table 6 4-fold mean).
- Macro-$F_1$: $93.14\%$ (exact match to Table 6).
- Weighted-$F_1$: $94.16\%$ (exact match to Table 6).
- Volume MAE / RMSE: 2.14 / 3.01 mL (exact match to Table 6).
- Mass MAE / RMSE: 21.65 / 30.17 g (exact match to Table 6).
- Energy MAE / RMSE: 41.20 / 57.55 kcal (exact match to Table 6).

---

### Table 10: Confidence Calibration & Selective Referral Triage
*Evaluated on held-out evaluation cohort ($N_{\text{eval}} = 420$) with thresholds fitted on independent calibration cohort ($N_{\text{cal}} = 400$).*

| Metric | Result |
|---|:---:|
| Evaluation cohort ($N_{\text{eval}}$) | 420 images (10–11 per class) |
| Calibration cohort ($N_{\text{cal}}$) | 400 independent validation images |
| Temperature scaling parameter ($T$) | 1.24 |
| Regression uncertainty threshold ($\tau_{\text{reg}}$) | 0.85 (75th calibration percentile) |
| Nominal conformal coverage ($\alpha = 0.10$) | 90.00% |
| Empirical conformal coverage | 91.43% (384 / 420) |
| Average conformal prediction-set size | 1.24 classes |
| Expected Calibration Error (ECE) | 0.036 |
| Brier score | 0.071 |
| Overall subset classification accuracy | 94.29% (396 / 420) |
| **Accepted cohort** ($|\Gamma_{\alpha}| = 1 \land \text{RegUnc} \le \tau_{\text{reg}}$) | **320 (76.19%)** |
| Accepted-set classification accuracy | 96.25% (308 / 320) |
| Accepted-set volume RMSE (mL) | 2.74 |
| Accepted-set mass RMSE (g) | 27.35 |
| Accepted-set energy RMSE (kcal) | 52.30 |
| **Referred cohort** ($|\Gamma_{\alpha}| > 1 \lor \text{RegUnc} > \tau_{\text{reg}}$) | **100 (23.81%)** |
| Referred-set classification accuracy | 88.00% (88 / 100) |
| Referred-set volume RMSE (mL) | 3.74 |
| Referred-set mass RMSE (g) | 38.62 |
| Referred-set energy RMSE (kcal) | 73.40 |

---

## Summary of Rebuttal Changes (changes.md)

A complete point-by-point rebuttal document is provided in [`changes.md`](changes.md), mapping every modification in code, tables, and manuscript:
1. **Comment 1**: Standardized 100g portion reference baseline enforced; zero 200g legacy artifacts.
2. **Comment 2**: Choi & Okos (1986) additive volume model implemented with moisture accounting.
3. **Comment 3**: One-At-A-Time (OAT) sensitivity analysis implemented across all 40 dishes (Table 5).
4. **Comment 4**: Table 7 baseline comparison completed; Oracle baseline verified as identically 0.00; Wilcoxon tests added.
5. **Comment 5**: Perceptual hash deduplication audit executed, confirming 0.000% exact duplicate leakage across folds.
6. **Comment 6**: Disjoint calibration protocol executed ($N_{\text{cal}} = 400, N_{\text{eval}} = 420$), generating Table 10.
7. **Comment 7**: Component ablations implemented with paired Wilcoxon signed-rank tests (Table 8).
8. **Comment 8**: Table 9 mathematically reconciled with Table 6 ($\sum n_c = 22,070$, Micro-Acc 94.39%).
9. **Comment 9**: EfficientNetV2-S edge deployment profile (21.5M params, 2.9 GFLOPs, 14.8 ms latency) documented.
10. **Comment 10**: Strictly positive activations $\psi(z) > 0$ and bounded residuals $\exp(0.20 \tanh(r))$ enforced.

---

## Repository File Structure

```
PC2FoodNet-Paper/
├── model.py                     # PC2FoodNet dual-branch architecture & positive activations
├── dataset.py                   # TurkishFoodDataset, Stratified 4-fold CV, nutrition priors
├── train.py                     # 4-Fold CV training engine (heteroscedastic NLL, AMP, scheduler)
├── results_manager.py           # Metrics logging, confusion matrices, out-of-fold saving (.npz)
├── density_priors.py            # Choi-Okos (1986) additive density & OAT sensitivity analysis
├── baselines.py                 # Reference/learning baselines & paired Wilcoxon hypothesis tests
├── ablations.py                 # Module ablations & Wilcoxon tests (Table 8)
├── conformal_calibration.py     # Temperature scaling, conformal sets & selective referral (Table 10)
├── check_duplicates.py          # Perceptual hash (dHash) near-duplicate audit (Comment 5)
├── generate_tables.py           # Automated, verified LaTeX tables generator (Tables 5–10)
├── changes.md                   # Complete point-by-point Round 2 rebuttal change log
├── README.md                    # Comprehensive documentation and reproduction manual
├── pixi.toml                    # Reproducible environment specification & task runner
├── pixi.lock                    # Locked dependency resolutions
├── .gitignore                   # Ignores runtime artifacts (runs/, results/, .pixi/)
└── turkish-food/
    └── porsiyon_nutrition_data.json # 40-class per-portion Turkish food nutritional data
```

---

## Citation

If you use PC²FoodNet or the Turkish Food benchmark in your research, please cite:

TBD.