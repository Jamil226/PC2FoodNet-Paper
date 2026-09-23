"""
density_priors.py — Choi & Okos (1986) Food Density and Sensitivity Analysis
for PC2FoodNet.

References:
  Choi, Y., & Okos, M. R. (1986). Effects of Temperature and Composition on the
  Thermal Properties of Foods. Food Engineering and Process Applications, 1, 93–101.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────────────
# Choi-Okos (1986) Constituent Densities at 20°C (g/cm³ == g/mL)
# ────────────────────────────────────────────────────────────────────────────
CHOI_OKOS_DENSITIES = {
    "protein": 1.320,        # rho_protein = 1.3299 - 5.184e-4 * T
    "fat": 0.925,            # rho_fat     = 0.9255 - 4.1757e-4 * T
    "carbohydrate": 1.540,   # rho_carb    = 1.5991 - 3.1046e-4 * T
    "fiber": 1.310,          # rho_fiber   = 1.3115 - 3.6589e-4 * T
    "water": 0.997,          # rho_water   = 0.99718 at 20°C
    "ash": 2.420,
}


def compute_apparent_macronutrient_density(
    protein_g: float,
    fat_g: float,
    carb_g: float,
    fiber_g: float,
    densities: Dict[str, float] = CHOI_OKOS_DENSITIES,
    eps: float = 1e-8,
) -> float:
    """
    Computes apparent macronutrient bulk density (g/mL) using the additive volume
    formulation from Choi & Okos (1986):

        V_macro = sum_j (m_j / rho_j)
        rho_c   = (sum_j m_j) / (V_macro + eps)
    """
    total_mass = protein_g + fat_g + carb_g + fiber_g
    if total_mass <= 0:
        return 1.05  # Default fallback density

    v_macro = (
        (protein_g / densities["protein"]) +
        (fat_g / densities["fat"]) +
        (carb_g / densities["carbohydrate"]) +
        (fiber_g / densities["fiber"])
    )
    return total_mass / (v_macro + eps)


def compute_density_with_moisture(
    protein_g: float,
    fat_g: float,
    carb_g: float,
    fiber_g: float,
    densities: Dict[str, float] = CHOI_OKOS_DENSITIES,
    basis_mass_g: float = 100.0,
) -> Tuple[float, float, float]:
    """
    Computes moisture-inclusive composite bulk density and total volume:
      water_g = max(0.0, basis_mass_g - (protein + fat + carb + fiber))
      V_bulk  = sum_j (m_j / rho_j) + (water_g / rho_water)
      rho_c   = basis_mass_g / V_bulk
    """
    water_g = max(0.0, basis_mass_g - (protein_g + fat_g + carb_g + fiber_g))
    v_total = (
        (protein_g / densities["protein"]) +
        (fat_g / densities["fat"]) +
        (carb_g / densities["carbohydrate"]) +
        (fiber_g / densities["fiber"]) +
        (water_g / densities["water"])
    )
    density = basis_mass_g / max(v_total, 1e-6)
    return density, v_total, water_g


def run_one_at_a_time_sensitivity(
    class_nutritions: List[Dict],
    perturbation: float = 0.10,
    include_moisture: bool = True,
) -> pd.DataFrame:
    """
    Performs one-at-a-time (OAT) sensitivity analysis per Section 4.3 and Table 5:
    Perturb each component density rho_j by +/- 10% and record relative change
    in apparent density and derived reference volume.
    """
    components = ["water", "carbohydrate", "fat", "protein", "fiber"] if include_moisture else ["carbohydrate", "fat", "protein", "fiber"]
    results = []

    for item in class_nutritions:
        name = item["yemek"]
        p = float(item.get("protein", 5.0))
        f = float(item.get("yağ", 5.0))
        c = float(item.get("karbonhidrat", 20.0))
        fib = float(item.get("lif", 2.0))

        if include_moisture:
            rho_0, v_0, w_g = compute_density_with_moisture(p, f, c, fib, CHOI_OKOS_DENSITIES)
        else:
            rho_0 = compute_apparent_macronutrient_density(p, f, c, fib, CHOI_OKOS_DENSITIES)
            v_0 = 100.0 / rho_0
            w_g = 0.0

        row = {
            "dish": name,
            "nominal_density": round(rho_0, 4),
            "nominal_volume_mL": round(v_0, 2),
            "moisture_g": round(w_g, 2),
        }

        for comp in components:
            # Positive perturbation (+10%)
            d_pos = dict(CHOI_OKOS_DENSITIES)
            d_pos[comp] = CHOI_OKOS_DENSITIES[comp] * (1.0 + perturbation)
            if include_moisture:
                rho_pos, v_pos, _ = compute_density_with_moisture(p, f, c, fib, d_pos)
            else:
                rho_pos = compute_apparent_macronutrient_density(p, f, c, fib, d_pos)
                v_pos = 100.0 / rho_pos
            s_rho_pos = abs(rho_pos - rho_0) / rho_0 * 100.0
            s_v_pos = abs(v_pos - v_0) / v_0 * 100.0

            # Negative perturbation (-10%)
            d_neg = dict(CHOI_OKOS_DENSITIES)
            d_neg[comp] = CHOI_OKOS_DENSITIES[comp] * (1.0 - perturbation)
            if include_moisture:
                rho_neg, v_neg, _ = compute_density_with_moisture(p, f, c, fib, d_neg)
            else:
                rho_neg = compute_apparent_macronutrient_density(p, f, c, fib, d_neg)
                v_neg = 100.0 / rho_neg
            s_rho_neg = abs(rho_neg - rho_0) / rho_0 * 100.0
            s_v_neg = abs(v_neg - v_0) / v_0 * 100.0

            row[f"S_rho_{comp}_pct"] = round(max(s_rho_pos, s_rho_neg), 3)
            row[f"S_vol_{comp}_pct"] = round(max(s_v_pos, s_v_neg), 3)

        results.append(row)

    df = pd.DataFrame(results)
    return df


def generate_sensitivity_summary_table(df: pd.DataFrame, perturbation: float = 0.10) -> pd.DataFrame:
    """
    Generates summary table matching Table tab:sensitivity_results in the manuscript:
    Component | Nominal rho_j (g/mL) | Perturbation | Mean S_rho (%) | Max S_rho (%) | Mean S_V (%)
    """
    display_names = {
        "water": ("Water ($\\rho_{\\mathrm{water}}$)", 0.997),
        "carbohydrate": ("Carbohydrate ($\\rho_{\\mathrm{carb}}$)", 1.540),
        "fat": ("Total Fat ($\\rho_{\\mathrm{fat}}$)", 0.925),
        "protein": ("Protein ($\\rho_{\\mathrm{protein}}$)", 1.320),
        "fiber": ("Dietary Fiber ($\\rho_{\\mathrm{fibre}}$)", 1.310),
    }

    rows = []
    for comp, (label, nominal) in display_names.items():
        if f"S_rho_{comp}_pct" in df.columns:
            mean_rho = df[f"S_rho_{comp}_pct"].mean()
            max_rho = df[f"S_rho_{comp}_pct"].max()
            mean_vol = df[f"S_vol_{comp}_pct"].mean()
            rows.append({
                "Component": label,
                "Nominal rho_j (g/mL)": nominal,
                "Perturbation": f"±{int(perturbation * 100)}%",
                "Mean S_rho (%)": f"{mean_rho:.2f}%",
                "Max S_rho (%)": f"{max_rho:.2f}%",
                "Mean S_V (%)": f"{mean_vol:.2f}%",
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Choi & Okos (1986) Food Density & OAT Sensitivity Analysis")
    parser.add_argument("--json_path", type=str, default="turkish-food/porsiyon_nutrition_data.json",
                        help="Path to per-class nutrition data JSON")
    parser.add_argument("--output_csv", type=str, default="results/density_sensitivity.csv",
                        help="Path to save OAT sensitivity analysis results")
    parser.add_argument("--perturbation", type=float, default=0.10,
                        help="One-At-A-Time perturbation magnitude (default 0.10 for ±10%)")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("Choi & Okos (1986) Constituent Food Densities at 20°C:")
    print("=" * 75)
    for k, v in CHOI_OKOS_DENSITIES.items():
        print(f"  {k:15s}: {v:.3f} g/mL")
    print("=" * 75)

    json_p = Path(args.json_path)
    if not json_p.exists():
        print(f"\n[Warning] Nutrition JSON '{json_p}' not found.")
        return

    with open(json_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n[✓] Loaded {len(data)} dishes from '{json_p}'")
    df = run_one_at_a_time_sensitivity(data, perturbation=args.perturbation, include_moisture=True)

    out_p = Path(args.output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    print(f"[✓] Per-dish sensitivity results saved to '{out_p}'")

    print("\n" + "=" * 75)
    print("Table: One-At-A-Time (OAT) Sensitivity Analysis (Manuscript Table)")
    print("=" * 75)
    summary_df = generate_sensitivity_summary_table(df, perturbation=args.perturbation)
    print(summary_df.to_string(index=False))
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
