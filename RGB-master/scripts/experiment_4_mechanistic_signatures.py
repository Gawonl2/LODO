"""
Experiment 4: Mechanistic Signature of Helpful vs Confidence-Only Documents

Compares layer-wise drift profiles by E2 taxonomy category.

Two data sources are used:
  A) Sweep data — 3 checkpoints (layer 0, 16, 32). Sufficient for layer-32
     comparison and a coarse drift trajectory.
  B) Detailed case studies — full 33-layer drift curves.  Provides the
     integration-horizon analysis.

Outputs
-------
results/E4_mechanistic_signatures.csv
tables/E4_layer_drift_stats.tex
tables/E4_mechanistic_tests.tex
plots/E4_layerwise_drift_by_taxonomy.png
plots/E4_layer32_drift_by_taxonomy.png
plots/E4_auc_late_layers_by_taxonomy.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


CATEGORY_ORDER = [
    "factuality-critical",
    "confidence-only",
    "factuality-disrupting",
    "fact-only",
    "neutral",
    "factuality-weak",
]

COLORS = {
    "factuality-critical":   "#5b9bd5",
    "confidence-only":     "#e07b54",
    "factuality-disrupting": "#ffc000",
    "fact-only":           "#70ad47",
    "neutral":             "#bfbfbf",
    "factuality-weak":        "#9e480e",
}

THR = -2.0


def assign_category(row):
    lp, fd = row["logprob_degradation"], row["fact_degradation"]
    if lp < THR:
        return "factuality-critical" if fd > 0 else ("confidence-only" if fd == 0 else "factuality-disrupting")
    else:
        return "fact-only" if fd > 0 else ("neutral" if fd == 0 else "factuality-weak")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",         default=None, help="analysis_table.csv")
    p.add_argument("--case_studies",  default="detailed_case_studies.json")
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--threshold",     type=float, default=-2.0)
    return p.parse_args()


def mwu_str(a, b):
    """Return Mann-Whitney U test string; handles degenerate cases."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    try:
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        return round(float(stat), 1), round(float(p), 4)
    except Exception:
        return float("nan"), float("nan")


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent.parent
    out  = Path(args.output_dir)
    for d in ["results", "tables", "plots"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # A) Sweep data: 3-checkpoint layer comparison
    # -----------------------------------------------------------------------
    if args.input:
        df = pd.read_csv(args.input)
    else:
        df = pd.read_csv(base / "results" / "analysis" / "analysis_table.csv")

    df["category"] = df.apply(assign_category, axis=1)

    # AUC approximation with 3 checkpoints: trapezoid over [0, 16, 32]
    layer_cols = ["layer_0_drift", "layer_16_drift", "layer_32_drift"]
    layer_xs   = np.array([0, 16, 32], dtype=float)
    vals = df[layer_cols].values  # shape (n, 3)
    # manual trapezoid: sum of (dx * (y_left + y_right) / 2)
    diffs = np.diff(layer_xs)
    df["auc_late_approx"] = (
        diffs[0] * (vals[:, 0] + vals[:, 1]) / 2 +
        diffs[1] * (vals[:, 1] + vals[:, 2]) / 2
    )

    # Per-category stats
    stat_rows = []
    for cat in CATEGORY_ORDER:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        for lcol, lx in zip(layer_cols, layer_xs):
            vals = sub[lcol].dropna()
            stat_rows.append({
                "category": cat,
                "layer":    lx,
                "n":        len(vals),
                "mean_drift": vals.mean(),
                "median_drift": vals.median(),
                "max_drift": vals.max() if len(vals) else float("nan"),
            })

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(out / "results" / "E4_mechanistic_signatures.csv", index=False)
    print("Saved results/E4_mechanistic_signatures.csv")

    # -----------------------------------------------------------------------
    # LaTeX table: Layer-32 drift stats by category
    # -----------------------------------------------------------------------
    layer32_stats = df.groupby("category")["layer_32_drift"].agg(["mean", "median", "max", "count"])
    auc_stats     = df.groupby("category")["auc_late_approx"].agg(["mean", "median"])

    tex_rows = []
    for cat in CATEGORY_ORDER:
        if cat not in layer32_stats.index:
            continue
        r  = layer32_stats.loc[cat]
        ra = auc_stats.loc[cat] if cat in auc_stats.index else {"mean": float("nan"), "median": float("nan")}
        tex_rows.append(
            f"  {cat} & {int(r['count'])} & "
            f"{r['mean']:.2f} & {r['median']:.2f} & {r['max']:.2f} & "
            f"{ra['mean']:.1f} \\\\"
        )

    tex1 = (
        r"\begin{table}[h]" + "\n"
        r"\centering\small" + "\n"
        r"\begin{tabular}{lrrrrr}" + "\n"
        r"\toprule" + "\n"
        r"Category & $n$ & Mean $\Delta_{L2}^{32}$ & Median & Max & Mean AUC\textsubscript{[0,32]} \\" + "\n"
        r"\midrule" + "\n"
        + "\n".join(tex_rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\caption{Layer-32 representation drift and AUC by importance category. "
        r"AUC approximated via trapezoid rule over checkpoints at layers 0, 16, 32.}" + "\n"
        r"\label{tab:e4_layer_drift_stats}" + "\n"
        r"\end{table}" + "\n"
    )
    (out / "tables" / "E4_layer_drift_stats.tex").write_text(tex1)
    print("Saved tables/E4_layer_drift_stats.tex")

    # -----------------------------------------------------------------------
    # LaTeX table: Statistical tests
    # -----------------------------------------------------------------------
    test_pairs = [
        ("factuality-critical", "confidence-only",  "layer_32_drift"),
        ("factuality-critical", "confidence-only",  "auc_late_approx"),
        ("confidence-only",   "neutral",           "layer_32_drift"),
        ("factuality-critical", "neutral",            "layer_32_drift"),
    ]
    test_rows = []
    for catA, catB, metric in test_pairs:
        a_vals = df[df["category"] == catA][metric].dropna().values
        b_vals = df[df["category"] == catB][metric].dropna().values
        stat, p = mwu_str(a_vals, b_vals)
        p_str = f"{p:.4f}" if not (isinstance(p, float) and np.isnan(p)) else r"\textemdash"
        s_str = f"{stat:.1f}" if not (isinstance(stat, float) and np.isnan(stat)) else r"\textemdash"
        sig = "**" if isinstance(p, float) and p < 0.01 else ("*" if isinstance(p, float) and p < 0.05 else "")
        test_rows.append(
            f"  {catA} vs {catB} & {metric.replace('_', ' ')} "
            f"& {len(a_vals)} & {len(b_vals)} & {s_str} & {p_str}{sig} \\\\"
        )

    tex2 = (
        r"\begin{table}[h]" + "\n"
        r"\centering\small" + "\n"
        r"\begin{tabular}{llrrrr}" + "\n"
        r"\toprule" + "\n"
        r"Comparison & Metric & $n_A$ & $n_B$ & $U$ & $p$ \\" + "\n"
        r"\midrule" + "\n"
        + "\n".join(test_rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\caption{Mann-Whitney U tests for drift differences between importance categories. "
        r"* $p < 0.05$, ** $p < 0.01$.}" + "\n"
        r"\label{tab:e4_mechanistic_tests}" + "\n"
        r"\end{table}" + "\n"
    )
    (out / "tables" / "E4_mechanistic_tests.tex").write_text(tex2)
    print("Saved tables/E4_mechanistic_tests.tex")

    # -----------------------------------------------------------------------
    # Plot 1: Coarse layer trajectory (3-point) by category mean
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cat in CATEGORY_ORDER:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        means = [sub[c].mean() for c in layer_cols]
        ax.plot(layer_xs, means, marker="o", color=COLORS[cat],
                label=f"{cat} (n={len(sub)})", linewidth=1.8, alpha=0.85)

    ax.axvspan(16, 20, alpha=0.08, color="gray", label="Integration horizon\n(est. layers 18–20)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 drift")
    ax.set_title("Layer-wise Representation Drift by Importance Category")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "plots" / "E4_layerwise_drift_by_taxonomy.png", dpi=150)
    plt.close(fig)
    print("Saved plots/E4_layerwise_drift_by_taxonomy.png")

    # -----------------------------------------------------------------------
    # Plot 2: Box plot — layer-32 drift by category
    # -----------------------------------------------------------------------
    cats_present = [c for c in CATEGORY_ORDER if c in df["category"].values]
    data = [df[df["category"] == c]["layer_32_drift"].dropna().values for c in cats_present]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(data, tick_labels=[c.replace("-", "\n") for c in cats_present],
                    patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, cat in zip(bp["boxes"], cats_present):
        patch.set_facecolor(COLORS[cat])
        patch.set_alpha(0.8)
    ax.set_ylabel("L2 drift at layer 32")
    ax.set_title("Layer-32 Representation Drift by Importance Category")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "plots" / "E4_layer32_drift_by_taxonomy.png", dpi=150)
    plt.close(fig)
    print("Saved plots/E4_layer32_drift_by_taxonomy.png")

    # -----------------------------------------------------------------------
    # Plot 3: AUC[0,32] by category
    # -----------------------------------------------------------------------
    auc_data = [df[df["category"] == c]["auc_late_approx"].dropna().values for c in cats_present]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(auc_data, tick_labels=[c.replace("-", "\n") for c in cats_present],
                    patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, cat in zip(bp["boxes"], cats_present):
        patch.set_facecolor(COLORS[cat])
        patch.set_alpha(0.8)
    ax.set_ylabel("AUC drift [layers 0–32] (trapezoid)")
    ax.set_title("Area Under Drift Curve by Importance Category")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "plots" / "E4_auc_late_layers_by_taxonomy.png", dpi=150)
    plt.close(fig)
    print("Saved plots/E4_auc_late_layers_by_taxonomy.png")

    # -----------------------------------------------------------------------
    # B) Full 33-layer analysis from detailed_case_studies.json
    # -----------------------------------------------------------------------
    case_path = base / args.case_studies
    if case_path.exists():
        with open(case_path) as f:
            cases = json.load(f)

        fig, ax = plt.subplots(figsize=(8, 5))
        group_colors = {"Top Divergence": "#e07b54", "Average Baseline": "#5b9bd5"}
        layers = list(range(33))

        for case in cases:
            grp   = case["group"]
            drifts = case["layer_drifts"]
            ax.plot(layers, drifts, color=group_colors[grp], alpha=0.4, linewidth=0.9)

        # Mean curves
        for grp, color in group_colors.items():
            grp_cases = [c for c in cases if c["group"] == grp]
            means = [np.mean([c["layer_drifts"][l] for c in grp_cases]) for l in layers]
            ax.plot(layers, means, color=color, linewidth=2.5, label=f"{grp} (mean)")

        ax.axvspan(18, 20, alpha=0.12, color="gray", label="Integration horizon")
        ax.set_xlabel("Layer")
        ax.set_ylabel("L2 drift")
        ax.set_title("Full 33-Layer Drift Trajectory (case studies)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / "plots" / "E4_full_33layer_drift_case_studies.png", dpi=150)
        plt.close(fig)
        print("Saved plots/E4_full_33layer_drift_case_studies.png")

    print("\n=== E4 Summary ===")
    for cat in cats_present:
        sub = df[df["category"] == cat]
        print(f"  {cat}: n={len(sub)}, "
              f"mean L32={sub['layer_32_drift'].mean():.2f}, "
              f"mean AUC={sub['auc_late_approx'].mean():.1f}")


if __name__ == "__main__":
    main()
