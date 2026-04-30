"""
Experiment 2: Factuality-Aware Importance Taxonomy

Classifies every ablated document into six mutually exclusive categories
based on logprob degradation and factual degradation:

  1. factuality-critical     logprob < -2  AND  fact > 0
  2. confidence-only       logprob < -2  AND  fact == 0
  3. harmful-influential   logprob < -2  AND  fact < 0
  4. fact-only             logprob >= -2 AND  fact > 0
  5. neutral               logprob >= -2 AND  fact == 0
  6. factuality-weak          logprob >= -2 AND  fact < 0

Outputs
-------
results/E2_importance_taxonomy.csv
tables/E2_taxonomy_overall.tex
tables/E2_taxonomy_by_passage_num.tex
tables/E2_taxonomy_by_baseline_correctness.tex
plots/E2_taxonomy_stacked_by_passage_num.png
plots/E2_logprob_vs_fact_taxonomy.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np


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
    "confidence-only":       "#e07b54",
    "factuality-disrupting": "#ffc000",
    "fact-only":             "#70ad47",
    "neutral":               "#bfbfbf",
    "factuality-weak":       "#9e480e",
}


def assign_category(row, thr):
    lp = row["logprob_degradation"]
    fd = row["fact_degradation"]
    if lp < thr:
        if fd > 0:    return "factuality-critical"
        elif fd == 0: return "confidence-only"
        else:         return "factuality-disrupting"
    else:
        if fd > 0:    return "fact-only"
        elif fd == 0: return "neutral"
        else:         return "factuality-weak"


def make_freq_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    pcts   = counts / counts.sum()
    return pd.DataFrame({"count": counts, "pct": pcts})


def freq_to_latex(freq_df: pd.DataFrame, caption: str, label: str) -> str:
    rows = []
    for cat in CATEGORY_ORDER:
        c = freq_df.loc[cat, "count"] if cat in freq_df.index else 0
        p = freq_df.loc[cat, "pct"]   if cat in freq_df.index else 0.0
        cat_tex = cat.replace("-", "\\-")
        rows.append(f"  {cat_tex} & {c} & {p*100:.1f}\\% \\\\")

    return (
        r"\begin{table}[h]" + "\n"
        r"\centering\small" + "\n"
        r"\begin{tabular}{lrr}" + "\n"
        r"\toprule" + "\n"
        r"Category & Count & \%" + r" \\" + "\n"
        r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        rf"\caption{{{caption}}}" + "\n"
        rf"\label{{{label}}}" + "\n"
        r"\end{table}" + "\n"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       default=None)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--threshold",   type=float, default=-2.0)
    p.add_argument("--passage_nums", nargs="+", type=int, default=[3, 5, 7, 10])
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    for d in ["results", "tables", "plots"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    if args.input:
        df = pd.read_csv(args.input)
    else:
        base = Path(__file__).resolve().parent.parent
        df = pd.read_csv(base / "results" / "analysis" / "analysis_table.csv")

    thr = args.threshold
    df["category"] = df.apply(lambda r: assign_category(r, thr), axis=1)

    # -----------------------------------------------------------------------
    # Overall CSV
    # -----------------------------------------------------------------------
    df.to_csv(out / "results" / "E2_importance_taxonomy.csv", index=False)
    print(f"Saved results/E2_importance_taxonomy.csv")

    # -----------------------------------------------------------------------
    # Table 1: Overall
    # -----------------------------------------------------------------------
    freq_all = make_freq_table(df)
    tex1 = freq_to_latex(
        freq_all,
        caption="Factuality-aware importance taxonomy across all ablations.",
        label="tab:e2_taxonomy_overall",
    )
    (out / "tables" / "E2_taxonomy_overall.tex").write_text(tex1)
    print("Saved tables/E2_taxonomy_overall.tex")

    # -----------------------------------------------------------------------
    # Table 2: By passage_num (sweep only)
    # -----------------------------------------------------------------------
    sweep = df[df["source"] == "sweep"].copy()
    if not sweep.empty and sweep["passage_num"].notna().any():
        rows = []
        for pn in sorted(sweep["passage_num"].dropna().unique()):
            sub = sweep[sweep["passage_num"] == pn]
            ft = make_freq_table(sub)
            for cat in CATEGORY_ORDER:
                c = int(ft.loc[cat, "count"]) if cat in ft.index else 0
                p = ft.loc[cat, "pct"]        if cat in ft.index else 0.0
                rows.append({"passage_num": int(pn), "category": cat, "count": c, "pct": p})
        pn_df = pd.DataFrame(rows)
        # Wide pivot for LaTeX
        pivot = pn_df.pivot(index="category", columns="passage_num", values="pct").reindex(CATEGORY_ORDER)
        pivot = pivot.fillna(0)

        tex2_rows = []
        for cat in CATEGORY_ORDER:
            vals = " & ".join(f"{pivot.loc[cat, c]*100:.1f}\\%" for c in sorted(pivot.columns))
            tex2_rows.append(f"  {cat} & {vals} \\\\")

        col_header = " & ".join(f"$n={c}$" for c in sorted(pivot.columns))
        tex2 = (
            r"\begin{table}[h]" + "\n"
            r"\centering\small" + "\n"
            rf"\begin{{tabular}}{{l{'r'*len(pivot.columns)}}}" + "\n"
            r"\toprule" + "\n"
            f"Category & {col_header} \\\\\n"
            r"\midrule" + "\n"
            + "\n".join(tex2_rows) + "\n"
            r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
            r"\caption{Importance taxonomy by passage count (percentages). "
            r"Factual degradation is only observable when the baseline is correct, "
            r"explaining the low factuality-critical rate at small $n$.}" + "\n"
            r"\label{tab:e2_taxonomy_by_passage_num}" + "\n"
            r"\end{table}" + "\n"
        )
        (out / "tables" / "E2_taxonomy_by_passage_num.tex").write_text(tex2)
        print("Saved tables/E2_taxonomy_by_passage_num.tex")

    # -----------------------------------------------------------------------
    # Table 3: By baseline correctness
    # -----------------------------------------------------------------------
    tex3_rows = []
    for correct, label in [(True, "Baseline Correct"), (False, "Baseline Incorrect")]:
        sub = df[df["baseline_correct"] == correct]
        if sub.empty:
            continue
        ft = make_freq_table(sub)
        for cat in CATEGORY_ORDER:
            c = int(ft.loc[cat, "count"]) if cat in ft.index else 0
            p = ft.loc[cat, "pct"]        if cat in ft.index else 0.0
            tex3_rows.append(f"  {label} & {cat} & {c} & {p*100:.1f}\\% \\\\")

    tex3 = (
        r"\begin{table}[h]" + "\n"
        r"\centering\small" + "\n"
        r"\begin{tabular}{llrr}" + "\n"
        r"\toprule" + "\n"
        r"Baseline & Category & Count & \% \\" + "\n"
        r"\midrule" + "\n"
        + "\n".join(tex3_rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\caption{Importance taxonomy split by baseline correctness. "
        r"Helpful-important cases require a correct baseline, so they only "
        r"appear in the ``Baseline Correct'' partition.}" + "\n"
        r"\label{tab:e2_taxonomy_by_baseline}" + "\n"
        r"\end{table}" + "\n"
    )
    (out / "tables" / "E2_taxonomy_by_baseline_correctness.tex").write_text(tex3)
    print("Saved tables/E2_taxonomy_by_baseline_correctness.tex")

    # -----------------------------------------------------------------------
    # Plot 1: Stacked bar by passage_num
    # -----------------------------------------------------------------------
    if not sweep.empty and sweep["passage_num"].notna().any():
        pn_vals = sorted(sweep["passage_num"].dropna().unique())
        cat_fracs = {cat: [] for cat in CATEGORY_ORDER}
        for pn in pn_vals:
            sub = sweep[sweep["passage_num"] == pn]
            ft  = make_freq_table(sub)
            for cat in CATEGORY_ORDER:
                cat_fracs[cat].append(ft.loc[cat, "pct"] if cat in ft.index else 0.0)

        x     = np.arange(len(pn_vals))
        width = 0.55
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bottoms = np.zeros(len(pn_vals))
        for cat in CATEGORY_ORDER:
            vals = np.array(cat_fracs[cat])
            ax.bar(x, vals, width, bottom=bottoms, color=COLORS[cat],
                   label=cat, alpha=0.9)
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels([f"n={int(pn)}" for pn in pn_vals])
        ax.set_ylabel("Fraction of ablations")
        ax.set_title("Document Importance Taxonomy by Passage Count")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
        ax.legend(loc="lower left", fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "plots" / "E2_taxonomy_stacked_by_passage_num.png", dpi=150)
        plt.close(fig)
        print("Saved plots/E2_taxonomy_stacked_by_passage_num.png")

    # -----------------------------------------------------------------------
    # Plot 2: Scatter logprob vs fact colored by category
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for cat in CATEGORY_ORDER:
        sub = df[df["category"] == cat]
        ax.scatter(sub["logprob_degradation"], sub["fact_degradation"],
                   color=COLORS[cat], label=cat, alpha=0.6, s=30, edgecolors="none")

    ax.axvline(thr, color="black", linestyle="--", linewidth=0.8, label=f"threshold={thr}")
    ax.axhline(0,   color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$\Delta_{\mathrm{logprob}}$")
    ax.set_ylabel(r"$\Delta_{\mathrm{fact}}$")
    ax.set_title("Logprob Degradation vs Factual Degradation (taxonomy)")
    ax.legend(fontsize=8, markerscale=1.4)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "plots" / "E2_logprob_vs_fact_taxonomy.png", dpi=150)
    plt.close(fig)
    print("Saved plots/E2_logprob_vs_fact_taxonomy.png")

    # Summary
    print("\n=== E2 Summary ===")
    freq_all["pct_str"] = (freq_all["pct"] * 100).map("{:.1f}%".format)
    print(freq_all[["count", "pct_str"]].to_string())


if __name__ == "__main__":
    main()
