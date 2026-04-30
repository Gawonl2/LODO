"""
Experiment 1: Collapse Divergence Rate (CDR)

Formalizes the qualitative collapse divergence finding into a quantitative
metric: among documents that cause strong logprob collapse, what fraction
also cause factual degradation?

CDR  = P(delta_fact == 0 | delta_logprob < threshold)
HCR  = P(delta_fact  > 0 | delta_logprob < threshold)
HMCR = P(delta_fact  < 0 | delta_logprob < threshold)

Outputs
-------
results/E1_collapse_divergence_rates.csv
tables/E1_collapse_divergence_rates.tex
plots/E1_cdr_by_passage_num.png
plots/E1_cdr_by_doc_type.png  (only if doc_type available)
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def compute_rates(df: pd.DataFrame, threshold: float) -> dict:
    lp_imp = df[df["logprob_degradation"] < threshold]
    n = len(lp_imp)
    if n == 0:
        return {"n_logprob_important": 0, "CDR": float("nan"),
                "HCR": float("nan"), "HMCR": float("nan")}
    cdr  = (lp_imp["fact_degradation"] == 0).sum() / n
    hcr  = (lp_imp["fact_degradation"]  > 0).sum() / n
    hmcr = (lp_imp["fact_degradation"]  < 0).sum() / n
    return {
        "n_total":            len(df),
        "n_logprob_important": n,
        "logprob_important_rate": n / len(df),
        "n_fact_useful":      (df["fact_degradation"] > 0).sum(),
        "CDR":  round(cdr,  4),
        "HCR":  round(hcr,  4),
        "HMCR": round(hmcr, 4),
    }


def latex_pct(v):
    if pd.isna(v):
        return r"\textemdash"
    return f"{v*100:.1f}\\%"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       default=None, help="Path to analysis_table.csv")
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--threshold",   type=float, default=-2.0)
    p.add_argument("--passage_nums", nargs="+", type=int, default=[3, 5, 7, 10])
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    if args.input:
        df = pd.read_csv(args.input)
    else:
        base = Path(__file__).resolve().parent.parent
        csv = base / "results" / "analysis" / "analysis_table.csv"
        df = pd.read_csv(csv)

    # Use sweep data (has passage_num and doc_type)
    sweep = df[df["source"] == "sweep"].copy()

    thr = args.threshold

    # -----------------------------------------------------------------------
    # 1. Overall rates
    # -----------------------------------------------------------------------
    rows = []
    overall = compute_rates(df, thr)
    overall["group"] = "Overall (all sources)"
    overall["passage_num"] = "all"
    rows.append(overall)

    sweep_overall = compute_rates(sweep, thr)
    sweep_overall["group"] = "Sweep only"
    sweep_overall["passage_num"] = "all"
    rows.append(sweep_overall)

    # -----------------------------------------------------------------------
    # 2. By passage_num
    # -----------------------------------------------------------------------
    pn_data = []
    for pn in sorted(sweep["passage_num"].dropna().unique()):
        sub = sweep[sweep["passage_num"] == pn]
        r = compute_rates(sub, thr)
        r["group"] = f"passage_num={int(pn)}"
        r["passage_num"] = int(pn)
        rows.append(r)
        pn_data.append({"passage_num": int(pn), **r})

    # -----------------------------------------------------------------------
    # 3. By doc_type (positive vs negative)
    # -----------------------------------------------------------------------
    has_doctype = sweep["doc_type"].nunique() > 1 and "unknown" not in sweep["doc_type"].values
    dt_data = []
    if has_doctype:
        for dt, sub in sweep.groupby("doc_type"):
            r = compute_rates(sub, thr)
            r["group"] = f"doc_type={dt}"
            r["passage_num"] = "all"
            rows.append(r)
            dt_data.append({"doc_type": dt, **r})

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    result_df = pd.DataFrame(rows)
    csv_path = out / "results" / "E1_collapse_divergence_rates.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # -----------------------------------------------------------------------
    # LaTeX table
    # -----------------------------------------------------------------------
    tex_rows = []
    for _, row in result_df.iterrows():
        n_imp = int(row["n_logprob_important"]) if not pd.isna(row["n_logprob_important"]) else 0
        n_tot = int(row["n_total"]) if not pd.isna(row["n_total"]) else 0
        tex_rows.append(
            f"  {row['group']} & {n_tot} & {n_imp} & "
            f"{latex_pct(row['CDR'])} & {latex_pct(row['HCR'])} & {latex_pct(row['HMCR'])} \\\\"
        )

    tex = r"""\begin{table}[h]
\centering
\small
\begin{tabular}{lrrrrr}
\toprule
Group & $N$ & $N_{\text{logprob-imp}}$ & CDR & FCR & FDCR \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\caption{Collapse Divergence Rate (CDR), Factuality-Critical Rate (FCR), and
Factuality-Disrupting Rate (FDCR) at $\Delta_{\text{logprob}} < """ + str(thr) + r"""$.
CDR = fraction of logprob-important documents with zero factual degradation.}
\label{tab:e1_cdr}
\end{table}
"""
    tex_path = out / "tables" / "E1_collapse_divergence_rates.tex"
    tex_path.write_text(tex)
    print(f"Saved {tex_path}")

    # -----------------------------------------------------------------------
    # Plot 1: CDR / HCR / HMCR by passage_num (stacked bar)
    # -----------------------------------------------------------------------
    if pn_data:
        pn_df = pd.DataFrame(pn_data).sort_values("passage_num")
        x = np.arange(len(pn_df))
        width = 0.55

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x, pn_df["CDR"],  width, label="CDR (confidence-only / logprob-important)",
               color="#e07b54", alpha=0.9)
        ax.bar(x, pn_df["HCR"],  width, bottom=pn_df["CDR"],
               label="FCR (factuality-critical / logprob-important)", color="#5b9bd5", alpha=0.9)
        ax.bar(x, pn_df["HMCR"], width,
               bottom=pn_df["CDR"] + pn_df["HCR"],
               label="FDCR (factuality-disrupting / logprob-important)", color="#70ad47", alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([f"n={int(r['passage_num'])}\n({int(r['n_logprob_important'])} logprob-imp)"
                            for _, r in pn_df.iterrows()])
        ax.set_ylabel("Fraction of logprob-important documents")
        ax.set_title(f"Collapse Divergence by Passage Count (threshold={thr})")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        plot1 = out / "plots" / "E1_cdr_by_passage_num.png"
        fig.savefig(plot1, dpi=150)
        plt.close(fig)
        print(f"Saved {plot1}")

    # -----------------------------------------------------------------------
    # Plot 2: CDR by doc_type
    # -----------------------------------------------------------------------
    if dt_data:
        dt_df = pd.DataFrame(dt_data)
        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.arange(len(dt_df))
        ax.bar(x, dt_df["CDR"],  0.4, label="CDR",  color="#e07b54", alpha=0.9)
        ax.bar(x, dt_df["HCR"],  0.4, bottom=dt_df["CDR"],
               label="HCR", color="#5b9bd5", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(dt_df["doc_type"])
        ax.set_ylabel("Fraction of logprob-important documents")
        ax.set_title("Collapse Divergence by Document Type")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        plot2 = out / "plots" / "E1_cdr_by_doc_type.png"
        fig.savefig(plot2, dpi=150)
        plt.close(fig)
        print(f"Saved {plot2}")
    else:
        print("NOTE: doc_type not available — skipping E1_cdr_by_doc_type.png")

    # Print summary to stdout
    print("\n=== E1 Summary ===")
    for _, row in result_df.iterrows():
        print(f"  {row['group']}: CDR={row['CDR']:.3f}  HCR={row['HCR']:.3f}  HMCR={row['HMCR']:.3f}")


if __name__ == "__main__":
    main()
