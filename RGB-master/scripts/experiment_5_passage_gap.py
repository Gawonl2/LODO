"""
Experiment 5: Passage-Number Dependence of the Influence-Usefulness Gap

For each passage_num in {3, 5, 7, 10}, tracks:
  - logprob-important rate
  - factually-useful rate
  - confidence-only rate (CDR × logprob-important rate)
  - helpful-important rate
  - baseline accuracy
  - average layer-32 drift
  - gap = logprob-important rate - factually-useful rate

Outputs
-------
results/E5_passage_gap_metrics.csv
tables/E5_passage_gap_metrics.tex
plots/E5_influence_usefulness_gap_by_passage.png
plots/E5_cdr_vs_baseline_accuracy.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


THR = -2.0


def assign_category(lp, fd):
    if lp < THR:
        return "factuality-critical" if fd > 0 else ("confidence-only" if fd == 0 else "factuality-disrupting")
    else:
        return "fact-only" if fd > 0 else ("neutral" if fd == 0 else "factuality-weak")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",        default=None)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--threshold",    type=float, default=-2.0)
    p.add_argument("--passage_nums", nargs="+", type=int, default=[3, 5, 7, 10])
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output_dir)
    for d in ["results", "tables", "plots"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    if args.input:
        df = pd.read_csv(args.input)
    else:
        base = Path(__file__).resolve().parent.parent
        df = pd.read_csv(base / "results" / "analysis" / "analysis_table.csv")

    sweep = df[df["source"] == "sweep"].copy()
    sweep["category"] = sweep.apply(
        lambda r: assign_category(r["logprob_degradation"], r["fact_degradation"]), axis=1
    )

    # -----------------------------------------------------------------------
    # Per passage_num metrics
    # -----------------------------------------------------------------------
    rows = []
    for pn in sorted(sweep["passage_num"].dropna().unique()):
        sub = sweep[sweep["passage_num"] == pn]
        n   = len(sub)

        logprob_imp   = (sub["logprob_degradation"] < args.threshold).sum()
        fact_useful   = (sub["fact_degradation"] > 0).sum()
        conf_only     = ((sub["logprob_degradation"] < args.threshold) & (sub["fact_degradation"] == 0)).sum()
        helpful_imp   = ((sub["logprob_degradation"] < args.threshold) & (sub["fact_degradation"] > 0)).sum()
        harmful_inf   = ((sub["logprob_degradation"] < args.threshold) & (sub["fact_degradation"] < 0)).sum()

        # Baseline accuracy = fraction of queries with baseline_fact_score > 0
        query_df = sub.drop_duplicates(subset=["query_id"])
        baseline_acc = (query_df["baseline_fact_score"] > 0).mean()

        cdr = conf_only / logprob_imp if logprob_imp > 0 else float("nan")

        rows.append({
            "passage_num":          int(pn),
            "n_ablations":          n,
            "n_queries":            sub["query_id"].nunique(),
            "logprob_important_n":  int(logprob_imp),
            "logprob_important_rate": logprob_imp / n,
            "fact_useful_n":        int(fact_useful),
            "fact_useful_rate":     fact_useful / n,
            "helpful_important_rate": helpful_imp / n,
            "confidence_only_rate": conf_only / n,
            "harmful_influential_rate": harmful_inf / n,
            "CDR":                  cdr,
            "gap":                  (logprob_imp - fact_useful) / n,
            "baseline_accuracy":    round(baseline_acc, 3),
            "mean_layer32_drift":   sub["layer_32_drift"].mean(),
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out / "results" / "E5_passage_gap_metrics.csv", index=False)
    print("Saved results/E5_passage_gap_metrics.csv")

    # -----------------------------------------------------------------------
    # LaTeX table
    # -----------------------------------------------------------------------
    tex_rows = []
    for _, r in result_df.iterrows():
        tex_rows.append(
            f"  {int(r['passage_num'])} & {int(r['n_ablations'])} & "
            f"{r['baseline_accuracy']*100:.0f}\\% & "
            f"{r['logprob_important_rate']*100:.1f}\\% & "
            f"{r['fact_useful_rate']*100:.1f}\\% & "
            f"{r['gap']*100:.1f}\\% & "
            f"{r['CDR']*100:.1f}\\% & "
            f"{r['mean_layer32_drift']:.2f} \\\\"
        )

    tex = (
        r"\begin{table}[h]" + "\n"
        r"\centering\small" + "\n"
        r"\begin{tabular}{rrrrrrrr}" + "\n"
        r"\toprule" + "\n"
        r"$n$ & ablations & base acc & logprob-imp & fact-useful & gap & CDR & mean $\Delta_{L2}^{32}$ \\" + "\n"
        r"\midrule" + "\n"
        + "\n".join(tex_rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\caption{Influence-usefulness gap by passage count. "
        r"Gap $=$ logprob-important rate $-$ fact-useful rate. "
        r"CDR $=$ fraction of logprob-important documents with zero factual degradation.}" + "\n"
        r"\label{tab:e5_passage_gap}" + "\n"
        r"\end{table}" + "\n"
    )
    (out / "tables" / "E5_passage_gap_metrics.tex").write_text(tex)
    print("Saved tables/E5_passage_gap_metrics.tex")

    # -----------------------------------------------------------------------
    # Plot 1: Influence vs usefulness gap by passage_num
    # -----------------------------------------------------------------------
    pns = result_df["passage_num"].values
    x   = np.arange(len(pns))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, result_df["logprob_important_rate"], "o-", color="#e07b54",
            linewidth=2, label="Logprob-important rate", markersize=7)
    ax.plot(x, result_df["fact_useful_rate"], "s-", color="#5b9bd5",
            linewidth=2, label="Fact-useful rate", markersize=7)
    ax.plot(x, result_df["helpful_important_rate"], "^--", color="#70ad47",
            linewidth=1.5, label="Helpful-important rate", markersize=6, alpha=0.8)
    ax.fill_between(x,
                    result_df["fact_useful_rate"],
                    result_df["logprob_important_rate"],
                    alpha=0.12, color="#e07b54", label="Influence-usefulness gap")

    ax2 = ax.twinx()
    ax2.plot(x, result_df["baseline_accuracy"], "D:", color="black",
             linewidth=1.2, label="Baseline accuracy", markersize=5, alpha=0.6)
    ax2.set_ylabel("Baseline accuracy")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.set_ylim(0, 0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={int(p)}" for p in pns])
    ax.set_ylabel("Rate (fraction of ablations)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Influence-Usefulness Gap by Passage Count")
    ax.grid(alpha=0.2)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "plots" / "E5_influence_usefulness_gap_by_passage.png", dpi=150)
    plt.close(fig)
    print("Saved plots/E5_influence_usefulness_gap_by_passage.png")

    # -----------------------------------------------------------------------
    # Plot 2: CDR vs baseline accuracy scatter
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(result_df["baseline_accuracy"],
                    result_df["CDR"],
                    c=result_df["passage_num"],
                    cmap="RdYlBu_r", s=120, zorder=5)
    for _, r in result_df.iterrows():
        ax.annotate(f"n={int(r['passage_num'])}",
                    (r["baseline_accuracy"], r["CDR"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    plt.colorbar(sc, ax=ax, label="passage_num")
    ax.set_xlabel("Baseline accuracy (fraction of queries)")
    ax.set_ylabel("CDR (collapse divergence rate)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("CDR vs Baseline Accuracy by Passage Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "plots" / "E5_cdr_vs_baseline_accuracy.png", dpi=150)
    plt.close(fig)
    print("Saved plots/E5_cdr_vs_baseline_accuracy.png")

    print("\n=== E5 Summary ===")
    print(result_df[["passage_num", "logprob_important_rate", "fact_useful_rate",
                      "gap", "CDR", "baseline_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
