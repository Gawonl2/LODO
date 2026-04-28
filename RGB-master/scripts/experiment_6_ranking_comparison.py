"""
Experiment 6: Factuality-Aware Ranking vs Logprob-Only Ranking

Compares four document ranking scores per query:
  A) Logprob-only:           max(0, -delta_logprob)
  B) Fact-token logprob:     max(0, -fact_token_delta_logprob)  [if available]
  C) Factuality-aware causal: max(0,-delta_logprob) * max(0, delta_fact)
  D) Harmfulness-aware:       max(0,-delta_logprob) * max(0,-delta_fact)

Target for retrieval:  delta_fact > 0  (helpful-important)
Secondary target:      delta_fact < 0  (harmful)

Metrics: Precision@1, Precision@3, Recall@1, Recall@3, MRR, Spearman r

Reports:
  1) Across all queries
  2) Only among queries with at least one delta_fact > 0 (non-trivial queries)

Outputs
-------
results/E6_ranking_comparison.csv
tables/E6_ranking_metrics.tex
plots/E6_precision_by_ranking_method.png
plots/E6_score_vs_delta_fact_scatter.png
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# scoring functions
# ---------------------------------------------------------------------------

def score_logprob_only(lp):
    return max(0.0, -lp)

def score_fact_aware(lp, fd):
    return max(0.0, -lp) * max(0.0, float(fd))

def score_harm_aware(lp, fd):
    return max(0.0, -lp) * max(0.0, -float(fd))


# ---------------------------------------------------------------------------
# retrieval metrics
# ---------------------------------------------------------------------------

def precision_at_k(ranked_targets, k):
    top = list(ranked_targets[:k])
    return sum(top) / k if len(top) > 0 else 0.0

def recall_at_k(ranked_targets, total_pos, k):
    if total_pos == 0:
        return float("nan")
    return sum(list(ranked_targets[:k])) / total_pos

def mrr(ranked_targets):
    for i, t in enumerate(ranked_targets, 1):
        if int(t) > 0:
            return 1.0 / i
    return 0.0


def evaluate_ranking(query_df: pd.DataFrame, score_col: str, target_col: str, ks=(1, 3)):
    """Evaluate ranking within a single query."""
    if len(query_df) == 0:
        return {f"P@{k}": float("nan") for k in ks} | {f"R@{k}": float("nan") for k in ks} | {"MRR": float("nan")}

    sorted_df = query_df.sort_values(score_col, ascending=False)
    targets   = sorted_df[target_col].values
    total_pos = targets.sum()

    result = {}
    for k in ks:
        result[f"P@{k}"] = precision_at_k(targets, k)
        result[f"R@{k}"] = recall_at_k(targets, total_pos, k)
    result["MRR"] = mrr(targets)
    return result


def aggregate_metrics(per_query: list[dict], ks=(1, 3)) -> dict:
    keys = [f"P@{k}" for k in ks] + [f"R@{k}" for k in ks] + ["MRR"]
    agg  = {}
    for k in keys:
        vals = [r[k] for r in per_query if not np.isnan(r[k])]
        agg[k] = np.mean(vals) if vals else float("nan")
    return agg


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       default=None)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--threshold",   type=float, default=-2.0)
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

    # Build scores
    df["score_logprob"]   = df["logprob_degradation"].apply(lambda lp: max(0.0, -lp))
    df["score_fact_aware"] = df.apply(
        lambda r: score_fact_aware(r["logprob_degradation"], r["fact_degradation"]), axis=1)
    df["score_harm_aware"] = df.apply(
        lambda r: score_harm_aware(r["logprob_degradation"], r["fact_degradation"]), axis=1)
    df["target_helpful"] = (df["fact_degradation"] > 0).astype(int)
    df["target_harmful"] = (df["fact_degradation"] < 0).astype(int)

    scoring_methods = {
        "Logprob-only":          "score_logprob",
        "Factuality-aware":      "score_fact_aware",
        "Harmfulness-aware":     "score_harm_aware",
    }

    # Use consistent query key: source + query_id + passage_num
    df["qkey"] = df["source"] + "_q" + df["query_id"].astype(str) + "_p" + df["passage_num"].astype(str)

    # Queries with at least one helpful target
    queries_with_helpful = set(df[df["target_helpful"] == 1]["qkey"].unique())
    print(f"Total queries: {df['qkey'].nunique()}")
    print(f"Queries with ≥1 helpful doc: {len(queries_with_helpful)}")

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    result_rows = []
    for method_name, score_col in scoring_methods.items():
        for target_name, target_col in [("helpful", "target_helpful"), ("harmful", "target_harmful")]:
            # All queries
            pq_all = []
            for qkey, qdf in df.groupby("qkey"):
                pq_all.append(evaluate_ranking(qdf, score_col, target_col))

            # Non-trivial queries only (at least one positive)
            pq_nontrivial = []
            for qkey, qdf in df.groupby("qkey"):
                if qdf[target_col].sum() > 0:
                    pq_nontrivial.append(evaluate_ranking(qdf, score_col, target_col))

            agg_all = aggregate_metrics(pq_all)
            agg_nt  = aggregate_metrics(pq_nontrivial)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r_all, _ = spearmanr(df[score_col], df["fact_degradation"])

            result_rows.append({
                "method": method_name,
                "target": target_name,
                "eval_set": "all_queries",
                "n_queries": len(pq_all),
                "n_queries_with_pos": len(pq_nontrivial),
                "spearman_r": round(r_all, 3),
                **{k: round(v, 3) for k, v in agg_all.items()},
            })
            result_rows.append({
                "method": method_name,
                "target": target_name,
                "eval_set": "nontrivial_queries",
                "n_queries": len(pq_nontrivial),
                "n_queries_with_pos": len(pq_nontrivial),
                "spearman_r": round(r_all, 3),
                **{k: round(v, 3) for k, v in agg_nt.items()},
            })

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(out / "results" / "E6_ranking_comparison.csv", index=False)
    print("Saved results/E6_ranking_comparison.csv")

    # -----------------------------------------------------------------------
    # LaTeX table — helpful target, all queries + nontrivial
    # -----------------------------------------------------------------------
    tex_rows = []
    for _, r in result_df[result_df["target"] == "helpful"].iterrows():
        tag = "all" if r["eval_set"] == "all_queries" else f"nontrivial ($n_q$={int(r['n_queries'])})"
        tex_rows.append(
            f"  {r['method']} & {tag} & "
            f"{r.get('P@1', float('nan')):.3f} & "
            f"{r.get('P@3', float('nan')):.3f} & "
            f"{r.get('R@1', float('nan')):.3f} & "
            f"{r.get('R@3', float('nan')):.3f} & "
            f"{r.get('MRR', float('nan')):.3f} & "
            f"{r.get('spearman_r', float('nan')):.3f} \\\\"
        )

    tex = (
        r"\begin{table}[h]" + "\n"
        r"\centering\small" + "\n"
        r"\begin{tabular}{llrrrrrr}" + "\n"
        r"\toprule" + "\n"
        r"Method & Eval set & P@1 & P@3 & R@1 & R@3 & MRR & Spearman $r$ \\" + "\n"
        r"\midrule" + "\n"
        + "\n".join(tex_rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\caption{Ranking evaluation for identifying factually useful documents "
        r"($\Delta_{\text{fact}} > 0$). ``All'' includes queries with no positive; "
        r"``nontrivial'' restricts to queries with $\geq 1$ factually useful document.}" + "\n"
        r"\label{tab:e6_ranking}" + "\n"
        r"\end{table}" + "\n"
    )
    (out / "tables" / "E6_ranking_metrics.tex").write_text(tex)
    print("Saved tables/E6_ranking_metrics.tex")

    # -----------------------------------------------------------------------
    # Plot 1: Precision@1 and Precision@3 by method (helpful, nontrivial)
    # -----------------------------------------------------------------------
    plot_df = result_df[
        (result_df["target"] == "helpful") &
        (result_df["eval_set"] == "nontrivial_queries")
    ]
    if not plot_df.empty:
        methods = plot_df["method"].values
        x = np.arange(len(methods))
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.3
        ax.bar(x - w/2, plot_df["P@1"].values, w, label="P@1", color="#5b9bd5", alpha=0.9)
        ax.bar(x + w/2, plot_df["P@3"].values, w, label="P@3", color="#e07b54", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=10, ha="right")
        ax.set_ylabel("Precision")
        ax.set_title("Precision@K for Identifying Factually Useful Documents\n(nontrivial queries only)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / "plots" / "E6_precision_by_ranking_method.png", dpi=150)
        plt.close(fig)
        print("Saved plots/E6_precision_by_ranking_method.png")

    # -----------------------------------------------------------------------
    # Plot 2: Score vs delta_fact scatter by method
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(scoring_methods), figsize=(14, 4))
    for ax, (name, col) in zip(axes, scoring_methods.items()):
        ax.scatter(df[col], df["fact_degradation"],
                   c=df["target_helpful"], cmap="RdYlBu_r",
                   alpha=0.5, s=20, edgecolors="none")
        ax.set_xlabel(f"Score ({name})")
        ax.set_ylabel(r"$\Delta_{\mathrm{fact}}$")
        ax.set_title(name)
        ax.grid(alpha=0.15)

    fig.suptitle("Ranking Score vs Factual Degradation", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "plots" / "E6_score_vs_delta_fact_scatter.png", dpi=150)
    plt.close(fig)
    print("Saved plots/E6_score_vs_delta_fact_scatter.png")

    print("\n=== E6 Summary ===")
    print(result_df[result_df["eval_set"] == "nontrivial_queries"]
          [["method", "target", "P@1", "P@3", "MRR", "spearman_r"]].to_string(index=False))


if __name__ == "__main__":
    main()
