"""
Experiment 3: Whole-Sequence vs Fact-Token Logprob Drop

Uses token-level data from detailed_case_studies.json (10 cases).
Factual tokens are identified via regex heuristics (no spaCy required):
  - tokens containing digits
  - tokens that start with a capital letter (after stripping whitespace)
  - tokens overlapping with ground-truth answer keywords if available

Computes per-case:
  - whole_seq_delta_logprob  (sum of all token deltas)
  - fact_token_delta_logprob (sum over factual tokens only)
  - nonfact_token_delta_logprob
  - fraction of total drop attributable to factual tokens

Cross-references with en_refine LODO results to get fact_degradation per case.

Outputs
-------
results/E3_fact_token_logprob.csv
tables/E3_fact_token_prediction_metrics.tex
plots/E3_fact_token_vs_whole_sequence_auroc.png
plots/E3_token_type_logprob_drop_boxplot.png

NOTE: n=10 cases only — not enough for AUROC. Reports Spearman correlation
and per-token-type distribution; flags the limitation explicitly.
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Token classification heuristics (no spaCy)
# ---------------------------------------------------------------------------

_DIGIT_RE   = re.compile(r"\d")
_CAPITAL_RE = re.compile(r"^[A-Z]")

def is_factual_token(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return False
    # numbers, dates, ordinals
    if _DIGIT_RE.search(t):
        return True
    # capitalized words (proper nouns, named entities)
    if _CAPITAL_RE.match(t) and len(t) > 1:
        return True
    return False


def compute_token_deltas(tokens, baseline_lps, ablated_lps):
    assert len(tokens) == len(baseline_lps) == len(ablated_lps)
    deltas = [a - b for b, a in zip(baseline_lps, ablated_lps)]
    fact_idx    = [i for i, t in enumerate(tokens) if is_factual_token(t)]
    nonfact_idx = [i for i in range(len(tokens)) if i not in set(fact_idx)]
    return {
        "whole_seq_delta":    sum(deltas),
        "fact_delta":         sum(deltas[i] for i in fact_idx)  if fact_idx    else float("nan"),
        "nonfact_delta":      sum(deltas[i] for i in nonfact_idx) if nonfact_idx else float("nan"),
        "n_fact_tokens":      len(fact_idx),
        "n_nonfact_tokens":   len(nonfact_idx),
        "n_total_tokens":     len(tokens),
        "fact_token_names":   [tokens[i] for i in fact_idx],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case_studies",  default="detailed_case_studies.json")
    p.add_argument("--en_refine",     default="lodo_results_en_refine_llama3.json")
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--threshold",     type=float, default=-2.0)
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent.parent
    out  = Path(args.output_dir)
    for d in ["results", "tables", "plots"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    case_path = base / args.case_studies
    refine_path = base / args.en_refine

    if not case_path.exists():
        print(f"TODO: detailed_case_studies.json not found at {case_path}")
        print("Experiment 3 requires token-level data. Skipping.")
        (out / "results" / "E3_fact_token_logprob.csv").write_text("# TODO: requires detailed_case_studies.json\n")
        return

    with open(case_path) as f:
        cases = json.load(f)

    # Load en_refine for fact_degradation cross-reference
    fact_deg_lookup = {}   # (query_id, removed_doc_index) -> fact_degradation
    if refine_path.exists():
        with open(refine_path) as f:
            refine = json.load(f)
        for i, item in enumerate(refine):
            qid = item["id"]
            for abl in item["lodo_results"]:
                fact_deg_lookup[(qid, abl["removed_doc_index"])] = abl["fact_degradation"]

    rows = []
    token_rows = []   # per-token rows for box plot

    for case in cases:
        qid   = case["query_id"]
        didx  = case["removed_doc_index"]
        group = case["group"]
        tokens   = case["tokens"]
        bl_lps   = case["baseline_token_logprobs"]
        abl_lps  = case["ablated_token_logprobs"]
        orig_deg = case["original_logprob_deg"]

        td = compute_token_deltas(tokens, bl_lps, abl_lps)

        # fact_degradation from cross-reference
        fact_deg = fact_deg_lookup.get((qid, didx), float("nan"))

        rows.append({
            "query_id":       qid,
            "removed_doc_idx": didx,
            "group":          group,
            "fact_degradation": fact_deg,
            "original_logprob_deg": orig_deg,
            **{k: v for k, v in td.items() if k != "fact_token_names"},
            "fact_tokens": str(td["fact_token_names"]),
        })

        # Per-token rows for distribution plot
        deltas_bl  = np.array(bl_lps)
        deltas_abl = np.array(abl_lps)
        for i, (tok, bl, ab) in enumerate(zip(tokens, bl_lps, abl_lps)):
            token_rows.append({
                "query_id": qid,
                "group": group,
                "token": tok,
                "token_type": "factual" if is_factual_token(tok) else "structural",
                "baseline_logprob": bl,
                "ablated_logprob": ab,
                "delta_logprob": ab - bl,
            })

    result_df = pd.DataFrame(rows)
    token_df  = pd.DataFrame(token_rows)

    result_df.to_csv(out / "results" / "E3_fact_token_logprob.csv", index=False)
    print("Saved results/E3_fact_token_logprob.csv")

    # -----------------------------------------------------------------------
    # Correlation table
    # -----------------------------------------------------------------------
    valid = result_df.dropna(subset=["fact_degradation", "whole_seq_delta", "fact_delta"])
    n_cases = len(valid)

    metrics = {}
    for col, label in [("whole_seq_delta", "Whole-sequence"), ("fact_delta", "Fact-token"), ("nonfact_delta", "Non-fact-token")]:
        sub = valid.dropna(subset=[col])
        if len(sub) < 3:
            metrics[label] = {"spearman_r": float("nan"), "spearman_p": float("nan"), "n": len(sub)}
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, p = spearmanr(sub[col], sub["fact_degradation"])
        metrics[label] = {"spearman_r": round(r, 3), "spearman_p": round(p, 3), "n": len(sub)}

    tex_rows = []
    for label, m in metrics.items():
        r_str = f"{m['spearman_r']:.3f}" if not np.isnan(m['spearman_r']) else r"\textemdash"
        p_str = f"{m['spearman_p']:.3f}" if not np.isnan(m['spearman_p']) else r"\textemdash"
        tex_rows.append(f"  {label} & {m['n']} & {r_str} & {p_str} \\\\")

    tex = (
        r"\begin{table}[h]" + "\n"
        r"\centering\small" + "\n"
        r"\begin{tabular}{lrrr}" + "\n"
        r"\toprule" + "\n"
        r"Predictor & $n$ & Spearman $r$ & $p$ \\" + "\n"
        r"\midrule" + "\n"
        + "\n".join(tex_rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        rf"\caption{{Spearman correlation with $\Delta_{{\text{{fact}}}}$ for whole-sequence vs "
        r"fact-token logprob degradation ($n=10$ case studies). "
        r"Factual tokens identified via heuristics: digits, dates, capitalized proper-noun tokens.}}" + "\n"
        r"\label{tab:e3_fact_token_metrics}" + "\n"
        r"\end{table}" + "\n"
        r"% NOTE: n=10 cases — results are indicative, not statistically conclusive." + "\n"
    )
    (out / "tables" / "E3_fact_token_prediction_metrics.tex").write_text(tex)
    print("Saved tables/E3_fact_token_prediction_metrics.tex")

    # -----------------------------------------------------------------------
    # Plot 1: Scatter — fact_delta vs whole_seq_delta, colored by group
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"Top Divergence": "#e07b54", "Average Baseline": "#5b9bd5"}

    for ax, (xcol, xlabel) in zip(axes, [
        ("whole_seq_delta", "Whole-sequence $\\Delta_{logprob}$"),
        ("fact_delta",      "Fact-token $\\Delta_{logprob}$"),
    ]):
        for grp, sub in result_df.groupby("group"):
            ax.scatter(sub[xcol], sub["original_logprob_deg"],
                       color=colors.get(grp, "gray"), label=grp, s=60, alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("original logprob degradation")
        ax.set_title(f"{xlabel}\nvs original logprob deg")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle("Fact-Token vs Whole-Sequence Logprob Drop (n=10 case studies)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "plots" / "E3_fact_token_vs_whole_sequence_auroc.png", dpi=150)
    plt.close(fig)
    print("Saved plots/E3_fact_token_vs_whole_sequence_auroc.png")

    # -----------------------------------------------------------------------
    # Plot 2: Boxplot of per-token delta by type and group
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, grp in zip(axes, ["Top Divergence", "Average Baseline"]):
        sub = token_df[token_df["group"] == grp]
        data = [sub[sub["token_type"] == "factual"]["delta_logprob"].values,
                sub[sub["token_type"] == "structural"]["delta_logprob"].values]
        bp = ax.boxplot(data, tick_labels=["Factual tokens", "Structural tokens"],
                        patch_artist=True, widths=0.5, medianprops=dict(color="black"))
        bp["boxes"][0].set_facecolor("#e07b54")
        bp["boxes"][1].set_facecolor("#5b9bd5")
        ax.set_title(f"Token logprob delta\n({grp})")
        ax.set_ylabel("$\\Delta_{logprob}$ (ablated - baseline)")
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Per-Token Logprob Change by Token Type", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "plots" / "E3_token_type_logprob_drop_boxplot.png", dpi=150)
    plt.close(fig)
    print("Saved plots/E3_token_type_logprob_drop_boxplot.png")

    # Summary
    print("\n=== E3 Summary ===")
    print(f"Cases analyzed: {len(result_df)}")
    print(f"Factual token fraction: "
          f"{result_df['n_fact_tokens'].sum() / result_df['n_total_tokens'].sum():.2%}")
    for label, m in metrics.items():
        print(f"  {label}: Spearman r={m['spearman_r']}, p={m['spearman_p']}")
    print("NOTE: n=10 is indicative only; see limitation note in table caption.")


if __name__ == "__main__":
    main()
