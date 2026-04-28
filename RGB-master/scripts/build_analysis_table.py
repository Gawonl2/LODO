"""
Build a unified flat analysis table from all existing LODO JSON results.

Outputs:
  <output_dir>/analysis_table.csv  — one row per ablation
  <output_dir>/data_summary.txt    — record counts by source / passage_num
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_sweep(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)

    rows = []
    for item in data:
        qid   = item["query_id"]
        pnum  = item["passage_num"]
        bfact = item["baseline_fact_score"]
        blogp = item["baseline_logprob"]
        for abl in item["ablations"]:
            drift = abl["representation_drift_l2"]
            rows.append({
                "source":           "sweep",
                "query_id":         qid,
                "passage_num":      pnum,
                "doc_idx":          abl["doc_idx"],
                "is_positive_doc":  bool(abl["is_positive_doc"]),
                "doc_type":         "positive" if abl["is_positive_doc"] else "negative",
                "logprob_degradation":  abl["logprob_degradation"],
                "fact_degradation":     abl["fact_degradation"],
                "layer_0_drift":    drift.get("layer_0",  float("nan")),
                "layer_16_drift":   drift.get("layer_16", float("nan")),
                "layer_32_drift":   drift.get("layer_32", float("nan")),
                "is_causally_important": bool(abl["is_causally_important"]),
                "baseline_fact_score": bfact,
                "baseline_correct":    bfact > 0,
                "baseline_logprob":    blogp,
            })
    return rows


def load_en_refine(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)

    rows = []
    for item in data:
        qid   = item["id"]
        bfact = item["baseline_fact_score"]
        blogp = item["baseline_logprob"]
        for abl in item["lodo_results"]:
            drift = abl["representation_drift_l2"]
            rows.append({
                "source":           "en_refine",
                "query_id":         qid,
                "passage_num":      None,      # not available
                "doc_idx":          abl["removed_doc_index"],
                "is_positive_doc":  None,
                "doc_type":         "unknown",
                "logprob_degradation":  abl["logprob_degradation"],
                "fact_degradation":     abl["fact_degradation"],
                "layer_0_drift":    drift.get("layer_0",  float("nan")),
                "layer_16_drift":   drift.get("layer_16", float("nan")),
                "layer_32_drift":   drift.get("layer_32", float("nan")),
                "is_causally_important": bool(abl["is_causally_important"]),
                "baseline_fact_score": bfact,
                "baseline_correct":    bfact > 0,
                "baseline_logprob":    blogp,
            })
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep",      default="lodo_passage_sweep_en_counter_mid_llama3.json")
    p.add_argument("--en_refine",  default="lodo_results_en_refine_llama3.json")
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    base = Path(__file__).resolve().parent.parent   # RGB-master/
    os.chdir(base)

    rows = []
    if Path(args.sweep).exists():
        rows += load_sweep(args.sweep)
        print(f"Loaded sweep:     {len([r for r in rows if r['source']=='sweep'])} ablations")
    else:
        print(f"WARNING: sweep file not found: {args.sweep}", file=sys.stderr)

    if Path(args.en_refine).exists():
        refine_rows = load_en_refine(args.en_refine)
        rows += refine_rows
        print(f"Loaded en_refine: {len(refine_rows)} ablations")
    else:
        print(f"WARNING: en_refine file not found: {args.en_refine}", file=sys.stderr)

    if not rows:
        print("ERROR: no data loaded", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    out_dir = Path(args.output_dir) if args.output_dir else Path("results") / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "analysis_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}  ({len(df)} rows)")

    # summary
    summary_lines = [
        f"Total ablations: {len(df)}",
        "",
        "By source:",
    ]
    for src, grp in df.groupby("source"):
        summary_lines.append(f"  {src}: {len(grp)}")
    summary_lines += ["", "By passage_num (sweep only):"]
    sweep_df = df[df["source"] == "sweep"]
    for pn, grp in sweep_df.groupby("passage_num"):
        summary_lines.append(f"  n={pn}: {len(grp)} ablations")
    summary_lines += ["", "logprob < -2.0:", f"  {(df['logprob_degradation'] < -2.0).sum()}"]
    summary_lines += ["fact_degradation > 0:", f"  {(df['fact_degradation'] > 0).sum()}"]
    summary_lines += ["fact_degradation < 0:", f"  {(df['fact_degradation'] < 0).sum()}"]

    summary_path = out_dir / "data_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"Saved: {summary_path}")
    print("\n" + "\n".join(summary_lines))

    return str(csv_path)


if __name__ == "__main__":
    main()
