"""
Experiment 7: Position-Controlled LODO

Checks whether position-controlled LODO results already exist.
If so, analyzes them. If not, prints instructions for rerunning.

Outputs (if data exists)
------------------------
results/E7_position_controlled_lodo.csv
tables/E7_position_controlled_comparison.tex
plots/E7_standard_vs_controlled_l2.png
plots/E7_integration_horizon_controlled.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       default="lodo_position_controlled_en_counter_mid_llama3.json",
                   help="Position-controlled LODO output file")
    p.add_argument("--output_dir",  required=True)
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent.parent
    out  = Path(args.output_dir)
    for d in ["results", "tables", "plots"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    data_path = base / args.input

    if not data_path.exists():
        msg = (
            "# Experiment 7: Position-Controlled LODO\n"
            "# STATUS: TODO — model inference required\n"
            "#\n"
            "# The position-controlled LODO output file was not found:\n"
            f"#   {data_path}\n"
            "#\n"
            "# To generate it, run (from RGB-master/ with the LODO conda env):\n"
            "#\n"
            "#   python run_lodo_position_controlled.py \\\n"
            "#       --dataset en_counter_mid \\\n"
            "#       --modelname llama3 \\\n"
            "#       --max_queries 10\n"
            "#\n"
            "# This requires a GPU and the Llama-3.1-8B-Instruct model loaded.\n"
            "# Estimated runtime: ~30-60 minutes on A100.\n"
            "#\n"
            "# After running, re-execute this script to generate the analysis.\n"
            "#\n"
            "# Expected output keys in each ablation:\n"
            "#   standard_lodo[i].representation_drift_l2\n"
            "#   controlled_lodo[i].representation_drift_l2\n"
            "#   controlled_lodo[i].positional_artifact_drift_l2\n"
            "#\n"
            "# Key question: does delta_PE = standard_L2 - controlled_L2 approach zero\n"
            "# at the integration horizon (layers 18-20)? If yes, the late-layer drift\n"
            "# is semantic, not a positional encoding artifact.\n"
        )
        stub = out / "results" / "E7_position_controlled_lodo.csv"
        stub.write_text(msg)
        print(f"TODO: {data_path} not found.")
        print(f"Wrote instruction stub to {stub}")
        print("Run run_lodo_position_controlled.py first, then re-run this script.")
        return

    # -----------------------------------------------------------------------
    # Data exists — analyze
    # -----------------------------------------------------------------------
    with open(data_path) as f:
        data = json.load(f)

    rows = []
    for item in data:
        qid = item["id"]
        std_list  = item.get("standard_lodo", [])
        ctrl_list = item.get("controlled_lodo", [])
        for std, ctrl in zip(std_list, ctrl_list):
            did = std["removed_doc_index"]
            std_drift  = std["representation_drift_l2"]
            ctrl_drift = ctrl["representation_drift_l2"]
            pe_drift   = ctrl.get("positional_artifact_drift_l2", {})
            for layer_key in std_drift:
                layer_n = int(layer_key.replace("layer_", ""))
                rows.append({
                    "query_id":      qid,
                    "doc_idx":       did,
                    "layer":         layer_n,
                    "standard_l2":   std_drift[layer_key],
                    "controlled_l2": ctrl_drift.get(layer_key, float("nan")),
                    "delta_pe":      pe_drift.get(layer_key, float("nan")),
                    "std_logprob_deg":  std["logprob_degradation"],
                    "std_fact_deg":     std["fact_degradation"],
                    "ctrl_logprob_deg": ctrl["logprob_degradation"],
                    "ctrl_fact_deg":    ctrl["fact_degradation"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(out / "results" / "E7_position_controlled_lodo.csv", index=False)
    print("Saved results/E7_position_controlled_lodo.csv")

    # Mean drift by layer
    layer_stats = df.groupby("layer").agg(
        std_mean=("standard_l2",   "mean"),
        ctrl_mean=("controlled_l2", "mean"),
        pe_mean=("delta_pe",        "mean"),
    ).reset_index()

    # LaTeX table
    tex_rows = []
    for _, r in layer_stats.iterrows():
        tex_rows.append(
            f"  {int(r['layer'])} & {r['std_mean']:.3f} & "
            f"{r['ctrl_mean']:.3f} & {r['pe_mean']:.3f} \\\\"
        )
    tex = (
        r"\begin{table}[h]" + "\n"
        r"\centering\small" + "\n"
        r"\begin{tabular}{rrrr}" + "\n"
        r"\toprule" + "\n"
        r"Layer & Standard $\Delta_{L2}$ & Controlled $\Delta_{L2}$ & $\Delta_{\text{PE}}$ \\" + "\n"
        r"\midrule" + "\n"
        + "\n".join(tex_rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\caption{Standard vs position-controlled LODO drift. "
        r"$\Delta_{\text{PE}}$ = positional encoding artifact = standard $-$ controlled.}" + "\n"
        r"\label{tab:e7_position_controlled}" + "\n"
        r"\end{table}" + "\n"
    )
    (out / "tables" / "E7_position_controlled_comparison.tex").write_text(tex)

    # Plot 1: Standard vs controlled by layer
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layer_stats["layer"], layer_stats["std_mean"], "o-", color="#e07b54",
            label="Standard LODO", linewidth=2)
    ax.plot(layer_stats["layer"], layer_stats["ctrl_mean"], "s-", color="#5b9bd5",
            label="Position-controlled LODO", linewidth=2)
    ax.fill_between(layer_stats["layer"],
                    layer_stats["ctrl_mean"],
                    layer_stats["std_mean"],
                    alpha=0.15, color="#e07b54", label="$\\Delta_{PE}$ artifact")
    ax.axvspan(18, 20, alpha=0.1, color="gray", label="Integration horizon")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 drift")
    ax.set_title("Standard vs Position-Controlled LODO Drift")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "plots" / "E7_standard_vs_controlled_l2.png", dpi=150)
    plt.close(fig)

    # Plot 2: PE artifact curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layer_stats["layer"], layer_stats["pe_mean"], "o-", color="#9e480e",
            label="$\\Delta_{PE}$ (positional artifact)", linewidth=2)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axvspan(18, 20, alpha=0.1, color="gray", label="Integration horizon")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean $\\Delta_{PE}$ drift")
    ax.set_title("Positional Encoding Artifact by Layer")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "plots" / "E7_integration_horizon_controlled.png", dpi=150)
    plt.close(fig)

    print("Saved E7 plots and tables.")


if __name__ == "__main__":
    main()
