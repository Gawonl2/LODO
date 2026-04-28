"""
Position-Controlled LODO Experiment

Addresses reviewer concern: in standard LODO, removing document i shifts all
subsequent document positions, inflating L2 drift via positional encoding
artifacts rather than semantic absence.

Fix: instead of removing doc i, replace it with a neutral placeholder of
similar length. This keeps document count and positions identical, isolating
semantic absence from positional re-encoding.

Placeholder strategy: "[DOCUMENT REMOVED]" padded to approximate token count
of the removed document to minimize attention-pattern changes due to length.
"""
import os
import json
import argparse
import tqdm
import yaml
import numpy as np
from evalue import checkanswer
from models.models import Llama3

PLACEHOLDER = "[DOCUMENT REMOVED]"


def pad_placeholder(original_doc: str) -> str:
    """Return a placeholder roughly matching the word count of the original doc."""
    word_count = len(original_doc.split())
    filler = " ".join([PLACEHOLDER] * max(1, word_count // 3))
    return filler


def compute_l2_drift(states1: dict, states2: dict) -> dict:
    drift = {}
    for layer, s1 in states1.items():
        if layer in states2:
            drift[layer] = float(np.linalg.norm(np.array(s1) - np.array(s2)))
    return drift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   type=str,   default='en_counter_mid')
    parser.add_argument('--modelname', type=str,   default='llama3')
    parser.add_argument('--temp',      type=float, default=0.7)
    parser.add_argument('--max_queries', type=int, default=10)
    args = parser.parse_args()

    instances = []
    with open(f'data/{args.dataset}.json', 'r', encoding='utf-8') as f:
        for line in f:
            instances.append(json.loads(line))

    lang_key = args.dataset[:2]
    prompt_config = yaml.load(
        open('config/instruction.yaml', 'r', encoding='utf-8'),
        Loader=yaml.FullLoader
    )[lang_key]
    system_prompt = prompt_config['system']
    instruction   = prompt_config['instruction']

    if args.modelname == 'llama3':
        model = Llama3()
    else:
        print("Only llama3 supported for LODO mechanistic evaluation.")
        return

    results = []
    output_file = f'lodo_position_controlled_{args.dataset}_{args.modelname}.json'

    outer_bar = tqdm.tqdm(instances[:args.max_queries], desc="Total Queries")
    for i, instance in enumerate(outer_bar):
        query           = instance['query']
        ans_ground_truth = instance['answer']
        docs = instance.get('positive', []) + instance.get('negative', [])

        if docs and isinstance(docs[0], list):
            docs = [d[0] for d in docs]

        docs_text_full = '\n'.join(docs)
        text_full      = instruction.format(QUERY=query, DOCS=docs_text_full)

        baseline_answer      = model.generate(text_full, temperature=args.temp, system=system_prompt)
        baseline_fact_labels = checkanswer(baseline_answer, ans_ground_truth)
        baseline_fact_score  = 1 if 1 in baseline_fact_labels and 0 not in baseline_fact_labels else 0
        baseline_features    = model.get_logprob_and_states(text_full, baseline_answer, system=system_prompt)

        # --- Standard LODO (removal) ---
        standard_results = []
        # --- Position-Controlled LODO (placeholder) ---
        controlled_results = []

        inner_bar = tqdm.tqdm(docs, desc=f"Query {i+1} Ablations", leave=False)
        for doc_idx, doc in enumerate(inner_bar):
            # Standard: remove doc
            ablated_docs_standard = docs[:doc_idx] + docs[doc_idx+1:]
            text_ablated_standard = instruction.format(
                QUERY=query,
                DOCS='\n'.join(ablated_docs_standard)
            )

            # Position-controlled: replace with placeholder
            placeholder_doc = pad_placeholder(doc)
            ablated_docs_controlled = docs[:doc_idx] + [placeholder_doc] + docs[doc_idx+1:]
            text_ablated_controlled = instruction.format(
                QUERY=query,
                DOCS='\n'.join(ablated_docs_controlled)
            )

            # --- Standard ablation metrics ---
            std_answer       = model.generate(text_ablated_standard, temperature=args.temp, system=system_prompt)
            std_fact_labels  = checkanswer(std_answer, ans_ground_truth)
            std_fact_score   = 1 if 1 in std_fact_labels and 0 not in std_fact_labels else 0
            std_features     = model.get_logprob_and_states(text_ablated_standard, baseline_answer, system=system_prompt)

            std_logprob_deg  = std_features['logprob'] - baseline_features['logprob']
            std_fact_deg     = baseline_fact_score - std_fact_score
            std_drift        = compute_l2_drift(baseline_features['hidden_states_mean'], std_features['hidden_states_mean'])

            standard_results.append({
                "removed_doc_index": doc_idx,
                "method": "standard_removal",
                "logprob_degradation": std_logprob_deg,
                "fact_degradation":    std_fact_deg,
                "representation_drift_l2": std_drift,
                "is_causally_important": bool(std_fact_deg > 0 or std_logprob_deg < -2.0)
            })

            # --- Position-controlled ablation metrics ---
            ctrl_answer      = model.generate(text_ablated_controlled, temperature=args.temp, system=system_prompt)
            ctrl_fact_labels = checkanswer(ctrl_answer, ans_ground_truth)
            ctrl_fact_score  = 1 if 1 in ctrl_fact_labels and 0 not in ctrl_fact_labels else 0
            ctrl_features    = model.get_logprob_and_states(text_ablated_controlled, baseline_answer, system=system_prompt)

            ctrl_logprob_deg = ctrl_features['logprob'] - baseline_features['logprob']
            ctrl_fact_deg    = baseline_fact_score - ctrl_fact_score
            ctrl_drift       = compute_l2_drift(baseline_features['hidden_states_mean'], ctrl_features['hidden_states_mean'])

            controlled_results.append({
                "removed_doc_index": doc_idx,
                "method": "position_controlled_placeholder",
                "logprob_degradation": ctrl_logprob_deg,
                "fact_degradation":    ctrl_fact_deg,
                "representation_drift_l2": ctrl_drift,
                "is_causally_important": bool(ctrl_fact_deg > 0 or ctrl_logprob_deg < -2.0),
                # Residual = standard drift minus controlled drift = positional encoding artifact
                "positional_artifact_drift_l2": {
                    k: std_drift.get(k, 0) - ctrl_drift.get(k, 0)
                    for k in ctrl_drift
                }
            })

        results.append({
            "id":                  instance['id'],
            "query":               query,
            "baseline_fact_score": baseline_fact_score,
            "baseline_logprob":    baseline_features['logprob'],
            "standard_lodo":       standard_results,
            "controlled_lodo":     controlled_results,
        })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Results saved to {output_file}")


if __name__ == "__main__":
    main()
