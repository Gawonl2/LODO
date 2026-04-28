"""
LODO Passage-Number Sweep Experiment

Runs LODO on en_counter_mid (or en_mid) with varying numbers of retrieved
documents (passage_num in [3, 5, 7, 10]) to measure how document count affects:
  - Fraction of causally important documents
  - Logprob degradation magnitude
  - Layer-wise representation drift

Hypothesis: with fewer documents each doc carries more weight → higher causal
importance rate. With more documents, redundancy increases → importance is sparser.
"""
import os
import json
import argparse
import random
import tqdm
import yaml
import numpy as np
from evalue import checkanswer
from models.models import Llama3


def compute_l2_drift(states1: dict, states2: dict) -> dict:
    return {
        layer: float(np.linalg.norm(np.array(s1) - np.array(states2[layer])))
        for layer, s1 in states1.items()
        if layer in states2
    }


def run_lodo_for_passage_num(model, instance, passage_num, system_prompt,
                              instruction, temp, seed=42):
    """Run full LODO for one query at a fixed passage_num. Returns per-ablation results."""
    rng = random.Random(seed)
    query            = instance['query']
    ans_ground_truth = instance['answer']

    pos_docs = instance.get('positive', [])
    neg_docs = instance.get('negative', [])
    if pos_docs and isinstance(pos_docs[0], list):
        pos_docs = [d[0] for d in pos_docs]
    if neg_docs and isinstance(neg_docs[0], list):
        neg_docs = [d[0] for d in neg_docs]

    # Always include at least 1 positive doc; fill the rest with negatives.
    # Mirrors the standard RGB sampling strategy.
    n_pos = max(1, passage_num // 2)
    n_neg = passage_num - n_pos

    sampled_pos = rng.sample(pos_docs, min(n_pos, len(pos_docs)))
    sampled_neg = rng.sample(neg_docs, min(n_neg, len(neg_docs)))
    docs = sampled_pos + sampled_neg
    rng.shuffle(docs)

    # Baseline
    docs_text  = '\n'.join(docs)
    text_full  = instruction.format(QUERY=query, DOCS=docs_text)
    baseline_answer     = model.generate(text_full, temperature=temp, system=system_prompt)
    baseline_labels     = checkanswer(baseline_answer, ans_ground_truth)
    baseline_fact_score = 1 if 1 in baseline_labels and 0 not in baseline_labels else 0
    baseline_features   = model.get_logprob_and_states(text_full, baseline_answer,
                                                        system=system_prompt)

    ablation_results = []
    for doc_idx, doc in enumerate(docs):
        ablated_docs  = docs[:doc_idx] + docs[doc_idx+1:]
        text_ablated  = instruction.format(QUERY=query, DOCS='\n'.join(ablated_docs))

        abl_answer      = model.generate(text_ablated, temperature=temp, system=system_prompt)
        abl_labels      = checkanswer(abl_answer, ans_ground_truth)
        abl_fact_score  = 1 if 1 in abl_labels and 0 not in abl_labels else 0
        abl_features    = model.get_logprob_and_states(text_ablated, baseline_answer,
                                                        system=system_prompt)

        logprob_deg = abl_features['logprob'] - baseline_features['logprob']
        fact_deg    = baseline_fact_score - abl_fact_score
        drift       = compute_l2_drift(baseline_features['hidden_states_mean'],
                                       abl_features['hidden_states_mean'])

        ablation_results.append({
            "doc_idx":              doc_idx,
            "is_positive_doc":      doc in sampled_pos,
            "logprob_degradation":  logprob_deg,
            "fact_degradation":     fact_deg,
            "representation_drift_l2": drift,
            "is_causally_important": bool(fact_deg > 0 or logprob_deg < -2.0),
        })

    return {
        "query_id":            instance['id'],
        "passage_num":         passage_num,
        "n_pos_sampled":       len(sampled_pos),
        "n_neg_sampled":       len(sampled_neg),
        "baseline_fact_score": baseline_fact_score,
        "baseline_logprob":    baseline_features['logprob'],
        "ablations":           ablation_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',     type=str,   default='en_counter_mid')
    parser.add_argument('--modelname',   type=str,   default='llama3')
    parser.add_argument('--temp',        type=float, default=0.7)
    parser.add_argument('--max_queries', type=int,   default=10)
    parser.add_argument('--passage_nums', type=int,  nargs='+', default=[3, 5, 7, 10])
    parser.add_argument('--seed',        type=int,   default=42)
    args = parser.parse_args()

    instances = []
    with open(f'data/{args.dataset}.json', 'r', encoding='utf-8') as f:
        for line in f:
            instances.append(json.loads(line))
    instances = instances[:args.max_queries]

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
        print("Only llama3 supported.")
        return

    all_results = []
    output_file = f'lodo_passage_sweep_{args.dataset}_{args.modelname}.json'

    for passage_num in args.passage_nums:
        print(f"\n{'='*50}")
        print(f"Running LODO with passage_num={passage_num}")
        print(f"{'='*50}")
        bar = tqdm.tqdm(instances, desc=f"passage_num={passage_num}")
        for instance in bar:
            result = run_lodo_for_passage_num(
                model, instance, passage_num,
                system_prompt, instruction, args.temp, args.seed
            )
            all_results.append(result)
            # Save incrementally
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Results saved to {output_file}")
    print_summary(all_results, args.passage_nums)


def print_summary(results, passage_nums):
    print("\n=== SUMMARY ===")
    print(f"{'passage_num':>12} {'queries':>8} {'total_abl':>10} {'causal%':>8} "
          f"{'mean_logdeg':>12} {'mean_drift32':>13}")
    for pn in passage_nums:
        subset = [r for r in results if r['passage_num'] == pn]
        ablations = [a for r in subset for a in r['ablations']]
        if not ablations:
            continue
        causal_pct = 100 * sum(a['is_causally_important'] for a in ablations) / len(ablations)
        mean_logdeg = np.mean([a['logprob_degradation'] for a in ablations])
        drifts = [a['representation_drift_l2'].get('layer_32',
                  a['representation_drift_l2'].get('layer_31', 0)) for a in ablations]
        mean_drift = np.mean(drifts)
        print(f"{pn:>12} {len(subset):>8} {len(ablations):>10} {causal_pct:>8.2f} "
              f"{mean_logdeg:>12.4f} {mean_drift:>13.4f}")


if __name__ == "__main__":
    main()
