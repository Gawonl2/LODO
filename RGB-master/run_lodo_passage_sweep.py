"""
LODO Passage-Number Sweep — Mixed Setting (Experiment 3)

For each query, constructs a context of:
  - 1 ground-truth document  (from en_mid positive)
  - (passage_num - 1) counter-factual documents  (from en_counter_mid positive)

This ensures the baseline is solvable (model can see the correct answer) while
the majority of the context is adversarially misleading. The design directly
tests whether logprob-only attribution can identify the single factually useful
document among n-1 misleading ones.

Each ablation result includes:
  is_gt_doc  — True if the removed document was the ground-truth document
  doc_type   — "ground-truth" | "counter-factual"
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


def run_lodo_for_passage_num(model, counter_instance, gt_instance, passage_num,
                              system_prompt, instruction, temp, seed=42):
    """Run full LODO for one query at a fixed passage_num. Returns per-ablation results.

    Context = 1 ground-truth doc (from en_mid) + (passage_num-1) counter-factual docs
    (from en_counter_mid positive). Each ablation records whether the removed doc
    was the ground-truth document.
    """
    rng = random.Random(seed)
    query            = counter_instance['query']
    ans_ground_truth = counter_instance['answer']

    # Ground-truth docs from en_mid positive field.
    gt_docs = gt_instance.get('positive', [])
    if gt_docs and isinstance(gt_docs[0], list):
        gt_docs = [d[0] for d in gt_docs]

    # Counter-factual docs from en_counter_mid positive field.
    counter_docs = counter_instance.get('positive', [])
    if counter_docs and isinstance(counter_docs[0], list):
        counter_docs = [d[0] for d in counter_docs]

    if not gt_docs:
        return None

    # 1 ground-truth doc + (passage_num - 1) counter-factual docs.
    gt_doc = rng.choice(gt_docs)
    n_counter = min(passage_num - 1, len(counter_docs))
    sampled_counter = rng.sample(counter_docs, n_counter)

    docs = [gt_doc] + sampled_counter
    gt_doc_text = gt_doc
    rng.shuffle(docs)

    n_docs = len(docs)

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

        is_gt = (doc == gt_doc_text)
        ablation_results.append({
            "doc_idx":              doc_idx,
            "is_gt_doc":           is_gt,
            "doc_type":            "ground-truth" if is_gt else "counter-factual",
            "logprob_degradation":  logprob_deg,
            "fact_degradation":     fact_deg,
            "representation_drift_l2": drift,
            "is_causally_important": bool(fact_deg != 0 or logprob_deg < -2.0),
        })

    return {
        "query_id":            counter_instance['id'],
        "passage_num":         passage_num,
        "n_docs_sampled":      n_docs,
        "n_gt_docs":           1,
        "n_counter_docs":      n_counter,
        "baseline_fact_score": baseline_fact_score,
        "baseline_logprob":    baseline_features['logprob'],
        "ablations":           ablation_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--counter_dataset', type=str, default='en_counter_mid')
    parser.add_argument('--gt_dataset',      type=str, default='en_mid')
    parser.add_argument('--modelname',       type=str, default='llama3')
    parser.add_argument('--temp',            type=float, default=0.7)
    parser.add_argument('--max_queries',     type=int,   default=10)
    parser.add_argument('--passage_nums',    type=int,   nargs='+', default=[3, 5, 7, 10])
    parser.add_argument('--seed',            type=int,   default=42)
    args = parser.parse_args()

    # Load counter-factual instances (en_counter_mid).
    counter_instances = []
    with open(f'data/{args.counter_dataset}.json', 'r', encoding='utf-8') as f:
        for line in f:
            counter_instances.append(json.loads(line))
    counter_instances = counter_instances[:args.max_queries]

    # Load ground-truth instances (en_mid), indexed by query ID.
    gt_by_id = {}
    with open(f'data/{args.gt_dataset}.json', 'r', encoding='utf-8') as f:
        for line in f:
            inst = json.loads(line)
            gt_by_id[inst['id']] = inst

    lang_key = args.counter_dataset[:2]
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
    output_file = f'lodo_passage_sweep_mixed_{args.modelname}.json'

    for passage_num in args.passage_nums:
        print(f"\n{'='*50}")
        print(f"Running LODO with passage_num={passage_num} (1 GT + {passage_num-1} counter-factual)")
        print(f"{'='*50}")
        bar = tqdm.tqdm(counter_instances, desc=f"passage_num={passage_num}")
        for counter_inst in bar:
            qid = counter_inst['id']
            gt_inst = gt_by_id.get(qid)
            if gt_inst is None:
                print(f"  WARNING: no en_mid instance for query {qid}, skipping")
                continue
            result = run_lodo_for_passage_num(
                model, counter_inst, gt_inst, passage_num,
                system_prompt, instruction, args.temp, args.seed
            )
            if result is None:
                print(f"  WARNING: no GT docs for query {qid}, skipping")
                continue
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
