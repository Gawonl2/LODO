import os
import json
import argparse
import tqdm
import yaml
import numpy as np
from evalue import checkanswer
from models.models import Llama3

def compute_l2_drift(states_dict1, states_dict2):
    drift = {}
    for layer, s1 in states_dict1.items():
        if layer in states_dict2:
            s2 = states_dict2[layer]
            drift[layer] = float(np.linalg.norm(np.array(s1) - np.array(s2)))
    return drift

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='en_refine')
    parser.add_argument('--modelname', type=str, default='llama3')
    parser.add_argument('--temp', type=float, default=0.7)
    args = parser.parse_args()

    instances = []
    with open(f'data/{args.dataset}.json', 'r', encoding='utf-8') as f:
        for line in f:
            instances.append(json.loads(line))
            
    prompt_config = yaml.load(open('config/instruction.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)[args.dataset[:2]]
    system_prompt = prompt_config['system']
    instruction = prompt_config['instruction']

    if args.modelname == 'llama3':
        model = Llama3()
    else:
        print("Only llama3 supported for LODO mechanistic evaluation right now.")
        return

    results = []
    output_file = f'lodo_results_{args.dataset}_{args.modelname}.json'
    
    # Process just the first 10 for the demo/stats to prevent massive run times
    outer_bar = tqdm.tqdm(instances[:10], desc="Total Queries")
    for i, instance in enumerate(outer_bar):
        query = instance['query']
        ans_ground_truth = instance['answer']
        docs = instance.get('positive', []) + instance.get('negative', [])
        
        outer_bar.set_postfix_str("Generating Baseline...")
        
        # Format docs into flat list of strings
        if len(docs) > 0 and isinstance(docs[0], list):
            docs = [d[0] for d in docs]
            
        # 1. Baseline Full Context Generation
        docs_text_full = '\n'.join(docs)
        text_full = instruction.format(QUERY=query, DOCS=docs_text_full)
        
        baseline_answer = model.generate(text_full, temperature=args.temp, system=system_prompt)
        baseline_fact_labels = checkanswer(baseline_answer, ans_ground_truth)
        baseline_fact_score = 1 if 1 in baseline_fact_labels and 0 not in baseline_fact_labels else 0
        
        # 1b. Baseline logprob and states
        baseline_features = model.get_logprob_and_states(text_full, baseline_answer, system=system_prompt)
        
        lodo_results = []
        
        # 2. LODO iteration
        inner_bar = tqdm.tqdm(docs, desc=f"Query {i+1} Ablations", leave=False)
        for doc_idx, doc in enumerate(inner_bar):
            outer_bar.set_postfix_str(f"Ablating Doc {doc_idx+1}/{len(docs)}")
            
            ablated_docs = docs[:doc_idx] + docs[doc_idx+1:]
            
            docs_text_ablated = '\n'.join(ablated_docs)
            text_ablated = instruction.format(QUERY=query, DOCS=docs_text_ablated)
            
            # 2a. New Generation Fact Score Without Doc
            ablated_answer = model.generate(text_ablated, temperature=args.temp, system=system_prompt)
            ablated_fact_labels = checkanswer(ablated_answer, ans_ground_truth)
            new_fact_score = 1 if 1 in ablated_fact_labels and 0 not in ablated_fact_labels else 0
            
            # 2b. Likelihood Degradation & Mechanistic Drift of the BASELINE answer using ablated context
            ablated_features = model.get_logprob_and_states(text_ablated, baseline_answer, system=system_prompt)
            
            logprob_degradation = ablated_features['logprob'] - baseline_features['logprob']
            fact_degradation = baseline_fact_score - new_fact_score
            
            representation_drift = compute_l2_drift(
                baseline_features['hidden_states_mean'], 
                ablated_features['hidden_states_mean']
            )
            
            lodo_results.append({
                "removed_doc_index": doc_idx,
                "removed_doc_text": doc[:100] + '...', # truncated for json size
                "new_generated_answer": ablated_answer,
                "new_fact_score": new_fact_score,
                "target_logprob_without_doc": ablated_features['logprob'],
                "logprob_degradation": logprob_degradation,
                "fact_degradation": fact_degradation,
                "representation_drift_l2": representation_drift,
                "is_causally_important": bool(fact_degradation > 0 or logprob_degradation < -2.0)
            })

        results.append({
            "id": instance['id'],
            "query": query,
            "baseline_answer": baseline_answer,
            "baseline_fact_score": baseline_fact_score,
            "baseline_logprob": baseline_features['logprob'],
            "lodo_results": lodo_results
        })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
