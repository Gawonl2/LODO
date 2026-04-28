import json
import numpy as np
import tqdm
from models.models import Llama3
import yaml
import os

def main():
    print("Loading initial LODO results to identify Top 5 vs Average 5 cases...")
    with open('lodo_results_en_refine_llama3.json', 'r', encoding='utf-8') as f:
        lodo_results = json.load(f)
        
    ablations = []
    for query in lodo_results:
        for l in query['lodo_results']:
            ablations.append({
                'query_id': query['id'],
                'query_text': query['query'],
                'baseline_answer': query['baseline_answer'],
                'removed_doc_index': l['removed_doc_index'],
                'logprob_deg': l['logprob_degradation'],
                'fact_deg': l['fact_degradation']
            })
            
    # Top 5 most negative logprob degradation with fact_deg == 0 (Collapse Divergence cases)
    divergence_cases = [a for a in ablations if a['fact_deg'] == 0]
    divergence_cases = sorted(divergence_cases, key=lambda x: x['logprob_deg'])
    top_5 = divergence_cases[:5]
    
    # 5 Average cases (logprob degradation closest to 0)
    avg_cases = sorted(ablations, key=lambda x: abs(x['logprob_deg']))
    avg_5 = avg_cases[:5]
    
    targets = top_5 + avg_5
    for i, t in enumerate(targets):
        t['group'] = 'Top Divergence' if i < 5 else 'Average Baseline'
        
    print(f"Selected exactly {len(targets)} ablations to Deep Dive.")
    
    # Load Dataset
    dataset = []
    with open('data/en_refine.json', 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    dataset_dict = {item['id']: item for item in dataset}
    
    # Load Model
    print("Loading Llama-3 Model into VRAM...")
    model = Llama3('meta-llama/Llama-3.1-8B-Instruct')
    prompt_config = yaml.load(open('config/instruction.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)['en']
    instruction = prompt_config['instruction']
    sys_prompt = prompt_config['system']
    
    detailed_results = []
    
    for meta in tqdm.tqdm(targets, desc="Processing specific ablations"):
        instance = dataset_dict[meta['query_id']]
        docs = instance.get('positive', []) + instance.get('negative', [])
        if len(docs) > 0 and isinstance(docs[0], list):
            docs = [d[0] for d in docs]
            
        # BASELINE
        baseline_text = instruction.format(QUERY=meta['query_text'], DOCS='\n'.join(docs))
        base_features = model.get_logprob_and_states(
            baseline_text, meta['baseline_answer'], system=sys_prompt, 
            return_all_layers=True, return_token_logprobs=True
        )
        
        # ABLATED
        ablated_docs = docs[:meta['removed_doc_index']] + docs[meta['removed_doc_index']+1:]
        ablated_text = instruction.format(QUERY=meta['query_text'], DOCS='\n'.join(ablated_docs))
        abl_features = model.get_logprob_and_states(
            ablated_text, meta['baseline_answer'], system=sys_prompt, 
            return_all_layers=True, return_token_logprobs=True
        )
        
        # L2 Drift for all layers
        layer_drifts = []
        # sort layers numerically to ensure sequential mapping
        b_states = base_features['hidden_states_mean']
        a_states = abl_features['hidden_states_mean']
        
        for i in range(len(b_states)):
            k = f"layer_{i}"
            b = np.array(b_states[k])
            a = np.array(a_states[k])
            drift = float(np.linalg.norm(b - a))
            layer_drifts.append(drift)
            
        detailed_results.append({
            'query_id': meta['query_id'],
            'removed_doc_index': meta['removed_doc_index'],
            'group': meta['group'],
            'original_logprob_deg': meta['logprob_deg'],
            'tokens': base_features['tokens_text'],            
            'baseline_token_logprobs': base_features['token_logprobs'],
            'ablated_token_logprobs': abl_features['token_logprobs'],
            'layer_drifts': layer_drifts
        })
        
    with open('detailed_case_studies.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2)
        
    print("Detailed extraction complete! Saved to detailed_case_studies.json")

if __name__ == '__main__':
    main()
