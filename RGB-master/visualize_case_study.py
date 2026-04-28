import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    try:
        with open('detailed_case_studies.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("detailed_case_studies.json not found! Please ensure run_detailed_case_study.py finishes first.")
        return
        
    sns.set_theme(style="whitegrid", context="talk")
    
    # --- PLOT 1: Token-wise Logprob Trajectory ---
    # We will pick 1 top example and 1 average example to show clearly
    top_case = next(d for d in data if d['group'] == 'Top Divergence')
    avg_case = next(d for d in data if d['group'] == 'Average Baseline')
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    for i, case in enumerate([top_case, avg_case]):
        ax = axes[i]
        tokens = case['tokens']
        # Clean up tokens for display (Llama-3 tokenizer has Ġ or plain strings, we'll try to keep them clean)
        clean_tokens = [t.replace('Ġ', ' ') for t in tokens]
        
        b_lp = case['baseline_token_logprobs']
        a_lp = case['ablated_token_logprobs']
        
        x = np.arange(len(tokens))
        ax.plot(x, b_lp, 'o-', color='#3498db', label='Baseline (All Docs)', linewidth=2.5, markersize=6)
        ax.plot(x, a_lp, 'x--', color='#e74c3c', label='Ablated (Doc Removed)', linewidth=2.5, markersize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(clean_tokens, rotation=45, ha='right', fontsize=11)
        ax.set_title(f"Token-wise Logprob: {case['group']} (Query: {case['query_id']})")
        ax.set_ylabel('Log Probability')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('plots/6_tokenwise_trajectory.png', dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
    plt.close()

    # --- PLOT 2: Layer-wise Drift Curve (All 33 Layers) ---
    plt.figure(figsize=(12, 7))
    
    for case in data:
        drifts = case['layer_drifts']
        x = np.arange(len(drifts))
        
        if case['group'] == 'Top Divergence':
            plt.plot(x, drifts, color='#e74c3c', alpha=0.8, linewidth=2.5)
        else:
            plt.plot(x, drifts, color='#95a5a6', alpha=0.6, linewidth=2.0)
            
    # Add custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#e74c3c', lw=3),
                    Line2D([0], [0], color='#95a5a6', lw=3)]
    plt.legend(custom_lines, ['Top Divergence (Critical Docs)', 'Average Baseline (Redundant Docs)'], loc='upper left')
    
    plt.title('Mechanistic Trajectory: Layer-wise L2 Drift (Layers 0-32)')
    plt.xlabel('Transformer Layer Depth')
    plt.ylabel('Representation Drift (L2 Distance)')
    
    plt.xticks(np.arange(0, 33, 2))
    plt.xlim(0, 32)
    
    plt.tight_layout()
    plt.savefig('plots/7_layerwise_drift_curve.png', dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
    plt.close()
    
    print("\nCase study visualizations successfully saved to the 'plots/' directory!")

if __name__ == '__main__':
    main()
