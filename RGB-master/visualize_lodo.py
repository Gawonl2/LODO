import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Flatten the JSON into a list of document ablations
    rows = []
    for query in data:
        qid = query['id']
        q_text = query['query']
        base_logprob = query['baseline_logprob']
        base_fact = query['baseline_fact_score']
        
        for lodo in query['lodo_results']:
            row = {
                'query_id': qid,
                'doc_index': lodo['removed_doc_index'],
                'target_logprob_without_doc': lodo['target_logprob_without_doc'],
                'logprob_degradation': lodo['logprob_degradation'],
                'fact_degradation': lodo['fact_degradation'],
                'is_causally_important': lodo['is_causally_important']
            }
            # Add layer drifts
            drift = lodo.get('representation_drift_l2', {})
            for layer, val in drift.items():
                row[layer] = val
                
            rows.append(row)
            
    return pd.DataFrame(rows)

def main():
    filepath = 'lodo_results_en_refine_llama3.json'
    if not os.path.exists(filepath):
        print(f"File {filepath} not found. Ensure the LODO script has finished running!")
        return
        
    df = load_data(filepath)
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    # Set overall aesthetic
    sns.set_theme(style="whitegrid", context="talk")
    
    # 1. Box plots for Logprobs (Outlier detection)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df['logprob_degradation'], color='skyblue')
    plt.title('Distribution of Logprob Degradation\n(Outliers indicate critical docs)')
    plt.ylabel('Logprob Degradation')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['target_logprob_without_doc'], color='lightgreen')
    plt.title('Target Logprob Without Doc\n(Identify overall collapse)')
    plt.ylabel('Target Logprob')
    
    plt.tight_layout()
    plt.savefig('plots/1_logprob_outliers.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plots for Representation Drift across Layers (Outliers)
    layer_cols = [c for c in df.columns if c.startswith('layer_')]
    if layer_cols:
        # Sort layers logically (e.g. 0, 16, 32)
        layer_cols = sorted(layer_cols, key=lambda x: int(x.split('_')[1]))
        
        plt.figure(figsize=(10, 6))
        # Melt DataFrame for easier seaborn plotting
        melted_drift = df.melt(value_vars=layer_cols, var_name='Layer', value_name='L2 Drift')
        sns.boxplot(x='Layer', y='L2 Drift', data=melted_drift, palette='pastel')
        plt.title('Representation Drift (L2) Across Transformer Layers')
        plt.savefig('plots/2_layer_drift_outliers.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    # 3. Bar chart for Fact Degradation (-1, 0, 1)
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='fact_degradation', data=df, palette='viridis')
    plt.title('Count of Fact Degradation Events\n(1=Lost Fact, 0=No Change, -1=Gained Fact)')
    plt.xlabel('Fact Degradation')
    
    # Annotate bar counts
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12, xytext=(0, 5), textcoords='offset points')
                    
    plt.savefig('plots/3_fact_degradation_counts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Box plot: Target Logprob by Causal Importance
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='is_causally_important', y='target_logprob_without_doc', data=df, palette='Set2')
    plt.title('Target Logprob Without Doc\nvs. Causal Importance')
    plt.xlabel('Is Causally Important?')
    plt.ylabel('Target Logprob Without Doc')
    plt.savefig('plots/4_logprob_by_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. BONUS: Scatter plot of Layer 32 Drift vs Logprob Degradation
    # Colored by Causal Importance to show divergence
    if 'layer_32' in df.columns:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x='logprob_degradation', 
            y='layer_32', 
            hue='is_causally_important', 
            style='fact_degradation',
            data=df, 
            alpha=0.8,
            s=100,
            palette={True: '#e74c3c', False: '#95a5a6'} # Red vs Gray
        )
        plt.title('Mechanistic Divergence:\nLikelihood Collapse vs Final Layer Drift')
        plt.xlabel('Logprob Degradation (More negative = Likelihood Collapse)')
        plt.ylabel('Layer 32 Representation Drift (L2 Distance)')
        
        # Add a vertical dotted line at our -2.0 threshold
        plt.axvline(x=-2.0, color='r', linestyle='--', alpha=0.5, label='Logprob Threshold (-2.0)')
        
        # Move legend outside
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('plots/5_mechanistic_divergence.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    print("\nVisualizations successfully saved to the 'plots/' directory!")

if __name__ == '__main__':
    main()
