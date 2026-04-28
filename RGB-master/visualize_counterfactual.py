import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

STYLES = ['Base', 'Academic', 'Confident', 'Conversational', 'Narrative']
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_summary(noise):
    path = f'result-en/multiple-runs/counter-factual/noise-{noise}/summary.json'
    with open(path) as f:
        return json.load(f)

noise0   = load_summary('0')
noise04  = load_summary('0.4')

# --------------------------------------------------------------------------
# Plot CF-1: Grouped bar — Accuracy by style and noise level
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = [('accuracy_mean', 'accuracy_std', 'Accuracy (%)'),
           ('reject_mean',   'reject_std',   'Reject (%)'),
           ('wrong_mean',    'wrong_std',    'Wrong (%)')]

x = np.arange(len(STYLES))
width = 0.35

for ax, (mean_key, std_key, label) in zip(axes, metrics):
    m0  = [s[mean_key] for s in noise0]
    s0  = [s[std_key]  for s in noise0]
    m04 = [s[mean_key] for s in noise04]
    s04 = [s[std_key]  for s in noise04]

    bars0  = ax.bar(x - width/2, m0,  width, yerr=s0,  capsize=4,
                    color='#4C72B0', alpha=0.85, label='noise=0.0')
    bars04 = ax.bar(x + width/2, m04, width, yerr=s04, capsize=4,
                    color='#DD8452', alpha=0.85, label='noise=0.4')

    ax.set_xticks(x)
    ax.set_xticklabels(STYLES, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(m0), max(m04)) * 1.25)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Counter-Factual Experiment: Style vs. Noise Level\n(en_counter_mid, Llama-3, 10 runs each)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/CF1_style_noise_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved CF1")

# --------------------------------------------------------------------------
# Plot CF-2: Box plots — raw run distributions per style (noise=0 vs 0.4)
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

for col, (mean_key, std_key, label) in enumerate(metrics):
    for row, (summary, noise_label) in enumerate([(noise0, 'noise=0.0'), (noise04, 'noise=0.4')]):
        ax = axes[row][col]
        data = [s['raw_results'] for s in summary]
        key_map = {'accuracy_mean': 'accuracy', 'reject_mean': 'reject', 'wrong_mean': 'wrong'}
        raw_key = key_map[mean_key]
        values = [[r[raw_key] for r in style_data] for style_data in data]

        bp = ax.boxplot(values, patch_artist=True, medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(STYLES)+1))
        ax.set_xticklabels(STYLES, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f'{label} | {noise_label}', fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

fig.suptitle('Distribution of Outcomes Across 10 Runs per Style\n(en_counter_mid, Counter-Factual Setting)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/CF2_boxplot_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved CF2")

# --------------------------------------------------------------------------
# Plot CF-3: Delta (noise=0 → noise=0.4) — stacked diverging bar
# --------------------------------------------------------------------------
delta_acc  = [n04['accuracy_mean']  - n00['accuracy_mean']  for n00, n04 in zip(noise0, noise04)]
delta_rej  = [n04['reject_mean']    - n00['reject_mean']    for n00, n04 in zip(noise0, noise04)]
delta_wrong= [n04['wrong_mean']     - n00['wrong_mean']     for n00, n04 in zip(noise0, noise04)]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(STYLES))
width = 0.25
ax.bar(x - width, delta_acc,   width, color='#55A868', label='Δ Accuracy', alpha=0.85)
ax.bar(x,         delta_rej,   width, color='#4C72B0', label='Δ Reject',   alpha=0.85)
ax.bar(x + width, delta_wrong, width, color='#C44E52', label='Δ Wrong',    alpha=0.85)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(STYLES, fontsize=11)
ax.set_ylabel('Change (noise 0.0 → 0.4)', fontsize=11)
ax.set_title('Effect of Noise on Outcome by Style\n(Δ = noise-0.4 minus noise-0.0)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/CF3_noise_delta.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved CF3")

# --------------------------------------------------------------------------
# Plot CF-4: en_mid vs en_counter_mid accuracy comparison (dataset motivation)
# --------------------------------------------------------------------------
en_mid_noise0   = {'Base': 96, 'Academic': 96, 'Confident': 96, 'Conversational': 92, 'Narrative': 96}
en_mid_noise04  = {'Base': 96, 'Academic': 90, 'Confident': 92, 'Conversational': 92, 'Narrative': 90}

counter_acc_noise0  = {s['style']: s['accuracy_mean'] for s in noise0}
counter_acc_noise04 = {s['style']: s['accuracy_mean'] for s in noise04}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (mid_data, counter_data, noise_label) in zip(axes, [
        (en_mid_noise0,  counter_acc_noise0,  'noise=0.0'),
        (en_mid_noise04, counter_acc_noise04, 'noise=0.4')]):
    x = np.arange(len(STYLES))
    width = 0.35
    mid_vals     = [mid_data[s]     for s in STYLES]
    counter_vals = [counter_data[s] for s in STYLES]

    ax.bar(x - width/2, mid_vals,     width, color='#55A868', alpha=0.85, label='en_mid (easy)')
    ax.bar(x + width/2, counter_vals, width, color='#C44E52', alpha=0.85, label='en_counter_mid (counterfactual)')
    ax.set_xticks(x)
    ax.set_xticklabels(STYLES, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_title(f'Dataset Difficulty Comparison | {noise_label}', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(50, color='gray', linestyle='--', linewidth=0.8, label='chance')

fig.suptitle('Why en_counter_mid? — en_mid Accuracy Is Near-Ceiling, Masking Document Effects',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/CF4_dataset_motivation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved CF4")

# --------------------------------------------------------------------------
# Plot CF-5: Scatter — accuracy vs wrong rate across all styles & runs (noise=0.4)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for i, (style_data, style, color) in enumerate(zip(noise04, STYLES, COLORS)):
    accs   = [r['accuracy'] for r in style_data['raw_results']]
    wrongs = [r['wrong']    for r in style_data['raw_results']]
    ax.scatter(accs, wrongs, color=color, label=style, alpha=0.75, s=60, edgecolors='white', linewidths=0.5)

ax.set_xlabel('Accuracy (%)', fontsize=11)
ax.set_ylabel('Wrong Rate (%)', fontsize=11)
ax.set_title('Accuracy vs. Wrong Rate per Run (noise=0.4)\nen_counter_mid Counter-Factual', fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/CF5_acc_vs_wrong_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved CF5")

# --------------------------------------------------------------------------
# Plot CF-6: Statistical significance heatmap (pairwise t-tests on accuracy)
# --------------------------------------------------------------------------
from scipy.stats import ttest_ind

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (summary, noise_label) in zip(axes, [(noise0, 'noise=0.0'), (noise04, 'noise=0.4')]):
    n = len(STYLES)
    pmat = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                a = [r['accuracy'] for r in summary[i]['raw_results']]
                b = [r['accuracy'] for r in summary[j]['raw_results']]
                _, p = ttest_ind(a, b)
                pmat[i, j] = p

    im = ax.imshow(pmat, vmin=0, vmax=0.1, cmap='RdYlGn_r')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(STYLES, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(STYLES, fontsize=9)
    ax.set_title(f'Pairwise t-test p-values (accuracy)\n{noise_label}', fontsize=10, fontweight='bold')
    for i in range(n):
        for j in range(n):
            val = pmat[i, j]
            sig = '**' if val < 0.01 else ('*' if val < 0.05 else f'{val:.2f}')
            ax.text(j, i, sig, ha='center', va='center', fontsize=8,
                    color='white' if val < 0.03 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle('Pairwise Statistical Significance of Style Differences\n(* p<0.05, ** p<0.01)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/CF6_pairwise_significance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved CF6")

print("\nAll counter-factual plots saved to", PLOTS_DIR)
