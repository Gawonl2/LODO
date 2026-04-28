import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

with open('lodo_passage_sweep_en_counter_mid_llama3.json') as f:
    data = json.load(f)

PASSAGE_NUMS = [3, 5, 7, 10]
COLORS = ['#C44E52', '#4C72B0', '#55A868', '#8172B2']

def get_ablations(pn):
    return [a for r in data if r['passage_num'] == pn for a in r['ablations']]

def get_drift(a):
    return a['representation_drift_l2'].get('layer_32', a['representation_drift_l2'].get('layer_31', 0))

# -----------------------------------------------------------------------
# Plot S1: Causal importance % and mean logprob degradation vs passage_num
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

causal_pcts   = []
mean_logdegs  = []
mean_drifts   = []
std_logdegs   = []
std_drifts    = []

for pn in PASSAGE_NUMS:
    ablations = get_ablations(pn)
    causal_pcts.append(100 * sum(a['is_causally_important'] for a in ablations) / len(ablations))
    ldegs = [a['logprob_degradation'] for a in ablations]
    drifts = [get_drift(a) for a in ablations]
    mean_logdegs.append(np.mean(ldegs))
    mean_drifts.append(np.mean(drifts))
    std_logdegs.append(np.std(ldegs))
    std_drifts.append(np.std(drifts))

axes[0].plot(PASSAGE_NUMS, causal_pcts, 'o-', color='#C44E52', linewidth=2, markersize=8)
axes[0].fill_between(PASSAGE_NUMS, causal_pcts, alpha=0.15, color='#C44E52')
axes[0].set_xlabel('Number of Retrieved Documents', fontsize=10)
axes[0].set_ylabel('Causally Important (%)', fontsize=10)
axes[0].set_title('Causal Importance Rate\nvs. Passage Count', fontsize=11, fontweight='bold')
axes[0].set_xticks(PASSAGE_NUMS)
axes[0].grid(alpha=0.3)

axes[1].errorbar(PASSAGE_NUMS, mean_logdegs, yerr=std_logdegs, fmt='o-',
                 color='#4C72B0', linewidth=2, markersize=8, capsize=5)
axes[1].axhline(-2.0, color='gray', linestyle='--', linewidth=1, label='importance threshold')
axes[1].set_xlabel('Number of Retrieved Documents', fontsize=10)
axes[1].set_ylabel('Mean Logprob Degradation', fontsize=10)
axes[1].set_title('Mean Logprob Degradation\nvs. Passage Count', fontsize=11, fontweight='bold')
axes[1].set_xticks(PASSAGE_NUMS)
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

axes[2].errorbar(PASSAGE_NUMS, mean_drifts, yerr=std_drifts, fmt='o-',
                 color='#55A868', linewidth=2, markersize=8, capsize=5)
axes[2].set_xlabel('Number of Retrieved Documents', fontsize=10)
axes[2].set_ylabel('Mean L2 Drift (Layer 32)', fontsize=10)
axes[2].set_title('Mean Representation Drift\nvs. Passage Count', fontsize=11, fontweight='bold')
axes[2].set_xticks(PASSAGE_NUMS)
axes[2].grid(alpha=0.3)

fig.suptitle('LODO Passage-Number Sweep on en_counter_mid (Llama-3.1-8B, 10 queries)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/S1_passage_sweep_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved S1")

# -----------------------------------------------------------------------
# Plot S2: Box plots of logprob degradation per passage_num
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

logdeg_data = [a['logprob_degradation'] for pn in PASSAGE_NUMS for a in get_ablations(pn)]
drift_data  = [get_drift(a) for pn in PASSAGE_NUMS for a in get_ablations(pn)]

for ax, (vals_list, ylabel, title) in zip(axes, [
    ([[a['logprob_degradation'] for a in get_ablations(pn)] for pn in PASSAGE_NUMS],
     'Logprob Degradation', 'Logprob Degradation Distribution'),
    ([[get_drift(a) for a in get_ablations(pn)] for pn in PASSAGE_NUMS],
     'L2 Drift (Layer 32)', 'Representation Drift Distribution'),
]):
    bp = ax.boxplot(vals_list, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels([f'n={pn}' for pn in PASSAGE_NUMS], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

axes[0].axhline(-2.0, color='gray', linestyle='--', linewidth=1, label='importance threshold')
axes[0].legend(fontsize=8)

fig.suptitle('Distribution of LODO Degradation Metrics by Passage Count\n(en_counter_mid, Llama-3.1-8B)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/S2_sweep_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved S2")

# -----------------------------------------------------------------------
# Plot S3: Positive vs negative doc causal importance breakdown
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

pos_causal_pct = []
neg_causal_pct = []
for pn in PASSAGE_NUMS:
    ablations = get_ablations(pn)
    pos_abl = [a for a in ablations if a['is_positive_doc']]
    neg_abl = [a for a in ablations if not a['is_positive_doc']]
    pos_causal_pct.append(100 * sum(a['is_causally_important'] for a in pos_abl) / len(pos_abl) if pos_abl else 0)
    neg_causal_pct.append(100 * sum(a['is_causally_important'] for a in neg_abl) / len(neg_abl) if neg_abl else 0)

x = np.arange(len(PASSAGE_NUMS))
width = 0.35
ax.bar(x - width/2, pos_causal_pct, width, color='#55A868', alpha=0.85, label='Positive doc (ground truth)')
ax.bar(x + width/2, neg_causal_pct, width, color='#C44E52', alpha=0.85, label='Negative doc (noise/counter)')
ax.set_xticks(x)
ax.set_xticklabels([f'n={pn}' for pn in PASSAGE_NUMS], fontsize=11)
ax.set_ylabel('Causal Importance Rate (%)', fontsize=11)
ax.set_title('Causal Importance Rate by Document Type\n(Positive vs. Negative/Counterfactual)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/S3_pos_vs_neg_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved S3")

# -----------------------------------------------------------------------
# Plot S4: Scatter — logprob degradation vs L2 drift, all passage_nums
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, pn, color in zip(axes.flatten(), PASSAGE_NUMS, COLORS):
    ablations = get_ablations(pn)
    ldegs  = [a['logprob_degradation'] for a in ablations]
    drifts = [get_drift(a) for a in ablations]
    causal = [a['is_causally_important'] for a in ablations]
    is_pos = [a['is_positive_doc'] for a in ablations]

    markers = ['^' if p else 'o' for p in is_pos]
    for ldeg, drift, c, m in zip(ldegs, drifts, causal, markers):
        fc = color if c else 'lightgray'
        ax.scatter(ldeg, drift, color=fc, marker=m, s=60,
                   edgecolors=color if c else 'gray', linewidths=0.8, alpha=0.85)

    ax.axvline(-2.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Logprob Degradation', fontsize=9)
    ax.set_ylabel('L2 Drift (Layer 32)', fontsize=9)
    ax.set_title(f'passage_num={pn}', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)

    causal_patch  = mpatches.Patch(color=color, label='Causally important')
    neutral_patch = mpatches.Patch(color='lightgray', label='Not important')
    tri = plt.Line2D([0],[0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Positive doc')
    circle = plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Negative doc')
    ax.legend(handles=[causal_patch, neutral_patch, tri, circle], fontsize=7, loc='upper right')

fig.suptitle('Logprob Degradation vs. Representation Drift per Passage Count\n'
             'Triangle=positive doc, Circle=negative doc; colored=causally important',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/S4_scatter_per_passage.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved S4")

# -----------------------------------------------------------------------
# Plot S5: Baseline fact score vs passage_num (showing context sufficiency)
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
for pn, color in zip(PASSAGE_NUMS, COLORS):
    subset = [r for r in data if r['passage_num'] == pn]
    bf = [r['baseline_fact_score'] for r in subset]
    ax.bar(pn, 100*sum(bf)/len(bf), width=1.5, color=color, alpha=0.85, label=f'n={pn}')
    ax.text(pn, 100*sum(bf)/len(bf) + 1, f"{100*sum(bf)/len(bf):.0f}%", ha='center', fontsize=10)

ax.set_xlabel('Number of Retrieved Documents', fontsize=11)
ax.set_ylabel('Baseline Accuracy (%)', fontsize=11)
ax.set_title('Baseline Model Accuracy vs. Passage Count\n'
             '(en_counter_mid: model must resist counterfactual evidence)', fontsize=11, fontweight='bold')
ax.set_xticks(PASSAGE_NUMS)
ax.set_ylim(0, 40)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/S5_baseline_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved S5")

print(f"\nAll sweep plots saved to {PLOTS_DIR}/")
