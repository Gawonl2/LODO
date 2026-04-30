import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

with open('lodo_passage_sweep_mixed_llama3.json') as f:
    data = json.load(f)

PASSAGE_NUMS = [3, 5, 7, 10]
COLORS = ['#C44E52', '#4C72B0', '#55A868', '#8172B2']
GT_COLOR  = '#5b9bd5'   # blue for ground-truth docs
CF_COLOR  = '#e07b54'   # orange for counter-factual docs

def get_ablations(pn):
    return [a for r in data if r['passage_num'] == pn for a in r['ablations']]

def get_drift(a):
    return a['representation_drift_l2'].get('layer_32', a['representation_drift_l2'].get('layer_31', 0))

# -----------------------------------------------------------------------
# Plot S1: Baseline accuracy, logprob-important rate, and GT-doc detection
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

baseline_accs    = []
logprob_imp_pcts = []
gt_detected_pcts = []  # fraction of GT docs that are logprob-important

for pn in PASSAGE_NUMS:
    subset = [r for r in data if r['passage_num'] == pn]
    ablations = [a for r in subset for a in r['ablations']]
    gt_ablations = [a for a in ablations if a.get('is_gt_doc', False)]

    baseline_accs.append(100 * sum(r['baseline_fact_score'] for r in subset) / len(subset))
    logprob_imp_pcts.append(100 * sum(a['logprob_degradation'] < -2.0 for a in ablations) / len(ablations))
    if gt_ablations:
        gt_detected_pcts.append(100 * sum(a['logprob_degradation'] < -2.0 for a in gt_ablations) / len(gt_ablations))
    else:
        gt_detected_pcts.append(0.0)

axes[0].bar(PASSAGE_NUMS, baseline_accs, width=1.5, color=COLORS, alpha=0.85)
for pn, acc in zip(PASSAGE_NUMS, baseline_accs):
    axes[0].text(pn, acc + 1, f'{acc:.0f}%', ha='center', fontsize=10)
axes[0].set_xlabel('Number of Retrieved Documents', fontsize=10)
axes[0].set_ylabel('Baseline Accuracy (%)', fontsize=10)
axes[0].set_title('Baseline Accuracy\n(1 GT + n-1 Counter-factual)', fontsize=11, fontweight='bold')
axes[0].set_xticks(PASSAGE_NUMS)
axes[0].set_ylim(0, 120)
axes[0].grid(axis='y', alpha=0.3)

axes[1].plot(PASSAGE_NUMS, logprob_imp_pcts, 'o-', color='#C44E52', linewidth=2,
             markersize=8, label='All docs (logprob-imp)')
axes[1].plot(PASSAGE_NUMS, gt_detected_pcts,  's--', color='#5b9bd5', linewidth=2,
             markersize=8, label='GT doc (logprob-imp)')
axes[1].axhline(100, color='gray', linestyle=':', linewidth=1)
axes[1].set_xlabel('Number of Retrieved Documents', fontsize=10)
axes[1].set_ylabel('Logprob-Important Rate (%)', fontsize=10)
axes[1].set_title('Logprob-Important Rate\nvs. Passage Count', fontsize=11, fontweight='bold')
axes[1].set_xticks(PASSAGE_NUMS)
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# GT doc factuality-critical rate (fact_degradation > 0 when GT doc is removed)
gt_fact_crit_pcts = []
for pn in PASSAGE_NUMS:
    subset = [r for r in data if r['passage_num'] == pn]
    gt_ablations = [a for r in subset for a in r['ablations'] if a.get('is_gt_doc', False)]
    if gt_ablations:
        gt_fact_crit_pcts.append(100 * sum(a['fact_degradation'] > 0 for a in gt_ablations) / len(gt_ablations))
    else:
        gt_fact_crit_pcts.append(0.0)

axes[2].bar(PASSAGE_NUMS, gt_fact_crit_pcts, width=1.5, color=GT_COLOR, alpha=0.85)
for pn, pct in zip(PASSAGE_NUMS, gt_fact_crit_pcts):
    axes[2].text(pn, pct + 1, f'{pct:.0f}%', ha='center', fontsize=10)
axes[2].set_xlabel('Number of Retrieved Documents', fontsize=10)
axes[2].set_ylabel('Factuality-Critical Rate (%)', fontsize=10)
axes[2].set_title('GT Doc: Factuality-Critical Rate\n(fact degrades when GT removed)', fontsize=11, fontweight='bold')
axes[2].set_xticks(PASSAGE_NUMS)
axes[2].set_ylim(0, 120)
axes[2].grid(axis='y', alpha=0.3)

fig.suptitle('Mixed Setting: 1 GT doc + (n-1) Counter-factual docs (Llama-3.1-8B, 10 queries)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/S1_mixed_sweep_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved S1")

# -----------------------------------------------------------------------
# Plot S2: Logprob degradation by doc type per passage_num
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, len(PASSAGE_NUMS), figsize=(14, 4.5), sharey=True)
for ax, pn, color in zip(axes, PASSAGE_NUMS, COLORS):
    subset = [r for r in data if r['passage_num'] == pn]
    gt_ldegs = [a['logprob_degradation'] for r in subset for a in r['ablations'] if a.get('is_gt_doc')]
    cf_ldegs = [a['logprob_degradation'] for r in subset for a in r['ablations'] if not a.get('is_gt_doc')]
    bp_data = [gt_ldegs, cf_ldegs] if gt_ldegs else [cf_ldegs]
    bp = ax.boxplot(bp_data, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    bp_colors = [GT_COLOR, CF_COLOR] if gt_ldegs else [CF_COLOR]
    for patch, c in zip(bp['boxes'], bp_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticklabels(['GT', 'Counter-fact'] if gt_ldegs else ['Counter-fact'], fontsize=9)
    ax.axhline(-2.0, color='gray', linestyle='--', linewidth=1)
    ax.set_title(f'n={pn}', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

axes[0].set_ylabel('Logprob Degradation', fontsize=10)
fig.suptitle('Logprob Degradation by Doc Type per Passage Count\n'
             '(dashed = importance threshold; GT=ground-truth doc)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/S2_logprob_by_doctype.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved S2")

# -----------------------------------------------------------------------
# Plot S3: Scatter logprob vs fact degradation, colored by doc type
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, pn, color in zip(axes.flatten(), PASSAGE_NUMS, COLORS):
    subset = [r for r in data if r['passage_num'] == pn]
    gt_abl = [a for r in subset for a in r['ablations'] if a.get('is_gt_doc')]
    cf_abl = [a for r in subset for a in r['ablations'] if not a.get('is_gt_doc')]

    ax.scatter([a['logprob_degradation'] for a in cf_abl],
               [a['fact_degradation'] for a in cf_abl],
               color=CF_COLOR, alpha=0.6, s=30, label='Counter-factual', edgecolors='none')
    ax.scatter([a['logprob_degradation'] for a in gt_abl],
               [a['fact_degradation'] for a in gt_abl],
               color=GT_COLOR, alpha=0.9, s=80, marker='*', label='Ground-truth', edgecolors='navy', linewidths=0.5)

    ax.axvline(-2.0, color='black', linestyle='--', linewidth=0.8)
    ax.axhline(0, color='black', linestyle=':', linewidth=0.8)
    ax.set_xlabel(r'$\Delta_{\mathrm{logprob}}$', fontsize=9)
    ax.set_ylabel(r'$\Delta_{\mathrm{fact}}$', fontsize=9)
    ax.set_title(f'n={pn}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

fig.suptitle('Logprob vs Factual Degradation by Doc Type (★=GT doc)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/S3_logprob_vs_fact_by_doctype.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved S3")

# -----------------------------------------------------------------------
# Plot S4: Baseline accuracy and GT factuality-critical rate summary
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(PASSAGE_NUMS))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_accs, width, color='#5b9bd5', alpha=0.85,
               label='Baseline Accuracy')
bars2 = ax.bar(x + width/2, gt_fact_crit_pcts, width, color='#70ad47', alpha=0.85,
               label='GT Doc Factuality-Critical Rate')

for bar, val in zip(bars1, baseline_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.0f}%', ha='center', fontsize=9)
for bar, val in zip(bars2, gt_fact_crit_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.0f}%', ha='center', fontsize=9)

ax.set_xlabel('Number of Retrieved Documents (1 GT + n-1 Counter-factual)', fontsize=11)
ax.set_ylabel('Rate (%)', fontsize=11)
ax.set_title('Mixed Setting: When Does Removing the GT Doc Matter?', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'n={pn}' for pn in PASSAGE_NUMS])
ax.set_ylim(0, 130)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/S4_baseline_vs_gt_critical.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved S4")

print(f"\nAll sweep plots saved to {PLOTS_DIR}/")
