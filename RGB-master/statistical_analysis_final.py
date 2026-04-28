"""Statistical analysis of counter-factual and LODO results for final report."""
import json
import numpy as np
from scipy import stats
import os

STYLES = ['Base', 'Academic', 'Confident', 'Conversational', 'Narrative']

def load_summary(noise):
    with open(f'result-en/multiple-runs/counter-factual/noise-{noise}/summary.json') as f:
        return json.load(f)

noise0  = load_summary('0')
noise04 = load_summary('0.4')

print("=" * 60)
print("COUNTER-FACTUAL EXPERIMENT STATISTICS")
print("=" * 60)

for noise_label, summary in [('noise=0.0', noise0), ('noise=0.4', noise04)]:
    print(f"\n--- {noise_label} ---")
    print(f"{'Style':<15} {'Acc mean':>9} {'Acc std':>8} {'Rej mean':>9} {'Rej std':>8} {'Wrong mean':>11} {'Wrong std':>9}")
    for s in summary:
        print(f"{s['style']:<15} {s['accuracy_mean']:>9.1f} {s['accuracy_std']:>8.2f} "
              f"{s['reject_mean']:>9.1f} {s['reject_std']:>8.2f} "
              f"{s['wrong_mean']:>11.1f} {s['wrong_std']:>9.2f}")

print("\n\n--- Pairwise t-tests (accuracy, noise=0.0) ---")
for i, si in enumerate(noise0):
    for j, sj in enumerate(noise0):
        if i >= j:
            continue
        a = [r['accuracy'] for r in si['raw_results']]
        b = [r['accuracy'] for r in sj['raw_results']]
        t, p = stats.ttest_ind(a, b)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print(f"  {si['style']} vs {sj['style']}: t={t:.2f}, p={p:.3f} {sig}")

print("\n--- Pairwise t-tests (accuracy, noise=0.4) ---")
for i, si in enumerate(noise04):
    for j, sj in enumerate(noise04):
        if i >= j:
            continue
        a = [r['accuracy'] for r in si['raw_results']]
        b = [r['accuracy'] for r in sj['raw_results']]
        t, p = stats.ttest_ind(a, b)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print(f"  {si['style']} vs {sj['style']}: t={t:.2f}, p={p:.3f} {sig}")

print("\n--- Noise effect (noise=0 vs noise=0.4 per style, paired t-test on accuracy) ---")
for s0, s04 in zip(noise0, noise04):
    a0  = [r['accuracy'] for r in s0['raw_results']]
    a04 = [r['accuracy'] for r in s04['raw_results']]
    t, p = stats.ttest_rel(a0, a04)
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
    delta = np.mean(a04) - np.mean(a0)
    print(f"  {s0['style']:<15}: Δacc={delta:+.1f}, t={t:.2f}, p={p:.3f} {sig}")

print("\n\n=" * 60)
print("LODO MECHANISTIC STATISTICS (en_refine)")
print("=" * 60)

with open('lodo_results_en_refine_llama3.json') as f:
    lodo = json.load(f)

all_logprob_deg = []
all_fact_deg    = []
all_drift_last  = []
causal_count    = 0
total_ablations = 0

for query in lodo:
    for r in query['lodo_results']:
        all_logprob_deg.append(r['logprob_degradation'])
        all_fact_deg.append(r['fact_degradation'])
        if 'layer_32' in r['representation_drift_l2']:
            all_drift_last.append(r['representation_drift_l2']['layer_32'])
        elif 'layer_31' in r['representation_drift_l2']:
            all_drift_last.append(r['representation_drift_l2']['layer_31'])
        if r['is_causally_important']:
            causal_count += 1
        total_ablations += 1

print(f"\nTotal queries: {len(lodo)}")
print(f"Total ablations: {total_ablations}")
print(f"Causally important: {causal_count} / {total_ablations} ({100*causal_count/total_ablations:.1f}%)")
print(f"\nLogprob degradation:")
print(f"  mean={np.mean(all_logprob_deg):.4f}, std={np.std(all_logprob_deg):.4f}")
print(f"  min={np.min(all_logprob_deg):.4f}, max={np.max(all_logprob_deg):.4f}")
print(f"\nFact degradation distribution: {dict(zip(*np.unique(all_fact_deg, return_counts=True)))}")
print(f"\nFinal-layer L2 drift:")
print(f"  mean={np.mean(all_drift_last):.4f}, std={np.std(all_drift_last):.4f}")
print(f"  max={np.max(all_drift_last):.4f}")

print("\n--- Detailed case studies ---")
with open('detailed_case_studies.json') as f:
    cases = json.load(f)

top_div = [c for c in cases if c.get('case_type') == 'top_divergence']
avg_base = [c for c in cases if c.get('case_type') == 'average_baseline']

if top_div and avg_base:
    td_drift = [c['last_layer_drift'] for c in top_div if 'last_layer_drift' in c]
    ab_drift = [c['last_layer_drift'] for c in avg_base if 'last_layer_drift' in c]
    if td_drift and ab_drift:
        print(f"Top divergence last-layer drift: mean={np.mean(td_drift):.4f}")
        print(f"Average baseline last-layer drift: mean={np.mean(ab_drift):.4f}")
        t, p = stats.ttest_ind(td_drift, ab_drift)
        print(f"t-test: t={t:.2f}, p={p:.4f}")
else:
    print("Case study structure:", list(cases[0].keys()) if cases else "empty")
