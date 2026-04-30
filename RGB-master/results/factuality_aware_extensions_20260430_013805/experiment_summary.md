# Factuality-Aware Extension Experiments: Summary

Generated: 2026-04-30 01:38:11
Output directory: /home/gawon/lodo/RGB-master/results/factuality_aware_extensions_20260430_013805

---

## Connection Between New Experiments and Pre-Existing Findings

| Experiment | Pre-existing finding extended |
|------------|-------------------------------|
| E1: Collapse Divergence Rate | Collapse divergence (qualitative) → quantified as CDR |
| E2: Importance Taxonomy | Sparsity → breakdown of *types* of sparse cases |
| E3: Fact-Token Logprob | Token-level locality → predicts factual degradation better? |
| E4: Mechanistic Signatures | Integration horizon → different for helpful vs confidence-only? |
| E5: Passage-Number Gap | Passage sweep → does redundancy change the influence-usefulness gap? |
| E6: Ranking Comparison | Sparsity + CDR → does logprob-only ranking find useful docs? |
| E7: Position-Controlled | Integration horizon → semantic or positional artifact? |

---

## E1: Collapse Divergence Rate

**CDR (overall): 64.8%** — Among all documents that caused strong
logprob collapse (Δ_logprob < -2.0), this fraction had *zero* factual degradation.
This directly shows that logprob-only attribution marks the majority of
'influential' documents as important even when they are factually irrelevant.

| Group | CDR | FCR | FDCR |
|-------|-----|-----|------|
| Overall (all sources) | 64.8% | 22.5% | 12.7% |
| Sweep only | 63.2% | 23.5% | 13.2% |
| passage_num=3 | 57.1% | 23.8% | 19.1% |
| passage_num=5 | 64.7% | 23.5% | 11.8% |
| passage_num=7 | 58.8% | 23.5% | 17.6% |
| passage_num=10 | 76.9% | 23.1% | 0.0% |
| doc_type=counter-factual | 71.2% | 11.5% | 17.3% |
| doc_type=ground-truth | 37.5% | 62.5% | 0.0% |

---

## E2: Importance Taxonomy

Distribution across all ablations:

| Category | Count | % |
|----------|-------|---|
| neutral | 378 | 79.9% |
| confidence-only | 46 | 9.7% |
| factuality-weak | 17 | 3.6% |
| factuality-critical | 16 | 3.4% |
| factuality-disrupting | 9 | 1.9% |
| fact-only | 7 | 1.5% |

**Key finding:** 'Confidence-only' documents (logprob collapses but fact unchanged)
are far more common than 'helpful-important' documents. This confirms that
logprob-based attribution inflates apparent document importance.

---

## E3: Fact-Token vs Whole-Sequence Logprob Drop

Token-level data available for 10 case studies from `detailed_case_studies.json`.
See plots/E3_token_type_logprob_drop_boxplot.png for distribution.

---

## E4: Mechanistic Signatures by Category

Layer-32 drift by category (mean across sweep ablations):

| Category | Mean L2@32 |
|----------|-----------|
| confidence-only | 14.05 |
| fact-only | 7.96 |
| factuality-critical | 15.68 |
| factuality-disrupting | 16.73 |
| factuality-weak | 10.03 |
| neutral | 5.65 |

Full 33-layer trajectory available in plots/E4_full_33layer_drift_case_studies.png.
The integration horizon (layers 18–20) is visible in both helpful and confidence-only cases,
but the magnitude differs — see tables/E4_mechanistic_tests.tex for Mann-Whitney results.

---

## E5: Passage-Number Dependence of Influence-Usefulness Gap

| n | logprob-imp rate | fact-useful rate | gap | CDR | baseline acc |
|---|-----------------|-----------------|-----|-----|-------------|
| 3 | 70.0% | 16.7% | 53.3% | 57.1% | 30% |
| 5 | 38.6% | 11.4% | 27.3% | 64.7% | 30% |
| 7 | 30.9% | 7.3% | 23.6% | 58.8% | 20% |
| 10 | 18.8% | 11.6% | 7.2% | 76.9% | 30% |

**Key finding:** The influence-usefulness gap persists across all passage counts.
Logprob importance stays frequent while factual usefulness remains sparse.
Factual usefulness rises only when baseline accuracy rises (more positive docs present).

---

## E6: Ranking Comparison

Evaluated on 11 queries with ≥1 factually useful doc.

| Method | P@1 | MRR | Spearman r |
|--------|-----|-----|-----------|
| Logprob-only | 0.727 | 0.826 | 0.063 |
| Factuality-aware | 1.000 | 1.000 | 0.638 |
| Harmfulness-aware | 0.455 | 0.652 | -0.633 |

---


---

## Recommended Figures and Tables for Final Report

| File | Where to insert |
|------|-----------------|
| plots/E1_cdr_by_passage_num.png | §LODO Mechanistic Evaluation — Collapse Divergence |
| tables/E1_collapse_divergence_rates.tex | §LODO — CDR table |
| plots/E2_taxonomy_stacked_by_passage_num.png | §Taxonomy — stacked bar |
| tables/E2_taxonomy_by_passage_num.tex | §Taxonomy — passage breakdown |
| tables/E2_taxonomy_by_baseline_correctness.tex | §Taxonomy — baseline correctness |
| plots/E2_logprob_vs_fact_taxonomy.png | §Taxonomy — scatter |
| plots/E3_token_type_logprob_drop_boxplot.png | §Token-level locality |
| plots/E4_layerwise_drift_by_taxonomy.png | §Integration Horizon |
| tables/E4_mechanistic_tests.tex | §Integration Horizon — statistical tests |
| plots/E4_full_33layer_drift_case_studies.png | §Integration Horizon — full curve |
| plots/E5_influence_usefulness_gap_by_passage.png | §Passage sweep |
| tables/E5_passage_gap_metrics.tex | §Passage sweep — gap table |
| tables/E6_ranking_metrics.tex | §Ranking comparison |
| plots/E6_precision_by_ranking_method.png | §Ranking comparison — bar |