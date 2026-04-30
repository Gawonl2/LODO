# Factuality-Aware Extension Experiments: Summary

Generated: 2026-04-30 00:42:32
Output directory: /home/gawon/lodo/RGB-master/results/factuality_aware_extensions_20260430_004226

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

**CDR (overall): 98.6%** — Among all documents that caused strong
logprob collapse (Δ_logprob < -2.0), this fraction had *zero* factual degradation.
This directly shows that logprob-only attribution marks the majority of
'influential' documents as important even when they are factually irrelevant.

| Group | CDR | HCR | HMCR |
|-------|-----|-----|------|
| Overall (all sources) | 98.6% | 0.0% | 1.4% |
| Sweep only | 98.5% | 0.0% | 1.5% |
| passage_num=3 | 100.0% | 0.0% | 0.0% |
| passage_num=5 | 100.0% | 0.0% | 0.0% |
| passage_num=7 | 95.7% | 0.0% | 4.3% |
| passage_num=10 | 100.0% | 0.0% | 0.0% |

---

## E2: Importance Taxonomy

Distribution across all ablations:

| Category | Count | % |
|----------|-------|---|
| neutral | 379 | 83.1% |
| confidence-only | 69 | 15.1% |
| factuality-weak | 5 | 1.1% |
| fact-only | 2 | 0.4% |
| factuality-disrupting | 1 | 0.2% |

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
| confidence-only | 15.61 |
| fact-only | 5.58 |
| harmful-influential | 12.90 |
| harmful-weak | 7.58 |
| neutral | 5.90 |

Full 33-layer trajectory available in plots/E4_full_33layer_drift_case_studies.png.
The integration horizon (layers 18–20) is visible in both helpful and confidence-only cases,
but the magnitude differs — see tables/E4_mechanistic_tests.tex for Mann-Whitney results.

---

## E5: Passage-Number Dependence of Influence-Usefulness Gap

| n | logprob-imp rate | fact-useful rate | gap | CDR | baseline acc |
|---|-----------------|-----------------|-----|-----|-------------|
| 3 | 53.6% | 0.0% | 53.6% | 100.0% | 0% |
| 5 | 30.0% | 0.0% | 30.0% | 100.0% | 0% |
| 7 | 46.0% | 0.0% | 46.0% | 95.7% | 0% |
| 10 | 27.0% | 1.6% | 25.4% | 100.0% | 10% |

**Key finding:** The influence-usefulness gap persists across all passage counts.
Logprob importance stays frequent while factual usefulness remains sparse.
Factual usefulness rises only when baseline accuracy rises (more positive docs present).

---

## E6: Ranking Comparison

Evaluated on 1 queries with ≥1 factually useful doc.

| Method | P@1 | MRR | Spearman r |
|--------|-----|-----|-----------|
| Logprob-only | 0.000 | 0.333 | 0.018 |
| Factuality-aware | 1.000 | 1.000 | 0.503 |
| Harmfulness-aware | 0.000 | 0.200 | -0.612 |

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