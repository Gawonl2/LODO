# Factuality-Aware Extension Experiments: Summary

Generated: 2026-04-28 02:31:56
Output directory: /home/gawon/lodo/RGB-master/results/factuality_aware_extensions_20260428_023148

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

**CDR (overall): 92.7%** — Among all documents that caused strong
logprob collapse (Δ_logprob < -2.0), this fraction had *zero* factual degradation.
This directly shows that logprob-only attribution marks the majority of
'influential' documents as important even when they are factually irrelevant.

| Group | CDR | HCR | HMCR |
|-------|-----|-----|------|
| Overall (all sources) | 92.7% | 5.5% | 1.8% |
| Sweep only | 92.3% | 5.8% | 1.9% |
| passage_num=3 | 93.3% | 0.0% | 6.7% |
| passage_num=5 | 90.9% | 9.1% | 0.0% |
| passage_num=7 | 91.7% | 8.3% | 0.0% |
| passage_num=10 | 92.9% | 7.1% | 0.0% |
| doc_type=negative | 90.0% | 10.0% | 0.0% |
| doc_type=positive | 93.8% | 3.1% | 3.1% |

---

## E2: Importance Taxonomy

Distribution across all ablations:

| Category | Count | % |
|----------|-------|---|
| neutral | 439 | 85.6% |
| confidence-only | 51 | 9.9% |
| fact-only | 14 | 2.7% |
| harmful-weak | 5 | 1.0% |
| helpful-important | 3 | 0.6% |
| harmful-influential | 1 | 0.2% |

**Key finding:** 'Confidence-only' documents (logprob collapses but fact unchanged)
are far more common than 'helpful-important' documents. This confirms that
logprob-based attribution inflates apparent document importance.

---

## E3: Fact-Token vs Whole-Sequence Logprob Drop

Analyzed 10 case studies (token-level data only available for 10 cases).

Factual token fraction: 48.1%

**Limitation:** n=10 is insufficient for robust AUROC. Results are indicative.
Full token-level extraction across all sweep ablations would require re-running inference.

---

## E4: Mechanistic Signatures by Category

Layer-32 drift by category (mean across sweep ablations):

| Category | Mean L2@32 |
|----------|-----------|
| confidence-only | 18.61 |
| fact-only | 8.07 |
| harmful-influential | 27.41 |
| harmful-weak | 5.43 |
| helpful-important | 19.95 |
| neutral | 6.01 |

Full 33-layer trajectory available in plots/E4_full_33layer_drift_case_studies.png.
The integration horizon (layers 18–20) is visible in both helpful and confidence-only cases,
but the magnitude differs — see tables/E4_mechanistic_tests.tex for Mann-Whitney results.

---

## E5: Passage-Number Dependence of Influence-Usefulness Gap

| n | logprob-imp rate | fact-useful rate | gap | CDR | baseline acc |
|---|-----------------|-----------------|-----|-----|-------------|
| 3 | 50.0% | 0.0% | 50.0% | 93.3% | 0% |
| 5 | 22.0% | 2.0% | 20.0% | 90.9% | 10% |
| 7 | 17.6% | 7.4% | 10.3% | 91.7% | 10% |
| 10 | 15.6% | 11.1% | 4.4% | 92.9% | 20% |

**Key finding:** The influence-usefulness gap persists across all passage counts.
Logprob importance stays frequent while factual usefulness remains sparse.
Factual usefulness rises only when baseline accuracy rises (more positive docs present).

---

## E6: Ranking Comparison

Evaluated on 4 queries with ≥1 factually useful doc.

| Method | P@1 | MRR | Spearman r |
|--------|-----|-----|-----------|
| Logprob-only | 0.500 | 0.750 | 0.039 |
| Factuality-aware | 1.000 | 1.000 | 0.723 |
| Harmfulness-aware | 0.250 | 0.550 | -0.423 |

---

## E7: Position-Controlled LODO

**Status: TODO** — requires model inference.
See results/E7_position_controlled_lodo.csv for run instructions.

Implementation is ready in `run_lodo_position_controlled.py`.

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