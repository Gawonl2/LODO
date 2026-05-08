# Final Project Summary

## Project Title
**Influence Is Not Usefulness: Factuality-Aware Document Attribution in Retrieval-Augmented Generation**

## One-Sentence Summary
This project shows that in RAG systems, a retrieved document can strongly influence model confidence without actually changing factual correctness, so influence-based attribution alone is not enough.

---

## 1) Motivation and Core Question

Most RAG attribution methods rank documents by how much model behavior changes when a document is removed (for example, output likelihood drop).  
The project asks a stricter question:

> Does an influential document actually matter for factual correctness?

To answer this, the project separates **causal influence** from **factual impact** and evaluates them independently.

---

## 2) Main Contributions

1. **Conceptual contribution**  
   Formalized the gap between:
   - influence on model likelihood, and
   - factual effect on answer correctness.

2. **Methodological contribution**  
   Built a **factuality-aware LODO (Leave-One-Document-Out)** diagnostic framework with three signals:
   - factual degradation (`Δfact`)
   - log-probability collapse (`Δlogprob`)
   - representation drift (`ΔL2`)

3. **Empirical contribution**  
   In a controlled mixed retrieval setting, demonstrated that many likelihood-influential documents are not factually useful.

---

## 3) Experimental Setup

### Data Setting
- Benchmark base: **RGB**
- Subsets used:
  - `en_mid` (documents support correct answers)
  - `en_counter_mid` (counterfactual documents with wrong facts)

### Mixed Retrieval Construction
For each query and passage count `n in {3, 5, 7, 10}`:
- 1 ground-truth (GT) document from `en_mid`
- `n-1` counterfactual (CF) documents from `en_counter_mid`
- documents shuffled before prompting

This creates a controlled stress-test where correct evidence exists, but misleading evidence dominates.

### Model and Evaluation
- Model: **Llama-3.1-8B-Instruct**
- Generation temperature: `0.7`
- `Δlogprob` and `ΔL2` computed with teacher forcing for deterministic measurement
- LODO influence threshold: `τ = 2.0` (severe likelihood collapse cutoff)

### Scale
- 10 shared queries
- 198 total LODO ablations in the final mixed sweep

---

## 4) Code and Script Guide

### Repository Layout (What to use)
- Main project code and outputs: `RGB-master/`
- Core model wrappers: `RGB-master/models/models.py`
- Main experiment orchestrator: `RGB-master/scripts/run_all_factuality_aware_experiments.py`
- Main outputs:
  - figures: `RGB-master/plots/`
  - experiment bundles: `RGB-master/results/factuality_aware_extensions_*/`

### Core Scripts by Purpose
- **Main mixed-setting LODO sweep**
  - `RGB-master/run_lodo_passage_sweep.py`
  - Builds the mixed context (1 ground-truth + n-1 counterfactual docs), runs LODO ablations, and writes:
    - `lodo_passage_sweep_mixed_llama3.json`

- **Position-controlled LODO (artifact control)**
  - `RGB-master/run_lodo_position_controlled.py`
  - Replaces removed docs with placeholders to separate semantic removal from positional shift artifacts.
  - Writes:
    - `lodo_position_controlled_<dataset>_<model>.json`

- **Factuality-aware extension experiments (E1-E7)**
  - `RGB-master/scripts/run_all_factuality_aware_experiments.py`
  - Runs analysis pipeline and produces:
    - `analysis_table.csv`
    - per-experiment CSV/TEX summaries
    - plots
    - `experiment_summary.md`

- **Per-experiment analysis scripts**
  - `RGB-master/scripts/experiment_1_collapse_divergence.py`
  - `RGB-master/scripts/experiment_2_taxonomy.py`
  - `RGB-master/scripts/experiment_3_fact_token_logprob.py`
  - `RGB-master/scripts/experiment_4_mechanistic_signatures.py`
  - `RGB-master/scripts/experiment_5_passage_gap.py`
  - `RGB-master/scripts/experiment_6_ranking_comparison.py`
  - `RGB-master/scripts/experiment_7_position_controlled.py`
  - `RGB-master/scripts/build_analysis_table.py`

- **Visualization scripts**
  - `RGB-master/visualize_counterfactual.py`
  - `RGB-master/visualize_passage_sweep.py`
  - `RGB-master/visualize_lodo.py`
  - `RGB-master/visualize_case_study.py`

### Recommended Run Order
Run from `RGB-master/`:

1. **Generate mixed LODO sweep**
   - `python run_lodo_passage_sweep.py --modelname llama3 --passage_nums 3 5 7 10 --max_queries 10`

2. **Run full factuality-aware analysis bundle (E1-E7)**
   - `python scripts/run_all_factuality_aware_experiments.py --input_sweep lodo_passage_sweep_mixed_llama3.json --input_refine lodo_results_en_refine_llama3.json`

3. **(Optional) Run position-controlled LODO**
   - `python run_lodo_position_controlled.py --dataset en_counter_mid --modelname llama3 --max_queries 10`

4. **Regenerate plots**
   - `python visualize_counterfactual.py`
   - `python visualize_passage_sweep.py`

### Legacy / Supporting Scripts
- `RGB-master/evalue.py`: baseline RGB evaluation pipeline.
- `RGB-master/fact_evalue.py`, `RGB-master/reject_evalue.py`: fact-checking/rejection evaluation.
- `RGB-master/run_lodo_experiments.py`: earlier LODO run used for initial mechanistic outputs.
- `RGB-master/run_detailed_case_study.py`: detailed token/layer-level case extraction.

---

## 5) Method: Factuality-Aware LODO

For each query:
1. Generate baseline answer with full context.
2. Remove one document at a time.
3. Recompute:
   - factual change (`Δfact`)
   - baseline-answer likelihood drop (`Δlogprob`)
   - layer-wise hidden-state drift (`ΔL2`)
4. Assign each ablated document to a taxonomy category.

### Taxonomy (Influence x Factuality)
- **Factuality-critical**: high influence, hurts factuality when removed
- **Confidence-only**: high influence, no factual change
- **Factuality-disrupting**: high influence, factuality improves when removed
- **Fact-only**, **Neutral**, **Factuality-weak** for non-collapsing cases

---

## 6) Key Findings

### Finding 1: Most influential documents are not factuality-impacting
- Likelihood-influential documents (`Δlogprob < -2.0`): **68 / 198**
- Among those 68:
  - **Confidence-only (CDR): 63.2%**
  - **Factuality-critical (FCR): 23.5%**
  - **Factuality-disrupting (FDCR): 13.2%**

Interpretation: large likelihood collapse often means the model reacted, not that factual correctness changed.

### Finding 2: GT and CF documents have different factual roles
- For likelihood-influential **GT** docs:
  - Factuality-critical: **62.5%**
  - Confidence-only: **37.5%**
- For likelihood-influential **CF** docs:
  - Confidence-only: **71.2%**
  - Factuality-critical: **11.5%**
  - Factuality-disrupting: **17.3%**

Interpretation: the same influence signal can correspond to opposite factual roles.

### Finding 3: The influence-factuality gap narrows with more passages, but remains
- As passage count increases, single-document influence weakens (dilution effect).
- Example trend:
  - LI rate drops from **70.0% (n=3)** to **18.8% (n=10)**
- But factuality-impacting documents remain much rarer, so mismatch persists.

### Finding 4: Representation drift also cannot determine factual direction
- Late-layer drift is large for factuality-critical, confidence-only, and factuality-disrupting groups.
- Neutral group has much lower drift.

Interpretation: drift is a magnitude-of-influence signal, not a direction-of-factual-effect signal.

---

## 7) Project-Level Conclusion

The project supports one central claim:

> **Influence is not usefulness.**

Likelihood collapse and hidden-state drift reliably indicate that a document affects model behavior, but they do not tell whether that effect helps, harms, or leaves factual correctness unchanged.  
Therefore, practical RAG attribution should explicitly separate:
- influence diagnostics, and
- factual impact diagnostics.

---

## 8) Limitations

- LODO is post-hoc and computationally expensive (one ablation per document).
- `Δfact` depends on answer matching quality.
- Mixed retrieval setting is controlled and adversarial, not a full real-world distribution.
- One-document removal measures marginal effects, not full interaction effects.
- Experiments are limited to one model family and one benchmark setting.

---

## 9) Practical Output of This Project

The project delivers:
- a formal factuality-aware attribution framework,
- reproducible experimental scripts and result tables,
- empirical evidence that challenges likelihood-only attribution interpretation,
- and a practical diagnostic lens for auditing RAG evidence use.

In short, this project reframes attribution quality from  
**"Did the model react?"** to **"Did the document actually matter for factual correctness?"**

## 10) Use of AI
From one of my past researches, I was introduced to RGB benchmark. From then, I always wanted to find out what makes document really important. 
To make this a proper research question, I did quick literature review through **Gemini** and Google Scholar, finding how current methods (SOTA) define document importance. 
After, I used **chatGPT** to formulate my thoughts into proper and structured RQs. For example, I asked "What do you think about adding factuality component when defining Document Importance in RAG context?".
In this process, more detailed literature review was needed, so I deliberatly read and selected papers that I think it mattered the most. I used **notebookLM** for getting help with connecting them to my work. 

The project base is from RGB-master branch. Utilizing this source code, I used **Claude Code** to re-construct overall structure of my project folder where I explicitly prompted roles to each script needed. 
While I used Claude Code for big-picture, I used **Cursor** for editing individual .py files as it is in IDE format, making it easier for file-level editing. 

For writing reports, I used **ChatGPT** as a writing assistant. When I first start drafting a paper, I usually create the \section{} structure and build a high-level outline. Then, I rapidly write down my ideas in both Korean and English, moving back and forth across different sections rather than writing in a strictly chronological order. GPT helps me translate phrases that are in Korean, and finds parts where I need to make stronger connections between sections or sentences.
