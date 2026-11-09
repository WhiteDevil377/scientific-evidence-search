# Measured retrieval results

Generated on 9 November 2026 by the bundled implementation. These are local research-benchmark results, not production measurements.

Corpus: **5,183 abstracts**. Method-selection set: **809 queries** from BEIR train judgments. Evaluation set: **300 queries** from BEIR's public test split.

Selected method: **hybrid_rrf**, chosen using validation nDCG@10 before calculating test scores. No encoder fine-tuning or test-guided hyperparameter search was performed.

## Validation

| Method | nDCG at 10 | Recall at 10 | MRR at 10 |
|---|---:|---:|---:|
| bm25 | 0.6683 | 0.7909 | 0.6357 |
| dense_minilm | 0.6602 | 0.7856 | 0.6261 |
| hybrid_rrf | 0.6968 | 0.8199 | 0.6636 |

## Test

| Method | nDCG at 10 | Recall at 10 | MRR at 10 |
|---|---:|---:|---:|
| bm25 | 0.6634 | 0.7876 | 0.6309 |
| dense_minilm | 0.6451 | 0.7833 | 0.6047 |
| hybrid_rrf | 0.6802 | 0.8139 | 0.6415 |

On this evaluation split, hybrid ranking improves nDCG@10 by **0.0168 absolute** over BM25. This is one small, public benchmark; statistical significance and performance on other corpora were not established.

## Local sequential query timing

50 validation queries; median **4.60 ms**, p95 **6.45 ms**. Includes query encoding, lexical/dense scoring, and fusion. Excludes model and index loading, indexing, downloads, and HTTP network overhead. Timings depend on the machine.

## Model and indexing

Frozen model: `sentence-transformers/all-MiniLM-L6-v2`.
Pinned revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
Embedding dimension: 384. Maximum sequence length: 256 wordpieces.
Truncated abstracts: 3,681.
Source archive SHA-256: `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.

## Environment

- numpy: 2.3.5
- scipy: 1.18.1
- scikit-learn: 1.9.1
- onnxruntime: 1.31.1
- tokenizers: 0.23.3
- joblib: 1.6.0
- threadpoolctl: 3.8.0

## Interpretation

The system retrieves relevant evidence documents; it does not predict whether the claim is supported or refuted. Rankings use the benchmark judgments, with unjudged documents treated as nonrelevant. Public benchmark and pretraining-overlap limitations are described in README.md. Per-query metrics are included so successes and failures can be inspected. No state-of-the-art, novel-algorithm, large-scale deployment, or causal-impact claim is made.

## Source and attribution

SciFact: Wadden et al. (2020), https://github.com/allenai/scifact. BEIR data and license: https://huggingface.co/datasets/BeIR/scifact, CC BY-SA 4.0. Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2, Apache 2.0.
