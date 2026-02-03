# Model Optimization via Structured Pruning and Multi-Task Fine-Tuning

## Overview

Modern neural language models achieve strong performance but often come at the cost of **millions to billions of parameters**, making them expensive to train, deploy, and maintain. Smaller models are more resource-efficient, but typically suffer from lower robustness and accuracy.

This project explores **model optimization strategies** that reduce computational and memory costs **while preserving downstream task performance**, using:

- Multi-task fine-tuning across diverse NLP benchmarks
- Structured pruning (one-shot and iterative)
- Standardized evaluation via LM Evaluation Harness

---

## Motivation

**Why model optimization?**

- Large models → high memory, compute, and deployment costs  
- Small models → faster and cheaper, but often less robust  
- Goal: **bridge the gap** by pruning and fine-tuning models to retain performance with fewer parameters

---

## Datasets

We fine-tuned models on a **combined multi-task dataset (~3.5M rows)** to improve generalization and reduce per-task training time.

### Tasks Included

| Dataset | Task Type |
|------|---------|
| **SIQA** | Social commonsense reasoning |
| **PIQA** | Physical commonsense reasoning |
| **OpenBookQA** | Elementary science QA |
| **HotpotQA** | Multi-hop question answering |
| **BillSum** | Long-document summarization |

This combined training setup:
- Increased robustness across tasks
- Reduced the need for dataset-specific fine-tuning
- Improved training efficiency

---

## Evaluation Framework

We use **LM Evaluation Harness**, a widely adopted evaluation suite for language models.

**Why LM Eval Harness?**
- 60+ academic benchmarks
- Used by Hugging Face Open LLM Leaderboard
- Adopted by NVIDIA, Cohere, BigScience, Mosaic ML, and others
- Ensures standardized and reproducible evaluation

---

## Baseline vs Fine-Tuned Performance

### Table 1: Baseline and Fine-Tuned Accuracies

| Model | PIQA | OpenBookQA | MMLU | HellaSwag | GLUE |
|-----|-----|------------|------|-----------|------|
| GPT-2 | 0.6289 | 0.1640 | 0.2292 | 0.2892 | 0.4658 |
| Fine-Tuned GPT-2 | 0.6371 | 0.1600 | 0.2305 | 0.2902 | 0.4459 |
| OLMO-1B | 0.7067 | 0.2340 | 0.2572 | 0.4137 | 0.4809 |
| Fine-Tuned OLMO-1B | **0.7503** | **0.2500** | 0.2446 | **0.4697** | 0.4536 |

**Key takeaway:**  
Multi-task fine-tuning improves overall task robustness, especially for larger models like OLMO-1B.

---

## Pruning Methodology

### GPT-2 Pruning

We applied **structured pruning** to GPT-2 using two approaches:

#### One-Shot Structured Pruning
- Pruned once, then fine-tuned on WikiText-103
- **47% model sparsity**
- Validation perplexity: **166.41**

#### Iterative Structured Pruning
- Gradual pruning over multiple iterations
- 3 epochs of training per iteration
- **46% model sparsity**
- Validation perplexity: **121.41**

✅ Iterative pruning achieved **lower perplexity at similar sparsity**, indicating better retention of language modeling ability.

---

### OLMO-1B Pruning

- One-shot structured pruning
- Applied to **MLP linear layers**
- Fine-tuned on a 300K-row Wikipedia subset
- Achieved **50% model sparsity**

---

## Pruning Evaluation Results

### Table 2: GPT-2 Pruned Performance

| Model | PIQA | OpenBookQA | MMLU | HellaSwag | GLUE |
|-----|-----|------------|------|-----------|------|
| Fine-Tuned GPT-2 | 0.6371 | 0.1600 | 0.2305 | 0.2902 | 0.4459 |
| Pruned + Fine-Tuned GPT-2 (46%) | 0.5386 | 0.1580 | **0.2312** | 0.2660 | **0.5280** |
| Directly Pruned + Fine-Tuned GPT-2 | 0.5312 | 0.1423 | 0.2213 | 0.2621 | 0.4921 |

**Observation:**  
Even at ~46% sparsity, GPT-2 retains competitive performance and improves on GLUE after pruning + fine-tuning.

---

### Table 3: OLMO-1B Pruned Performance

| Model | PIQA | OpenBookQA | MMLU | HellaSwag | GLUE |
|-----|-----|------------|------|-----------|------|
| Fine-Tuned OLMO-1B | 0.7503 | 0.2500 | 0.2446 | 0.4697 | 0.4536 |
| Pruned + Fine-Tuned OLMO-1B (50%) | 0.5386 | 0.1580 | 0.2312 | 0.2660 | **0.5280** |

---

## Key Findings

- Multi-task fine-tuning improves robustness and reduces per-dataset training time
- Structured pruning achieves **~50% parameter reduction** with acceptable performance loss
- Iterative pruning outperforms one-shot pruning in language modeling quality
- Pruned models remain competitive on reasoning and classification benchmarks
- Significant **memory and deployment efficiency gains** without catastrophic degradation

---

## Future Work

- Explore mixed-granularity pruning (attention + MLP)
- Combine pruning with quantization
- Extend to instruction-tuned and RAG-based models
- Evaluate inference latency and memory footprint directly

---

## References

- Kornilova & Eidelman (2019). *BillSum: A Corpus for Automatic Summarization of US Legislation*
- LM Evaluation Harness
- WikiText-103
- Hugging Face Open LLM Leaderboard
