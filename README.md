# Episodic Memories Generation and Evaluation Benchmark for Large Language Models

Synthetic data generation and benchmark implementation for "Episodic Memories Generation and Evaluation Benchmark for Large Language Models" [2024]

## Quickstart

Three Jupyter notebooks are available in `epbench/experiments/`:

- `step_1_generation.ipynb`: generation of the documents with two rounds of verifications, and extraction of the selected question/answer pairs,
- `step_2_answering.ipynb`: predicting the answers given the document and the questions, using in-context, RAG, or fine-tuned models, and perform the evaluations,
- `step_3_results.ipynb`: extract the results, including the CD plots and the summarizing tables.

## Data

Data that have been generated using steps 1 and 2 are available at this address: https://figshare.com/s/7b634eff