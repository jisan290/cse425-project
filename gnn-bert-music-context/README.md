# GNN-Based BERT for Understanding Context from Music

CSE425 / EEE474 / CSE715 — Neural Networks — Group Project

A hybrid BERT + Graph Neural Network system for musical context
understanding on MagnaTagATune. A DistilBERT classifier maps tags/captions
to a 50-tag multi-label prediction; a 3-layer GraphSAGE network maps a
per-track segment graph (nodes = 5-second audio windows, edges = temporal
adjacency + MFCC/chroma cosine similarity) to the same label space; and a
cross-attention fusion model combines both.

## Team

| Member | Task | Key files |
|---|---|---|
| A | Task 1 — BERT tag classifier | `src/bert_encoder.py` |
| B | Task 2 — GNN on segment graphs | `src/audio_features.py`, `src/graph_builder.py`, `src/gnn_model.py` |
| C | Task 3 — GNN-BERT fusion | `src/fusion_model.py` |
| D | Task 4 (bonus), report, repo integration | `src/contrastive.py`, `train.py`, `evaluate.py` |

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU for training in a reasonable time (all models were
developed on a Kaggle T4).

## Data

MagnaTagATune is downloaded automatically via `kagglehub` on first run
(set `dataset.kagglehub_slug` in `config.yaml` to your dataset's
`owner/slug`, found in its Kaggle URL) — or attach it manually as a
Kaggle input and place the resulting CSVs under `data/processed/`.

**Known constraint:** the Kaggle mirror used for this project only has
audio files for 133 of the 25,863 annotated clips (108/14/11 train/val/test).
Task 1 (text-only) trains on the full 25,863-clip corpus; Tasks 2-4 (which
need audio-derived graphs) are restricted to the 133-clip subset. This is
documented in `report/final_report.pdf` — Task 2/3/4 results should be read
as a proof-of-concept given the n=11 test set, not a statistically powered
comparison.

## Running

```bash
python train.py --task 1      # BERT tag classifier
python train.py --task 2      # builds graphs, trains GraphSAGE + CNN baseline
python train.py --task 3      # fusion model + 4-way ablation
python train.py --task 4      # contrastive retrieval + zero-shot (bonus)
python train.py --task all    # runs all four in order

python evaluate.py            # prints the cross-task comparison table
```

Task 2 must run before Tasks 3 and 4, since it builds the cached segment
graphs in `data/processed/task2_graphs/` that both depend on.

## Repository structure

```
gnn-bert-music-context/
├── README.md, requirements.txt, config.yaml
├── data/{raw, processed, splits}/
├── notebooks/{eda.ipynb, demo_context.ipynb}
├── src/
│   ├── audio_features.py     # audio loading + MFCC/chroma segment features
│   ├── graph_builder.py      # graph construction, caching, PyG dataset
│   ├── bert_encoder.py       # Task 1 model + training loop
│   ├── gnn_model.py          # Task 2 GraphSAGE + CNN baseline
│   ├── fusion_model.py       # Task 3 cross-attention fusion + ablation
│   └── contrastive.py        # Task 4 dual-encoder + zero-shot (bonus)
├── train.py, evaluate.py
├── results/{metrics per task, plots/, retrieval_examples/}
└── report/final_report.pdf
```

## Honesty notes (see report for full discussion)

- Task 1's near-perfect score is a measurement artifact: MagnaTagATune has
  no natural-language captions, so the caption fed to BERT is literally
  built from the same tags used as labels. A masking ablation (in
  `src/bert_encoder.py`'s training pipeline / Section 4.3 of the report)
  quantifies this directly.
- With an 11-clip audio test set, no Task 2/3/4 metric should be treated
  as more precise than roughly ±1 clip.
