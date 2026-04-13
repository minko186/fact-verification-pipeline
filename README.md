# Fact Verification Pipeline

A three-stage pipeline for automated fact-checking using retrieval-augmented verification. Given raw text, the system extracts atomic claims, retrieves supporting or refuting evidence from a knowledge base, and classifies each claim as **SUPPORTS**, **REFUTES**, or **NOT ENOUGH INFO**.

## Pipeline Overview

```
Raw Text
  |
  v
[Stage 1: Fact Extraction]         FLAN-T5 seq2seq model
  |  "Einstein was born in Ulm"
  v
[Stage 2: Evidence Retrieval]       BM25 + Dense + Graph -> RRF -> NLI Reranker
  |  "Albert Einstein was born on 14 March 1879 in Ulm, Germany."
  v
[Stage 3: Fact Verification]        DeBERTa NLI classifier
  |  SUPPORTS (0.94)
  v
Final Verdict
```

## Project Structure

```
fact-verification-pipeline/
|
+-- 01_fact_extraction/             # Stage 1: Extract atomic claims from text
|   +-- src/
|   |   +-- config.yaml
|   |   +-- prepare.py              # Tokenize datasets -> Arrow files
|   |   +-- train.py                # Fine-tune FLAN-T5 (seq2seq)
|   |   +-- run_evaluation.py       # ROUGE + BERTScore evaluation
|   |   +-- inference.py            # Generate claims from evidence
|   |   +-- preprocessors/          # Per-dataset preprocessing
|   |   |   +-- preprocess_fever.py
|   |   |   +-- preprocess_vitaminc.py
|   |   |   +-- preprocess_wice.py
|   |   |   +-- preprocess_scifact.py
|   |   |   +-- preprocess_claimdecomp.py
|   |   +-- preprocess_fever_v1.py   # Legacy preprocessors
|   |   +-- preprocess_fever_v2.py
|   +-- notebooks/
|       +-- exploration.ipynb
|
+-- 02_evidence_retrieval/          # Stage 2: Retrieve evidence for claims
|   +-- src/
|   |   +-- config.yaml
|   |   +-- build_index.py          # Build all knowledge base indexes
|   |   +-- retrieve.py             # Retrieve evidence for claims
|   |   +-- evaluate_retrieval.py   # Recall@k and MRR evaluation
|   |   +-- kb/                     # Knowledge base modules
|   |   |   +-- parse_wiki.py       # Parse FEVER wiki-pages.zip
|   |   |   +-- bm25_index.py       # Sparse lexical index (BM25)
|   |   |   +-- dense_index.py      # Dense semantic index (ChromaDB)
|   |   |   +-- graph_index.py      # Entity co-occurrence graph (NetworkX)
|   |   +-- retrieval/              # Retrieval pipeline modules
|   |       +-- fusion.py           # Reciprocal Rank Fusion (RRF)
|   |       +-- reranker.py         # NLI cross-encoder reranker
|   |       +-- pipeline.py         # End-to-end retrieval orchestrator
|   +-- notebooks/
|       +-- exploration.ipynb
|
+-- 03_fact_verification/           # Stage 3: Classify claims as S/R/NEI
|   +-- src/
|       +-- config.yaml
|       +-- prepare.py              # Tokenize (claim, evidence) pairs -> Arrow
|       +-- train.py                # Fine-tune NLI classifier
|       +-- evaluate.py             # Accuracy, F1, confusion matrix
|       +-- inference.py            # Predict verdicts from retrieval output
|       +-- preprocessors/          # Per-dataset NLI preprocessing
|           +-- preprocess_fever_nli.py
|           +-- preprocess_vitaminc_nli.py
|           +-- preprocess_anli.py
|
+-- data/
|   +-- raw/                        # Unprocessed source data
|   |   +-- fever/                  # FEVER dataset (train.jsonl, wiki-pages.zip)
|   |   +-- HaluEval/              # HaluEval hallucination data
|   |   +-- LibreEval/             # LibreEval hallucination data
|   +-- processed/                  # Pipeline-generated artifacts
|       +-- fact_extraction/        # Tokenized Arrow datasets (stage 1)
|       +-- evidence_retrieval/     # Indexes, sentence records (stage 2)
|       +-- fact_verification/      # Tokenized datasets & results (stage 3)
|
+-- models/                         # Trained model checkpoints
+-- experiments/                    # W&B runs, eval result JSONs
+-- shared/                         # Shared utilities
|   +-- utils/
|       +-- cleaning.py             # Text cleaning (HTML, URLs, emoji removal)
|       +-- content_hash.py         # SHA-256 hashing for deduplication
|       +-- hc3.py                  # HC3 dataset conversion utility
|
+-- docs/                           # Documentation, diagrams, reports, posters
+-- notebooks/                      # Top-level exploration notebooks
+-- plots/                          # Generated visualizations
```

## Requirements

```bash
pip install torch transformers datasets evaluate sentencepiece
pip install sentence-transformers chromadb rank_bm25 networkx
pip install scikit-learn beautifulsoup4 pyyaml tqdm
pip install huggingface_hub wandb
```

A CUDA-capable GPU is recommended for training and dense index construction.

## Data Setup

The pipeline uses the [FEVER](https://fever.ai/) dataset as its primary data source.

1. **FEVER claims** are loaded automatically from HuggingFace Hub via the `datasets` library
2. **FEVER wiki-pages.zip** (1.6 GB) must be placed at `data/raw/fever/wiki-pages.zip` for the evidence retrieval knowledge base. Download from the [FEVER shared task](https://fever.ai/resources.html) or the HuggingFace dataset page

---

## Stage 1: Fact Extraction

Extracts atomic claims from evidence text using a fine-tuned FLAN-T5 seq2seq model. The model is prompted with `"extract fact: {evidence} Based on this, what is the claim?"` and generates a natural-language claim.

### Supported Datasets

| Dataset | Preprocessor | Description |
|---------|-------------|-------------|
| FEVER | `preprocess_fever.py` | Wikipedia-grounded claims (SUPPORTS only) |
| VitaminC | `preprocess_vitaminc.py` | Adversarial Wikipedia edits |
| WiCE | `preprocess_wice.py` | Fine-grained sub-sentence entailment |
| SciFact | `preprocess_scifact.py` | Biomedical claims from PubMed |
| ClaimDecomp | `preprocess_claimdecomp.py` | Complex claims decomposed to atomic subclaims |

All preprocessors normalize data to `{"evidence": str, "claim": str}` and push to HuggingFace Hub.

### Configuration

Edit `01_fact_extraction/src/config.yaml` to configure:
- **Model**: HuggingFace model name and Hub ID for pushing
- **Datasets**: List of HF Hub dataset IDs with optional sampling
- **Training args**: Batch size, learning rate, epochs, eval/save steps
- **Prompt template**: The instruction format for the seq2seq model

### Usage

```bash
cd 01_fact_extraction/src

# 1. Preprocess raw data and push to HF Hub
python preprocessors/preprocess_fever.py --repo minko186/fever-fact-extraction-supports

# 2. Tokenize datasets for training
python prepare.py

# 3. Train the model
python train.py

# 4. Evaluate (ROUGE + BERTScore)
python run_evaluation.py

# 5. Run inference on new text
python inference.py
```

### Output

- Trained model pushed to HuggingFace Hub
- Evaluation JSON report saved to `experiments/eval_results/`
- Metrics: ROUGE-1/2/L, BERTScore (precision, recall, F1)

---

## Stage 2: Evidence Retrieval

Retrieves evidence sentences from a Wikipedia knowledge base that can either support or refute a given claim. Uses a two-stage hybrid retrieval approach.

### Architecture

**Knowledge Base Construction** — Three co-indexed stores built from FEVER's `wiki-pages.zip`:

| Index | Library | Purpose |
|-------|---------|---------|
| BM25 (sparse) | `rank_bm25` | Lexical matching — good for names, dates, numbers |
| Dense (semantic) | `sentence-transformers` + ChromaDB | Semantic similarity — catches paraphrases |
| Graph (structural) | NetworkX | Entity co-occurrence — finds related articles via BFS |

**Two-Stage Retrieval**:

1. **Stage 1 — Hybrid candidate generation**: Each of the three indexes independently returns top-100 candidates for a claim. Results are merged using **Reciprocal Rank Fusion (RRF)** — a score-agnostic method that handles the incompatible scales of BM25 scores, cosine distances, and hop depths. Top 200 candidates are passed to the reranker.

2. **Stage 2 — NLI reranking**: A cross-encoder NLI model (`cross-encoder/nli-deberta-v3-base`) scores each `(claim, candidate)` pair. The key insight is **stance-neutral scoring**: relevance = P(ENTAILMENT) + P(CONTRADICTION). This ensures both supporting and refuting evidence ranks above unrelated text — only truly neutral (off-topic) evidence is filtered out.

### Configuration

Edit `02_evidence_retrieval/src/config.yaml` to configure:
- **Corpus**: Path to wiki-pages.zip, sentence records output path
- **Indexes**: Storage paths for BM25, ChromaDB, and GraphML files
- **Embedding model**: Sentence-transformer model name and batch size
- **Retrieval**: Top-k per channel, graph hop depth
- **Fusion**: RRF k parameter, candidate pool size
- **Reranker**: NLI model, batch size, final top-n evidence count

### Usage

```bash
cd 02_evidence_retrieval/src

# 1. Build all knowledge base indexes (skip flags available)
python build_index.py
python build_index.py --skip-parse --skip-bm25   # rebuild only dense + graph

# 2. Retrieve evidence for a single claim
python retrieve.py --claim "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."

# 3. Retrieve evidence for a batch of claims
python retrieve.py --input claims.jsonl --output results/

# 4. Evaluate retrieval quality against FEVER gold evidence
python evaluate_retrieval.py
python evaluate_retrieval.py --sample 1000   # evaluate on a random subset
```

### Output

JSONL file where each line contains:

```json
{
  "claim": "Nikolaj Coster-Waldau worked with Fox Broadcasting Company.",
  "evidence": ["Nikolaj Coster-Waldau appeared in the Fox pilot...", "..."],
  "evidence_ids": ["Nikolaj_Coster-Waldau_7", "Fox_Broadcasting_Company_0"],
  "reranker_scores": [0.94, 0.87],
  "source_channels": [["bm25", "dense"], ["dense", "graph"]]
}
```

This JSONL is consumed directly by Stage 3.

Evaluation metrics: Recall@5, Recall@10, Recall@20, Mean Reciprocal Rank (MRR).

---

## Stage 3: Fact Verification

Classifies each `(claim, evidence)` pair as **SUPPORTS**, **REFUTES**, or **NOT ENOUGH INFO** using a fine-tuned NLI sequence classification model.

### Supported Models

The config includes five pretrained models selectable via `--model-key`:

| Key | Model | Params | Notes |
|-----|-------|--------|-------|
| `deberta-base` | `microsoft/deberta-v3-base` | 86M | Strong baseline, fits on consumer GPU |
| `deberta-large` | `microsoft/deberta-v3-large` | 304M | Higher accuracy, needs ~16GB VRAM |
| `deberta-mnli-fever` | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | 86M | Pre-trained on FEVER+MNLI+ANLI, best zero-shot |
| `roberta-mnli` | `roberta-large-mnli` | 355M | Classic NLI workhorse |
| `cross-encoder-nli` | `cross-encoder/nli-deberta-v3-base` | 86M | Same model as stage 2 reranker |

### Supported Datasets

| Dataset | Preprocessor | Description |
|---------|-------------|-------------|
| FEVER | `preprocess_fever_nli.py` | Wikipedia claims — all 3 labels (S/R/NEI) |
| VitaminC | `preprocess_vitaminc_nli.py` | Adversarial Wikipedia edits — all 3 labels |
| ANLI | `preprocess_anli.py` | Adversarial NLI (R1+R2+R3) — mapped to S/R/NEI |

All preprocessors normalize data to `{"claim": str, "evidence": str, "label": str}` and push to HuggingFace Hub.

### Configuration

Edit `03_fact_verification/src/config.yaml` to configure:
- **Models**: Multiple model configs with HF model name, Hub ID, and max sequence length
- **Datasets**: List of HF Hub NLI dataset IDs with optional sampling
- **NEI strategy**: How to handle NOT ENOUGH INFO claims (`"empty"` or `"random"` evidence)
- **Training args**: Batch size, learning rate, epochs, early stopping via `metric_for_best_model`
- **Inference**: Path to stage 2 retrieval output, evidence join strategy

### Usage

```bash
cd 03_fact_verification/src

# 1. Preprocess FEVER for NLI (all 3 labels) and push to Hub
python preprocessors/preprocess_fever_nli.py --repo minko186/fever-nli
python preprocessors/preprocess_vitaminc_nli.py --repo minko186/vitaminc-nli   # optional
python preprocessors/preprocess_anli.py --repo minko186/anli-nli               # optional

# 2. Tokenize datasets for a specific model
python prepare.py --model-key deberta-base

# 3. Train the model
python train.py --model-key deberta-base

# 4. Evaluate (accuracy, F1, confusion matrix)
python evaluate.py --model-key deberta-base

# 5a. Run inference on stage 2 retrieval output (batch mode)
python inference.py --model-key deberta-base --input ../../data/processed/evidence_retrieval/results/fever-wiki-hybrid-v1.jsonl

# 5b. Run inference on a single claim (interactive mode)
python inference.py --model-key deberta-base \
  --claim "Einstein was born in Germany" \
  --evidence "Albert Einstein was born on 14 March 1879 in Ulm, Germany."
```

### Output

**Evaluation** — JSON report saved to `experiments/eval_results/`:

```json
{
  "accuracy": 0.89,
  "macro_f1": 0.87,
  "per_class": {
    "SUPPORTS":        {"precision": 0.91, "recall": 0.93, "f1": 0.92},
    "REFUTES":         {"precision": 0.88, "recall": 0.85, "f1": 0.86},
    "NOT ENOUGH INFO": {"precision": 0.82, "recall": 0.80, "f1": 0.81}
  },
  "confusion_matrix": [[...], [...], [...]],
  "confusion_labels": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
}
```

**Inference** — JSONL file with final verdicts:

```json
{
  "claim": "Einstein was born in Germany",
  "evidence": ["Albert Einstein was born on 14 March 1879 in Ulm, Germany."],
  "verdict": "SUPPORTS",
  "confidence": 0.94,
  "verdict_probs": {"SUPPORTS": 0.94, "REFUTES": 0.04, "NOT ENOUGH INFO": 0.02}
}
```

---

## End-to-End Pipeline

Running the full pipeline from scratch:

```bash
# ── Stage 1: Fact Extraction ─────────────────────────────────────────────────
cd 01_fact_extraction/src
python preprocessors/preprocess_fever.py --repo minko186/fever-fact-extraction-supports
python prepare.py
python train.py
python run_evaluation.py
python inference.py   # generates claims from evidence text

# ── Stage 2: Evidence Retrieval ──────────────────────────────────────────────
cd ../../02_evidence_retrieval/src
python build_index.py                             # build KB indexes (BM25, dense, graph)
python retrieve.py --input claims.jsonl           # retrieve evidence for claims
python evaluate_retrieval.py --sample 1000        # evaluate Recall@k

# ── Stage 3: Fact Verification ───────────────────────────────────────────────
cd ../../03_fact_verification/src
python preprocessors/preprocess_fever_nli.py --repo minko186/fever-nli
python prepare.py --model-key deberta-base
python train.py --model-key deberta-base
python evaluate.py --model-key deberta-base
python inference.py --model-key deberta-base --input ../../data/processed/evidence_retrieval/results/fever-wiki-hybrid-v1.jsonl
```

## Shared Utilities

Located in `shared/utils/`:

- **`cleaning.py`** — `remove_special_characters(text)`: strips HTML tags, URLs, emoji, special characters, and normalizes whitespace. Used by all preprocessors across all stages.
- **`content_hash.py`** — `generate_content_hash_id(text)`: SHA-256 hash for content deduplication.

## Experiment Tracking

- Training runs are logged via **Weights & Biases** (wandb) to `experiments/wandb/`
- Evaluation reports are saved as JSON to `experiments/eval_results/`
- Trained models are pushed to **HuggingFace Hub** for versioning and sharing
