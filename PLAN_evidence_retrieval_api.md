# Plan: Host the 2-Stage Retrieval Knowledge Base on HuggingFace Hub

## Goal

Keep all three KB indexes (sparse BM25, dense FAISS, graph) online on HuggingFace Hub
so the pipeline can pull them to any machine and run evidence retrieval without re-building
from the raw FEVER wiki dump.

---

## What is stored where

| Artifact | File(s) | HF Repo |
|----------|---------|---------|
| Sentence records (text lookup) | `sentence_records.jsonl` | `minko186/fever-wiki-sentences` |
| Dense index (semantic retrieval) | `faiss_index.faiss`, `faiss_index_ids.pkl`, `faiss_index_meta.json` | `minko186/fever-evidence-retrieval-kb` |
| Sparse index (lexical retrieval) | `bm25_index.pkl` | `minko186/fever-evidence-retrieval-kb` |
| Graph index (structural retrieval) | `entity_graph.graphml`, `entity_graph_sentences.pkl` | `minko186/fever-evidence-retrieval-kb` |

---

## Workflow

### Build once, push to Hub

```bash
cd 02_evidence_retrieval/src

# Option A — build everything fresh, then upload
python build_index.py --push-to-hub

# Option B — skip rebuild (indexes already built locally), just upload
python build_index.py --skip-parse --skip-bm25 --skip-dense --skip-graph --push-to-hub
```

`--push-to-hub` calls `upload_to_hub()` which:
1. Creates the HF repos if they don't exist
2. Uploads each artifact with a descriptive commit message
3. Skips any file that's missing locally (non-fatal)

### Pull on any new machine

```bash
cd 02_evidence_retrieval/src
python build_index.py --from-hub
```

`--from-hub` calls `download_from_hub()` which downloads all four artifact groups in order,
skipping files already present on disk.

---

## Artifacts breakdown

### 1. Sentence records — `minko186/fever-wiki-sentences`

`sentence_records.jsonl` — one line per sentence parsed from FEVER `wiki-pages.zip`:
```json
{"sentence_id": "Albert_Einstein_0", "article_title": "Albert_Einstein", "text": "Albert Einstein was born on 14 March 1879 in Ulm, Germany."}
```
Used at retrieval time as the text lookup table for all three indexes.

### 2. Dense index — FAISS IVF-PQ (`minko186/fever-evidence-retrieval-kb`)

- `faiss_index.faiss` — compressed IVF-PQ index (sentence embeddings from `all-mpnet-base-v2`)
- `faiss_index_ids.pkl` — ordered list of sentence IDs matching FAISS internal positions
- `faiss_index_meta.json` — index hyperparameters (model, dim, nlist, m, nprobe)

### 3. Sparse index — BM25 (`minko186/fever-evidence-retrieval-kb`)

- `bm25_index.pkl` — pickled `BM25Okapi` object + sentence ID list

### 4. Graph index — entity co-occurrence (`minko186/fever-evidence-retrieval-kb`)

- `entity_graph.graphml` — NetworkX graph (nodes = Wikipedia articles, edges = co-occurrence)
- `entity_graph_sentences.pkl` — `dict[article_title -> list[sentence_id]]` for BFS expansion

---

## Current state of HF Hub repos

| Artifact | HF Hub status | Local status |
|----------|--------------|--------------|
| `sentence_records.jsonl` | **✅ uploaded** (`minko186/fever-wiki-sentences`) | not present |
| FAISS index (3 files) | **❌ repo doesn't exist yet** | not built |
| BM25 index | **❌ repo doesn't exist yet** | not built |
| Graph index (2 files) | **❌ repo doesn't exist yet** | not built |
| `wiki-pages.zip` (source) | n/a | **❌ not on machine** |

## Status

- [x] `download_from_hub()` — downloads all 4 artifact groups including graph (was missing before)
- [x] `upload_to_hub()` — uploads all 4 artifact groups, auto-creates repos if needed
- [x] `--push-to-hub` CLI flag — triggers upload after a build (or standalone with all `--skip-*`)
- [x] `--from-hub` CLI flag — triggers download (existed, now fixed to include graph)
- [x] `graph_index.py` — co-occurrence scan rewritten to use Aho-Corasick (was O(25M×5.4M), now O(n×sentence_len))
- [x] BM25 index built locally (7 min, saved to `data/processed/evidence_retrieval/bm25_index.pkl`)
- [ ] FAISS index — **currently building** (~38 h on CPU, ~1 h on GPU)
- [ ] Graph index — will build after FAISS completes (Aho-Corasick, ~30–60 min)
- [ ] Push all to `minko186/fever-evidence-retrieval-kb` (triggered automatically via `--push-to-hub`)

---

## Notes

- **BM25 pickle size**: ~1–2 GB depending on corpus size; HF Hub supports LFS for large files automatically.
- **Graph GraphML size**: can be large (hundreds of MB); consider switching to `graph.gpickle` (binary, ~3–5× smaller) if load time becomes a bottleneck.
- **Re-uploading**: `upload_to_hub()` always overwrites. To avoid redundant uploads, the `--skip-*` flags prevent rebuilding but still allow `--push-to-hub` to re-upload existing files.
- **Token**: uses `huggingface-cli login` token via `get_token()`. Set `HF_TOKEN` env var for CI/headless environments.
