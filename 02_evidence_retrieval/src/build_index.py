"""
CLI entry point: builds knowledge base indexes or downloads pre-built ones from HF Hub.

Usage:
    python build_index.py                              # Build everything locally
    python build_index.py --skip-parse                 # Reuse existing sentence_records.jsonl
    python build_index.py --skip-dense --skip-graph    # Only rebuild BM25
    python build_index.py --from-hub                   # Download pre-built indexes from HF
"""

import argparse
import json
import os
import sys
import time
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from kb.parse_wiki import parse_wiki_pages, save_records, load_records
from kb.bm25_index import BM25Index
from kb.dense_index import DenseIndex
from kb.graph_index import GraphIndex


def resolve_path(base_dir, path):
    """Resolve a relative path against the config file's directory."""
    return os.path.normpath(os.path.join(base_dir, path))


def download_from_hub(config):
    """Download pre-built KB indexes from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download, get_token

    hub_cfg = config.get("hub", {})
    kb_repo = hub_cfg.get("kb_repo_id", "minko186/fever-evidence-retrieval-kb")
    records_repo = hub_cfg.get("records_dataset_id", "minko186/fever-wiki-sentences")
    token = get_token()

    # Download sentence records
    records_path = config["corpus"]["records_path"]
    records_dir = os.path.dirname(records_path)
    os.makedirs(records_dir, exist_ok=True)

    if not os.path.exists(records_path):
        print(f"Downloading sentence records from {records_repo}...")
        hf_hub_download(
            repo_id=records_repo,
            filename="sentence_records.jsonl",
            repo_type="dataset",
            local_dir=records_dir,
            token=token,
        )
        downloaded = os.path.join(records_dir, "sentence_records.jsonl")
        if os.path.abspath(downloaded) != os.path.abspath(records_path):
            os.rename(downloaded, records_path)
        print(f"  -> {records_path}")
    else:
        print(f"Sentence records already exist: {records_path}")

    # Download FAISS index + supporting files
    faiss_path = config["index"]["faiss_path"]
    faiss_dir = os.path.dirname(faiss_path)
    os.makedirs(faiss_dir, exist_ok=True)

    faiss_files = ["faiss_index.faiss", "faiss_index_ids.pkl", "faiss_index_meta.json"]
    for fname in faiss_files:
        local = os.path.join(faiss_dir, fname)
        if not os.path.exists(local):
            print(f"Downloading {fname} from {kb_repo}...")
            hf_hub_download(
                repo_id=kb_repo,
                filename=fname,
                repo_type="dataset",
                local_dir=faiss_dir,
                token=token,
            )
            print(f"  -> {local}")
        else:
            print(f"Already exists: {local}")

    # Optionally download BM25 index
    bm25_path = config["index"]["bm25_path"]
    if not os.path.exists(bm25_path):
        bm25_fname = os.path.basename(bm25_path)
        try:
            print(f"Downloading {bm25_fname} from {kb_repo}...")
            hf_hub_download(
                repo_id=kb_repo,
                filename=bm25_fname,
                repo_type="dataset",
                local_dir=os.path.dirname(bm25_path),
                token=token,
            )
            print(f"  -> {bm25_path}")
        except Exception as e:
            print(f"  BM25 index not available on Hub (optional): {e}")

    print("\nAll KB artifacts downloaded from Hub.")


def main():
    parser = argparse.ArgumentParser(description="Build knowledge base indexes")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--skip-parse", action="store_true", help="Skip wiki parsing (reuse existing records)")
    parser.add_argument("--skip-bm25", action="store_true", help="Skip BM25 index build")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense (FAISS) index build")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph index build")
    parser.add_argument("--from-hub", action="store_true", help="Download pre-built indexes from HF Hub instead of building locally")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    base_dir = os.path.dirname(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve all relative paths in config
    config["corpus"]["wiki_zip_path"] = resolve_path(base_dir, config["corpus"]["wiki_zip_path"])
    config["corpus"]["records_path"] = resolve_path(base_dir, config["corpus"]["records_path"])
    config["index"]["faiss_path"] = resolve_path(base_dir, config["index"]["faiss_path"])
    config["index"]["bm25_path"] = resolve_path(base_dir, config["index"]["bm25_path"])
    config["index"]["graph_path"] = resolve_path(base_dir, config["index"]["graph_path"])

    if args.from_hub:
        download_from_hub(config)
        return

    total_start = time.time()

    # ── Step 1: Parse wiki pages ──────────────────────────────────────────
    records_path = config["corpus"]["records_path"]

    if args.skip_parse:
        print("\n[1/4] Skipping wiki parsing — loading existing records...")
        records = load_records(records_path)
    else:
        print("\n[1/4] Parsing wiki pages...")
        step_start = time.time()
        records = parse_wiki_pages(
            config["corpus"]["wiki_zip_path"],
            min_length=config["corpus"].get("min_sentence_length", 10),
        )
        save_records(records, records_path)
        print(f"  Completed in {time.time() - step_start:.1f}s")

    # ── Step 2: BM25 index ────────────────────────────────────────────────
    if args.skip_bm25:
        print("\n[2/4] Skipping BM25 index build")
    else:
        print("\n[2/4] Building BM25 index...")
        step_start = time.time()
        bm25 = BM25Index(config)
        bm25.build(records)
        print(f"  Completed in {time.time() - step_start:.1f}s")

    # ── Step 3: Dense (FAISS) index ───────────────────────────────────────
    if args.skip_dense:
        print("\n[3/4] Skipping dense index build")
    else:
        print("\n[3/4] Building FAISS dense index (this may take a while)...")
        step_start = time.time()
        dense = DenseIndex(config)
        dense.build(records)
        print(f"  Completed in {time.time() - step_start:.1f}s")

    # ── Step 4: Graph index ───────────────────────────────────────────────
    if args.skip_graph:
        print("\n[4/4] Skipping graph index build")
    else:
        print("\n[4/4] Building knowledge graph index...")
        step_start = time.time()
        graph = GraphIndex(config)
        graph.build(records)
        print(f"  Completed in {time.time() - step_start:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\nAll indexes built in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
