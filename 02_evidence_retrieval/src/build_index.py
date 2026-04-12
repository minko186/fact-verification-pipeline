"""
CLI entry point: builds all three knowledge base indexes.

Usage:
    python build_index.py                         # Build everything
    python build_index.py --skip-parse            # Reuse existing sentence_records.jsonl
    python build_index.py --skip-dense --skip-graph  # Only rebuild BM25
"""

import argparse
import os
import sys
import time
import yaml

# Ensure kb package is importable
sys.path.insert(0, os.path.dirname(__file__))

from kb.parse_wiki import parse_wiki_pages, save_records, load_records
from kb.bm25_index import BM25Index
from kb.dense_index import DenseIndex
from kb.graph_index import GraphIndex


def resolve_path(base_dir, path):
    """Resolve a relative path against the config file's directory."""
    return os.path.normpath(os.path.join(base_dir, path))


def main():
    parser = argparse.ArgumentParser(description="Build knowledge base indexes")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--skip-parse", action="store_true", help="Skip wiki parsing (reuse existing records)")
    parser.add_argument("--skip-bm25", action="store_true", help="Skip BM25 index build")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense (ChromaDB) index build")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph index build")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    base_dir = os.path.dirname(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve all relative paths in config
    config["corpus"]["wiki_zip_path"] = resolve_path(base_dir, config["corpus"]["wiki_zip_path"])
    config["corpus"]["records_path"] = resolve_path(base_dir, config["corpus"]["records_path"])
    config["index"]["bm25_path"] = resolve_path(base_dir, config["index"]["bm25_path"])
    config["index"]["chroma_path"] = resolve_path(base_dir, config["index"]["chroma_path"])
    config["index"]["graph_path"] = resolve_path(base_dir, config["index"]["graph_path"])

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

    # ── Step 3: Dense (ChromaDB) index ────────────────────────────────────
    if args.skip_dense:
        print("\n[3/4] Skipping dense index build")
    else:
        print("\n[3/4] Building dense index (this may take a while)...")
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
