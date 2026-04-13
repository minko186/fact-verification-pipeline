"""
CLI entry point: builds knowledge base indexes or syncs them with HuggingFace Hub.

Usage:
    python build_index.py                              # Build everything locally
    python build_index.py --skip-parse                 # Reuse existing sentence_records.jsonl
    python build_index.py --skip-dense --skip-graph    # Only rebuild BM25

    python build_index.py --from-hub                   # Download all indexes from HF Hub
    python build_index.py --push-to-hub                # Build everything, then upload to HF Hub
    python build_index.py --skip-parse --skip-bm25 \\
        --skip-dense --skip-graph --push-to-hub        # Upload existing local indexes (no rebuild)

HF Hub repos (configured in config.yaml hub section):
    minko186/fever-evidence-retrieval-kb    — FAISS, BM25, graph indexes
    minko186/fever-wiki-sentences           — sentence_records.jsonl
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


def _hub_download_file(repo_id, filename, local_path, repo_type, token):
    """Download a single file from HF Hub to local_path if not already present."""
    from huggingface_hub import hf_hub_download

    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    if os.path.exists(local_path):
        print(f"  Already exists: {local_path}")
        return

    print(f"  Downloading {filename} from {repo_id}...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=local_dir,
        token=token,
    )
    # hf_hub_download saves to local_dir/<filename>; rename if needed
    downloaded = os.path.join(local_dir, filename)
    if os.path.abspath(downloaded) != os.path.abspath(local_path):
        os.rename(downloaded, local_path)
    print(f"  -> {local_path}")


def download_from_hub(config):
    """Download all pre-built KB indexes from HuggingFace Hub."""
    from huggingface_hub import get_token

    hub_cfg = config.get("hub", {})
    kb_repo = hub_cfg.get("kb_repo_id", "minko186/fever-evidence-retrieval-kb")
    records_repo = hub_cfg.get("records_dataset_id", "minko186/fever-wiki-sentences")
    token = get_token()

    print(f"\n[1/4] Sentence records  ({records_repo})")
    _hub_download_file(
        repo_id=records_repo,
        filename="sentence_records.jsonl",
        local_path=config["corpus"]["records_path"],
        repo_type="dataset",
        token=token,
    )

    print(f"\n[2/4] FAISS dense index  ({kb_repo})")
    faiss_base = os.path.splitext(config["index"]["faiss_path"])[0]
    try:
        for fname, local in [
            ("faiss_index.faiss",      config["index"]["faiss_path"]),
            ("faiss_index_ids.pkl",    faiss_base + "_ids.pkl"),
            ("faiss_index_meta.json",  faiss_base + "_meta.json"),
        ]:
            _hub_download_file(kb_repo, fname, local, "dataset", token)
    except Exception as e:
        print(f"  FAISS index not available on Hub (optional): {e}")

    print(f"\n[3/4] BM25 sparse index  ({kb_repo})")
    bm25_fname = os.path.basename(config["index"]["bm25_path"])
    try:
        _hub_download_file(kb_repo, bm25_fname, config["index"]["bm25_path"], "dataset", token)
    except Exception as e:
        print(f"  BM25 not available on Hub (optional): {e}")

    print(f"\n[4/4] Graph index  ({kb_repo})")
    graph_base = os.path.splitext(config["index"]["graph_path"])[0]
    graph_sentences_path = graph_base + "_sentences.pkl"
    graph_fname = os.path.basename(config["index"]["graph_path"])
    graph_sentences_fname = os.path.basename(graph_sentences_path)
    try:
        _hub_download_file(kb_repo, graph_fname, config["index"]["graph_path"], "dataset", token)
        _hub_download_file(kb_repo, graph_sentences_fname, graph_sentences_path, "dataset", token)
    except Exception as e:
        print(f"  Graph index not available on Hub (optional): {e}")

    print("\nAll KB artifacts downloaded from Hub.")


def upload_to_hub(config):
    """Upload all built KB indexes to HuggingFace Hub."""
    from huggingface_hub import HfApi, get_token

    hub_cfg = config.get("hub", {})
    kb_repo = hub_cfg.get("kb_repo_id", "minko186/fever-evidence-retrieval-kb")
    records_repo = hub_cfg.get("records_dataset_id", "minko186/fever-wiki-sentences")
    token = get_token()
    api = HfApi(token=token)

    def _ensure_repo(repo_id, repo_type):
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
        except Exception:
            print(f"  Creating repo {repo_id} ({repo_type})...")
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

    def _upload(local_path, repo_id, repo_type, description=""):
        if not os.path.exists(local_path):
            print(f"  Skipping {local_path} (not found)")
            return
        fname = os.path.basename(local_path)
        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"  Uploading {fname} ({size_mb:.1f} MB) -> {repo_id}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Upload {fname}" + (f" — {description}" if description else ""),
        )

    print(f"\n[1/4] Sentence records  -> {records_repo}")
    _ensure_repo(records_repo, "dataset")
    _upload(config["corpus"]["records_path"], records_repo, "dataset", "FEVER wiki sentence records")

    print(f"\n[2/4] FAISS dense index  -> {kb_repo}")
    _ensure_repo(kb_repo, "dataset")
    faiss_base = os.path.splitext(config["index"]["faiss_path"])[0]
    _upload(config["index"]["faiss_path"],        kb_repo, "dataset", "FAISS IVF-PQ index")
    _upload(faiss_base + "_ids.pkl",              kb_repo, "dataset", "FAISS sentence ID mapping")
    _upload(faiss_base + "_meta.json",            kb_repo, "dataset", "FAISS index metadata")

    print(f"\n[3/4] BM25 sparse index  -> {kb_repo}")
    _upload(config["index"]["bm25_path"], kb_repo, "dataset", "BM25Okapi index")

    print(f"\n[4/4] Graph index  -> {kb_repo}")
    graph_base = os.path.splitext(config["index"]["graph_path"])[0]
    _upload(config["index"]["graph_path"],        kb_repo, "dataset", "entity co-occurrence graph (GraphML)")
    _upload(graph_base + "_sentences.pkl",        kb_repo, "dataset", "article->sentence_id mapping")

    print(f"\nAll KB artifacts uploaded.")
    print(f"  Dense + sparse + graph: https://huggingface.co/datasets/{kb_repo}")
    print(f"  Sentence records:        https://huggingface.co/datasets/{records_repo}")


def main():
    parser = argparse.ArgumentParser(description="Build knowledge base indexes")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--skip-parse", action="store_true", help="Skip wiki parsing (reuse existing records)")
    parser.add_argument("--skip-bm25", action="store_true", help="Skip BM25 index build")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense (FAISS) index build")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph index build")
    parser.add_argument("--from-hub", action="store_true", help="Download pre-built indexes from HF Hub instead of building locally")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload all built KB indexes to HF Hub after building (or standalone)")
    parser.add_argument(
        "--push-only",
        action="store_true",
        help="Only upload existing local files to HF Hub (no build, no loading of sentence records)",
    )
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

    if args.push_only:
        upload_to_hub(config)
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

    if args.push_to_hub:
        upload_to_hub(config)


if __name__ == "__main__":
    main()
