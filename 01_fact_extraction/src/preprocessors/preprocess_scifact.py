"""
Preprocess SciFact into atomic (evidence, claim) pairs and push to HF Hub.

Downloads the official release tarball (same source as the paper) so this step
works without the legacy allenai/scifact Hub loading script (removed in
datasets 3+).

Usage:
    python preprocess_scifact.py --repo minko186/scifact-fact-extraction
    python preprocess_scifact.py --save-to-disk /path/to/out
"""

import sys
import os
import argparse
import tarfile
import urllib.request
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

# fact-verification-pipeline/ (src/preprocessors/ -> parents[3])
_PIPELINE_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = str(_PIPELINE_ROOT)
sys.path.insert(0, _REPO_ROOT)
from shared.utils.cleaning import remove_special_characters

SCIFACT_TARBALL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def default_data_dir() -> Path:
    return _PIPELINE_ROOT / "data" / "raw" / "scifact"


def ensure_scifact_files(data_dir: Path) -> Path:
    """Download and extract SciFact JSONL files into data_dir if missing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    inner = data_dir / "data"
    marker = inner / "corpus.jsonl"
    if marker.is_file():
        return inner
    tar_path = data_dir / "data.tar.gz"
    print(f"Downloading SciFact release to {tar_path} ...")
    urllib.request.urlretrieve(SCIFACT_TARBALL, tar_path)
    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    return inner


def build_corpus_lookup(corpus_rows):
    """Build {doc_id: list_of_sentences} from corpus examples."""
    lookup = {}
    for row in corpus_rows:
        lookup[str(row["doc_id"])] = row["abstract"]
    return lookup


def process_claims_rows(claim_rows, corpus_lookup, labels=("SUPPORT",)):
    """
    Each claim row has evidence: {doc_id: [{sentences, label}, ...]}.
    Emit one (evidence, claim) row per matching annotation.
    """
    rows = []
    for row in claim_rows:
        claim_text = (row.get("claim") or "").strip()
        if not claim_text:
            continue
        evidence_dict = row.get("evidence") or {}
        if not evidence_dict:
            continue
        for doc_id, ann_list in evidence_dict.items():
            if isinstance(ann_list, dict):
                ann_list = [ann_list]
            for annotation in ann_list:
                if annotation.get("label") not in labels:
                    continue
                abstract = corpus_lookup.get(str(doc_id), [])
                cited_ids = annotation.get("sentences", [])
                if not isinstance(abstract, list):
                    abstract = [str(abstract)]
                cited_sentences = [
                    str(abstract[i]).strip()
                    for i in cited_ids
                    if isinstance(i, int) and i < len(abstract)
                ]
                if not cited_sentences:
                    continue
                evidence_text = " ".join(cited_sentences).strip()
                evidence_text = remove_special_characters(evidence_text)
                claim_clean = remove_special_characters(claim_text)
                if evidence_text and claim_clean:
                    rows.append({"evidence": evidence_text, "claim": claim_clean})
    print(f"  -> {len(rows)} (evidence, claim) pairs")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Preprocess SciFact for fact extraction")
    parser.add_argument("--repo", default="minko186/scifact-fact-extraction",
                        help="HF Hub dataset repo to push to")
    parser.add_argument("--labels", nargs="+", default=["SUPPORT"],
                        help="Evidence labels to keep")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory for corpus.jsonl / claims_*.jsonl (default: repo data/raw/scifact)",
    )
    parser.add_argument(
        "--save-to-disk",
        default=None,
        help="If set, save DatasetDict here instead of pushing to the Hub",
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir) if args.data_dir else default_data_dir()
    json_dir = ensure_scifact_files(data_root)

    corpus_path = json_dir / "corpus.jsonl"
    print(f"Loading corpus from {corpus_path}")
    corpus_ds = load_dataset("json", data_files=str(corpus_path), split="train")
    corpus_lookup = build_corpus_lookup(corpus_ds)

    result_parts = {}
    label_tuple = tuple(args.labels)
    for split_file, out_key in [
        ("claims_train.jsonl", "train"),
        ("claims_dev.jsonl", "validation"),
    ]:
        path = json_dir / split_file
        if not path.is_file():
            print(f"Skip missing {path}")
            continue
        print(f"Processing {split_file} -> {out_key}")
        claims_ds = load_dataset("json", data_files=str(path), split="train")
        ds = process_claims_rows(claims_ds, corpus_lookup, label_tuple)
        if len(ds) > 0:
            result_parts[out_key] = ds

    result = DatasetDict(result_parts)

    print("\nFinal dataset:")
    print(result)
    if "train" in result and len(result["train"]) > 0:
        print("Sample:", result["train"][0])

    if args.save_to_disk:
        print(f"\nSaving to disk: {args.save_to_disk}")
        result.save_to_disk(args.save_to_disk)
    else:
        print(f"\nPushing to Hub: {args.repo}")
        result.push_to_hub(args.repo)
    print("Done.")


if __name__ == "__main__":
    main()
