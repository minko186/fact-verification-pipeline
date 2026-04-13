"""
Preprocess WiCE into atomic (evidence, claim) pairs and push to HF Hub.

The original google/wice Hub dataset is no longer loadable in many environments.
This script defaults to jon-tow/wice (config "claim"), which exposes list evidence,
supporting sentence indices, and claim text.

Usage:
    python preprocess_wice.py --repo minko186/wice-fact-extraction
    python preprocess_wice.py --save-to-disk /path/to/out
"""

import sys
import os
import argparse
from datasets import load_dataset, DatasetDict, Dataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
from shared.utils.cleaning import remove_special_characters


def split_into_sentences(text):
    """Rough sentence splitter — returns list of non-empty sentences."""
    import re
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def process_google_wice_split(split, labels=("SUPPORTS",)):
    """Legacy google/wice schema: text, chunks with sentence_used indices."""
    rows = []
    for example in split:
        full_text = example.get("text", "").strip()
        if not full_text:
            continue
        sentences = split_into_sentences(full_text)
        chunks = example.get("chunks", [])
        if not chunks:
            continue
        for chunk in chunks:
            if chunk.get("label") not in labels:
                continue
            subclaim = chunk.get("text", "").strip()
            if not subclaim:
                continue
            cited_indices = chunk.get("sentence_used", [])
            if cited_indices:
                cited = [sentences[i] for i in cited_indices if i < len(sentences)]
                evidence = " ".join(cited).strip()
            else:
                evidence = full_text
            evidence = remove_special_characters(evidence)
            subclaim = remove_special_characters(subclaim)
            if evidence and subclaim:
                rows.append({"evidence": evidence, "claim": subclaim})
    print(f"  -> {len(rows)} (evidence, claim) pairs")
    return Dataset.from_list(rows)


def process_jon_tow_split(split, labels=("supported", "partially_supported")):
    """jon-tow/wice claim schema: claim, evidence (list of lines), supporting_sentences."""
    rows = []
    for example in split:
        if example.get("label") not in labels:
            continue
        claim = (example.get("claim") or "").strip()
        ev_lines = example.get("evidence") or []
        spans = example.get("supporting_sentences") or []
        if not claim or not spans or not spans[0]:
            continue
        idxs = spans[0]
        if not isinstance(idxs, (list, tuple)):
            continue
        parts = []
        for i in idxs:
            if isinstance(i, int) and 0 <= i < len(ev_lines):
                parts.append(str(ev_lines[i]).strip())
        evidence = " ".join(parts).strip()
        evidence = remove_special_characters(evidence)
        claim = remove_special_characters(claim)
        if evidence and claim:
            rows.append({"evidence": evidence, "claim": claim})
    print(f"  -> {len(rows)} (evidence, claim) pairs")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Preprocess WiCE for fact extraction")
    parser.add_argument("--repo", default="minko186/wice-fact-extraction",
                        help="HF Hub dataset repo to push to")
    parser.add_argument(
        "--variant",
        choices=["jon_tow", "google"],
        default="jon_tow",
        help="Data source: jon-tow/wice (default) or legacy google/wice",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels to keep (defaults depend on --variant)",
    )
    parser.add_argument(
        "--save-to-disk",
        default=None,
        help="If set, save DatasetDict here instead of pushing to the Hub",
    )
    args = parser.parse_args()

    if args.variant == "google":
        label_tuple = tuple(args.labels or ["SUPPORTS"])
        print("Loading google/wice...")
        raw = load_dataset("google/wice")
        process_fn = lambda split: process_google_wice_split(split, label_tuple)
    else:
        label_tuple = tuple(args.labels or ["supported", "partially_supported"])
        print("Loading jon-tow/wice (config=claim)...")
        raw = load_dataset("jon-tow/wice", "claim")
        process_fn = lambda split: process_jon_tow_split(split, label_tuple)

    result = DatasetDict()
    for split_name, split in raw.items():
        print(f"Processing split: {split_name}")
        result[split_name] = process_fn(split)

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
