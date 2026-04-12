"""
Preprocess WiCE (google/wice) into atomic (evidence, claim) pairs and push to HF Hub.

WiCE provides fine-grained textual entailment labels at the sub-sentence level.
Each example contains a claim with annotated subclaims, each mapped to specific
Wikipedia sentences. This yields the highest-atomicity evidence-claim pairs of any
public dataset and is ideal as a second-stage fine-tuning source.

Dataset schema (relevant fields):
  - target: the full claim text
  - text: the Wikipedia evidence paragraph
  - chunks: list of {text: str, label: str, sentence_used: list[int]} annotations

Usage:
    python preprocess_wice.py --repo minko186/wice-fact-extraction
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
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def process_split(split, labels=("SUPPORTS",)):
    """
    For each example, extract individual subclaims from the 'chunks' annotation
    and pair each with its cited sentence(s) from the evidence paragraph.
    Emits one {"evidence": str, "claim": str} row per subclaim.
    """
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
                # Fall back to the full paragraph if no sentence indices provided
                evidence = full_text
            evidence = remove_special_characters(evidence)
            subclaim = remove_special_characters(subclaim)
            if evidence and subclaim:
                rows.append({"evidence": evidence, "claim": subclaim})
    print(f"  -> {len(rows)} (evidence, claim) pairs")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Preprocess WiCE for fact extraction")
    parser.add_argument("--repo", default="minko186/wice-fact-extraction",
                        help="HF Hub dataset repo to push to")
    parser.add_argument("--labels", nargs="+", default=["SUPPORTS"],
                        help="Chunk-level labels to keep")
    args = parser.parse_args()

    print("Loading google/wice...")
    raw = load_dataset("google/wice")

    result = DatasetDict()
    for split_name, split in raw.items():
        print(f"Processing split: {split_name}")
        result[split_name] = process_split(split, tuple(args.labels))

    print("\nFinal dataset:")
    print(result)
    if "train" in result and len(result["train"]) > 0:
        print("Sample:", result["train"][0])

    print(f"\nPushing to Hub: {args.repo}")
    result.push_to_hub(args.repo)
    print("Done.")


if __name__ == "__main__":
    main()
