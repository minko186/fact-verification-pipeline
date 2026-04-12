"""
Preprocess VitaminC (tals/vitaminc) into atomic (evidence, claim) pairs and push to HF Hub.

VitaminC contains Wikipedia passages with adversarial sentence-level edits that
flip claim labels. Every SUPPORTS pair is atomic and tightly grounded by construction,
making it an ideal complement to FEVER for training claim extraction models.

Usage:
    python preprocess_vitaminc.py --repo minko186/vitaminc-fact-extraction
"""

import sys
import os
import argparse
from datasets import load_dataset, DatasetDict, Dataset

# Allow importing from shared/utils/ regardless of working directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
from shared.utils.cleaning import remove_special_characters


def process_split(split, labels=("SUPPORTS",)):
    """Filter to the given labels and return atomic (evidence, claim) rows."""
    rows = []
    for example in split:
        if example.get("label") not in labels:
            continue
        evidence = example.get("evidence", "").strip()
        claim = example.get("claim", "").strip()
        if not evidence or not claim:
            continue
        evidence = remove_special_characters(evidence)
        claim = remove_special_characters(claim)
        if evidence and claim:
            rows.append({"evidence": evidence, "claim": claim})
    print(f"  -> {len(rows)} (evidence, claim) pairs")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Preprocess VitaminC for fact extraction")
    parser.add_argument("--repo", default="minko186/vitaminc-fact-extraction",
                        help="HF Hub dataset repo to push to")
    parser.add_argument("--labels", nargs="+", default=["SUPPORTS"],
                        help="Labels to keep (e.g. SUPPORTS REFUTES)")
    args = parser.parse_args()

    print("Loading tals/vitaminc...")
    raw = load_dataset("tals/vitaminc")

    result = DatasetDict()
    for split_name, split in raw.items():
        print(f"Processing split: {split_name}")
        result[split_name] = process_split(split, tuple(args.labels))

    print("\nFinal dataset:")
    print(result)
    if "train" in result:
        print("Sample:", result["train"][0])

    print(f"\nPushing to Hub: {args.repo}")
    result.push_to_hub(args.repo)
    print("Done.")


if __name__ == "__main__":
    main()
