"""
Preprocess VitaminC into (claim, evidence, label) triples for NLI fact verification.

VitaminC already has claim, evidence, and label columns with the standard
label set (SUPPORTS, REFUTES, NOT ENOUGH INFO). This preprocessor applies
text cleaning and pushes to HF Hub.

Output schema: {"claim": str, "evidence": str, "label": str}

Usage:
    python preprocess_vitaminc_nli.py
    python preprocess_vitaminc_nli.py --repo minko186/vitaminc-nli
"""

import argparse
import os
import sys

from datasets import load_dataset, DatasetDict, Dataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
from shared.utils.cleaning import remove_special_characters


# VitaminC uses these label strings natively
VALID_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}


def process_split(split):
    """
    Process a VitaminC split — keep all three labels, clean text.
    """
    rows = []
    for example in split:
        label = example.get("label")
        if label not in VALID_LABELS:
            continue

        claim = remove_special_characters(example["claim"].strip())
        evidence = remove_special_characters(example["evidence"].strip())

        if not claim:
            continue

        rows.append({
            "claim": claim,
            "evidence": evidence,
            "label": label,
        })

    print(f"  -> {len(rows)} (claim, evidence, label) triples extracted")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Preprocess VitaminC for NLI fact verification")
    parser.add_argument("--repo", default="minko186/vitaminc-nli",
                        help="HF Hub dataset repo to push to")
    args = parser.parse_args()

    print("Loading VitaminC dataset...")
    dataset = load_dataset("tals/vitaminc", trust_remote_code=True)

    result = DatasetDict()
    for split_name in dataset.keys():
        print(f"Processing split: {split_name}")
        result[split_name] = process_split(dataset[split_name])

    print("\nFinal dataset:")
    print(result)
    if "train" in result:
        print("Sample:", result["train"][0])

    print(f"\nPushing to Hub: {args.repo}")
    result.push_to_hub(args.repo)
    print("Done.")


if __name__ == "__main__":
    main()
