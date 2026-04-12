"""
Preprocess Adversarial NLI (ANLI) into (claim, evidence, label) triples
for NLI fact verification.

ANLI uses numeric labels and different column names:
  - premise -> evidence
  - hypothesis -> claim
  - label: 0 = entailment (SUPPORTS), 1 = neutral (NOT ENOUGH INFO), 2 = contradiction (REFUTES)

Processes all 3 rounds (R1, R2, R3) and concatenates them.

Output schema: {"claim": str, "evidence": str, "label": str}

Usage:
    python preprocess_anli.py
    python preprocess_anli.py --repo minko186/anli-nli
"""

import argparse
import os
import sys

from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
from shared.utils.cleaning import remove_special_characters


LABEL_MAP = {
    0: "SUPPORTS",         # entailment
    1: "NOT ENOUGH INFO",  # neutral
    2: "REFUTES",          # contradiction
}


def process_split(split):
    """
    Process an ANLI split — map labels and rename columns.
    """
    rows = []
    for example in split:
        label_id = example.get("label")
        if label_id not in LABEL_MAP:
            continue

        label = LABEL_MAP[label_id]
        claim = remove_special_characters(example["hypothesis"].strip())
        evidence = remove_special_characters(example["premise"].strip())

        if not claim or not evidence:
            continue

        rows.append({
            "claim": claim,
            "evidence": evidence,
            "label": label,
        })

    print(f"  -> {len(rows)} (claim, evidence, label) triples extracted")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Preprocess ANLI for NLI fact verification")
    parser.add_argument("--repo", default="minko186/anli-nli",
                        help="HF Hub dataset repo to push to")
    args = parser.parse_args()

    result = DatasetDict()

    for split_type in ["train", "dev", "test"]:
        round_parts = []
        for round_name in ["r1", "r2", "r3"]:
            split_key = f"{split_type}_{round_name}"
            print(f"Loading ANLI split: {split_key}")
            try:
                ds = load_dataset("facebook/anli", split=split_key, trust_remote_code=True)
                processed = process_split(ds)
                round_parts.append(processed)
            except Exception as e:
                print(f"  Skipping {split_key}: {e}")

        if round_parts:
            combined = concatenate_datasets(round_parts)
            combined = combined.shuffle(seed=42)
            result[split_type] = combined
            print(f"  {split_type} total: {len(combined)} examples")

    print("\nFinal dataset:")
    print(result)
    if "train" in result:
        print("Sample:", result["train"][0])

    print(f"\nPushing to Hub: {args.repo}")
    result.push_to_hub(args.repo)
    print("Done.")


if __name__ == "__main__":
    main()
