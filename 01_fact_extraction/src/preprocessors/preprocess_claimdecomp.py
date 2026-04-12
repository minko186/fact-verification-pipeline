"""
Preprocess ClaimDecomp (amandaask/ClaimDecomp) into atomic (evidence, claim) pairs
and push to HF Hub.

ClaimDecomp decomposes complex political/news claims into atomic sub-questions and
sub-claims, each paired with supporting passages. Training on this dataset teaches
the model to produce atomic, independent outputs rather than compound claims.

Dataset schema (relevant fields):
  - claim: the original complex claim
  - subclaims: list of atomic sub-claim strings
  - evidence: supporting passage(s) for the sub-claims (or the full claim context)

Note: the exact field names may differ across versions of the dataset. The
preprocessor handles the two most common schemas seen in the wild and falls back
gracefully if neither matches.

Usage:
    python preprocess_claimdecomp.py --repo minko186/claimdecomp-fact-extraction
"""

import sys
import os
import argparse
from datasets import load_dataset, DatasetDict, Dataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
from shared.utils.cleaning import remove_special_characters


def extract_pairs_from_example(example):
    """
    Try multiple known schemas to extract (evidence, subclaim) pairs.
    Returns a list of {"evidence": str, "claim": str} dicts.
    """
    pairs = []

    # Schema A: subclaims list + evidence string
    subclaims = example.get("subclaims") or example.get("atomic_claims") or []
    evidence = example.get("evidence") or example.get("passage") or example.get("context") or ""

    if isinstance(evidence, list):
        evidence = " ".join(str(e) for e in evidence)
    evidence = str(evidence).strip()

    if subclaims and evidence:
        for sub in subclaims:
            sub = str(sub).strip()
            if sub:
                e_clean = remove_special_characters(evidence)
                s_clean = remove_special_characters(sub)
                if e_clean and s_clean:
                    pairs.append({"evidence": e_clean, "claim": s_clean})
        return pairs

    # Schema B: single claim + evidence (fallback — treat the full claim as the label)
    claim = example.get("claim", "").strip()
    if claim and evidence:
        e_clean = remove_special_characters(evidence)
        c_clean = remove_special_characters(claim)
        if e_clean and c_clean:
            pairs.append({"evidence": e_clean, "claim": c_clean})

    return pairs


def process_split(split):
    rows = []
    for example in split:
        rows.extend(extract_pairs_from_example(example))
    print(f"  -> {len(rows)} (evidence, claim) pairs")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Preprocess ClaimDecomp for fact extraction")
    parser.add_argument("--repo", default="minko186/claimdecomp-fact-extraction",
                        help="HF Hub dataset repo to push to")
    args = parser.parse_args()

    print("Loading amandaask/ClaimDecomp...")
    raw = load_dataset("amandaask/ClaimDecomp")

    result = DatasetDict()
    for split_name, split in raw.items():
        print(f"Processing split: {split_name}")
        result[split_name] = process_split(split)

    print("\nFinal dataset:")
    print(result)
    for split_name in result:
        if len(result[split_name]) > 0:
            print(f"Sample ({split_name}):", result[split_name][0])
            break

    print(f"\nPushing to Hub: {args.repo}")
    result.push_to_hub(args.repo)
    print("Done.")


if __name__ == "__main__":
    main()
