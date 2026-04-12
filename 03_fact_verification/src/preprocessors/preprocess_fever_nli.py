"""
Preprocess FEVER v1.0 into (claim, evidence, label) triples for NLI fact verification.

Unlike the fact extraction preprocessor (which keeps only SUPPORTS), this keeps
all three labels: SUPPORTS, REFUTES, and NOT ENOUGH INFO.

For SUPPORTS/REFUTES claims, gold evidence sentences are resolved from wiki_pages.
For NOT ENOUGH INFO claims, the evidence field depends on --nei-strategy:
  - "empty": evidence = "" (standard FEVER baseline)
  - "random": sample a random non-gold Wikipedia sentence

Output schema: {"claim": str, "evidence": str, "label": str}

Usage:
    python preprocess_fever_nli.py --repo minko186/fever-nli
    python preprocess_fever_nli.py --repo minko186/fever-nli --nei-strategy random
"""

import argparse
import os
import sys
import random

from datasets import load_dataset, DatasetDict, Dataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
from shared.utils.cleaning import remove_special_characters


def create_wiki_lookup(wiki_dataset):
    """Build {page_id: raw_lines_text} dict for fast evidence lookup."""
    print("Building Wikipedia lookup dict...")
    wiki_lookup = {}
    for row in wiki_dataset["wikipedia_pages"]:
        wiki_lookup[row["id"]] = row["lines"]
    return wiki_lookup


def resolve_evidence(example, wiki_lookup):
    """
    Resolve gold evidence sentence text from a FEVER v1 example.

    FEVER v1 evidence is a nested list: [[ann_id, ev_id, wiki_url, sentence_id], ...]
    Returns the concatenated text of all unique evidence sentences, or "" if
    none can be resolved.
    """
    evidence_sets = example.get("evidence", [])
    seen_sentences = set()
    texts = []

    for annotation_group in evidence_sets:
        for ev in annotation_group:
            try:
                wiki_url = ev[2]
                sentence_id = ev[3]
            except (IndexError, TypeError):
                continue

            if wiki_url is None or sentence_id is None:
                continue

            key = (wiki_url, sentence_id)
            if key in seen_sentences:
                continue
            seen_sentences.add(key)

            if wiki_url not in wiki_lookup:
                continue

            lines = wiki_lookup[wiki_url].split("\n")
            target_prefix = f"{sentence_id}\t"
            line = next((l for l in lines if l.startswith(target_prefix)), None)
            if line:
                sentence_text = line.split("\t", 1)[1].strip()
                if sentence_text:
                    texts.append(sentence_text)

    return " ".join(texts)


def get_random_sentence(wiki_lookup, rng):
    """Sample a random sentence from the wiki lookup for NEI distractor evidence."""
    page_id = rng.choice(list(wiki_lookup.keys()))
    lines = wiki_lookup[page_id].split("\n")
    valid_lines = [l for l in lines if "\t" in l]
    if not valid_lines:
        return ""
    line = rng.choice(valid_lines)
    return line.split("\t", 1)[1].strip()


def process_split(split, wiki_lookup, nei_strategy="empty"):
    """
    Process a FEVER split into {claim, evidence, label} rows for all three labels.

    For SUPPORTS/REFUTES: resolves gold evidence from wiki_pages.
    For NOT ENOUGH INFO: uses nei_strategy to determine evidence text.
    """
    rng = random.Random(42)
    rows = []

    for example in split:
        label = example.get("label")
        if label not in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
            continue

        claim = example["claim"].strip()
        if not claim:
            continue

        if label in ("SUPPORTS", "REFUTES"):
            evidence = resolve_evidence(example, wiki_lookup)
            if not evidence.strip():
                continue
        else:
            # NOT ENOUGH INFO
            if nei_strategy == "random":
                evidence = get_random_sentence(wiki_lookup, rng)
            else:
                evidence = ""

        claim = remove_special_characters(claim)
        if evidence:
            evidence = remove_special_characters(evidence)

        rows.append({
            "claim": claim,
            "evidence": evidence,
            "label": label,
        })

    print(f"  -> {len(rows)} (claim, evidence, label) triples extracted")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Preprocess FEVER for NLI fact verification")
    parser.add_argument("--repo", default="minko186/fever-nli",
                        help="HF Hub dataset repo to push to")
    parser.add_argument("--nei-strategy", choices=["empty", "random"], default="empty",
                        help="How to handle NOT ENOUGH INFO evidence (default: empty)")
    args = parser.parse_args()

    print("Loading FEVER v1.0 claims...")
    claims_dataset = load_dataset("fever", "v1.0", trust_remote_code=True)

    print("Loading FEVER wiki_pages...")
    wiki_dataset = load_dataset("fever", "wiki_pages", trust_remote_code=True)
    wiki_lookup = create_wiki_lookup(wiki_dataset)

    result = DatasetDict()
    for split_name in ["train", "labelled_dev"]:
        if split_name not in claims_dataset:
            print(f"Split '{split_name}' not found, skipping.")
            continue
        print(f"Processing split: {split_name}")
        result[split_name] = process_split(
            claims_dataset[split_name], wiki_lookup, args.nei_strategy
        )

    print("\nFinal dataset:")
    print(result)
    if "train" in result:
        print("Sample:", result["train"][0])

    print(f"\nPushing to Hub: {args.repo}")
    result.push_to_hub(args.repo)
    print("Done.")


if __name__ == "__main__":
    main()
