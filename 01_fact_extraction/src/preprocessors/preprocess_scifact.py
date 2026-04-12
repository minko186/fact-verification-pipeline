"""
Preprocess SciFact (allenai/scifact) into atomic (evidence, claim) pairs and push to HF Hub.

SciFact contains ~1,400 expert-written biomedical claims paired with supporting or
refuting sentences from PubMed abstracts. Claims are carefully crafted to be atomic
and verifiable by a single cited sentence, providing domain diversity beyond FEVER's
Wikipedia-encyclopedic style.

Dataset structure:
  - claims split: id, claim, evidence (dict mapping doc_id -> {sentences, label})
  - corpus split: doc_id, title, abstract (list of sentences)

Usage:
    python preprocess_scifact.py --repo minko186/scifact-fact-extraction
"""

import sys
import os
import argparse
from datasets import load_dataset, DatasetDict, Dataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
from shared.utils.cleaning import remove_special_characters


def build_corpus_lookup(corpus_split):
    """Build {doc_id: list_of_sentences} from the corpus split."""
    lookup = {}
    for row in corpus_split:
        lookup[str(row["doc_id"])] = row["abstract"]
    return lookup


def process_claims(claims_split, corpus_lookup, labels=("SUPPORTS",)):
    """
    For each claim, find its cited abstract sentences and emit one
    {"evidence": str, "claim": str} row. Uses all cited sentences joined
    as the evidence string.
    """
    rows = []
    for example in claims_split:
        claim_text = example.get("claim", "").strip()
        if not claim_text:
            continue
        evidence_dict = example.get("evidence", {})
        if not evidence_dict:
            continue
        for doc_id, annotation in evidence_dict.items():
            if annotation.get("label") not in labels:
                continue
            abstract = corpus_lookup.get(str(doc_id), [])
            cited_ids = annotation.get("sentences", [])
            cited_sentences = [abstract[i] for i in cited_ids if i < len(abstract)]
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
    parser.add_argument("--labels", nargs="+", default=["SUPPORTS"],
                        help="Annotation labels to keep")
    args = parser.parse_args()

    print("Loading allenai/scifact corpus...")
    corpus = load_dataset("allenai/scifact", "corpus")
    corpus_lookup = build_corpus_lookup(corpus["train"])

    print("Loading allenai/scifact claims...")
    claims = load_dataset("allenai/scifact", "claims")

    result = DatasetDict()
    for split_name, split in claims.items():
        print(f"Processing split: {split_name}")
        result[split_name] = process_claims(split, corpus_lookup, tuple(args.labels))

    print("\nFinal dataset:")
    print(result)
    if "train" in result and len(result["train"]) > 0:
        print("Sample:", result["train"][0])

    print(f"\nPushing to Hub: {args.repo}")
    result.push_to_hub(args.repo)
    print("Done.")


if __name__ == "__main__":
    main()
