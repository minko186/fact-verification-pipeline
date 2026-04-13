"""
Preprocess FEVER v1.0 or v2.0 into atomic (evidence, claim) pairs and push to HF Hub.

Usage:
    python preprocess_fever.py --version v1  --repo minko186/fever-fact-extraction-supports
    python preprocess_fever.py --version v2  --repo minko186/fever-v2-fact-extraction

Replaces preprocess_fever_v1.py and preprocess_fever_v2.py.
Key fix over the original: emits one {"evidence": str, "claim": str} row per pair
instead of joining all claims for the same context with " || ".
"""

import argparse
from datasets import load_dataset, DatasetDict, Dataset


def create_wiki_lookup(wiki_dataset):
    """Build {page_id: raw_lines_text} dict for fast evidence lookup."""
    print("Building Wikipedia lookup dict...")
    wiki_lookup = {}
    for row in wiki_dataset["wikipedia_pages"]:
        wiki_lookup[row["id"]] = row["lines"]
    return wiki_lookup


def resolve_evidence_v1(example, wiki_lookup):
    """Return evidence sentence text for a FEVER v1 example, or '' if missing."""
    wiki_url = example.get("evidence_wiki_url")
    sentence_id = example.get("evidence_sentence_id")
    if wiki_url is None or sentence_id is None or wiki_url not in wiki_lookup:
        return ""
    lines = wiki_lookup[wiki_url].split("\n")
    target_prefix = f"{sentence_id}\t"
    line = next((l for l in lines if l.startswith(target_prefix)), None)
    if line:
        return line.split("\t", 1)[1]
    return ""


def resolve_evidence_v2(example, wiki_lookup):
    """Return evidence sentence text for a FEVER v2 example, or '' if missing."""
    try:
        ev_list = example["evidence"][0]  # first annotation group
        wiki_url = ev_list[2]
        sentence_id = ev_list[3]
    except (IndexError, TypeError, KeyError):
        return ""
    if wiki_url not in wiki_lookup:
        return ""
    lines = wiki_lookup[wiki_url].split("\n")
    target_prefix = f"{sentence_id}\t"
    line = next((l for l in lines if l.startswith(target_prefix)), None)
    if line:
        return line.split("\t", 1)[1]
    return ""


def process_split(split, wiki_lookup, version, labels=("SUPPORTS",)):
    """
    Filter a split to the given labels, resolve evidence text, and return a
    list of {"evidence": str, "claim": str} dicts — one row per pair.
    """
    rows = []
    for example in split:
        if example.get("label") not in labels:
            continue
        if version == "v1":
            evidence = resolve_evidence_v1(example, wiki_lookup)
        else:
            evidence = resolve_evidence_v2(example, wiki_lookup)
        if evidence.strip():
            rows.append({"evidence": evidence.strip(), "claim": example["claim"].strip()})
    print(f"  -> {len(rows)} (evidence, claim) pairs extracted")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Preprocess FEVER for fact extraction")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--repo", default="minko186/fever-fact-extraction-supports",
                        help="HF Hub dataset repo to push to")
    parser.add_argument("--labels", nargs="+", default=["SUPPORTS"],
                        help="Labels to keep (e.g. SUPPORTS REFUTES)")
    parser.add_argument(
        "--save-to-disk",
        default=None,
        help="If set, save DatasetDict here instead of pushing to the Hub",
    )
    args = parser.parse_args()

    fever_config = "v1.0" if args.version == "v1" else "v2.0"
    splits_map = {
        "v1": ["train", "dev_labelled"],
        "v2": ["train", "labelled_dev", "paper_dev", "paper_test"],
    }

    print(f"Loading FEVER {fever_config} claims...")
    claims_dataset = load_dataset("fever", fever_config, trust_remote_code=True)

    print("Loading FEVER wiki_pages...")
    wiki_dataset = load_dataset("fever", "wiki_pages", trust_remote_code=True)
    wiki_lookup = create_wiki_lookup(wiki_dataset)

    result = DatasetDict()
    for split_name in splits_map[args.version]:
        if split_name not in claims_dataset:
            print(f"Split '{split_name}' not found, skipping.")
            continue
        print(f"Processing split: {split_name}")
        result[split_name] = process_split(
            claims_dataset[split_name], wiki_lookup, args.version, tuple(args.labels)
        )

    print("\nFinal dataset:")
    print(result)
    if "train" in result:
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
