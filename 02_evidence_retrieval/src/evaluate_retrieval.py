"""
Evaluate retrieval quality against FEVER gold evidence.

Computes:
    - Recall@k: fraction of claims where at least one gold evidence sentence
      appears in the top-k retrieved results.
    - MRR (Mean Reciprocal Rank): average of 1/rank for the first gold hit.

Usage:
    python evaluate_retrieval.py --config config.yaml
    python evaluate_retrieval.py --config config.yaml --sample 500
"""

import argparse
import json
import os
import sys
import time

import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from retrieval.pipeline import EvidencePipeline


def load_fever_gold(fever_path, sample_size=None):
    """
    Load FEVER train.jsonl and extract gold evidence for verifiable claims.

    Each FEVER entry has:
        - claim: str
        - label: "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO"
        - evidence: nested list of [annotation_id, evidence_id, wiki_url, sentence_id]

    Returns:
        list of (claim, set of gold_sentence_ids)
        where gold_sentence_id = "{wiki_url}_{sentence_id}"
    """
    entries = []

    with open(fever_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            label = obj.get("label", "")
            if label not in ("SUPPORTS", "REFUTES"):
                continue

            claim = obj["claim"]
            evidence_sets = obj.get("evidence", [])

            gold_ids = set()
            for evidence_set in evidence_sets:
                for evidence in evidence_set:
                    # evidence format: [annotation_id, evidence_id, wiki_url, sentence_id]
                    wiki_url = evidence[2]
                    sentence_id = evidence[3]
                    if wiki_url and sentence_id is not None:
                        gold_ids.add(f"{wiki_url}_{sentence_id}")

            if gold_ids:
                entries.append((claim, gold_ids))

    if sample_size and sample_size < len(entries):
        import random
        random.seed(42)
        entries = random.sample(entries, sample_size)

    return entries


def recall_at_k(retrieved_ids, gold_ids, k):
    """Check if any gold ID appears in the top-k retrieved IDs."""
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & gold_ids else 0.0


def reciprocal_rank(retrieved_ids, gold_ids):
    """Find the rank of the first gold ID in the retrieved list."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            return 1.0 / rank
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval against FEVER gold")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--sample", type=int, default=None, help="Evaluate on N random claims")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    base_dir = os.path.dirname(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve paths
    fever_path = os.path.normpath(
        os.path.join(base_dir, config["evaluation"]["fever_train_path"])
    )
    eval_results_dir = os.path.normpath(
        os.path.join(base_dir, config["evaluation"]["eval_results_dir"])
    )
    recall_k_values = config["evaluation"]["recall_k_values"]

    # Load gold data
    print(f"Loading gold evidence from {fever_path}...")
    gold_data = load_fever_gold(fever_path, sample_size=args.sample)
    print(f"Loaded {len(gold_data)} verifiable claims")

    # Initialize pipeline
    pipeline = EvidencePipeline(config_path)

    # Evaluate
    print(f"\nEvaluating Recall@{recall_k_values} and MRR...")
    start_time = time.time()

    recall_totals = {k: 0.0 for k in recall_k_values}
    mrr_total = 0.0

    for claim, gold_ids in tqdm(gold_data, desc="Evaluating"):
        result = pipeline.retrieve(claim)
        retrieved_ids = result.evidence_ids

        for k in recall_k_values:
            recall_totals[k] += recall_at_k(retrieved_ids, gold_ids, k)

        mrr_total += reciprocal_rank(retrieved_ids, gold_ids)

    n = len(gold_data)
    elapsed = time.time() - start_time

    # Compute averages
    metrics = {
        "num_claims": n,
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_claim": round(elapsed / n, 3) if n > 0 else 0,
    }

    for k in recall_k_values:
        metrics[f"recall@{k}"] = round(recall_totals[k] / n, 4) if n > 0 else 0

    metrics["mrr"] = round(mrr_total / n, 4) if n > 0 else 0

    # Print results
    print(f"\n{'='*50}")
    print("Retrieval Evaluation Results")
    print(f"{'='*50}")
    print(f"Claims evaluated: {n}")
    print(f"Time: {elapsed:.1f}s ({metrics['seconds_per_claim']:.3f}s/claim)")
    print()
    for k in recall_k_values:
        print(f"  Recall@{k}: {metrics[f'recall@{k}']:.4f}")
    print(f"  MRR:       {metrics['mrr']:.4f}")
    print(f"{'='*50}")

    # Save report
    os.makedirs(eval_results_dir, exist_ok=True)
    run_name = config["output"].get("run_name", "eval")
    report_path = os.path.join(eval_results_dir, f"{run_name}_eval.json")

    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
