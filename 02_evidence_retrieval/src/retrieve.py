"""
CLI entry point: retrieve evidence for one or more claims.

Usage:
    # Single claim
    python retrieve.py --claim "Albert Einstein developed the theory of relativity."

    # Batch from JSONL (one claim per line, field "claim")
    python retrieve.py --input claims.jsonl --output results/

    # Override config
    python retrieve.py --config config.yaml --claim "..."
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from retrieval.pipeline import EvidencePipeline


def main():
    parser = argparse.ArgumentParser(description="Retrieve evidence for claims")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--claim", type=str, help="Single claim to retrieve evidence for")
    parser.add_argument("--input", type=str, help="Input JSONL file with claims")
    parser.add_argument("--output", type=str, help="Output directory for results JSONL")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    pipeline = EvidencePipeline(config_path)

    if args.claim:
        # Single claim mode
        result = pipeline.retrieve(args.claim)

        print(f"\nClaim: {result.claim}")
        print(f"Retrieved {len(result.evidence)} evidence sentences:\n")
        for i, (text, sid, score, sources) in enumerate(
            zip(
                result.evidence,
                result.evidence_ids,
                result.reranker_scores,
                result.source_channels,
            )
        ):
            print(f"  [{i+1}] (score={score:.4f}, sources={sources})")
            print(f"      ID: {sid}")
            print(f"      {text}\n")

        print(f"Metadata: {json.dumps(result.metadata, indent=2)}")

    elif args.input:
        # Batch mode
        claims = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                claims.append(obj["claim"])

        print(f"Loaded {len(claims)} claims from {args.input}")

        results = pipeline.retrieve_batch(claims)

        # Determine output path
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        base_dir = os.path.dirname(config_path)
        output_dir = args.output or os.path.normpath(
            os.path.join(base_dir, config["output"]["results_dir"])
        )
        os.makedirs(output_dir, exist_ok=True)

        run_name = config["output"].get("run_name", "results")
        output_path = os.path.join(output_dir, f"{run_name}.jsonl")

        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

        print(f"\nResults written to {output_path}")
        print(f"Total claims processed: {len(results)}")

    else:
        parser.print_help()
        print("\nError: provide either --claim or --input")
        sys.exit(1)


if __name__ == "__main__":
    main()
