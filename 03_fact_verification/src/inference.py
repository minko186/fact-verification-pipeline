"""
Run fact verification inference on claims with retrieved evidence.

Takes JSONL output from stage 2 (evidence retrieval) and predicts a verdict
for each (claim, evidence) pair. Also supports single-claim mode.

Input JSONL format (from stage 2):
    {"claim": "...", "evidence": ["sent1", "sent2", ...], "evidence_ids": [...], ...}

Output JSONL format:
    {"claim": "...", "evidence": [...], "verdict": "SUPPORTS", "confidence": 0.94,
     "verdict_probs": {"SUPPORTS": 0.94, "REFUTES": 0.04, "NOT ENOUGH INFO": 0.02}}

Usage:
    # Batch mode (from retrieval output)
    python inference.py --model-key deberta-base --input results.jsonl

    # Single claim mode
    python inference.py --model-key deberta-base --claim "Einstein was born in Germany" --evidence "Albert Einstein was born on 14 March 1879 in Ulm, Germany."
"""

import argparse
import json
import os
from datetime import datetime

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_model_config(config, model_key=None):
    """Resolve model configuration from config using model_key or default."""
    model_key = model_key or config["default_model"]
    if model_key not in config["models"]:
        available = ", ".join(config["models"].keys())
        raise ValueError(f"Unknown model key '{model_key}'. Available: {available}")
    return config["models"][model_key], model_key


def load_model(model_path, device=None):
    """Load tokenizer and model for inference."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model, device


def predict_batch(claims, evidences, tokenizer, model, device, max_length=512, batch_size=32):
    """
    Predict verdicts for a batch of (claim, evidence) pairs.

    Returns list of dicts: {"verdict": str, "confidence": float, "verdict_probs": dict}
    """
    id2label = model.config.id2label
    results = []

    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i : i + batch_size]
        batch_evidences = evidences[i : i + batch_size]

        inputs = tokenizer(
            batch_claims, batch_evidences,
            max_length=max_length, truncation=True,
            padding=True, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1).cpu()

        for j in range(len(batch_claims)):
            prob_dict = {id2label[k]: round(probs[j][k].item(), 4) for k in id2label}
            pred_idx = torch.argmax(probs[j]).item()
            results.append({
                "verdict": id2label[pred_idx],
                "confidence": round(probs[j][pred_idx].item(), 4),
                "verdict_probs": prob_dict,
            })

    return results


def load_retrieval_results(jsonl_path):
    """Read JSONL file from stage 2 evidence retrieval."""
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    print(f"Loaded {len(results)} retrieval results from {jsonl_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run NLI fact verification inference")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-key", default=None,
                        help="Model key from config (default: config's default_model)")
    parser.add_argument("--model-path", default=None,
                        help="Override model path/ID (default: model's hub_model_id)")

    # Batch mode
    parser.add_argument("--input", default=None,
                        help="Path to retrieval output JSONL (batch mode)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: from config)")

    # Single claim mode
    parser.add_argument("--claim", default=None,
                        help="Single claim text (single mode)")
    parser.add_argument("--evidence", default=None,
                        help="Single evidence text (single mode)")

    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_config, model_key = resolve_model_config(config, args.model_key)
    inference_cfg = config.get("inference", {})

    model_path = args.model_path or model_config["hub_model_id"]
    max_length = model_config.get("max_length", 512)
    batch_size = args.batch_size or inference_cfg.get("batch_size", 32)
    evidence_join = inference_cfg.get("evidence_join", " ")

    print(f"Loading model: {model_path}")
    tokenizer, model, device = load_model(model_path)
    print(f"Device: {device}")

    # ── Single claim mode ───────────────────────────────────────────────────────
    if args.claim is not None:
        evidence = args.evidence or ""
        print(f"\nClaim:    {args.claim}")
        print(f"Evidence: {evidence}")

        results = predict_batch(
            [args.claim], [evidence],
            tokenizer, model, device, max_length,
        )
        result = results[0]
        print(f"\nVerdict:    {result['verdict']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Probs:      {result['verdict_probs']}")
        return

    # ── Batch mode ──────────────────────────────────────────────────────────────
    input_path = args.input or inference_cfg.get("retrieval_input")
    if not input_path:
        parser.error("Either --input or --claim must be provided")

    output_dir = args.output or inference_cfg.get("output_dir", "../../data/processed/fact_verification/results")
    run_name = inference_cfg.get("run_name", "verdicts")

    retrieval_results = load_retrieval_results(input_path)

    # Prepare (claim, evidence) pairs
    claims = []
    evidences = []
    for r in retrieval_results:
        claims.append(r["claim"])
        ev_list = r.get("evidence", [])
        if isinstance(ev_list, list):
            evidences.append(evidence_join.join(ev_list))
        else:
            evidences.append(str(ev_list))

    print(f"\nPredicting verdicts for {len(claims)} claims...")
    verdicts = predict_batch(
        claims, evidences,
        tokenizer, model, device, max_length, batch_size,
    )

    # Merge verdicts with retrieval results
    output_rows = []
    for r, v in zip(retrieval_results, verdicts):
        output_row = {
            "claim": r["claim"],
            "evidence": r.get("evidence", []),
            "evidence_ids": r.get("evidence_ids", []),
            "reranker_scores": r.get("reranker_scores", []),
            "verdict": v["verdict"],
            "confidence": v["confidence"],
            "verdict_probs": v["verdict_probs"],
        }
        output_rows.append(output_row)

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"{run_name}_{timestamp}.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary
    verdict_counts = {}
    for v in verdicts:
        verdict_counts[v["verdict"]] = verdict_counts.get(v["verdict"], 0) + 1

    print(f"\nResults written to: {out_path}")
    print(f"Verdict distribution:")
    for label, count in sorted(verdict_counts.items()):
        print(f"  {label:20s}  {count:5d}  ({100*count/len(verdicts):.1f}%)")


if __name__ == "__main__":
    main()
