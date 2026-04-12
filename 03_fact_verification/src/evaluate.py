"""
Post-training evaluation for an NLI fact verification model.

Loads a trained model (from HF Hub or local path), runs it over an eval split,
and computes accuracy, macro F1, per-class precision/recall/F1, and confusion
matrix. Writes a JSON report to ../../experiments/eval_results/.

Usage:
    python evaluate.py
    python evaluate.py --model-key deberta-base
    python evaluate.py --model-path minko186/deberta-v3-base-fever-nli --data-path ../../data/processed/fact_verification/deberta-base-fever-nli/eval
"""

import argparse
import json
import os
from datetime import datetime

import torch
import yaml
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_from_disk
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


def evaluate_model(model_path, data_path, config, batch_size=32):
    """Run inference on eval split and collect predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    label2id = config["labels"]
    id2label = {v: k for k, v in label2id.items()}
    max_length = 512

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    print(f"Loading eval data from: {data_path}")
    dataset = load_from_disk(data_path)

    all_predictions = []
    all_labels = []

    # Check if dataset has text columns (claim/evidence) or is pre-tokenized
    has_text_columns = "claim" in dataset.column_names and "evidence" in dataset.column_names

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        if has_text_columns:
            inputs = tokenizer(
                batch["claim"], batch["evidence"],
                max_length=max_length, truncation=True,
                padding=True, return_tensors="pt",
            ).to(device)
            labels = [label2id[l] for l in batch["label"]]
        else:
            inputs = {
                "input_ids": torch.tensor(batch["input_ids"]).to(device),
                "attention_mask": torch.tensor(batch["attention_mask"]).to(device),
            }
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = torch.tensor(batch["token_type_ids"]).to(device)
            labels = batch["labels"]

        with torch.no_grad():
            outputs = model(**inputs)

        preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        all_predictions.extend(preds)
        all_labels.extend(labels)

        if (i // batch_size) % 20 == 0:
            print(f"  {i + len(preds)}/{len(dataset)} examples processed")

    return all_predictions, all_labels, id2label


def compute_all_metrics(predictions, labels, id2label):
    """Compute classification metrics from predictions and labels."""
    label_names = [id2label[i] for i in sorted(id2label.keys())]

    # Overall accuracy
    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)

    # sklearn classification report
    report = classification_report(
        labels, predictions,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    # Per-class metrics
    per_class = {}
    for label_name in label_names:
        if label_name in report:
            per_class[label_name] = {
                "precision": round(report[label_name]["precision"], 4),
                "recall": round(report[label_name]["recall"], 4),
                "f1": round(report[label_name]["f1-score"], 4),
                "support": report[label_name]["support"],
            }

    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=sorted(id2label.keys()))

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": label_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate NLI fact verification model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-key", default=None,
                        help="Model key from config (default: config's default_model)")
    parser.add_argument("--model-path", default=None,
                        help="Override model path/ID (default: model's hub_model_id)")
    parser.add_argument("--data-path", default=None,
                        help="Override path to Arrow eval split")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config)
    model_config, model_key = resolve_model_config(config, args.model_key)

    model_path = args.model_path or model_config["hub_model_id"]

    if args.data_path:
        data_path = args.data_path
    else:
        processed_dir = config["data"]["processed_dir"]
        run_name = config["data"]["run_name"]
        data_path = f"{processed_dir}/{run_name}/eval"

    predictions, labels, id2label = evaluate_model(
        model_path, data_path, config, batch_size=args.batch_size
    )

    metrics = compute_all_metrics(predictions, labels, id2label)

    # Add metadata
    metrics["model"] = model_path
    metrics["model_key"] = model_key
    metrics["data"] = data_path
    metrics["num_examples"] = len(predictions)
    metrics["timestamp"] = datetime.utcnow().isoformat()

    # Sample predictions for inspection
    metrics["samples"] = [
        {"prediction": id2label[p], "true_label": id2label[l]}
        for p, l in zip(predictions[:10], labels[:10])
    ]

    # Print summary
    print(f"\n{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']}")
    print(f"Macro F1:  {metrics['macro_f1']}")
    print(f"\nPer-class:")
    for label_name, m in metrics["per_class"].items():
        print(f"  {label_name:20s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  n={m['support']}")
    print(f"\nConfusion Matrix ({', '.join(metrics['confusion_labels'])}):")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")

    # Save report
    out_dir = config.get("evaluation", {}).get("eval_results_dir", "../../experiments/eval_results")
    os.makedirs(out_dir, exist_ok=True)
    run_name = config["data"]["run_name"]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{run_name}_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
