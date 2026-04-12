"""
Post-training evaluation for a fact extraction model.

Loads a trained model (from HF Hub or local path), runs it over a test split,
and computes ROUGE-1/2/L plus BERTScore. Writes a JSON report to
../../experiments/eval_results/<run_name>.json.

Usage:
    python evaluate.py
    python evaluate.py --config config.yaml --model-path minko186/flan-t5-base-fact-extraction-v2
    python evaluate.py --data-path ../../data/processed/fact_extraction/flan-t5-base-fever-vitaminc-wice/eval
"""

import argparse
import json
import os
from datetime import datetime

import torch
import yaml
import numpy as np
import evaluate as hf_evaluate
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_claims(examples, tokenizer, model, device, prompt_template,
                    max_input, max_output, num_beams):
    """Run batch generation and return decoded predictions."""
    inputs_text = [prompt_template.format(evidence=ev) for ev in examples["evidence"]]
    inputs = tokenizer(
        inputs_text,
        return_tensors="pt",
        max_length=max_input,
        truncation=True,
        padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_output,
            num_beams=num_beams,
            early_stopping=True,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def evaluate_model(model_path, data_path, config, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    prompt_template = config["prompt"]["template"]
    max_input = config["prompt"]["max_input_length"]
    max_output = config["prompt"]["max_output_length"]
    num_beams = config.get("inference", {}).get("num_beams", 4)
    eval_cfg = config.get("evaluation", {})
    metrics_to_run = eval_cfg.get("metrics", ["rouge"])
    bertscore_model = eval_cfg.get("bertscore_model", "microsoft/deberta-xlarge-mnli")

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    print(f"Loading test data from: {data_path}")
    dataset = load_from_disk(data_path)

    # If the dataset has evidence/claim columns use them directly;
    # otherwise it's a tokenized dataset — we can still decode labels.
    has_text_columns = "evidence" in dataset.column_names and "claim" in dataset.column_names

    all_predictions = []
    all_references = []

    if has_text_columns:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            preds = generate_claims(
                batch, tokenizer, model, device,
                prompt_template, max_input, max_output, num_beams,
            )
            all_predictions.extend(preds)
            all_references.extend(batch["claim"])
            if (i // batch_size) % 10 == 0:
                print(f"  {i + len(preds)}/{len(dataset)} examples processed")
    else:
        # Tokenized dataset — decode labels and run generation from input_ids
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_output,
                    num_beams=num_beams,
                    early_stopping=True,
                )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            labels = np.array(batch["labels"])
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            refs = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
            all_predictions.extend(preds)
            all_references.extend(refs)
            if (i // batch_size) % 10 == 0:
                print(f"  {i + len(preds)}/{len(dataset)} examples processed")

    results = {}

    if "rouge" in metrics_to_run:
        print("Computing ROUGE...")
        rouge = hf_evaluate.load("rouge")
        rouge_scores = rouge.compute(
            predictions=all_predictions,
            references=all_references,
            use_stemmer=True,
        )
        results["rouge"] = {k: round(v, 4) for k, v in rouge_scores.items()}
        print("ROUGE:", results["rouge"])

    if "bertscore" in metrics_to_run:
        try:
            print(f"Computing BERTScore (model: {bertscore_model})...")
            bertscore = hf_evaluate.load("bertscore")
            bs = bertscore.compute(
                predictions=all_predictions,
                references=all_references,
                model_type=bertscore_model,
                device=str(device),
            )
            results["bertscore"] = {
                "precision": round(float(np.mean(bs["precision"])), 4),
                "recall": round(float(np.mean(bs["recall"])), 4),
                "f1": round(float(np.mean(bs["f1"])), 4),
            }
            print("BERTScore:", results["bertscore"])
        except Exception as e:
            print(f"BERTScore failed: {e}")
            results["bertscore_error"] = str(e)

    results["model"] = model_path
    results["data"] = data_path
    results["num_examples"] = len(all_predictions)
    results["timestamp"] = datetime.utcnow().isoformat()

    # Sample predictions for inspection
    results["samples"] = [
        {"prediction": p, "reference": r}
        for p, r in zip(all_predictions[:5], all_references[:5])
    ]

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fact extraction model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-path", default=None,
                        help="Override model path/ID (default: model.hub_model_id from config)")
    parser.add_argument("--data-path", default=None,
                        help="Override path to Arrow eval split (default: derived from config)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config = load_config(args.config)

    model_path = args.model_path or config["model"]["hub_model_id"]

    if args.data_path:
        data_path = args.data_path
    else:
        processed_dir = config["data"]["processed_dir"]
        run_name = config["data"]["run_name"]
        data_path = f"{processed_dir}/{run_name}/eval"

    results = evaluate_model(model_path, data_path, config, batch_size=args.batch_size)

    out_dir = "../../experiments/eval_results"
    os.makedirs(out_dir, exist_ok=True)
    run_name = config["data"]["run_name"]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{run_name}_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
