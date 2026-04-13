"""
Fine-tune a seq2seq model for atomic claim extraction.

Reads configuration from config.yaml. Supports any AutoModelForSeq2SeqLM-compatible
model (FLAN-T5 variants, BART, etc.) via the model.type field in config.

Usage:
    python train.py
    python train.py --config path/to/config.yaml
"""

import argparse
import torch
import yaml
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from huggingface_hub import get_token


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _sanitize_token_ids_for_decode(ids: np.ndarray, tokenizer) -> np.ndarray:
    """
    Rust fast tokenizers reject out-of-range IDs (OverflowError / segfault).
    Clip to valid vocab indices and use a concrete integer dtype.
    """
    ids = np.asarray(ids)
    if ids.dtype.kind == "f":
        ids = np.rint(ids).astype(np.int64)
    else:
        ids = ids.astype(np.int64, copy=False)
    max_id = max(len(tokenizer) - 1, 0)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids = np.where(ids < 0, pad, ids)
    ids = np.clip(ids, 0, max_id)
    return ids


def make_compute_metrics(tokenizer, metric):
    """Return a compute_metrics function that closes over tokenizer and metric."""
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.asarray(preds)
        # Teacher-forcing logits (3D) vs. generated token ids (2D) from predict_with_generate.
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
        preds = _sanitize_token_ids_for_decode(preds, tokenizer)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = _sanitize_token_ids_for_decode(labels, tokenizer)
        decoded_preds = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
        return metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Train fact extraction model")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = config["model"]["name"]
    model_type = config["model"].get("type", "seq2seq")
    hub_model_id = config["model"]["hub_model_id"]

    print(f"Model: {model_name}  (type: {model_type})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Only 'seq2seq' is supported.")

    if config["training_args"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    processed_dir = config["data"]["processed_dir"]
    run_name = config["data"]["run_name"]
    data_path = f"{processed_dir}/{run_name}"

    print(f"Loading tokenized data from: {data_path}")
    train_dataset = load_from_disk(f"{data_path}/train")
    eval_dataset = load_from_disk(f"{data_path}/eval")

    rouge_metric = evaluate.load("rouge")
    compute_metrics = make_compute_metrics(tokenizer, rouge_metric)

    # Build training args — extract keys that Seq2SeqTrainingArguments accepts,
    # then inject hub_model_id from the model section.
    training_cfg = dict(config["training_args"])
    training_cfg.pop("gradient_checkpointing", None)  # handled manually above
    training_cfg["hub_model_id"] = hub_model_id
    # Eval with predict_with_generate needs explicit caps (defaults are too small / unstable).
    if "generation_max_length" not in training_cfg:
        mol = config.get("prompt", {}).get("max_output_length")
        if mol is not None:
            training_cfg["generation_max_length"] = mol
    if "generation_num_beams" not in training_cfg:
        training_cfg["generation_num_beams"] = config.get("inference", {}).get("num_beams", 4)
    push_to_hub = training_cfg.get("push_to_hub", False)
    hub_token = get_token() if push_to_hub else None

    training_args = Seq2SeqTrainingArguments(
        **training_cfg,
        hub_token=hub_token,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete")

    if push_to_hub:
        print(f"Pushing model to Hub: {hub_model_id}")
        trainer.push_to_hub()
        print("Model pushed")
    else:
        print("push_to_hub is false; model saved under output_dir only.")


if __name__ == "__main__":
    main()
