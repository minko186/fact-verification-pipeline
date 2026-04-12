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
from huggingface_hub import HfFolder


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_compute_metrics(tokenizer, metric):
    """Return a compute_metrics function that closes over tokenizer and metric."""
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple) or (hasattr(preds, "shape") and len(preds.shape) > 1):
            preds = np.argmax(preds, axis=-1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
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

    training_args = Seq2SeqTrainingArguments(
        **training_cfg,
        hub_token=HfFolder.get_token(),
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete")

    print(f"Pushing model to Hub: {hub_model_id}")
    trainer.push_to_hub()
    print("Model pushed")


if __name__ == "__main__":
    main()
