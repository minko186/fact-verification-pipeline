"""
Fine-tune a sequence classification model for NLI-based fact verification.

Reads configuration from config.yaml. Supports any AutoModelForSequenceClassification-
compatible model (DeBERTa, RoBERTa, etc.) via the models section in config.

The model classifies (claim, evidence) pairs into three labels:
  SUPPORTS (0), REFUTES (1), NOT ENOUGH INFO (2)

Usage:
    python train.py                          # uses default model from config
    python train.py --model-key deberta-base
    python train.py --config path/to/config.yaml --model-key deberta-large
"""

import argparse
import torch
import yaml
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from huggingface_hub import HfFolder


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


def make_compute_metrics():
    """Return a compute_metrics function for accuracy and macro F1."""
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(
            predictions=predictions, references=labels, average="macro"
        )
        return {
            "accuracy": acc["accuracy"],
            "macro_f1": f1["f1"],
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Train NLI fact verification model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-key", default=None,
                        help="Model key from config (default: config's default_model)")
    args = parser.parse_args()

    config = load_config(args.config)
    model_config, model_key = resolve_model_config(config, args.model_key)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = model_config["name"]
    hub_model_id = model_config["hub_model_id"]

    # Build label mappings
    label2id = config["labels"]  # {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    id2label = {v: k for k, v in label2id.items()}
    num_labels = config["num_labels"]

    print(f"Model: {model_name} (key: {model_key})")
    print(f"Labels: {label2id}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    ).to(device)

    if config["training_args"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    # Load tokenized data
    processed_dir = config["data"]["processed_dir"]
    run_name = config["data"]["run_name"]
    data_path = f"{processed_dir}/{run_name}"

    print(f"Loading tokenized data from: {data_path}")
    train_dataset = load_from_disk(f"{data_path}/train")
    eval_dataset = load_from_disk(f"{data_path}/eval")

    # Build training args
    training_cfg = dict(config["training_args"])
    training_cfg.pop("gradient_checkpointing", None)  # handled manually above
    training_cfg["hub_model_id"] = hub_model_id

    training_args = TrainingArguments(
        **training_cfg,
        hub_token=HfFolder.get_token(),
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    compute_metrics = make_compute_metrics()

    trainer = Trainer(
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
    print("Training complete.")

    print(f"Pushing model to Hub: {hub_model_id}")
    trainer.push_to_hub()
    print("Model pushed.")


if __name__ == "__main__":
    main()
