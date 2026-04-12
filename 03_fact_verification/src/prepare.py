"""
Tokenize one or more HF Hub NLI datasets and save Arrow files for training.

Reads the list of datasets from config.yaml, loads and concatenates them,
tokenizes (claim, evidence) pairs for sequence classification, and saves
train/eval splits to disk under data.processed_dir/data.run_name/.

Usage:
    python prepare.py                          # uses default model from config
    python prepare.py --model-key deberta-base
    python prepare.py --config path/to/config.yaml --model-key roberta-mnli
"""

import argparse
import yaml
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


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


def load_and_concat(datasets_cfg):
    """
    Load each dataset entry from config, apply optional sampling, and
    concatenate all of them into a single Dataset. Every source dataset
    must have 'claim', 'evidence', and 'label' columns (produced by the
    preprocessors in preprocessors/).
    """
    parts = []
    for entry in datasets_cfg:
        ds_id = entry["id"]
        sample_size = entry.get("sample_size", 0)
        print(f"Loading {ds_id}...")
        ds = load_dataset(ds_id, split="train")
        if sample_size and sample_size > 0 and sample_size < len(ds):
            ds = ds.shuffle(seed=42).select(range(sample_size))
        print(f"  {len(ds)} examples")
        parts.append(ds)

    combined = concatenate_datasets(parts)
    combined = combined.shuffle(seed=42)
    print(f"\nTotal examples after concatenation: {len(combined)}")
    return combined


def tokenize(dataset, tokenizer, max_length, label2id):
    """
    Tokenize (claim, evidence) pairs for sequence classification.

    The tokenizer handles pair encoding automatically:
    [CLS] claim [SEP] evidence [SEP] for BERT-style models.
    """
    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["claim"],
            examples["evidence"],
            max_length=max_length,
            truncation=True,
            padding=False,  # dynamic padding via DataCollatorWithPadding at train time
        )
        tokenized["labels"] = [label2id[l] for l in examples["label"]]
        return tokenized

    tokenized = dataset.map(preprocess_function, batched=True)

    # Keep only the columns needed for training
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized.column_names:
        columns_to_keep.append("token_type_ids")
    tokenized.set_format(type=None, columns=columns_to_keep)

    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Tokenize datasets for NLI fact verification")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-key", default=None,
                        help="Model key from config (default: config's default_model)")
    args = parser.parse_args()

    config = load_config(args.config)
    model_config, model_key = resolve_model_config(config, args.model_key)

    model_name = model_config["name"]
    max_length = model_config.get("max_length", 512)

    label2id = config["labels"]

    processed_dir = config["data"]["processed_dir"]
    run_name = config["data"]["run_name"]
    save_path = f"{processed_dir}/{run_name}"

    print(f"Model / tokenizer: {model_name} (key: {model_key})")
    print(f"Max length:        {max_length}")
    print(f"Labels:            {label2id}")
    print(f"Save path:         {save_path}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    combined = load_and_concat(config["datasets"])

    split = combined.train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    print(f"\nTokenizing {len(train_ds)} train examples...")
    tokenized_train = tokenize(train_ds, tokenizer, max_length, label2id)

    print(f"Tokenizing {len(eval_ds)} eval examples...")
    tokenized_eval = tokenize(eval_ds, tokenizer, max_length, label2id)

    tokenized_train.save_to_disk(f"{save_path}/train")
    tokenized_eval.save_to_disk(f"{save_path}/eval")

    print(f"\nSaved tokenized datasets to {save_path}/")
    print(f"  Train: {len(tokenized_train)} examples")
    print(f"  Eval:  {len(tokenized_eval)} examples")


if __name__ == "__main__":
    main()
