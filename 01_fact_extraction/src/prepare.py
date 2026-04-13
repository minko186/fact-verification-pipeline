"""
Tokenize one or more HF Hub datasets and save Arrow files for training.

Reads the list of datasets from config.yaml, loads and concatenates them,
applies the prompt template, tokenizes, and saves train/eval splits to disk
under data.processed_dir/data.run_name/.

Usage:
    python prepare.py              # uses config.yaml in the same directory
    python prepare.py --config path/to/config.yaml
"""

import argparse
import os
import yaml
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _datasetdict_to_dataset(ds_dict: DatasetDict):
    """Merge all splits into one Dataset (train first, then remaining keys)."""
    keys = list(ds_dict.keys())
    if "train" in keys:
        ordered = ["train"] + [k for k in keys if k != "train"]
    else:
        ordered = sorted(keys)
    chunks = [ds_dict[k] for k in ordered]
    if len(chunks) == 1:
        return chunks[0]
    return concatenate_datasets(chunks)


def load_one_source(entry):
    """
    Load a single dataset from HuggingFace Hub (id) or local disk (path).
    path may be relative to the config file's directory.
    """
    sample_size = entry.get("sample_size", 0)
    local_path = entry.get("path")
    if local_path:
        path = local_path
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(os.getcwd(), path))
        print(f"Loading from disk {path}...")
        raw = load_from_disk(path)
        if isinstance(raw, DatasetDict):
            ds = _datasetdict_to_dataset(raw)
        else:
            ds = raw
    else:
        ds_id = entry["id"]
        print(f"Loading {ds_id}...")
        ds = load_dataset(ds_id, split="train")

    if sample_size and sample_size > 0 and sample_size < len(ds):
        ds = ds.shuffle(seed=42).select(range(sample_size))
    print(f"  {len(ds)} examples")
    return ds


def load_and_concat(datasets_cfg):
    """
    Load each dataset entry from config, apply optional sampling, and
    concatenate all of them into a single Dataset. Every source dataset
    must already have 'evidence' and 'claim' columns (produced by the
    preprocessors in preprocessors/).
    """
    parts = []
    for entry in datasets_cfg:
        parts.append(load_one_source(entry))

    combined = concatenate_datasets(parts)
    combined = combined.shuffle(seed=42)
    print(f"\nTotal examples after concatenation: {len(combined)}")
    return combined


def tokenize(dataset, tokenizer, prompt_template, max_input, max_output):
    """Map tokenization over the dataset using the prompt template from config."""
    def preprocess_function(examples):
        evidences = [str(ev) if ev is not None else "" for ev in examples["evidence"]]
        claims = [str(c) if c is not None else "" for c in examples["claim"]]
        inputs = [prompt_template.format(evidence=ev) for ev in evidences]
        model_inputs = tokenizer(inputs, max_length=max_input, truncation=True)
        labels = tokenizer(
            text_target=claims,
            max_length=max_output,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(preprocess_function, batched=True)
    tokenized.set_format(type=None, columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Tokenize datasets for fact extraction training")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    model_name = config["model"]["name"]
    prompt_template = config["prompt"]["template"]
    max_input = config["prompt"]["max_input_length"]
    max_output = config["prompt"]["max_output_length"]

    processed_dir = config["data"]["processed_dir"]
    run_name = config["data"]["run_name"]
    save_path = f"{processed_dir}/{run_name}"

    print(f"Model / tokenizer: {model_name}")
    print(f"Prompt template:   {prompt_template}")
    print(f"Save path:         {save_path}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    combined = load_and_concat(config["datasets"])

    split = combined.train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    print(f"\nTokenizing {len(train_ds)} train examples...")
    tokenized_train = tokenize(train_ds, tokenizer, prompt_template, max_input, max_output)

    print(f"Tokenizing {len(eval_ds)} eval examples...")
    tokenized_eval = tokenize(eval_ds, tokenizer, prompt_template, max_input, max_output)

    tokenized_train.save_to_disk(f"{save_path}/train")
    tokenized_eval.save_to_disk(f"{save_path}/eval")

    print(f"\nSaved tokenized datasets to {save_path}/")


if __name__ == "__main__":
    main()
