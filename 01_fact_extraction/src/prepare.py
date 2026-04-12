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
import yaml
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_and_concat(datasets_cfg):
    """
    Load each dataset entry from config, apply optional sampling, and
    concatenate all of them into a single Dataset. Every source dataset
    must already have 'evidence' and 'claim' columns (produced by the
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


def tokenize(dataset, tokenizer, prompt_template, max_input, max_output):
    """Map tokenization over the dataset using the prompt template from config."""
    def preprocess_function(examples):
        inputs = [
            prompt_template.format(evidence=ev)
            for ev in examples["evidence"]
        ]
        model_inputs = tokenizer(inputs, max_length=max_input, truncation=True)
        labels = tokenizer(
            text_target=examples["claim"],
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
