import os
from datasets import load_dataset, DatasetDict, Dataset
from huggingface_hub import HfApi

    
repo_id = "minko186/fever-fact-extraction-supports"

def create_wiki_lookup(wiki_dataset):
    """
    Creates a dictionary for fast lookup of Wikipedia page content.
    The key is the page title (ID) and the value is the text content.
    """
    print("Creating a lookup dictionary for Wikipedia pages... This may take a moment.")
    wiki_lookup = {}
    # The actual data is in the 'wikipedia_pages' split
    for row in wiki_dataset["wikipedia_pages"]:
        wiki_lookup[row["id"]] = row["lines"]
    return wiki_lookup

def get_evidence_text(example, wiki_lookup):
    """
    For a given claim, finds all its evidence sentences from the wiki_lookup dictionary.
    Returns the concatenated context string.
    """
    context_sentences = []
    
    # Access the evidence fields directly from the example dictionary
    wiki_url = example.get("evidence_wiki_url")
    sentence_id = example.get("evidence_sentence_id")

    if wiki_url is not None and sentence_id is not None and wiki_url in wiki_lookup:
        page_content = wiki_lookup[wiki_url]
        lines = page_content.split("\n")

        # The sentence_id corresponds to the line number. We find the matching line.
        target_line_prefix = f"{sentence_id}\t"
        found_line = next((line for line in lines if line.startswith(target_line_prefix)), None)

        if found_line:
            sentence_text = found_line.split("\t", 1)[1]
            context_sentences.append(sentence_text)

    return " ".join(context_sentences)


def process_split_for_T5_extraction(split, wiki_lookup):
    """
    Processes a single split, grouping claims that share the same context.
    """
    print(f"Filtering split for 'SUPPORTS' labels...")
    processed_split = split.filter(
        lambda example: example["label"] in ["SUPPORTS"] and example.get("evidence_wiki_url")
    )

    context_to_claims = {}

    print(f"Grouping claims by context for split...")
    for example in processed_split:
        context = get_evidence_text(example, wiki_lookup)
        claim = example["claim"]

        if len(context) > 0:
            if context not in context_to_claims:
                context_to_claims[context] = set() # Use a set to store unique claims
            context_to_claims[context].add(claim) # Add to the set

    new_data = []
    for context, claims_set in context_to_claims.items():
        # Convert the set to a list and join the unique claims
        combined_claims = " || ".join(list(claims_set))
        new_data.append({"context": context, "claims": combined_claims})

    print(f"Created a new dataset with {len(new_data)} unique contexts for .")
    return Dataset.from_list(new_data)


def main():
    print("Loading FEVER v1.0 dataset for claims...")
    claims_dataset = load_dataset("fever", "v1.0", trust_remote_code=True)
    
    print("Loading FEVER wiki_pages dataset for evidence text...")
    wiki_dataset = load_dataset("fever", "wiki_pages", trust_remote_code=True)
    
    wiki_lookup = create_wiki_lookup(wiki_dataset)

    splits_to_process = ["train", "dev_labelled"] # Changed to 'dev_labelled'
    
    t5_ready_dataset = DatasetDict()
    for split_name in splits_to_process:
        if split_name in claims_dataset:
            t5_ready_dataset[split_name] = process_split_for_T5_extraction(claims_dataset[split_name], wiki_lookup)
        else:
            print(f"Split '{split_name}' not found in the dataset. Skipping.")

    print("\nDataset preparation complete! ✨")
    print("Final dataset structure:")
    print(t5_ready_dataset)
    
    if "train" in t5_ready_dataset and len(t5_ready_dataset["train"]) > 0:
        print("\nExample data point from the 'train' split:")
        print(t5_ready_dataset["train"][0])

    print(f"\nPushing dataset to the Hugging Face Hub at: {repo_id}")
    
    try:
        t5_ready_dataset.push_to_hub(repo_id)
        print("Dataset pushed successfully! 🎉")
        print(f"Access it here: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Failed to push to hub. Error: {e}")

if __name__ == "__main__":
    main()