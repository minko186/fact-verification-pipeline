import os
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi

def create_wiki_lookup(wiki_dataset):
    """
    Creates a dictionary for fast lookup of Wikipedia page content.
    The key is the page title (ID) and the value is the text content.
    """
    print("Creating a lookup dictionary for Wikipedia pages... This may take a moment.")
    wiki_lookup = {}
    # The actual data is in the 'wikipedia_pages' split
    for row in wiki_dataset['wikipedia_pages']:
        # The 'id' is the Wikipedia page title (e.g., "Barack_Obama")
        # 'lines' contains all sentences, each preceded by its line number and a tab.
        wiki_lookup[row['id']] = row['lines']
    return wiki_lookup

def process_split(split, wiki_lookup):
    """
    Processes a single split of the FEVER dataset to create context-claim pairs.
    """
    # 1. We only want claims that have evidence.
    print(f"Filtering '{split.split_info.name}' split for 'SUPPORTS' and 'REFUTES' labels...")
    processed_split = split.filter(
        lambda example: example['label'] in ['SUPPORTS', 'REFUTES'] and example.get('evidence')
    )

    def get_evidence_text(example):
        """
        For a given claim, finds all its evidence sentences from the wiki_lookup dictionary.
        """
        context_sentences = []
        # The 'evidence' field has a complex nested list structure. We'll parse it.
        # It's a list of evidence sets, we'll use all evidence from the first set.
        evidence_set = example['evidence'][0] if example['evidence'] else []
        
        for evidence in evidence_set:
            # Each evidence item is a list: [annotation_id, evidence_id, wiki_url, sentence_id]
            wiki_url = evidence[2]
            sentence_id = evidence[3]

            # Check if the evidence pointers are valid and the wiki page exists in our lookup
            if wiki_url is not None and sentence_id is not None and wiki_url in wiki_lookup:
                page_content = wiki_lookup[wiki_url]
                lines = page_content.split('\n')
                
                # The sentence_id corresponds to the line number. We find the matching line.
                target_line_prefix = f"{sentence_id}\t"
                found_line = next((line for line in lines if line.startswith(target_line_prefix)), None)
                
                if found_line:
                    # Extract the sentence text, removing the line number and tab.
                    sentence_text = found_line.split('\t', 1)[1]
                    context_sentences.append(sentence_text)
            
        return {
            "context": " ".join(context_sentences),
            "claim": example["claim"]
        }

    print(f"Mapping evidence text to claims for '{split.split_info.name}' split...")
    # Apply the function to create the new columns
    processed_split = processed_split.map(get_evidence_text, remove_columns=split.column_names)
    
    # *** NEW LOGIC: Filter out examples where no evidence text was found ***
    initial_rows = len(processed_split)
    processed_split = processed_split.filter(lambda example: len(example['context']) > 0)
    final_rows = len(processed_split)
    
    print(f"Filtered out {initial_rows - final_rows} examples with no found evidence from '{split.split_info.name}'.")
    
    return processed_split

def main():
    # 1. Load both the FEVER claims data and the Wikipedia pages data
    print("Loading FEVER v2.0 dataset for claims...")
    claims_dataset = load_dataset("fever", "v2.0")
    
    print("Loading FEVER wiki_pages dataset for evidence text...")
    wiki_dataset = load_dataset("fever", "wiki_pages")

    # 2. Create the fast lookup dictionary from the wiki pages
    wiki_lookup = create_wiki_lookup(wiki_dataset)

    # 3. Process the relevant splits
    splits_to_process = ['train', 'labelled_dev', 'paper_dev', 'paper_test']
    
    t5_ready_dataset = DatasetDict()
    for split_name in splits_to_process:
        if split_name in claims_dataset:
            t5_ready_dataset[split_name] = process_split(claims_dataset[split_name], wiki_lookup)
        else:
            print(f"Split '{split_name}' not found in the dataset. Skipping.")

    print("\nDataset preparation complete! ✨")
    print("Final dataset structure:")
    print(t5_ready_dataset)

    # 4. Push the final dataset to the Hugging Face Hub
    # Make sure to replace 'your-username'
    repo_id = "your-username/fever-claim-extraction-clean" # Changed the name slightly
    print(f"\nPushing dataset to the Hugging Face Hub at: {repo_id}")
    
    try:
        t5_ready_dataset.push_to_hub(repo_id)
        print("Dataset pushed successfully! 🎉")
        print(f"Access it here: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Failed to push to hub. Error: {e}")

if __name__ == "__main__":
    main()