import hashlib
from huggingface_hub import HfApi


def generate_content_hash_id(text: str) -> str:
    """
    Generate a unique ID for a document based on its content using SHA-256.

    Args:
        text (str): The content of the document.

    Returns:
        str: A unique SHA-256 hash string representing the content.
    """
    # Normalize text to avoid minor formatting causing different hashes
    normalized_text = text.strip().replace("\r\n", "\n").replace("\r", "\n")

    # Encode the normalized text to bytes
    text_bytes = normalized_text.encode("utf-8")

    # Generate the SHA-256 hash
    hash_object = hashlib.sha256(text_bytes)

    # Return the hexadecimal digest
    return hash_object.hexdigest()


def upload_dataset_to_collection(dataset, dataset_name, collection_title="AID Training Datasets", owner="polygraf-ai"):
    """Upload dataset and add to collection by title"""

    # 1. Upload dataset
    dataset.push_to_hub(dataset_name, private=True)
    print(f"✅ Uploaded dataset: {dataset_name}")

    # 2. Find collection slug by title
    api = HfApi()
    collections = api.list_collections(owner=owner)

    collection_slug = None
    for collection in collections:
        if collection.title == collection_title:
            collection_slug = collection.slug
            break

    if not collection_slug:
        print(f"❌ Collection '{collection_title}' not found")
        return

    # 3. Add to collection (skip if already exists)
    try:
        api.add_collection_item(collection_slug=collection_slug, item_id=dataset_name, item_type="dataset")
        print(f"✅ Added to collection: {collection_title}")
    except Exception as e:
        print(f"ℹ️ Dataset '{dataset_name}' is already in collection '{collection_title}'")
