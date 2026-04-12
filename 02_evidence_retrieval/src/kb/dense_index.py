import os
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .parse_wiki import SentenceRecord


class DenseIndex:
    def __init__(self, config):
        self.chroma_path = config["index"]["chroma_path"]
        self.collection_name = config["index"]["chroma_collection"]
        self.model_name = config["embedding"]["model_name"]
        self.batch_size = config["embedding"].get("batch_size", 512)
        self.device = config["embedding"].get("device", "cuda")

        self._model = None
        self._collection = None
        self._client = None

    def _get_model(self):
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _get_client(self):
        if self._client is None:
            os.makedirs(self.chroma_path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.chroma_path)
        return self._client

    def build(self, records):
        """
        Embed all sentence records and store in a persistent ChromaDB collection.
        """
        model = self._get_model()
        client = self._get_client()

        # Delete existing collection if it exists, to rebuild cleanly
        try:
            client.delete_collection(self.collection_name)
        except ValueError:
            pass

        collection = client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        total = len(records)
        print(f"Embedding and indexing {total:,} sentences (batch_size={self.batch_size})...")

        for start in tqdm(range(0, total, self.batch_size), desc="Indexing"):
            batch = records[start : start + self.batch_size]
            texts = [r.text for r in batch]
            ids = [r.sentence_id for r in batch]
            metadatas = [{"article": r.article_title, "line": r.line_number} for r in batch]

            embeddings = model.encode(texts, show_progress_bar=False).tolist()

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        self._collection = collection
        print(f"Dense index built: {collection.count():,} vectors in {self.chroma_path}")

    def load(self):
        """Open an existing persistent ChromaDB collection."""
        client = self._get_client()
        self._collection = client.get_collection(self.collection_name)
        print(
            f"Dense index loaded: {self._collection.count():,} vectors "
            f"from {self.chroma_path}"
        )

    def query(self, claim, top_k=100):
        """
        Embed the claim and query ChromaDB for nearest neighbors.

        Returns:
            list of (sentence_id, cosine_distance) sorted ascending by distance.
            Lower distance = more similar.
        """
        model = self._get_model()
        embedding = model.encode([claim]).tolist()

        results = self._collection.query(
            query_embeddings=embedding,
            n_results=top_k,
            include=["distances"],
        )

        ids = results["ids"][0]
        distances = results["distances"][0]

        return list(zip(ids, distances))
