"""
Dense retrieval index backed by FAISS with sentence-transformers embeddings.

Build workflow:
    1. Encode all sentence records with a sentence-transformer model on GPU.
    2. Train an IVF-PQ FAISS index on a sample, then add all vectors.
    3. Save index.faiss + sentence_ids.pkl to disk.

Query workflow:
    1. Encode the claim.
    2. Search the FAISS index for nearest neighbors.
    3. Return (sentence_id, distance) pairs.
"""

import json
import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .parse_wiki import SentenceRecord


class DenseIndex:
    def __init__(self, config):
        self.faiss_path = config["index"]["faiss_path"]
        self.model_name = config["embedding"]["model_name"]
        self.batch_size = config["embedding"].get("batch_size", 2048)
        self.device = config["embedding"].get("device", "cuda")

        self._model = None
        self._index = None
        self._sentence_ids = []

    def _get_model(self):
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def build(self, records):
        """
        Encode all records and build a compressed FAISS IVF-PQ index.
        """
        model = self._get_model()
        dim = model.get_sentence_embedding_dimension()
        total = len(records)

        self._sentence_ids = [r.sentence_id for r in records]

        print(f"Encoding {total:,} sentences (dim={dim}, batch_size={self.batch_size})...")

        all_embeddings = []
        for start in tqdm(range(0, total, self.batch_size), desc="Encoding"):
            batch_texts = [r.text for r in records[start : start + self.batch_size]]
            embs = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.append(embs.astype(np.float32))

        embeddings = np.vstack(all_embeddings)
        print(f"Embeddings shape: {embeddings.shape}")

        n_vectors = embeddings.shape[0]
        # IVF-PQ parameters: nlist clusters, m sub-quantizers of nbits each
        # nlist ~ sqrt(n) is a good rule of thumb, capped for practicality
        nlist = min(int(np.sqrt(n_vectors)), 4096)
        m = 48  # sub-quantizers (must divide dim=768 evenly: 768/48=16)
        nbits = 8

        print(f"Training IVF{nlist}_PQ{m}x{nbits} index on {min(n_vectors, 500_000):,} vectors...")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

        # Train on a random sample for speed
        train_size = min(n_vectors, 500_000)
        if train_size < n_vectors:
            rng = np.random.default_rng(42)
            train_indices = rng.choice(n_vectors, size=train_size, replace=False)
            train_vecs = embeddings[train_indices]
        else:
            train_vecs = embeddings

        index.train(train_vecs)
        print("Index trained. Adding vectors...")

        # Add in chunks to avoid memory spikes
        add_batch = 100_000
        for start in tqdm(range(0, n_vectors, add_batch), desc="Adding to index"):
            index.add(embeddings[start : start + add_batch])

        index.nprobe = 32

        # Save to disk
        os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)
        faiss.write_index(index, self.faiss_path)

        ids_path = self.faiss_path.replace(".faiss", "_ids.pkl")
        with open(ids_path, "wb") as f:
            pickle.dump(self._sentence_ids, f)

        meta_path = self.faiss_path.replace(".faiss", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "model_name": self.model_name,
                "dim": dim,
                "n_vectors": n_vectors,
                "nlist": nlist,
                "m": m,
                "nbits": nbits,
                "nprobe": 32,
            }, f, indent=2)

        self._index = index
        print(f"FAISS index saved: {n_vectors:,} vectors -> {self.faiss_path}")

    def load(self):
        """Load a previously built FAISS index and sentence ID mapping."""
        print(f"Loading FAISS index from {self.faiss_path}...")
        self._index = faiss.read_index(self.faiss_path)
        self._index.nprobe = 32

        ids_path = self.faiss_path.replace(".faiss", "_ids.pkl")
        with open(ids_path, "rb") as f:
            self._sentence_ids = pickle.load(f)

        print(f"FAISS index loaded: {self._index.ntotal:,} vectors")

    def query(self, claim, top_k=100):
        """
        Embed the claim and search the FAISS index for nearest neighbors.

        Returns:
            list of (sentence_id, similarity_score) sorted descending by score.
        """
        model = self._get_model()
        q_emb = model.encode([claim], normalize_embeddings=True).astype(np.float32)

        scores, indices = self._index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self._sentence_ids[idx], float(score)))

        return results
