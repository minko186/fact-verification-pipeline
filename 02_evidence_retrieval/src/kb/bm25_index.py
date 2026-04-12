import os
import pickle
import re
from rank_bm25 import BM25Okapi

from .parse_wiki import SentenceRecord


def _tokenize(text):
    """Simple whitespace + lowercasing tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self, config):
        self.index_path = config["index"]["bm25_path"]
        self._bm25 = None
        self._sentence_ids = []

    def build(self, records):
        """
        Build a BM25Okapi index from sentence records and persist to disk.
        """
        print(f"Building BM25 index over {len(records):,} sentences...")
        corpus = [_tokenize(r.text) for r in records]
        self._sentence_ids = [r.sentence_id for r in records]
        self._bm25 = BM25Okapi(corpus)

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "sentence_ids": self._sentence_ids,
            }, f)

        print(f"BM25 index saved to {self.index_path}")

    def load(self):
        """Load a previously built BM25 index from disk."""
        print(f"Loading BM25 index from {self.index_path}...")
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._sentence_ids = data["sentence_ids"]
        print(f"BM25 index loaded: {len(self._sentence_ids):,} sentences")

    def query(self, claim, top_k=100):
        """
        Query the BM25 index with a claim string.

        Returns:
            list of (sentence_id, bm25_score) sorted descending by score.
        """
        tokens = _tokenize(claim)
        scores = self._bm25.get_scores(tokens)

        # Get top-k indices by score
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [
            (self._sentence_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]
