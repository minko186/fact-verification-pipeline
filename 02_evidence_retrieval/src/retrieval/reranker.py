"""
NLI-based cross-encoder reranker for stance-neutral evidence scoring.

Uses an NLI model that outputs P(ENTAILMENT), P(NEUTRAL), P(CONTRADICTION).
Relevance score = P(ENTAILMENT) + P(CONTRADICTION), so both supporting
and refuting evidence are ranked above unrelated text.
"""

import torch
import numpy as np
from sentence_transformers import CrossEncoder


class NLIReranker:
    def __init__(self, model_name, device="cuda", batch_size=32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            print(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, claim, candidates, candidate_ids, top_n=10):
        """
        Score each (claim, candidate) pair using cross-encoder NLI.

        The NLI model outputs logits for [CONTRADICTION, NEUTRAL, ENTAILMENT].
        We compute: relevance = P(ENTAILMENT) + P(CONTRADICTION)
        This is stance-neutral: both supporting and refuting evidence score high.
        Only truly unrelated (NEUTRAL) evidence is down-ranked.

        Args:
            claim: the claim string to check.
            candidates: list of candidate sentence texts.
            candidate_ids: list of sentence_ids matching candidates.
            top_n: number of top results to return.

        Returns:
            list of (sentence_id, text, relevance_score) sorted descending.
        """
        if not candidates:
            return []

        model = self._get_model()

        # Create (claim, candidate) pairs
        pairs = [(claim, cand) for cand in candidates]

        # Score in batches
        all_logits = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            logits = model.predict(batch, apply_softmax=True)
            all_logits.append(logits)

        # Concatenate all batch results
        scores_matrix = np.vstack(all_logits)

        # NLI label order for cross-encoder/nli-deberta-v3-base:
        # [contradiction, neutral, entailment]
        # Relevance = P(entailment) + P(contradiction) = 1 - P(neutral)
        relevance_scores = 1.0 - scores_matrix[:, 1]

        # Sort by relevance descending
        ranked_indices = np.argsort(relevance_scores)[::-1][:top_n]

        results = [
            (
                candidate_ids[i],
                candidates[i],
                float(relevance_scores[i]),
            )
            for i in ranked_indices
        ]

        return results
