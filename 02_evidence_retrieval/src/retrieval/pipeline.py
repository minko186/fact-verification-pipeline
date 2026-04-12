"""
EvidencePipeline: orchestrates hybrid retrieval for a fact-checking pipeline.

Flow:
    claim -> [BM25 + Dense + Graph] -> RRF fusion -> NLI reranker -> RetrievalResult
"""

import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml
from tqdm import tqdm

from kb.parse_wiki import load_records
from kb.bm25_index import BM25Index
from kb.dense_index import DenseIndex
from kb.graph_index import GraphIndex
from retrieval.fusion import reciprocal_rank_fusion_with_sources
from retrieval.reranker import NLIReranker


@dataclass
class RetrievalResult:
    claim: str
    evidence: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    reranker_scores: list[float] = field(default_factory=list)
    source_channels: list[list[str]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _resolve_path(base_dir, path):
    return os.path.normpath(os.path.join(base_dir, path))


class EvidencePipeline:
    def __init__(self, config_path="config.yaml"):
        config_path = os.path.abspath(config_path)
        base_dir = os.path.dirname(config_path)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Resolve all paths relative to config location
        self.config["corpus"]["records_path"] = _resolve_path(
            base_dir, self.config["corpus"]["records_path"]
        )
        self.config["index"]["bm25_path"] = _resolve_path(
            base_dir, self.config["index"]["bm25_path"]
        )
        self.config["index"]["chroma_path"] = _resolve_path(
            base_dir, self.config["index"]["chroma_path"]
        )
        self.config["index"]["graph_path"] = _resolve_path(
            base_dir, self.config["index"]["graph_path"]
        )

        # Load sentence records for text lookup
        self._records_by_id = {}
        records = load_records(self.config["corpus"]["records_path"])
        for rec in records:
            self._records_by_id[rec.sentence_id] = rec.text

        # Load indexes
        self._bm25 = BM25Index(self.config)
        self._bm25.load()

        self._dense = DenseIndex(self.config)
        self._dense.load()

        self._graph = GraphIndex(self.config)
        self._graph.load()

        # Initialize reranker
        reranker_cfg = self.config["reranker"]
        self._reranker = NLIReranker(
            model_name=reranker_cfg["model_name"],
            device=reranker_cfg.get("device", "cuda"),
            batch_size=reranker_cfg.get("batch_size", 32),
        )

        # Retrieval params
        self._bm25_top_k = self.config["retrieval"]["bm25_top_k"]
        self._dense_top_k = self.config["retrieval"]["dense_top_k"]
        self._graph_top_k = self.config["retrieval"]["graph_top_k"]
        self._graph_max_hops = self.config["retrieval"]["graph_max_hops"]
        self._rrf_k = self.config["fusion"]["rrf_k"]
        self._pool_size = self.config["fusion"]["candidate_pool_size"]
        self._final_top_n = reranker_cfg.get("final_top_n", 10)

    def retrieve(self, claim):
        """
        Full hybrid retrieval pipeline for a single claim.

        Returns a RetrievalResult with the top-n evidence sentences.
        """
        start_time = time.time()

        # Stage 1: Run all three retrieval channels
        bm25_results = self._bm25.query(claim, top_k=self._bm25_top_k)
        dense_results = self._dense.query(claim, top_k=self._dense_top_k)
        graph_results = self._graph.query(
            claim, top_k=self._graph_top_k, max_hops=self._graph_max_hops
        )

        # Extract just the sentence IDs in rank order for RRF
        bm25_ids = [sid for sid, _ in bm25_results]
        dense_ids = [sid for sid, _ in dense_results]
        graph_ids = [sid for sid, _ in graph_results]

        # Fuse with RRF (tracks source channels)
        fused = reciprocal_rank_fusion_with_sources(
            ranked_lists=[bm25_ids, dense_ids, graph_ids],
            channel_names=["bm25", "dense", "graph"],
            k=self._rrf_k,
            pool_size=self._pool_size,
        )

        # Look up sentence texts for the fused candidates
        candidate_ids = []
        candidate_texts = []
        candidate_sources = []

        for sid, rrf_score, sources in fused:
            text = self._records_by_id.get(sid)
            if text:
                candidate_ids.append(sid)
                candidate_texts.append(text)
                candidate_sources.append(sources)

        # Stage 2: Rerank with NLI cross-encoder
        if candidate_texts:
            reranked = self._reranker.rerank(
                claim=claim,
                candidates=candidate_texts,
                candidate_ids=candidate_ids,
                top_n=self._final_top_n,
            )
        else:
            reranked = []

        # Build the result
        evidence = []
        evidence_ids = []
        reranker_scores = []
        source_channels = []

        # Build a lookup for source tracking
        id_to_sources = dict(zip(candidate_ids, candidate_sources))

        for sid, text, score in reranked:
            evidence.append(text)
            evidence_ids.append(sid)
            reranker_scores.append(round(score, 4))
            source_channels.append(id_to_sources.get(sid, []))

        elapsed = time.time() - start_time

        return RetrievalResult(
            claim=claim,
            evidence=evidence,
            evidence_ids=evidence_ids,
            reranker_scores=reranker_scores,
            source_channels=source_channels,
            metadata={
                "elapsed_seconds": round(elapsed, 3),
                "bm25_candidates": len(bm25_ids),
                "dense_candidates": len(dense_ids),
                "graph_candidates": len(graph_ids),
                "fused_candidates": len(candidate_ids),
            },
        )

    def retrieve_batch(self, claims, show_progress=True):
        """
        Run retrieval for a list of claims.

        Returns a list of RetrievalResult, one per claim.
        """
        results = []
        iterator = tqdm(claims, desc="Retrieving") if show_progress else claims

        for claim in iterator:
            result = self.retrieve(claim)
            results.append(result)

        return results
