"""
Reciprocal Rank Fusion (RRF) for merging ranked lists from multiple retrieval channels.

RRF is score-agnostic: it works on rank positions, not raw scores,
so it handles the incompatible scales of BM25, cosine distance, and hop depth.

Reference: Cormack, Clarke, Buettcher (2009). "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods."
"""

from collections import defaultdict


def reciprocal_rank_fusion(ranked_lists, k=60, pool_size=200):
    """
    Merge N ranked lists via RRF.

    Args:
        ranked_lists: list of lists, each containing sentence_ids in rank order.
                      e.g. [["id_a", "id_b", ...], ["id_c", "id_a", ...], ...]
        k: RRF constant (standard: 60). Higher k smooths rank differences.
        pool_size: max number of merged candidates to return.

    Returns:
        list of (sentence_id, rrf_score) sorted descending by rrf_score.
    """
    rrf_scores = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Sort by RRF score descending
    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return merged[:pool_size]


def reciprocal_rank_fusion_with_sources(ranked_lists, channel_names, k=60, pool_size=200):
    """
    Same as reciprocal_rank_fusion but also tracks which channels contributed
    each document. Useful for debugging and analysis.

    Args:
        ranked_lists: list of lists of sentence_ids in rank order.
        channel_names: list of channel name strings, same length as ranked_lists.
                       e.g. ["bm25", "dense", "graph"]
        k: RRF constant.
        pool_size: max merged candidates.

    Returns:
        list of (sentence_id, rrf_score, source_channels) sorted descending.
        source_channels is a list of channel names that returned this doc.
    """
    rrf_scores = defaultdict(float)
    doc_sources = defaultdict(list)

    for channel_name, ranked_list in zip(channel_names, ranked_lists):
        for rank, doc_id in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
            if channel_name not in doc_sources[doc_id]:
                doc_sources[doc_id].append(channel_name)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = [
        (doc_id, score, doc_sources[doc_id])
        for doc_id, score in merged[:pool_size]
    ]

    return results
