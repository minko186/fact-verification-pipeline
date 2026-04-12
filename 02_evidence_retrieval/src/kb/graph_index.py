import os
import re
import pickle
from collections import defaultdict
from itertools import combinations

import networkx as nx

from .parse_wiki import SentenceRecord


class GraphIndex:
    def __init__(self, config):
        self.graph_path = config["index"]["graph_path"]
        self._graph = None
        # Maps article_title -> list of sentence_ids belonging to that article
        self._article_sentences = {}

    def build(self, records):
        """
        Build an entity co-occurrence graph from sentence records.

        Nodes: Wikipedia article titles.
        Edges: Two titles are connected when one title is mentioned
               in a sentence belonging to the other's article.
               Edge weight = number of co-occurring sentences.

        Also builds an article_title -> [sentence_ids] lookup for retrieval.
        """
        print("Building knowledge graph...")

        # Build the title set for entity linking
        title_set = set()
        article_sentences = defaultdict(list)

        for rec in records:
            title_set.add(rec.article_title)
            article_sentences[rec.article_title].append(rec.sentence_id)

        self._article_sentences = dict(article_sentences)

        # Build a normalized lookup: lowercase title -> original title
        title_lower = {}
        for t in title_set:
            # FEVER titles use underscores for spaces
            normalized = t.replace("_", " ").lower()
            title_lower[normalized] = t

        G = nx.Graph()

        # Add all articles as nodes
        for title in title_set:
            G.add_node(title)

        # Scan sentences for mentions of other article titles
        print("Scanning sentences for entity co-occurrences...")
        edge_counts = defaultdict(int)

        for rec in records:
            text_lower = rec.text.lower()
            # Find which other article titles are mentioned in this sentence
            mentioned_titles = set()
            mentioned_titles.add(rec.article_title)

            for norm_title, orig_title in title_lower.items():
                if orig_title == rec.article_title:
                    continue
                if len(norm_title) < 3:
                    continue
                if norm_title in text_lower:
                    mentioned_titles.add(orig_title)

            # Create edges between all co-mentioned titles
            if len(mentioned_titles) > 1:
                for t1, t2 in combinations(sorted(mentioned_titles), 2):
                    edge_counts[(t1, t2)] += 1

        # Add weighted edges
        for (t1, t2), weight in edge_counts.items():
            G.add_edge(t1, t2, weight=weight)

        self._graph = G

        # Save graph and article-sentence mapping
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        nx.write_graphml(G, self.graph_path)

        mapping_path = self.graph_path.replace(".graphml", "_sentences.pkl")
        with open(mapping_path, "wb") as f:
            pickle.dump(self._article_sentences, f)

        print(
            f"Knowledge graph built: {G.number_of_nodes():,} nodes, "
            f"{G.number_of_edges():,} edges"
        )
        print(f"Saved to {self.graph_path}")

    def load(self):
        """Load graph and article-sentence mapping from disk."""
        print(f"Loading knowledge graph from {self.graph_path}...")
        self._graph = nx.read_graphml(self.graph_path)

        mapping_path = self.graph_path.replace(".graphml", "_sentences.pkl")
        with open(mapping_path, "rb") as f:
            self._article_sentences = pickle.load(f)

        print(
            f"Graph loaded: {self._graph.number_of_nodes():,} nodes, "
            f"{self._graph.number_of_edges():,} edges"
        )

    def _extract_entities(self, text):
        """
        Extract candidate entity names from claim text.

        Uses capitalized multi-word spans (proper nouns) and maps them
        to Wikipedia article title format (spaces -> underscores).
        """
        # Match sequences of capitalized words (2+ chars each)
        # e.g. "Barack Obama" or "Fox Broadcasting Company"
        spans = re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)

        # Also grab single capitalized words that might be entities
        singles = re.findall(r"\b([A-Z][a-z]{2,})\b", text)

        candidates = set()
        for span in spans:
            # Convert to Wikipedia title format
            candidates.add(span.replace(" ", "_"))

        for word in singles:
            candidates.add(word)

        return list(candidates)

    def query(self, claim, top_k=100, max_hops=2):
        """
        Extract entities from the claim, find matching graph nodes,
        BFS-expand to max_hops depth, and return sentence IDs from
        the expanded article set.

        Returns:
            list of (sentence_id, score) where score = 1.0/hop_depth.
            Hop 0 (direct match) = 1.0, hop 1 = 1.0, hop 2 = 0.5.
        """
        entities = self._extract_entities(claim)

        # Match entities to graph nodes (exact or partial match)
        seed_nodes = set()
        graph_nodes = set(self._graph.nodes()) if self._graph else set()

        for entity in entities:
            # Exact match
            if entity in graph_nodes:
                seed_nodes.add(entity)
                continue

            # Case-insensitive match
            entity_lower = entity.lower()
            for node in graph_nodes:
                if node.lower() == entity_lower:
                    seed_nodes.add(node)
                    break

        if not seed_nodes:
            return []

        # BFS expansion
        # node -> minimum hop depth from any seed
        visited = {}
        frontier = list(seed_nodes)
        for node in frontier:
            visited[node] = 0

        for hop in range(1, max_hops + 1):
            next_frontier = []
            for node in frontier:
                for neighbor in self._graph.neighbors(node):
                    if neighbor not in visited:
                        visited[neighbor] = hop
                        next_frontier.append(neighbor)
            frontier = next_frontier

        # Collect sentence IDs from all visited articles, scored by hop depth
        results = []
        for article, hop_depth in visited.items():
            score = 1.0 / max(hop_depth, 1)  # seed nodes get score 1.0
            sentence_ids = self._article_sentences.get(article, [])
            for sid in sentence_ids:
                results.append((sid, score))

        # Sort by score descending, then truncate
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
