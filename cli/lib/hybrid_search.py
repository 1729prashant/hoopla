import os
import json
from keyword_search_cli import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from typing import Any

SCORE_PRECISION = 3

def load_movies():
    with open('data/movies.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["movies"]


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            movies_json = 'data/movies.json'
            with open(movies_json, 'r', encoding='utf-8') as file:
                movies_data = json.load(file)
            self.idx.build(movies_data)
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
    def weighted_search(self, query,alpha, limit=5):
        #bm25_search_result = self._bm25_search(query, limit*500)
        docs_by_id = {d["id"]: d for d in self.documents}
        bm25_search_result = adapt_bm25_results(self._bm25_search(query, limit*500), docs_by_id)
        semantic_search_result = self.semantic_search.search_chunks(query, limit*500)

        combined = combine_search_results(bm25_search_result, semantic_search_result, alpha)
        return combined[:limit]

    
def normalize_scores(score_strings: list[str]) -> list[float]:
    # Convert string scores to floats, handling potential errors in a real application
    try:
        scores = [float(s) for s in score_strings]
    except ValueError as e:
        print(f"Error: Invalid score found in list. Scores must be numbers. ({e})")
        return []

    # Min-Max Normalization calculation
    min_score = min(scores)
    max_score = max(scores)
    
    if not scores:
        return []
        
    if max_score == min_score:
        return [1.0] * len(scores)
        
    normalized = [(s - min_score) / (max_score - min_score) for s in scores]
    return normalized


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float) -> float:
    """Calculates the weighted hybrid score."""
    # Alpha (a) determines the weight given to the semantic score.
    # 1-alpha determines the weight given to the BM25 score.
    # Alpha should be between 0 and 1.
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result['score'])

    normalized: list[float] = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results

def combine_search_results(bm25_results: list[dict], semantic_results: list[dict], alpha: float) -> list[dict]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    for result in semantic_normalized:
        doc_id = result["movie_idx"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["description"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def weighted_search_command(query: str, alpha: float, limit: int) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = searcher.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }

def format_search_result(doc_id: str, title: str, document: str, score: float, **metadata: Any) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }


def adapt_bm25_results(bm25_results, documents_by_id):
    adapted = []
    for doc_id, score in bm25_results:
        doc = documents_by_id[doc_id]
        adapted.append({
            "id": doc_id,
            "title": doc["title"],
            "document": doc["description"],
            "score": score,
        })
    return adapted