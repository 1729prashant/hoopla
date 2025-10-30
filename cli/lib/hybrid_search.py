import os
import json
from keyword_search_cli import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from typing import Any
from .llm_search import llm_rerank
import time
import re
import ast 

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

    def rrf_search(self, query, k, limit=5):
        docs_by_id = {d["id"]: d for d in self.documents}
        bm25_search_result = adapt_bm25_results(self._bm25_search(query, limit*500), docs_by_id)
        semantic_search_result = self.semantic_search.search_chunks(query, limit*500)
        rrf_combined = rrf_combine_search_results(bm25_search_result, semantic_search_result, k)
        return sorted(rrf_combined.values(), key=lambda x: x["rrf_score"], reverse=True)[:limit]
    
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



# Lower k values like 20: Gives more weight to top-ranked results, creating a steep drop-off in scores.
# Higher k values like 100: Creates a more gradual decline, giving lower-ranked results more influence.
def rrf_score(rank, k=60):
    return 1 / (k + rank)


def rrf_search_command(query: str, k: int, limit: int, rerank_method: str = None) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query


    match rerank_method:
        case "individual":
            results = searcher.rrf_search(query, k, limit*5)
            print(f"Reranking top {limit} results using individual method...")
            print(f"Reciprocal Rank Fusion Results for '{query}' (k={k})")
            for i, result in enumerate(results[:limit], start=1):
                rank = i
                title = result['title']
                rrf_score = result['rrf_score']
                bm25_rank = result['bm25_rank']
                semantic_rank = result['semantic_rank']

                # Limit the document text (description) for display
                document = result['document']
                display_text = document[:100] + '...' if len(document) > 100 else document

                llm_query = f"""Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {title} - {document}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Give me ONLY the number in your response, no other text or explanation.

    Score:"""
                rerank_score = llm_rerank(llm_query)
                time.sleep(10)

                # Print the required format
                print(f"{rank}. {title}")
                print(f"    Rerank Score: {str(rerank_score).rstrip()}/10")
                print(f"    RRF Score: {rrf_score:.3f}") # Use 3 decimal places for score
                print(f"    BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"    {display_text}\n")
        
        case "batch":
            results = searcher.rrf_search(query, k, limit*5)
            print(f"Reranking top {limit} results using batch method...")
            print(f"Reciprocal Rank Fusion Results for '{query}' (k={k})")

            llm_query = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{results}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
            rerank_score_list = llm_rerank(llm_query)

            match = re.search(r"\[(.*?)\]", rerank_score_list)
            llm_reranked = []
            if match: 
                list_str = "[" + match.group(1) + "]"
                llm_reranked = ast.literal_eval(list_str)
            
            for i, doc_id in enumerate(llm_reranked[:limit], start=1):
                doc = next((d for d in results if d["doc_id"] == doc_id), None)
                rank = i
                title = doc['title']
                rrf_score = doc['rrf_score']
                bm25_rank = doc['bm25_rank']
                semantic_rank = doc['semantic_rank']

                # Limit the document text (description) for display
                document = doc['document']
                display_text = document[:100] + '...' if len(document) > 100 else document

                # Print the required format
                print(f"{rank}. {title}")
                print(f"    Rerank Score: {i}")
                print(f"    RRF Score: {rrf_score:.3f}") # Use 3 decimal places for score
                print(f"    BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"    {display_text}\n")

        case _:
            results = searcher.rrf_search(query, k, limit)
            for i, result in enumerate(results, start=1):
                rank = i
                title = result['title']
                rrf_score = result['rrf_score']
                bm25_rank = result['bm25_rank']
                semantic_rank = result['semantic_rank']

                # Limit the document text (description) for display
                document = result['document']
                display_text = document[:100] + '...' if len(document) > 100 else document

                # Print the required format
                print(f"{rank}. {title}")
                print(f"    RRF Score: {rrf_score:.3f}") # Use 3 decimal places for score
                print(f"    BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"    {display_text}\n")



def rank_and_sort_dictionaries(data: list[dict], sort_key: str, k: int) -> list[dict]:
    """
    Sorts a list of dictionaries by a specified key in descending order
    and adds a 'rank' key, handling ties.
    """
    if not data:
        return []

    # 1. Sort the list by the specified key in descending order
    # The sorted() function returns a new list, leaving the original unchanged.
    sorted_data = sorted(data, key=lambda x: x.get(sort_key, 0), reverse=True)

    # 2. Add ranks, handling ties
    current_rank = 0
    previous_score = None

    for i, item in enumerate(sorted_data):
        current_score = item.get(sort_key)

        # Check for a tie: if the current score is the same as the previous score,
        # assign the same rank. Otherwise, assign the rank based on the index + 1.
        if current_score != previous_score:
            current_rank = i + 1  # Rank starts at 1
        
        item['rank'] = current_rank
        item['rrf_score'] = rrf_score(current_rank, k=k)
        previous_score = current_score

    return sorted_data


def rrf_combine_search_results(bm25_results: list[dict], semantic_results: list[dict], k: int) -> dict:
    
    # RRF uses ranks, so we bypass score normalization and go straight to ranking
    bm25_ranked = rank_and_sort_dictionaries(bm25_results, 'score', k)
    semantic_ranked = rank_and_sort_dictionaries(semantic_results, 'score', k)

    combined_scores = {}

    def update_combined(result_list, rank_key):
        for result in result_list:
            # Semantic search results use 'movie_idx', BM25 uses 'id'
            doc_id = result.get("id") or result.get("movie_idx") 
            
            if doc_id not in combined_scores:
                # Initialize the document data
                combined_scores[doc_id] = {
                    "doc_id": doc_id,
                    "title": result["title"],
                    "document": result.get("document", result.get("description")), # Choose document or description
                    "bm25_rank": 0,
                    "semantic_rank": 0,
                    "rrf_score": 0.0,
                }
            
            # Update the specific rank
            combined_scores[doc_id][rank_key] = result['rank']
            # Sum the RRF score
            combined_scores[doc_id]['rrf_score'] += result['rrf_score']

    # Process BM25 Results
    update_combined(bm25_ranked, 'bm25_rank')

    # Process Semantic Results
    update_combined(semantic_ranked, 'semantic_rank')
    
    return combined_scores
