import numpy as np
import json
import os
import re
from sentence_transformers import SentenceTransformer

class SemanticSearch:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}


    def generate_embedding(self, text):
        if len(text) == 0:
            raise ValueError("Please provide text input")
        
        embedding = self.model.encode([text])

        return embedding[0]


    def build_embeddings(self, documents): 
        self.documents = documents

        document_list = []
        for document in documents:
            self.document_map[document['id']] = document
            document_list.append(f"{document['title']}: {document['description']}")
        
        self.embeddings = self.model.encode(document_list, show_progress_bar=True)
        np.save('cache/movie_embeddings.npy', self.embeddings)

        return self.embeddings


    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc

        cached_file_path = 'cache/movie_embeddings.npy'
        try:
            self.embeddings = np.load(cached_file_path, allow_pickle=False)
        except FileNotFoundError:
            return self.build_embeddings(documents)
        except Exception as e:
            # optional: log e
            return self.build_embeddings(documents)

        if len(self.embeddings) != len(documents):
            return self.build_embeddings(documents)
        return self.embeddings

    def search(self, query, limit=5):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        # 1. Generate embedding for the query
        query_embedding = self.generate_embedding(query)

        # 2. Calculate cosine similarity between query and each document embedding
        # The search() method assumes a one-to-one, order-preserving correspondence between:
        # self.embeddings[i] ↔ self.documents[i].
        similarity_scores = []
        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            similarity_scores.append((score, self.documents[i]))

        # 3. Sort by similarity in descending order
        similarity_scores.sort(key=lambda x: x[0], reverse=True)

        # 4. Prepare top results up to limit
        results = []
        for score, doc in similarity_scores[:limit]:
            results.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"]
            })

        return results


def verify_model() -> None:
    try:
        minilm_semantic_search = SemanticSearch()
        print(f"Model loaded: {minilm_semantic_search.model}")
        print(f"Max sequence length: {minilm_semantic_search.model.max_seq_length}")
    except Exception as e:
        print(f"Error while loading model in SemanticSearch: {e}")


def embed(text: str) -> None:
    try:
        minilm_semantic_search = SemanticSearch()
        print(f"Text: {text}")
        embedding = minilm_semantic_search.generate_embedding(text)
        print(f"First 3 dimensions: {embedding[:3]}")
        print(f"Dimensions: {embedding.shape[0]}")
    except Exception as e:
        print(f"Error while loading model in SemanticSearch: {e}")


def verify_embeddings() -> None:
    try:
        minilm_semantic_load = SemanticSearch()
    except Exception as e:
        print(f"Error while loading model in SemanticSearch: {e}")
        return
    
    movies_file = 'data/movies.json'
    try:
        with open(movies_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            documents = data["movies"]
    except FileNotFoundError:
        print(f"Error: {movies_file} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {movies_file}.")
        return

    embeddings = minilm_semantic_load.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")




def embed_query_text(query: str):
    
    minilm_semantic_load = SemanticSearch()
    query_embedding = minilm_semantic_load.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {query_embedding[:5]}")
    print(f"Shape: {query_embedding.shape}")


def cosine_similarity(vec1, vec2):

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)



class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        chunks_list = []
        chunk_metadata = []

        for document in documents:
            if len(document['description'].strip()) == 0:
                continue

            doc_chunks = semantic_chunking(document['description'], 4, 1)
            chunks_list.extend(doc_chunks)
            for chunk_idx, chunk in enumerate(doc_chunks):
                chunk_metadata.append({
                    "movie_idx": document['id'],
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(doc_chunks)
                })

        self.chunk_embeddings = self.model.encode(chunks_list, show_progress_bar=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(chunks_list)}, f, indent=2)

        self.chunk_metadata = chunk_metadata
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        emb_path = "cache/chunk_embeddings.npy"
        meta_path = "cache/chunk_metadata.json"

        if os.path.exists(emb_path) and os.path.exists(meta_path):
            self.chunk_embeddings = np.load(emb_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)


    def search_chunks(self, query: str, limit: int = 10):
            # 1. Generate an embedding of the query
            query_embedding = self.generate_embedding(query)
            
            # 2. Populate an empty list to score "chunk score" dictionaries
            chunk_score_list = []
            
            # For each chunk embedding: Calculate the cosine similarity and append to list
            if self.chunk_embeddings is None or self.chunk_metadata is None:
                raise ValueError("Chunk embeddings and/or metadata not loaded. Call `load_or_create_chunk_embeddings` first.")

            for i, chunk_embedding in enumerate(self.chunk_embeddings):
                score = cosine_similarity(query_embedding, chunk_embedding)
                metadata = self.chunk_metadata[i]
                
                # Append a dictionary to the chunk score list
                chunk_score_list.append({
                    "chunk_idx": metadata["chunk_idx"],
                    # Note: movie_idx in metadata is the document 'id', which is an index in self.documents
                    "movie_idx": metadata["movie_idx"], 
                    "score": score
                })

            # 3. Create an empty dictionary that maps movie indexes to their scores (movie_idx: max_score)
            movie_scores = {}

            # 4. Update the movie score dictionary with the highest chunk score for that movie
            for chunk_score in chunk_score_list:
                movie_idx = chunk_score["movie_idx"]
                score = chunk_score["score"]
                
                # If the movie_idx is not in the movie score dictionary yet, 
                # or the new score is higher than the existing one, update the movie score.
                if movie_idx not in movie_scores or score > movie_scores[movie_idx]["score"]:
                    # Store the whole chunk_score dict for easy sorting/access
                    movie_scores[movie_idx] = chunk_score

            # 5. Sort the movie scores by score in descending order.
            # Convert dictionary values to a list of the chunk_score dictionaries
            sorted_movie_scores = sorted(
                movie_scores.values(), 
                key=lambda x: x["score"], 
                reverse=True
            )

            # 6. Filter down to the top limit movies.
            top_movie_results = sorted_movie_scores[:limit]
            
            # 7. Format the results using format_search_result. Limit the movie description to the first 100 characters.
            final_results = []
            for result_data in top_movie_results:
                movie_idx = result_data["movie_idx"]
                # Lookup the full document from self.documents using the movie_idx (which is the document 'id')
                doc = self.document_map[movie_idx] 
                
                formatted_result = format_search_result(
                    score=result_data["score"],
                    doc=doc,
                    chunk_idx=result_data["chunk_idx"],
                    movie_idx=movie_idx,
                    description_limit=100
                )
                final_results.append(formatted_result)

            # 8. Return the final list of results.
            return final_results


def semantic_chunking(query: str, max_chunk_size: int=4, overlap: int=1):
    # Strip leading and trailing whitespace from the input text
    text = query.strip() 

    # If there's nothing left after stripping, return an empty list.
    if not text:
        return []

    # Split sentences using the provided regex
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    # Handle single sentence without end punctuation & ensure stripping
    if len(sentences) == 1 and not re.search(r'[.!?]$', sentences[0]):
        # If there's only one sentence and it doesn't end with a punctuation mark, 
        # the original splitting might have failed to capture the full intent.
        # Since the original logic already stripped sentences, we just ensure 
        # the single sentence is used. If the text itself was the single sentence, 
        # it was already stripped via .strip() on 'text'.
        pass 
    
    # Re-strip all sentences for robustness (though the list comprehension does this, 
    # it's safer to ensure we filter out empty strings in the next step).
    sentences = [s.strip() for s in sentences]

    # Filter out any resulting empty strings
    sentences = [s for s in sentences if s]
    
    if not sentences:
        # This handles cases where splitting results in an empty list even if text was non-empty 
        # (e.g., text with only spaces, tabs, etc., which is largely covered by the initial check, 
        # but is good practice).
        return []

    chunked_sentences = []
    
    step = max_chunk_size - overlap
    if step <= 0:
        print("Error: overlap must be smaller than chunk size.")
        return [] # Changed return to [] instead of None for consistency

    i = 0
    # The original loop condition `i < len(sentences) - overlap` is correct for overlapping chunks
    while i < len(sentences):
        group = sentences[i:i + max_chunk_size]
        chunk = " ".join(group)
        
        # Strip leading and trailing whitespace from each sentence before appending it to the chunk list.
        # The sentences were already stripped, but the final chunk needs to be stripped, 
        # and then checked for content.
        stripped_chunk = chunk.strip() 

        # Only use chunks that still have content after the stripping.
        if stripped_chunk:
            chunked_sentences.append(stripped_chunk)
        
        # Move the index forward by the step size (max_chunk_size - overlap)
        if i + max_chunk_size >= len(sentences):
            # If the current group covered the remaining sentences, stop.
            break
        else:
            # Move to the start of the next chunk
            i += step

    return chunked_sentences


def format_search_result(score, doc, chunk_idx=None, movie_idx=None, description_limit=None):
    """
    Formats the search result dictionary.
    """
    description = doc.get("description", "")
    if description_limit is not None and len(description) > description_limit:
        description = description[:description_limit] + "..."

    result = {
        "score": score,
        "title": doc.get("title", "N/A"),
        "description": description
    }
    # Include chunk-specific metadata only if provided (for debugging/context)
    if chunk_idx is not None:
        result["chunk_idx"] = chunk_idx
    if movie_idx is not None:
        result["movie_idx"] = movie_idx

    return result