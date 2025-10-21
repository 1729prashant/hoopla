import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer

class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
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