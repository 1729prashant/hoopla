from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text):
        if len(text) == 0:
            raise ValueError("Please provide text input")
        
        embedding = self.model.encode([text])

        return embedding[0]


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