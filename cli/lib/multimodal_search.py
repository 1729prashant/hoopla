from PIL import Image
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
from typing import Union, List, Dict, Any
import torch

def load_movies():
    with open('data/movies.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["movies"]


class MultimodalSearch:
    """
    Provides functionality for generating and searching image and text embeddings.
    """
    def __init__(self, documents: List[Dict[str, Any]], model_name: str = "clip-ViT-B-32"):
        """
        Initializes the MultimodalSearch instance, loads the model, and encodes documents.

        Args:
            documents: A list of movie dictionaries.
            model_name: The name of the SentenceTransformer model to load.
        """

        # Determine device: use CUDA if available, otherwise CPU
        # to address issue where RuntimeError: Tensor for argument #1 'mat1' is on CPU, but expected it to be on GPU (while checking arguments for mm)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # macos M series

        self.documents: List[Dict[str, Any]] = documents
        self.model: SentenceTransformer = SentenceTransformer(model_name).to(self.device)        
        
        # 1. Create self.texts list from documents
        self.texts: List[str] = [
            f"{doc['title']}: {doc['description']}" for doc in self.documents
        ]

        # 2. Generate text embeddings
        print("Encoding movie documents (this may take a moment)...")
        # Ensure 'convert_to_tensor=True' for optimal use with util.cos_sim
        self.text_embeddings = self.model.encode(
            self.texts, 
            show_progress_bar=True,
            convert_to_tensor=True  # Use tensors for fast similarity calculation
        )

        # Note: If self.model is on GPU, self.text_embeddings will also be on GPU.
        # Ensure it is explicitly on the device if you want to be completely safe:
        self.text_embeddings = self.text_embeddings.to(self.device)

        print(f"Encoding complete. Generated {len(self.text_embeddings)} embeddings of shape {self.text_embeddings.shape[1]}.")


    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Generates an embedding vector for an image at the specified path.

        Args:
            image_path: The file path to the image.

        Returns:
            A numpy array representing the image embedding (e.g., a 512-dimensional vector).
        """
        try:
            # 1. Load the image using PIL
            img: Image.Image = Image.open(image_path)
            
            # 2. Pass the image to the model's encode method
            # The encode method expects an iterable (list) of items
            embeddings: List[np.ndarray] = self.model.encode([img], convert_to_tensor=True)
            
            # 3. Return the first/only element from the resulting list
            # **CRITICAL FIX: add .to(self.device) Ensure the embedding is moved to the model's device**
            # This is often handled by the model, but this line guarantees it.
            return embeddings[0].to(self.device)
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return np.array([])
        except Exception as e:
            print(f"An error occurred during image embedding: {e}")
            return np.array([])
        
            
    def search_with_image(self, image_path: str, limit: int = 5) -> List[Dict[str, Any]]:
            """
            Generates an embedding for an image and finds the top N most similar movies.

            Args:
                image_path: The file path to the query image.
                limit: The maximum number of results to return.

            Returns:
                A list of dictionaries containing search results.
            """
            # 1. Generate embedding for the provided image
            image_embedding = self.embed_image(image_path)
            
            # 2. Calculate cosine similarity between the image embedding and all text embeddings
            # util.cos_sim is highly optimized for this operation on tensors.
            # It returns a 1xN tensor of scores.
            cosine_scores = util.cos_sim(image_embedding, self.text_embeddings)[0]
            
            # 3. Create a list of (score, index) pairs
            # Get the top K indices and scores from the tensor
            top_results = util.semantic_search(
                image_embedding, self.text_embeddings, top_k=limit
            )[0] # semantic_search returns a list of lists, we take the first element
            
            # 4. Format and sort the results
            results = []
            for rank, result in enumerate(top_results):
                corpus_id = result['corpus_id']
                score = result['score']
                doc = self.documents[corpus_id]
                
                results.append({
                    "document_id": doc.get('id', corpus_id),
                    "title": doc['title'],
                    "description": doc['description'],
                    "similarity_score": score,
                    "rank": rank + 1
                })

            return results


def verify_image_embedding(image_path: str) -> None:
    """
    Creates a MultimodalSearch instance, generates an embedding for the image,
    and prints the shape of the resulting embedding.

    Args:
        image_path: The file path to the image.
    """
    # 1. Create an instance of MultimodalSearch
    # Use dummy documents just to initialize the class for this verification function
    searcher: MultimodalSearch = MultimodalSearch(documents=load_movies()[:1])

    # 2. Generate an embedding for the image
    embedding: np.ndarray = searcher.embed_image(image_path)

    # 3. Print the shape of the embedding
    if embedding.size > 0:
        print(f"Embedding shape: {embedding.shape[0]} dimensions")
    else:
        print("Could not generate embedding.")


def image_search_command(image_path: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Loads movie data, initializes MultimodalSearch, and runs the image search.

    Args:
        image_path: The path to the query image.
        limit: The number of results to return.
        
    Returns:
        A list of search result dictionaries.
    """
    # 1. Load the movie dataset
    documents = load_movies()
    if not documents:
        print("Error: Could not load any movie documents.")
        return []

    # 2. Create an instance of MultimodalSearch (this initializes embeddings)
    searcher = MultimodalSearch(documents=documents)
    
    # 3. Call the search_with_image method
    results = searcher.search_with_image(image_path, limit)
    
    return results
