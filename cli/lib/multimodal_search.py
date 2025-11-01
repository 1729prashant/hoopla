from PIL import Image
from sentence_transformers import SentenceTransformer
from typing import Union, List
import numpy as np

class MultimodalSearch:
    """
    Provides functionality for generating embeddings for multimodal data
    (e.g., images and text) using a SentenceTransformer model.
    """
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """
        Initializes the MultimodalSearch instance.

        Args:
            model_name: The name of the SentenceTransformer model to load.
        """
        # Load the SentenceTransformer model (will download if not present)
        # We assume a suitable model (like a CLIP model) is used for multimodal tasks.
        self.model: SentenceTransformer = SentenceTransformer(model_name)

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
            embeddings: List[np.ndarray] = self.model.encode([img], convert_to_numpy=True)
            
            # 3. Return the first/only element from the resulting list
            return embeddings[0]
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return np.array([])
        except Exception as e:
            print(f"An error occurred during image embedding: {e}")
            return np.array([])


def verify_image_embedding(image_path: str) -> None:
    """
    Creates a MultimodalSearch instance, generates an embedding for the image,
    and prints the shape of the resulting embedding.

    Args:
        image_path: The file path to the image.
    """
    # 1. Create an instance of MultimodalSearch
    searcher: MultimodalSearch = MultimodalSearch()

    # 2. Generate an embedding for the image
    embedding: np.ndarray = searcher.embed_image(image_path)

    # 3. Print the shape of the embedding
    if embedding.size > 0:
        print(f"Embedding shape: {embedding.shape[0]} dimensions")
    else:
        print("Could not generate embedding.")

if __name__ == '__main__':
    # Simple test run (requires a placeholder image for actual execution)
    # verify_image_embedding("path/to/your/image.jpg")
    pass