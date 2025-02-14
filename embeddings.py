from typing import List
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

class E5Embeddings(Embeddings):
    """E5 embeddings wrapper for langchain."""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        """Initialize the E5 model."""
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for a query."""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist() 