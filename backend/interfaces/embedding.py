from abc import abstractmethod
from typing import List
from backend.interfaces.base import BaseModelInterface


class EmbeddingModel(BaseModelInterface):
    """Abstract interface for offline Semantic Text Embeddings (e.g., all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name=model_name, target_hardware="Hexagon NPU")

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Transforms text into a fixed-dimensional dense vector embedding.

        Returns:
            List[float]: 384-dimensional dense semantic embedding.
        """
        pass
