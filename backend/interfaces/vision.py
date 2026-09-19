from abc import abstractmethod
from typing import List
from backend.interfaces.base import BaseModelInterface


class VisionModel(BaseModelInterface):
    """Abstract interface for offline Visual Semantic Encoding (e.g., OpenAI-CLIP)."""

    def __init__(self, model_name: str = "OpenAI-Clip"):
        super().__init__(model_name=model_name, target_hardware="Hexagon NPU")

    @abstractmethod
    def encode_image(self, image_bytes: bytes) -> List[float]:
        """
        Encodes a raw desktop or window screenshot into a normalized embedding vector.

        Returns:
            List[float]: 512-dimensional normalized visual feature vector.
        """
        pass
