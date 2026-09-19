from abc import abstractmethod
from typing import Any, Dict, List
from backend.interfaces.base import BaseModelInterface


class OCRModel(BaseModelInterface):
    """Abstract interface for offline Optical Character Recognition (e.g., EasyOCR)."""

    def __init__(self, model_name: str = "EasyOCR"):
        super().__init__(model_name=model_name, target_hardware="Hexagon NPU")

    @abstractmethod
    def extract_text(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Detects and extracts text from an image.

        Returns:
            List of detected text items, each containing:
                - text (str): Recognized string
                - bbox (list): Bounding box coordinates [[x1, y1], [x2, y2], ...]
                - confidence (float): Recognition confidence score
        """
        pass
