from typing import Dict, Any
from backend.interfaces.base import BaseModelInterface, ModelStatus
from backend.interfaces.speech import SpeechModel
from backend.interfaces.vision import VisionModel
from backend.interfaces.ocr import OCRModel
from backend.interfaces.embedding import EmbeddingModel
from backend.interfaces.language import LanguageModel


class ModelRegistry:
    """Manages the lifecycle and state inspection of all NEXUS AI models."""

    def __init__(self):
        self.speech: SpeechModel | None = None
        self.vision: VisionModel | None = None
        self.ocr: OCRModel | None = None
        self.embedding: EmbeddingModel | None = None
        self.language: LanguageModel | None = None

    def get_summary_status(self) -> str:
        """Returns the high-level status of the AI model ecosystem."""
        models = [self.speech, self.vision, self.ocr, self.embedding, self.language]
        if any(m is not None and m.is_loaded for m in models):
            return "loaded"
        return "not_loaded"

    def register_model(self, modality: str, model: BaseModelInterface):
        """Registers a concrete model adapter to the specified modality."""
        if hasattr(self, modality):
            setattr(self, modality, model)

    def get_detailed_status(self) -> Dict[str, str]:
        """Returns dictionary of all 5 modalities and their load states."""
        return {
            "speech": self.speech.status.value if self.speech else ModelStatus.NOT_LOADED.value,
            "vision": self.vision.status.value if self.vision else ModelStatus.NOT_LOADED.value,
            "ocr": self.ocr.status.value if self.ocr else ModelStatus.NOT_LOADED.value,
            "embedding": self.embedding.status.value if self.embedding else ModelStatus.NOT_LOADED.value,
            "language": self.language.status.value if self.language else ModelStatus.NOT_LOADED.value,
        }


# Global registry instance reflecting current uninitialized state
registry = ModelRegistry()
model_registry = registry

__all__ = [
    "BaseModelInterface",
    "ModelStatus",
    "SpeechModel",
    "VisionModel",
    "OCRModel",
    "EmbeddingModel",
    "LanguageModel",
    "ModelRegistry",
    "registry",
    "model_registry",
]
