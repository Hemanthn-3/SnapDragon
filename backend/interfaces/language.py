from abc import abstractmethod
from typing import Any, Dict, Iterator, Optional
from backend.interfaces.base import BaseModelInterface, ModelStatus


class LanguageModel(BaseModelInterface):
    """Abstract interface for offline Small Language Model Reasoning (e.g., Llama-3.2-1B)."""

    def __init__(self, model_name: str = "Llama-v3.2-1B-Instruct"):
        super().__init__(model_name=model_name, target_hardware="Hexagon NPU")

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generates text or structured action plans from the provided prompt.

        Returns:
            str: Generated completion or structured response.
        """
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Streams generated text tokens one by one.

        Yields:
            str: Generated text chunk/token.
        """
        pass

    @abstractmethod
    def health_check(self) -> ModelStatus:
        """
        Verifies local model integrity, loaded state, and hardware execution provider.

        Returns:
            ModelStatus: Current diagnostic status of the language model.
        """
        pass

