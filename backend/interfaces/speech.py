from abc import abstractmethod
from typing import Any, Dict
from backend.interfaces.base import BaseModelInterface


class SpeechModel(BaseModelInterface):
    """Abstract interface for offline Automatic Speech Recognition (e.g., Whisper)."""

    def __init__(self, model_name: str = "Whisper-Small-Quantized"):
        super().__init__(model_name=model_name, target_hardware="Hexagon NPU")

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcribes raw PCM audio bytes into text.

        Returns:
            Dict containing:
                - text (str): Recognized transcript
                - language (str): Detected language
                - segments (list): Timestamped utterance segments
        """
        pass
