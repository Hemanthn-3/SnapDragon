from abc import abstractmethod
from typing import Any, Dict
from backend.interfaces.base import BaseModelInterface


class SpeechModel(BaseModelInterface):
    """
    Abstract interface for offline Automatic Speech Recognition.

    Implementations MUST:
    - Perform real model inference. Hard-coded transcripts are prohibited.
    - Return empty string for silence, not a fake transcript.
    - Return an explicit error dict (success=False) for inference failures.
    - Never make network calls.
    - Accurately report execution_provider and runtime.
    """

    def __init__(self, model_name: str = "openai/whisper-small"):
        super().__init__(model_name=model_name, target_hardware="CPU")

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcribes raw audio bytes into text using real local inference.

        Args:
            audio_bytes: WAV (RIFF) or raw 16-bit PCM bytes.
            sample_rate: Hint for raw PCM; WAV header takes precedence.

        Returns:
            Dict containing at minimum:
                text (str)              : Transcribed text. Empty string for silence.
                language (str)          : Detected language code (e.g. "en").
                duration_seconds (float): Audio duration in seconds.
                energy (float)          : RMS energy of the audio.
                silence (bool)          : True if audio was below energy threshold.
                model (str)             : Model identifier.
                runtime (str)           : Runtime library used.
                execution_provider (str): Active provider ("CPUExecutionProvider" etc).
                latency_ms (float)      : Wall-clock inference time in milliseconds.
                success (bool)          : False if inference failed.
                error (str, optional)   : Error message if success=False.
        """
        pass

    def health_check(self) -> Dict[str, Any]:
        """Returns model health and status metadata."""
        return {
            "model": self.model_name,
            "status": self.status.value,
            "is_loaded": self.is_loaded,
            "runtime": "unknown",
            "execution_provider": "unknown",
            "target_hardware": self.target_hardware,
            "blockers": [],
        }
