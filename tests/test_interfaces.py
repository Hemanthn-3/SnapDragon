import pytest
from backend.interfaces.base import BaseModelInterface, ModelStatus
from backend.interfaces.speech import SpeechModel
from backend.interfaces.vision import VisionModel
from backend.interfaces.ocr import OCRModel
from backend.interfaces.embedding import EmbeddingModel
from backend.interfaces.language import LanguageModel
from backend.interfaces import registry


def test_cannot_instantiate_abstract_interfaces():
    """Verifies that abstract model interfaces enforce the ABC contract and cannot be directly instantiated."""
    for cls in [SpeechModel, VisionModel, OCRModel, EmbeddingModel, LanguageModel]:
        with pytest.raises(TypeError):
            cls()


def test_model_status_defaults():
    """Verifies that model implementations have clean status tracking."""
    class DummySpeech(SpeechModel):
        def load(self):
            self._status = ModelStatus.READY

        def unload(self):
            self._status = ModelStatus.NOT_LOADED

        def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000):
            return {"text": "dummy"}

    model = DummySpeech()
    assert model.status == ModelStatus.NOT_LOADED
    assert model.is_loaded is False

    model.load()
    assert model.status == ModelStatus.READY
    assert model.is_loaded is True

    model.unload()
    assert model.status == ModelStatus.NOT_LOADED
    assert model.is_loaded is False


def test_registry_unloaded_state():
    """Verifies that in Phase 1, registry reports all 5 AI models as not_loaded."""
    assert registry.get_summary_status() == "not_loaded"
    details = registry.get_detailed_status()
    assert details["speech"] == "not_loaded"
    assert details["vision"] == "not_loaded"
    assert details["ocr"] == "not_loaded"
    assert details["embedding"] == "not_loaded"
    assert details["language"] == "not_loaded"
