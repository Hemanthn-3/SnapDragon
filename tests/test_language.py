"""
NEXUS Phase 4: Local Language Model Tests
Tests:
- model loading
- inference (question-answering with context)
- timeout
- malformed request / input
- unavailable model
- streaming interface contract
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.models_local.llama_language import LocalLlamaModel
from backend.routes_llm import local_llama_model
from backend.interfaces.base import ModelStatus

client = TestClient(app)


def test_model_loading_and_health():
    """Verify LocalLlamaModel initialization and health check behavior."""
    model = LocalLlamaModel(auto_load=False)
    status = model.health_check()
    assert status == ModelStatus.NOT_LOADED
    assert model.model_name == "Llama-v3.2-1B-Instruct"
    assert model.target_hardware == "Hexagon NPU"
    assert model.is_loaded is False


def test_inference_question_answering():
    """Verify POST /llm/query generates grounded answers from context."""
    test_question = "What is the continuous TOPS performance of the Hexagon NPU?"
    test_context = [
        "The dedicated Qualcomm Hexagon NPU delivers 45 TOPS of continuous AI matrix computation.",
        "The Oryon CPU cluster operates at 3.8 GHz peak frequency."
    ]
    mock_answer = "The Qualcomm Hexagon NPU delivers 45 TOPS of continuous AI matrix computation."

    with patch.object(local_llama_model, "health_check", return_value=ModelStatus.READY):
        with patch.object(local_llama_model, "generate", return_value=mock_answer) as mock_gen:
            response = client.post(
                "/llm/query",
                json={
                    "question": test_question,
                    "context": test_context,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == mock_answer
            assert data["sources"] == test_context
            mock_gen.assert_called_once()
            call_kwargs = mock_gen.call_args[1]
            assert "Context:" in call_kwargs["prompt"]
            assert test_question in call_kwargs["prompt"]


def test_inference_timeout():
    """Verify POST /llm/query returns 408 when generation times out."""
    with patch.object(local_llama_model, "health_check", return_value=ModelStatus.READY):
        with patch.object(local_llama_model, "generate", side_effect=TimeoutError("Inference timed out after 30.0s")):
            response = client.post(
                "/llm/query",
                json={
                    "question": "What is the CPU frequency?",
                    "context": ["CPU runs at 3.8 GHz"],
                },
            )
            assert response.status_code == 408
            assert "timed out" in response.json()["detail"].lower()


def test_malformed_request_validation():
    """Verify validation errors for missing/blank questions or invalid types."""
    # Blank question
    resp = client.post("/llm/query", json={"question": "   ", "context": ["test"]})
    assert resp.status_code == 422

    # Missing question field
    resp = client.post("/llm/query", json={"context": ["test"]})
    assert resp.status_code == 422

    # Context not a list
    resp = client.post("/llm/query", json={"question": "valid?", "context": "not-a-list"})
    assert resp.status_code == 422


def test_unavailable_model_handling():
    """Verify POST /llm/query returns 503 when local model weights cannot be loaded."""
    with patch.object(local_llama_model, "health_check", return_value=ModelStatus.NOT_LOADED):
        with patch.object(local_llama_model, "load", return_value=False):
            local_llama_model._load_error = "Weights missing"
            response = client.post(
                "/llm/query",
                json={
                    "question": "Where are the files stored?",
                    "context": ["Local directory"],
                },
            )
            assert response.status_code == 503
            assert "unavailable" in response.json()["detail"].lower()


def test_streaming_interface_contract():
    """Verify that stream() yields tokens iteratively."""
    model = LocalLlamaModel(auto_load=False)
    tokens_to_yield = ["The", " answer", " is", " 45", " TOPS", "."]

    with patch.object(model, "load", return_value=True):
        model._is_loaded = True
        model._model = MagicMock()

        mock_tokenizer = MagicMock()
        mock_stream = MagicMock()
        mock_stream.decode.side_effect = tokens_to_yield
        mock_tokenizer.create_stream.return_value = mock_stream
        mock_tokenizer.encode.return_value = [1, 2, 3]
        model._tokenizer = mock_tokenizer

        with patch("onnxruntime_genai.GeneratorParams") as mock_params_cls:
            with patch("onnxruntime_genai.Generator") as mock_gen_cls:
                mock_gen_instance = MagicMock()
                mock_gen_instance.is_done.side_effect = [False, False, False, False, False, False, True]
                mock_gen_instance.get_next_tokens.return_value = [42]
                mock_gen_cls.return_value = mock_gen_instance

                stream_iter = model.stream(prompt="Test prompt", max_tokens=10)
                collected = list(stream_iter)
                assert "".join(collected) == "The answer is 45 TOPS."
