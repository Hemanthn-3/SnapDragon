"""
NEXUS Local Language Model Adapter: Llama-3.2-1B-Instruct (INT4)
Runs 100% locally via onnxruntime-genai.
Complies with offline-first execution, strict sandboxing, and no cloud fallbacks.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from backend.interfaces.base import ModelStatus
from backend.interfaces.language import LanguageModel
from backend.logger import get_logger

logger = get_logger("nexus.llm")

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "llm" / "Llama-3.2-1B-Instruct" / "cpu_and_mobile" / "cpu-int4-rtn-block-32-acc-level-4"


class LocalLlamaModel(LanguageModel):
    """
    Concrete adapter for offline execution of Llama-3.2-1B-Instruct via onnxruntime-genai.
    """

    def __init__(
        self,
        model_name: str = "Llama-v3.2-1B-Instruct",
        model_dir: Optional[Path] = None,
        auto_load: bool = False,
    ):
        super().__init__(model_name=model_name)
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self._model = None
        self._tokenizer = None
        self._execution_provider = "Unloaded"
        self._is_loaded = False
        self._load_error: Optional[str] = None

        if auto_load:
            self.load()

    def load(self) -> bool:
        """Loads the ONNX GenAI model and tokenizer into memory."""
        if self._is_loaded and self._model is not None:
            return True

        if not self.model_dir.exists():
            self._load_error = f"Model directory does not exist: {self.model_dir}"
            logger.warning(self._load_error)
            return False

        # Verify essential files
        required_files = ["model.onnx", "genai_config.json"]
        for rf in required_files:
            if not (self.model_dir / rf).exists():
                self._load_error = f"Missing required model asset: {rf} in {self.model_dir}"
                logger.warning(self._load_error)
                return False

        try:
            import onnxruntime_genai as og

            t0 = time.time()
            logger.info(f"Loading local GenAI language model from {self.model_dir}...")
            self._model = og.Model(str(self.model_dir))
            self._tokenizer = og.Tokenizer(self._model)

            # Detect provider
            # Note: on Windows AMD64, execution runs on CPU; Snapdragon runs on QNN
            self._execution_provider = "CPUExecutionProvider"
            self._is_loaded = True
            self._load_error = None
            elapsed = (time.time() - t0) * 1000
            logger.info(
                f"Local language model '{self.model_name}' loaded successfully in {elapsed:.1f}ms "
                f"(Provider: {self._execution_provider})"
            )
            return True
        except Exception as e:
            self._is_loaded = False
            self._model = None
            self._tokenizer = None
            self._load_error = f"Failed to load language model: {str(e)}"
            logger.error(self._load_error)
            return False

    @property
    def execution_provider(self) -> str:
        return self._execution_provider

    def unload(self) -> None:
        """Unloads model resources from memory."""
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._execution_provider = "Unloaded"
        logger.info(f"Model '{self.model_name}' unloaded.")

    def format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Formats the input into Llama-3.2 Instruct template."""
        sys_content = (
            system_prompt
            or "You are NEXUS, an offline multimodal work agent. Answer the user prompt accurately based strictly on available context."
        )
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{sys_content}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        timeout_seconds: Optional[float] = 30.0,
        **kwargs: Any,
    ) -> str:
        """
        Synchronously generates completion tokens with strict offline execution and timeout protection.
        """
        if not self._is_loaded or self._model is None:
            # Attempt auto-load if not already attempted
            if not self.load():
                raise RuntimeError(
                    f"Local language model '{self.model_name}' is not loaded: {self._load_error or 'Weights missing'}"
                )

        import onnxruntime_genai as og

        formatted = self.format_prompt(prompt, system_prompt)
        tokens = self._tokenizer.encode(formatted)

        params = og.GeneratorParams(self._model)
        params.set_search_options(
            max_length=len(tokens) + max_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
        )

        generator = og.Generator(self._model, params)
        generator.append_tokens(tokens)
        tokenizer_stream = self._tokenizer.create_stream()

        generated_pieces = []
        t0 = time.time()

        while not generator.is_done():
            if timeout_seconds and (time.time() - t0) > timeout_seconds:
                del generator
                raise TimeoutError(f"Inference timed out after {timeout_seconds}s")

            generator.generate_next_token()
            token = generator.get_next_tokens()[0]
            new_text = tokenizer_stream.decode(token)
            generated_pieces.append(new_text)

        del generator
        return "".join(generated_pieces).strip()

    def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Iteratively yields generated tokens one by one for streaming UX.
        """
        if not self._is_loaded or self._model is None:
            if not self.load():
                raise RuntimeError(
                    f"Local language model '{self.model_name}' is not loaded: {self._load_error or 'Weights missing'}"
                )

        import onnxruntime_genai as og

        formatted = self.format_prompt(prompt, system_prompt)
        tokens = self._tokenizer.encode(formatted)

        params = og.GeneratorParams(self._model)
        params.set_search_options(
            max_length=len(tokens) + max_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
        )

        generator = og.Generator(self._model, params)
        generator.append_tokens(tokens)
        tokenizer_stream = self._tokenizer.create_stream()

        try:
            while not generator.is_done():
                generator.generate_next_token()
                token = generator.get_next_tokens()[0]
                new_text = tokenizer_stream.decode(token)
                if new_text:
                    yield new_text
        finally:
            del generator

    def health_check(self) -> ModelStatus:
        """Returns the current operational status of the local language model."""
        if not self._is_loaded or self._model is None:
            self._status = ModelStatus.NOT_LOADED
            return self._status
        self._status = ModelStatus.READY
        return self._status


local_llama_model = LocalLlamaModel(auto_load=False)

