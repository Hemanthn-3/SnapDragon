from pathlib import Path
from typing import List, Optional
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from backend.config import settings
from backend.errors import ModelNotLoadedError, NexusException
from backend.interfaces.base import ModelStatus
from backend.interfaces.embedding import EmbeddingModel
from backend.logger import logger

MODEL_DIR = settings.BASE_DIR / "models" / "embeddings" / "all-MiniLM-L6-v2"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
ONNX_MODEL_PATH = MODEL_DIR / "onnx" / "model_quint8_avx2.onnx"


class LocalMiniLMEmbeddingModel(EmbeddingModel):
    """
    Concrete implementation of all-MiniLM-L6-v2 embedding model.
    Runs 100% locally via ONNX Runtime without any cloud network requests.
    Outputs 384-dimensional dense vectors normalized under L2 norm.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model_path: Optional[Path] = None,
        tokenizer_path: Optional[Path] = None,
    ):
        super().__init__(model_name=model_name)
        self.model_path = model_path or ONNX_MODEL_PATH
        self.tokenizer_path = tokenizer_path or TOKENIZER_PATH
        self.tokenizer: Optional[Tokenizer] = None
        self.session: Optional[ort.InferenceSession] = None
        self.max_length: int = 256

    def load(self) -> None:
        """Initializes tokenizer and ONNX Runtime session."""
        if self.is_loaded:
            return

        if not self.tokenizer_path.exists():
            raise NexusException(
                f"Tokenizer not found at {self.tokenizer_path}. Run model setup first.",
                status_code=500,
            )
        if not self.model_path.exists():
            raise NexusException(
                f"ONNX model not found at {self.model_path}. Run model setup first.",
                status_code=500,
            )

        logger.info(f"Loading local embedding model: {self.model_name} from {self.model_path}")
        self._status = ModelStatus.LOADING

        try:
            # 1. Load Tokenizer
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            self.tokenizer.enable_padding(length=self.max_length, pad_id=0, pad_token="[PAD]")
            self.tokenizer.enable_truncation(max_length=self.max_length)

            # 2. Load ONNX Session (Target QNN on ARM64; CPUExecutionProvider fallback on x64)
            available_providers = ort.get_available_providers()
            providers = ["CPUExecutionProvider"]
            if "QNNExecutionProvider" in available_providers:
                providers.insert(0, "QNNExecutionProvider")

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=providers)

            self._status = ModelStatus.READY
            logger.info(f"Local embedding model '{self.model_name}' loaded successfully (Providers: {self.session.get_providers()})")
        except Exception as e:
            self._status = ModelStatus.FAILED
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise NexusException(f"Failed to initialize embedding model: {e}", status_code=500)

    def unload(self) -> None:
        """Unloads weights and frees compute handles."""
        self.session = None
        self.tokenizer = None
        self._status = ModelStatus.NOT_LOADED
        logger.info(f"Local embedding model '{self.model_name}' unloaded.")

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string into a 384-dimensional normalized vector."""
        vectors = self.embed_batch([text])
        return vectors[0].tolist()

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a batch of texts into an [N, 384] float32 numpy array.
        Performs mean pooling with attention masking and L2 normalization.
        """
        if not self.is_loaded or self.session is None or self.tokenizer is None:
            # Auto-load on first inference call
            self.load()

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        # Tokenize batch
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        # Run ONNX inference
        outputs = self.session.run(None, inputs)
        token_embeddings = outputs[0]  # Shape: [batch_size, seq_len, 384]

        # Mean pooling: sum token embeddings according to attention mask
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = sum_embeddings / sum_mask

        # L2 Normalize
        norms = np.linalg.norm(mean_pooled, ord=2, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        normalized_embeddings = (mean_pooled / norms).astype(np.float32)

        return normalized_embeddings


# Default singleton instance
local_embedding_model = LocalMiniLMEmbeddingModel()
