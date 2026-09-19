from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict


class ModelStatus(str, Enum):
    """Lifecycle states of an AI model within NEXUS."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class BaseModelInterface(ABC):
    """Abstract base class defining the contract for all NEXUS model adapters."""

    def __init__(self, model_name: str, target_hardware: str = "Hexagon NPU"):
        self.model_name = model_name
        self.target_hardware = target_hardware
        self._status: ModelStatus = ModelStatus.NOT_LOADED

    @property
    def status(self) -> ModelStatus:
        """Returns the current lifecycle status of the model."""
        return self._status

    @property
    def is_loaded(self) -> bool:
        """Returns True if the model is initialized and ready for inference."""
        return self._status == ModelStatus.READY

    @abstractmethod
    def load(self) -> None:
        """Loads model weights and initializes execution provider context."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unloads weights from device memory and frees compute handles."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Returns hardware, status, and metadata descriptor for the model."""
        return {
            "model_name": self.model_name,
            "target_hardware": self.target_hardware,
            "status": self.status.value,
            "is_loaded": self.is_loaded,
        }
