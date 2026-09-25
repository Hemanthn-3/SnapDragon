"""
NEXUS Local Vision Model Adapter: Genuine ONNX Neural Network Inference
Dual-head architecture: 1000-class ImageNet classification + 512-dimensional visual embedding.
Supports PNG, JPG, JPEG, and raster images extracted from PDFs.
Performs local visual inspection, strictly demarcating OBSERVED (measurable facts)
from INFERRED (neural semantic deductions). Zero unrestricted camera monitoring.
"""

import io
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from backend.interfaces.base import ModelStatus
from backend.interfaces.vision import VisionModel
from backend.logger import get_logger

logger = get_logger("nexus.vision")

DEFAULT_VISION_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "vision"
DEFAULT_MODEL_PATH = DEFAULT_VISION_DIR / "resnet18_vision.onnx"
DEFAULT_CLASSES_PATH = DEFAULT_VISION_DIR / "imagenet_classes.json"

# Standard ImageNet normalization coefficients
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LocalClipVisionModel(VisionModel):
    """
    Local multimodal vision understanding adapter executing real ONNX Runtime inference.
    Executes real neural-network weights on Qualcomm Hexagon NPU (QNNExecutionProvider)
    or CPUExecutionProvider fallback on non-Snapdragon systems.
    Zero external cloud calls; zero heuristic substitute vectors.
    """

    def __init__(
        self,
        model_name: str = "ResNet-18-Vision",
        model_path: Optional[Union[str, Path]] = None,
        classes_path: Optional[Union[str, Path]] = None,
        auto_load: bool = False,
    ):
        super().__init__(model_name=model_name)
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.classes_path = Path(classes_path) if classes_path else DEFAULT_CLASSES_PATH
        self._session = None
        self._classes: List[str] = []
        self._execution_provider = "Unloaded"
        self._blockers: List[str] = []
        self._detect_environment()

        if auto_load:
            self.load()

    def _detect_environment(self) -> None:
        """Evaluates execution hardware and logs compatibility blockers."""
        machine = platform.machine().upper()
        system = platform.system()
        self._blockers = []

        if machine not in ["ARM64", "AARCH64"]:
            self._blockers.append(
                f"Architecture Mismatch: Current host is {machine} ({system}). "
                f"Qualcomm Hexagon NPU is physically absent on non-Snapdragon silicon."
            )
            self._blockers.append(
                "QNN Runtime Blocker: QNN Execution Provider targets Snapdragon X Elite/Plus "
                "Hexagon Tensor Processor on Windows 11 ARM64 with QnnHtp.dll."
            )
            self.target_hardware = "CPU (x86_64 Fallback)"
        else:
            self.target_hardware = "Hexagon NPU"

    @property
    def blockers(self) -> List[str]:
        return list(self._blockers)

    @property
    def execution_provider(self) -> str:
        return self._execution_provider

    def _load_classes(self) -> None:
        """Loads 1000 ImageNet category labels from JSON."""
        if self._classes:
            return
        if self.classes_path.exists():
            try:
                with open(self.classes_path, "r", encoding="utf-8") as f:
                    self._classes = json.load(f)
                return
            except Exception as e:
                logger.warning(f"Could not load classes from {self.classes_path}: {e}")
        # Fallback to indexed names
        self._classes = [f"category_{i}" for i in range(1000)]

    def _get_class_name(self, index: int) -> str:
        """Safe lookup for class category names."""
        if 0 <= index < len(self._classes):
            return self._classes[index]
        return f"class_{index}"

    def load(self) -> bool:
        """
        Initializes the ONNX Runtime inference session with real model weights.
        Attempts QNNExecutionProvider on Snapdragon ARM64; falls back to CPUExecutionProvider.
        """
        t0 = time.perf_counter()
        logger.info(f"Loading local vision model '{self.model_name}' from {self.model_path}...")

        if not self.model_path.exists():
            self._status = ModelStatus.FAILED
            logger.error(f"Vision model file not found at: {self.model_path}")
            return False

        try:
            import onnxruntime as ort

            self._load_classes()

            providers = []
            available = ort.get_available_providers()

            # Attempt QNN on ARM64 Snapdragon only if QnnHtp backend is available
            if "QNNExecutionProvider" in available and not self._blockers:
                providers.append((
                    "QNNExecutionProvider",
                    {
                        "backend_path": "QnnHtp.dll",
                        "htp_performance_mode": "sustained_high_performance",
                    },
                ))

            # Always supply CPUExecutionProvider as fallback or primary
            providers.append("CPUExecutionProvider")

            sess = ort.InferenceSession(str(self.model_path), providers=providers)
            actual_provider = sess.get_providers()[0]

            self._session = sess
            self._execution_provider = actual_provider

            if actual_provider == "QNNExecutionProvider":
                self.target_hardware = "Hexagon NPU"
            else:
                self.target_hardware = "CPU (PyTorch/ORT Fallback)"

            self._status = ModelStatus.READY
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(
                f"Local vision engine ready in {elapsed}ms "
                f"(Provider: {self._execution_provider}, Hardware: {self.target_hardware})"
            )
            return True

        except Exception as e:
            self._status = ModelStatus.FAILED
            self._session = None
            self._execution_provider = "Unloaded"
            logger.error(f"Failed to initialize vision ONNX session: {e}", exc_info=True)
            return False

    def unload(self) -> None:
        self._session = None
        self._status = ModelStatus.NOT_LOADED
        self._execution_provider = "Unloaded"
        logger.info(f"Vision model '{self.model_name}' unloaded.")

    def get_status(self) -> Dict[str, Any]:
        """Returns diagnostic and architectural metadata for the vision model."""
        return {
            "model_name": self.model_name,
            "architecture": "ResNet-18 (Dual-Head: ImageNet-1k + 512-dim Embedding)",
            "framework": "ONNX Runtime",
            "model_path": str(self.model_path),
            "target_hardware": "Snapdragon X Elite Hexagon NPU",
            "current_hardware": self.target_hardware,
            "execution_provider": self.execution_provider,
            "status": self.status.value,
            "is_loaded": self.is_loaded,
            "supported_formats": ["jpg", "jpeg", "png", "pdf"],
            "blockers": self.blockers,
            "epistemic_separation": {
                "observed": "Measurable optical data (dimensions, aspect ratio, color channel statistics, luminance, contrast, entropy, dominant palette)",
                "inferred": "Semantic deductions from genuine neural network inference (top predictions, classification categories, confidence)",
            },
            "capabilities_boundary": (
                "Optical features and dominant colors were measured directly from pixel arrays. "
                "Semantic categories and 512-dim visual embeddings were inferred via ResNet-18 neural network. "
                "Model is trained on ImageNet-1k general categories; fine-grained industrial defect detection requires dedicated inspection models. "
                "Alphanumeric text must be verified via the dedicated OCR pipeline."
            ),
        }

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        """
        Standard ImageNet preprocessing:
        1. RGB conversion
        2. Bicubic resize to 224x224
        3. Convert to float32 [0.0, 1.0]
        4. Normalize with ImageNet mean and std
        5. Transpose to CHW (3, 224, 224) and add batch dim -> (1, 3, 224, 224)
        """
        img_rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        normalized = (arr - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(normalized, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0)
        return batch.astype(np.float32)

    def encode_image(self, image_bytes: bytes) -> List[float]:
        """
        Encodes image bytes into a normalized 512-dimensional visual embedding vector
        via real neural-network inference (ResNet-18 avgpool feature representation).
        """
        if not image_bytes:
            raise ValueError("Image bytes payload is empty.")

        if self.status != ModelStatus.READY or self._session is None:
            if not self.load():
                raise RuntimeError(
                    f"Vision model failed to load from '{self.model_path}'. "
                    "Ensure ONNX weights are present."
                )

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                input_tensor = self._preprocess(img)
        except Exception as e:
            raise ValueError(f"Unable to decode image bytes: {e}") from e

        # Real ONNX Runtime inference
        outputs = self._session.run(["embedding"], {"input": input_tensor})
        raw_emb = outputs[0][0].astype(np.float32)  # shape (512,)

        # L2 normalize
        norm = float(np.linalg.norm(raw_emb))
        if norm > 0:
            raw_emb = raw_emb / norm

        return raw_emb.tolist()

    def inspect_image(
        self,
        image_bytes: Optional[bytes] = None,
        filename: str = "image.png",
        prompt: Optional[str] = None,
        image_input: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Inspects an image, extracting physical measurable optical facts (OBSERVED)
        and genuine neural-network semantic deductions (INFERRED).
        Strictly distinguishes optical measurements from neural classifications.
        """
        raw_target = image_bytes if image_bytes is not None else image_input
        if raw_target is None:
            raise ValueError("Image payload is empty or not provided.")

        if isinstance(raw_target, (str, Path)):
            p = Path(raw_target)
            if not p.exists():
                raise FileNotFoundError(f"Image file not found at '{p}'")
            filename = p.name
            with open(p, "rb") as f:
                data = f.read()
        elif isinstance(raw_target, bytes):
            data = raw_target
        elif isinstance(raw_target, Image.Image):
            bio = io.BytesIO()
            fmt = raw_target.format or "PNG"
            raw_target.save(bio, format=fmt)
            data = bio.getvalue()
        else:
            raise TypeError(f"Unsupported image input type: {type(raw_target)}")

        if not data:
            raise ValueError("Image bytes payload is empty.")

        # Ensure model is ready
        if self.status != ModelStatus.READY or self._session is None:
            if not self.load():
                raise RuntimeError(
                    f"Vision model failed to load from '{self.model_path}'. "
                    "Ensure ONNX weights are present."
                )

        t_total_start = time.perf_counter()

        try:
            pil_img = Image.open(io.BytesIO(data))
        except Exception as e:
            raise ValueError(f"Unable to decode image data: {e}") from e

        with pil_img as img:
            width, height = img.size
            format_name = img.format or Path(filename).suffix.lstrip(".").upper() or "PNG"
            mode = img.mode
            aspect_ratio = round(width / float(height), 2) if height > 0 else 1.0

            # Convert to RGB for optical analysis
            rgb_img = img.convert("RGB")
            arr = np.array(rgb_img, dtype=np.float32)

            # Channel statistics
            r_channel = arr[:, :, 0]
            g_channel = arr[:, :, 1]
            b_channel = arr[:, :, 2]

            r_mean, r_std = float(np.mean(r_channel)), float(np.std(r_channel))
            g_mean, g_std = float(np.mean(g_channel)), float(np.std(g_channel))
            b_mean, b_std = float(np.mean(b_channel)), float(np.std(b_channel))

            # Perceived luminance (standard Rec. 601 formula)
            luminance = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
            brightness = round(float(np.mean(luminance)), 2)
            contrast = round(float(np.std(luminance)), 2)

            # Calculate Shannon entropy (visual complexity)
            hist, _ = np.histogram(luminance, bins=256, range=(0, 256), density=True)
            hist = hist[hist > 0]
            entropy = round(float(-np.sum(hist * np.log2(hist))), 2)

            # Dominant palette via median-cut quantization
            quantized = rgb_img.quantize(colors=4, method=Image.Quantize.MEDIANCUT).convert("RGB")
            palette_arr = np.array(quantized).reshape(-1, 3)
            unique_colors, counts = np.unique(palette_arr, axis=0, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]

            dominant_palette = []
            for idx in sorted_indices[:4]:
                color = unique_colors[idx]
                pct = round((counts[idx] / len(palette_arr)) * 100, 1)
                hex_val = f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"
                dominant_palette.append({
                    "rgb": [int(color[0]), int(color[1]), int(color[2])],
                    "hex": hex_val,
                    "coverage_percent": pct,
                })

            # OBSERVED: Strictly objective optical measurements
            observed: Dict[str, Any] = {
                "filename": filename,
                "dimensions": {"width": width, "height": height},
                "aspect_ratio": aspect_ratio,
                "format": format_name,
                "color_mode": mode,
                "channel_statistics": {
                    "red": {"mean": round(r_mean, 2), "std": round(r_std, 2)},
                    "green": {"mean": round(g_mean, 2), "std": round(g_std, 2)},
                    "blue": {"mean": round(b_mean, 2), "std": round(b_std, 2)},
                },
                "brightness": brightness,
                "contrast": contrast,
                "complexity_entropy": entropy,
                "dominant_palette": dominant_palette,
            }

            # -----------------------------------------------------------------
            # REAL NEURAL INFERENCE (ResNet-18 via ONNX Runtime)
            # -----------------------------------------------------------------
            t_prep = time.perf_counter()
            input_tensor = self._preprocess(rgb_img)
            prep_ms = round((time.perf_counter() - t_prep) * 1000, 2)

            t_infer = time.perf_counter()
            logits, embedding = self._session.run(["logits", "embedding"], {"input": input_tensor})
            infer_ms = round((time.perf_counter() - t_infer) * 1000, 2)

            t_post = time.perf_counter()
            # Softmax to get genuine class probabilities
            logits_1d = logits[0]
            shifted = logits_1d - np.max(logits_1d)
            exp_logits = np.exp(shifted)
            probabilities = exp_logits / np.sum(exp_logits)

            # Top 5 predictions
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            top_candidates = []
            for class_idx in top5_indices:
                top_candidates.append({
                    "label": self._get_class_name(int(class_idx)),
                    "confidence": round(float(probabilities[class_idx]), 4),
                    "class_index": int(class_idx),
                })

            primary_pred = top_candidates[0]
            visual_category = primary_pred["label"]
            confidence = primary_pred["confidence"]
            semantic_tags = [c["label"] for c in top_candidates]

            # Candid descriptive synthesis
            candidate_summary = ", ".join([f"{c['label']} ({c['confidence']*100:.1f}%)" for c in top_candidates[:3]])
            description = (
                f"Neural vision model classified image as '{visual_category}' with {confidence*100:.1f}% confidence. "
                f"Top visual candidates: {candidate_summary}. "
                f"Image dimensions: {width}x{height} ({format_name}), luminance {brightness}/255."
            )
            if prompt:
                description += f" Visual inquiry: '{prompt}'."

            inferred: Dict[str, Any] = {
                "visual_category": visual_category,
                "primary_classification": visual_category,
                "confidence": confidence,
                "semantic_tags": semantic_tags,
                "top_candidates": top_candidates,
                "description": description,
                "capabilities_boundary": (
                    "Optical features and dominant colors were measured directly from pixel arrays. "
                    "Visual features and categories were generated via genuine neural network inference (ResNet-18 ImageNet-1k). "
                    "Model is trained on general object and scene categories; fine-grained industrial surface crack or microscopic metallurgical defect detection requires specialized inspection models. "
                    "Alphanumeric text must be verified via the dedicated OCR pipeline."
                ),
            }

            post_ms = round((time.perf_counter() - t_post) * 1000, 2)
            total_elapsed_ms = round((time.perf_counter() - t_total_start) * 1000, 2)

            logger.info(
                f"[VISION] Inspected '{filename}' in {total_elapsed_ms}ms "
                f"(prep={prep_ms}ms, infer={infer_ms}ms, post={post_ms}ms) "
                f"-> Primary: {visual_category} ({confidence*100:.1f}%)"
            )

            return {
                "observed": observed,
                "inferred": inferred,
                "analysis": description,
                "duration_ms": total_elapsed_ms,
                "preprocessing_ms": prep_ms,
                "inference_ms": infer_ms,
                "postprocessing_ms": post_ms,
                "hardware": self.target_hardware,
                "execution_provider": self.execution_provider,
            }

    def extract_images_from_pdf(
        self,
        pdf_bytes: bytes,
        page_number: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts embedded raster images from PDF pages using pypdf.
        Returns list of extracted image records with binary data and metadata.
        """
        import pypdf

        extracted: List[Dict[str, Any]] = []
        try:
            bio = io.BytesIO(pdf_bytes)
            reader = pypdf.PdfReader(bio)
            total_pages = len(reader.pages)

            pages_to_check = [page_number - 1] if page_number and 1 <= page_number <= total_pages else list(range(total_pages))

            for p_idx in pages_to_check:
                page = reader.pages[p_idx]
                current_page_num = p_idx + 1

                for img_idx, image_file in enumerate(page.images, 1):
                    img_name = getattr(image_file, "name", f"image_p{current_page_num}_{img_idx}.png")
                    img_data = image_file.data
                    try:
                        with Image.open(io.BytesIO(img_data)) as pil_img:
                            w, h = pil_img.size
                            fmt = pil_img.format or "PNG"
                    except Exception:
                        w, h, fmt = 0, 0, "UNKNOWN"

                    extracted.append({
                        "page_number": current_page_num,
                        "image_index": img_idx,
                        "image_name": img_name,
                        "image_bytes": img_data,
                        "size_bytes": len(img_data),
                        "dimensions": {"width": w, "height": h},
                        "format": fmt,
                    })

            logger.info(f"[PDF_IMAGE_EXTRACT] Extracted {len(extracted)} embedded images from PDF ({total_pages} pages).")
            return extracted
        except Exception as e:
            logger.error(f"Failed to extract images from PDF: {e}", exc_info=True)
            return []


# Global singleton instance
local_clip_vision = LocalClipVisionModel()
LocalVisionModel = LocalClipVisionModel
