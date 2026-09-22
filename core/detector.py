"""
core/detector.py

YOLO Object Detection Engine for DarkLabel Modern:
- Resolves hardware devices cleanly for PyTorch CPU / CUDA / MPS builds.
- Wraps Ultralytics YOLO with fallback device handling so invalid device strings are never passed.
- Retains weights_path attribute across initialization and model loading.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from core.annotation_models import Annotation, AnnotationSource, ShapeType

logger = logging.getLogger(__name__)


class HardwareDeviceResolver:
    """Resolves and selects optimal hardware compute devices for PyTorch/Ultralytics."""

    @staticmethod
    def resolve_device(preferred: Optional[str] = None) -> str:
        """
        Returns a valid PyTorch/Ultralytics device string:
        'cpu', '0' (for cuda:0), or 'mps'.
        Never returns 'autodetect'.
        """
        import torch

        if not preferred or preferred.strip().lower() in ("auto detect", "autodetect", "auto", ""):
            if torch.cuda.is_available():
                return "0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        pref_clean = preferred.strip().lower()
        if "cuda" in pref_clean or "gpu" in pref_clean:
            return "0" if torch.cuda.is_available() else "cpu"
        elif "mps" in pref_clean:
            return "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
        return "cpu"


class YOLODetector:
    """Wrapper around Ultralytics YOLO models for inference and auto-labeling."""

    def __init__(self):
        self.model: Optional[Any] = None
        self.weights_path: str = ""
        self.target_device: str = "cpu"
        self.device_name: str = "CPU (Unloaded)"

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self, weights_path: str, preferred_device: Optional[str] = None) -> bool:
        """Loads YOLO weights file and maps inference to a verified hardware device string."""
        self.weights_path = str(weights_path or "")
        if not ULTRALYTICS_AVAILABLE:
            logger.error("Ultralytics package is not installed.")
            self.device_name = "Ultralytics Missing"
            return False

        try:
            self.target_device = HardwareDeviceResolver.resolve_device(preferred_device)
            self.model = YOLO(self.weights_path)
            self.device_name = self.target_device.upper() if self.target_device != "0" else "CUDA:0"
            logger.info(f"Loaded YOLO weights: {self.weights_path} on target: {self.target_device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO weights from {self.weights_path}: {e}")
            self.model = None
            self.device_name = "Load Failed"
            return False

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        allowed_classes: Optional[List[int]] = None
    ) -> List[Annotation]:
        """Runs inference on a single BGR image frame and returns detected Annotations."""
        if not self.is_loaded or frame is None:
            return []

        try:
            # Ensure valid device target fallback
            run_device = self.target_device if self.target_device in ("cpu", "0", "mps") else "cpu"

            results = self.model.predict(
                source=frame,
                conf=conf_thresh,
                iou=iou_thresh,
                device=run_device,
                verbose=False
            )

            annotations: List[Annotation] = []
            if not results:
                return annotations

            r = results[0]
            boxes = r.boxes

            for box in boxes:
                cls_id = int(box.cls[0].item())
                if allowed_classes is not None and len(allowed_classes) > 0 and cls_id not in allowed_classes:
                    continue

                conf = float(box.conf[0].item())
                class_name = r.names.get(cls_id, f"class_{cls_id}") if hasattr(r, "names") else f"class_{cls_id}"

                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(float, xyxy)

                x = max(0.0, x1)
                y = max(0.0, y1)
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)

                ann = Annotation(
                    track_id=1,
                    class_id=cls_id,
                    class_name=class_name,
                    frame_index=frame_index,
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    shape_type=ShapeType.BBOX,
                    is_keyframe=True,
                    source=AnnotationSource.AI,
                    confidence=round(conf, 4)
                )
                annotations.append(ann)

            return annotations

        except Exception as e:
            logger.error(f"YOLO inference execution error on frame {frame_index}: {e}")
            return []