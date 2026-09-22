"""
core/tracker.py

OpenCV single-object visual tracking engine for DarkLabel Modern:
- Wraps CSRT and KCF tracking algorithms with version-independent API adapters.
- Implements failure protection: detects loss of tracking lock, out-of-boundary departures,
  and radical bounding-box dimension collapse.
- Provides step_forward_track: initializes on frame N and predicts object geometry on frame N+1.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple
import cv2
import numpy as np

from core.annotation_manager import AnnotationManager
from core.annotation_models import Annotation, AnnotationSource


class TrackerType(str, Enum):
    CSRT = "CSRT"
    KCF = "KCF"


class VisualTracker:
    """
    OpenCV visual tracking wrapper featuring drift and failure detection heuristics.
    """

    def __init__(self, tracker_type: TrackerType = TrackerType.CSRT):
        self.tracker_type = tracker_type
        self.tracker: Optional[cv2.Tracker] = None
        self.initialized: bool = False
        self.initial_area: float = 1.0
        self.last_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def _create_tracker(self) -> cv2.Tracker:
        """Factory method handling OpenCV legacy and contrib API variations."""
        tt = self.tracker_type.value

        if tt == TrackerType.CSRT.value:
            if hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()
            if hasattr(cv2, "TrackerCSRT") and hasattr(cv2.TrackerCSRT, "create"):
                return cv2.TrackerCSRT.create()
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
                return cv2.legacy.TrackerCSRT_create()

        elif tt == TrackerType.KCF.value:
            if hasattr(cv2, "TrackerKCF_create"):
                return cv2.TrackerKCF_create()
            if hasattr(cv2, "TrackerKCF") and hasattr(cv2.TrackerKCF, "create"):
                return cv2.TrackerKCF.create()
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
                return cv2.legacy.TrackerKCF_create()

        # Fallback to MIL if CSRT/KCF are unavailable
        if hasattr(cv2, "TrackerMIL_create"):
            return cv2.TrackerMIL_create()
        if hasattr(cv2, "TrackerMIL") and hasattr(cv2.TrackerMIL, "create"):
            return cv2.TrackerMIL.create()

        raise RuntimeError("No compatible OpenCV tracking algorithms found in cv2 environment.")

    def init(self, frame_bgr: np.ndarray, bbox_xywh: Tuple[float, float, float, float]) -> bool:
        """Initializes the tracker on a bounding box region."""
        if frame_bgr is None or frame_bgr.size == 0:
            return False

        h_img, w_img = frame_bgr.shape[:2]
        x, y, w, h = bbox_xywh

        # Coordinate clamping
        x_c = max(0.0, min(float(x), float(w_img - 2)))
        y_c = max(0.0, min(float(y), float(h_img - 2)))
        w_c = max(2.0, min(float(w), float(w_img - x_c)))
        h_c = max(2.0, min(float(h), float(h_img - y_c)))

        int_rect = (int(x_c), int(y_c), int(w_c), int(h_c))
        if int_rect[2] <= 1 or int_rect[3] <= 1:
            return False

        try:
            self.tracker = self._create_tracker()
            self.tracker.init(frame_bgr, int_rect)
            self.initialized = True
            self.initial_area = float(int_rect[2] * int_rect[3])
            self.last_bbox = (float(int_rect[0]), float(int_rect[1]), float(int_rect[2]), float(int_rect[3]))
            return True
        except Exception:
            self.initialized = False
            return False

    def update(self, frame_bgr: np.ndarray) -> Tuple[bool, Tuple[float, float, float, float], float]:
        """
        Updates tracking on the next frame with validation checks.
        Returns: (success: bool, (x, y, w, h), estimated_confidence: float)
        """
        if not self.initialized or self.tracker is None or frame_bgr is None or frame_bgr.size == 0:
            return (False, (0.0, 0.0, 0.0, 0.0), 0.0)

        h_img, w_img = frame_bgr.shape[:2]

        try:
            success, box = self.tracker.update(frame_bgr)
            if not success:
                return (False, self.last_bbox, 0.0)

            x, y, w, h = box
            curr_area = float(w * h)

            # Heuristic 1: Detect radical dimension collapse or explosive expansion
            area_ratio = curr_area / max(1.0, self.initial_area)
            if area_ratio < 0.05 or area_ratio > 20.0:
                return (False, self.last_bbox, 0.0)

            # Heuristic 2: Boundary collision
            if x + w < 0 or x > w_img or y + h < 0 or y > h_img:
                return (False, self.last_bbox, 0.0)

            # Clamp coordinates
            x_c = max(0.0, min(float(x), float(w_img - 1)))
            y_c = max(0.0, min(float(y), float(h_img - 1)))
            w_c = max(1.0, min(float(w), float(w_img - x_c)))
            h_c = max(1.0, min(float(h), float(h_img - y_c)))

            # Normalized pseudo-confidence heuristic based on area stability
            stability_conf = max(0.3, min(1.0, 1.0 - abs(1.0 - area_ratio) * 0.2))

            self.last_bbox = (x_c, y_c, w_c, h_c)
            return (True, self.last_bbox, round(stability_conf, 4))
        except Exception:
            return (False, self.last_bbox, 0.0)

    @classmethod
    def step_forward_track(
        cls,
        manager: AnnotationManager,
        track_id: int,
        source_frame: int,
        target_frame: int,
        source_img: np.ndarray,
        target_img: np.ndarray,
        tracker_type: TrackerType = TrackerType.CSRT
    ) -> Tuple[bool, Optional[Annotation], str]:
        """
        Executes single-step visual tracking between two consecutive frames (Enter key workflow).
        Returns: (success, predicted_annotation, status_message)
        """
        with manager.lock:
            active_box = manager.get_annotation(source_frame, track_id)
            if not active_box:
                return (False, None, f"No active box found for Track #{track_id} on frame {source_frame}.")

            tracker = cls(tracker_type)
            if not tracker.init(source_img, active_box.to_xywh()):
                return (False, None, "Failed to initialize visual tracker on source region.")

            success, (tx, ty, tw, th), conf = tracker.update(target_img)
            if not success:
                return (False, None, "Visual tracker lost target lock on subsequent frame.")

            tracked_ann = Annotation(
                track_id=track_id,
                class_id=active_box.class_id,
                class_name=active_box.class_name,
                frame_index=target_frame,
                x=tx,
                y=ty,
                width=tw,
                height=th,
                shape_type=active_box.shape_type,
                is_keyframe=False,
                interpolated=False,
                source=AnnotationSource.TRACKER,
                confidence=conf,
                occluded=active_box.occluded,
                outside=False,
                attributes=dict(active_box.attributes)
            )

            manager.add_or_update_annotation(tracked_ann, auto_create_track=False)
            return (True, tracked_ann, f"Tracked #{track_id} forward to frame {target_frame} (conf: {conf:.2f}).")