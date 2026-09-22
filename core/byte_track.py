"""
core/byte_track.py

Asynchronous Batch Auto-Tracking Worker:
- Runs YOLO inference across a designated frame range in a background QThread.
- Emits accurate 0-100% progress metrics back to the UI progress bar.
"""

from __future__ import annotations

import traceback
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.annotation_manager import AnnotationManager
from core.annotation_models import Annotation, AnnotationSource
from core.detector import YOLODetector


class BatchAutoTrackWorker(QThread):
    """Background worker thread executing batch YOLO inference and tracking."""

    progress = Signal(int, int, str)  # frames_processed, total_target_frames, status_message
    finished = Signal(int, int)       # frames_processed, unique_tracks_created
    error = Signal(str)

    def __init__(
        self,
        weights_path: str,
        video_path: str,
        annotation_manager: AnnotationManager,
        start_frame: int,
        end_frame: int,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        allowed_classes: Optional[List[int]] = None,
        preferred_device: Optional[str] = None
    ):
        super().__init__()
        self.weights_path = weights_path
        self.video_path = video_path
        self.annotation_manager = annotation_manager
        self.start_frame = max(0, start_frame)
        self.end_frame = max(self.start_frame, end_frame)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.allowed_classes = allowed_classes
        self.preferred_device = preferred_device

        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        cap = None
        try:
            # 1. Initialize local detector instance for this thread
            detector = YOLODetector()
            if self.weights_path:
                success = detector.load_model(self.weights_path, self.preferred_device)
                if not success:
                    self.error.emit(f"Batch worker failed to load YOLO weights from: {self.weights_path}")
                    return
            elif not detector.is_loaded:
                self.error.emit("No YOLO model loaded. Please load weights before batch tracking.")
                return

            # 2. Open dedicated video capture handle
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"Batch worker failed to open video file: {self.video_path}")
                return

            total_target_frames = max(1, (self.end_frame - self.start_frame) + 1)
            frames_processed = 0
            unique_tracks_set = set()

            active_track_map = {}
            next_generated_id = 1

            with self.annotation_manager.lock:
                existing_tracks = list(self.annotation_manager.track_manager.tracks.keys())
                if existing_tracks:
                    next_generated_id = max(existing_tracks) + 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            for f_idx in range(self.start_frame, self.end_frame + 1):
                if self._is_cancelled:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    # Seek fallback in case sequential read drifted
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        frames_processed += 1
                        self.progress.emit(
                            frames_processed,
                            total_target_frames,
                            f"Frame {f_idx} (skipped / read error)"
                        )
                        continue

                boxes = detector.detect_frame(
                    frame=frame,
                    frame_index=f_idx,
                    conf_thresh=self.conf_thresh,
                    iou_thresh=self.iou_thresh,
                    allowed_classes=self.allowed_classes
                )

                with self.annotation_manager.lock:
                    for box in boxes:
                        matched_tid = None
                        b_cx, b_cy = box.center

                        for tid, (last_cx, last_cy, last_cls) in list(active_track_map.items()):
                            if last_cls == box.class_id:
                                dist = ((b_cx - last_cx) ** 2 + (b_cy - last_cy) ** 2) ** 0.5
                                if dist < 80.0:
                                    matched_tid = tid
                                    break

                        if matched_tid is None:
                            matched_tid = next_generated_id
                            next_generated_id += 1

                        active_track_map[matched_tid] = (b_cx, b_cy, box.class_id)
                        unique_tracks_set.add(matched_tid)

                        box.track_id = matched_tid
                        box.is_keyframe = True
                        box.source = AnnotationSource.AI

                        self.annotation_manager.add_or_update_annotation(box, auto_create_track=True)

                frames_processed += 1
                msg = f"Frame {f_idx}/{self.end_frame} ({len(unique_tracks_set)} tracks)"
                self.progress.emit(frames_processed, total_target_frames, msg)

            # Ensure 100% emission upon completion
            self.progress.emit(total_target_frames, total_target_frames, "Processing finished. Updating annotations...")
            self.finished.emit(frames_processed, len(unique_tracks_set))

        except Exception as e:
            err_trace = traceback.format_exc()
            self.error.emit(f"Batch tracking error:\n{str(e)}\n\n{err_trace}")
        finally:
            if cap is not None:
                cap.release()