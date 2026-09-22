"""
utils/video_utils.py

Video Processing & Preview Overlay Utilities for DarkLabel Modern:
- extract_subclip: Cuts a video sequence temporally between [start_frame, end_frame].
- render_annotated_video: Renders preview/verification videos with burned-in overlays
  (bounding boxes, labels, confidence, historical trajectory trails, and timecode banners).
"""

from __future__ import annotations

from collections import deque
import time
from typing import Callable, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from core.annotation_manager import AnnotationManager
from core.annotation_models import Annotation, ShapeType
from gui.canvas_items import get_track_color


def extract_subclip(
    video_provider,
    start_frame: int,
    end_frame: int,
    out_path: str,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Tuple[bool, int]:
    """
    Extracts a temporal sub-clip from start_frame to end_frame into a standalone MP4 video.
    Returns (success_flag, total_frames_written).
    """
    total = video_provider.total_frames
    start_f = max(0, min(start_frame, total - 1))
    end_f = max(start_f, min(end_frame, total - 1))
    total_to_process = end_f - start_f + 1

    width = video_provider.width
    height = video_provider.height
    fps = video_provider.fps if video_provider.fps > 0 else 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        return (False, 0)

    start_time = time.time()
    processed_count = 0

    try:
        for f_idx in range(start_f, end_f + 1):
            if cancel_check and cancel_check():
                writer.release()
                return (False, processed_count)

            frame = video_provider.get_frame_at(f_idx)
            if frame is None:
                break

            writer.write(frame)
            processed_count += 1

            if progress_callback:
                elapsed = max(1e-4, time.time() - start_time)
                current_fps = processed_count / elapsed
                progress_callback(processed_count, total_to_process, current_fps)

        return (True, processed_count)
    finally:
        writer.release()


def render_annotated_video(
    video_provider,
    manager: AnnotationManager,
    out_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    show_ids: bool = True,
    show_labels: bool = True,
    show_conf: bool = True,
    show_trajectory: bool = True,
    trajectory_length: int = 30,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Tuple[bool, int]:
    """
    Renders an audit/verification video with burned-in annotations, trajectories, and timecode banners.
    Returns (success_flag, total_frames_processed).
    """
    total = video_provider.total_frames
    start_f = max(0, min(start_frame, total - 1))
    end_f = total - 1 if end_frame is None else max(start_f, min(end_frame, total - 1))
    total_to_process = end_f - start_f + 1

    width = video_provider.width
    height = video_provider.height
    fps = video_provider.fps if video_provider.fps > 0 else 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        return (False, 0)

    # Trajectory history buffers: track_id -> deque of (center_x, center_y)
    trajectories: Dict[int, deque] = {}

    start_time = time.time()
    processed_count = 0

    try:
        for f_idx in range(start_f, end_f + 1):
            if cancel_check and cancel_check():
                writer.release()
                return (False, processed_count)

            frame = video_provider.get_frame_at(f_idx)
            if frame is None:
                break

            with manager.lock:
                boxes = manager.get_annotations(f_idx, visible_only=True, include_outside=False)

            active_track_ids_in_frame: Set[int] = set()

            # 1. Draw Historical Trajectory Paths
            if show_trajectory:
                for b in boxes:
                    if b.shape_type != ShapeType.BBOX:
                        continue
                    active_track_ids_in_frame.add(b.track_id)
                    cx, cy = b.center

                    if b.track_id not in trajectories:
                        trajectories[b.track_id] = deque(maxlen=trajectory_length)
                    trajectories[b.track_id].append((int(cx), int(cy)))

                for tid, pts in trajectories.items():
                    if len(pts) > 1:
                        qcolor = get_track_color(tid)
                        bgr_color = (qcolor.blue(), qcolor.green(), qcolor.red())
                        for i in range(1, len(pts)):
                            thickness = max(1, int(2.5 * (i / len(pts))))
                            cv2.line(frame, pts[i - 1], pts[i], bgr_color, thickness)

            # 2. Draw Bounding Boxes and Text Overlays
            for b in boxes:
                if b.shape_type != ShapeType.BBOX:
                    continue

                qcolor = get_track_color(b.track_id)
                bgr_color = (qcolor.blue(), qcolor.green(), qcolor.red())

                x1, y1 = int(b.x), int(b.y)
                x2, y2 = int(b.x + b.width), int(b.y + b.height)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)

                # Format label string
                tag_parts = []
                if show_ids:
                    tag_parts.append(f"#{b.track_id}")
                if show_labels:
                    tag_parts.append(b.class_name)
                if show_conf and b.confidence < 0.99:
                    tag_parts.append(f"{b.confidence:.2f}")

                tag_text = " ".join(tag_parts)

                if tag_text:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_thickness = 1
                    (text_w, text_h), baseline = cv2.getTextSize(tag_text, font, font_scale, font_thickness)

                    tag_y1 = max(0, y1 - text_h - 6)
                    tag_y2 = y1
                    tag_x2 = min(width, x1 + text_w + 8)

                    # Filled badge background
                    cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), bgr_color, -1)

                    # Text label (white or black contrast)
                    luminance = (qcolor.red() * 299 + qcolor.green() * 587 + qcolor.blue() * 114) / 1000
                    txt_color = (0, 0, 0) if luminance > 140 else (255, 255, 255)
                    cv2.putText(
                        frame,
                        tag_text,
                        (x1 + 4, tag_y2 - baseline - 1),
                        font,
                        font_scale,
                        txt_color,
                        font_thickness,
                        cv2.LINE_AA
                    )

            # 3. Draw Bottom Telemetry Banner
            banner_h = 24
            cv2.rectangle(frame, (0, height - banner_h), (width, height), (18, 19, 22), -1)
            timecode_sec = f_idx / fps
            tc_str = f"{int(timecode_sec // 3600):02}:{int((timecode_sec % 3600) // 60):02}:{int(timecode_sec % 60):02}:{int(f_idx % fps):02}"
            banner_text = f"Frame: {f_idx:,} / {total - 1:,} | Time: {tc_str} | Active: {len(boxes)}"
            cv2.putText(
                frame,
                banner_text,
                (12, height - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 229, 255),
                1,
                cv2.LINE_AA
            )

            writer.write(frame)
            processed_count += 1

            if progress_callback:
                elapsed = max(1e-4, time.time() - start_time)
                current_fps = processed_count / elapsed
                progress_callback(processed_count, total_to_process, current_fps)

        return (True, processed_count)
    finally:
        writer.release()