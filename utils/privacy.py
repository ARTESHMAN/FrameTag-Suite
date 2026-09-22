"""
utils/privacy.py

Video Anonymization & Privacy Processing Engine for DarkLabel Modern:
- Supported masking modes: GAUSSIAN_BLUR, PIXELATED_MOSAIC, SOLID_BLACKOUT.
- Scoped anonymization: filter by specific Category IDs or Track IDs.
- Operates non-destructively: writes to a new video container without touching source media.
- Supports progress reporting and cooperative cancellation for long video rendering.
"""

from __future__ import annotations

from enum import Enum
import time
from typing import Callable, List, Optional, Set, Tuple
import cv2
import numpy as np

from core.annotation_manager import AnnotationManager
from core.annotation_models import ShapeType


class PrivacyMode(str, Enum):
    GAUSSIAN_BLUR = "blur"
    PIXELATED_MOSAIC = "mosaic"
    SOLID_BLACKOUT = "blackout"


def apply_privacy_mask(
    image_bgr: np.ndarray,
    bbox_xywh: Tuple[float, float, float, float],
    mode: PrivacyMode = PrivacyMode.GAUSSIAN_BLUR,
    blur_kernel_divisor: int = 4,
    mosaic_downscale_factor: int = 12
) -> None:
    """
    Applies an in-place visual privacy mask to a bounding box region within an image.
    """
    h_img, w_img = image_bgr.shape[:2]
    x, y, w, h = bbox_xywh

    x1 = max(0, min(int(x), w_img - 1))
    y1 = max(0, min(int(y), h_img - 1))
    x2 = max(0, min(int(x + w), w_img))
    y2 = max(0, min(int(y + h), h_img))

    roi_w = x2 - x1
    roi_h = y2 - y1

    if roi_w <= 2 or roi_h <= 2:
        return

    roi = image_bgr[y1:y2, x1:x2]

    if mode == PrivacyMode.GAUSSIAN_BLUR:
        # Adaptive odd kernel size based on ROI dimensions
        kw = max(3, (roi_w // blur_kernel_divisor) | 1)
        kh = max(3, (roi_h // blur_kernel_divisor) | 1)
        image_bgr[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kw, kh), 0)

    elif mode == PrivacyMode.PIXELATED_MOSAIC:
        # Downscale then upscale using nearest-neighbor interpolation
        down_w = max(1, roi_w // mosaic_downscale_factor)
        down_h = max(1, roi_h // mosaic_downscale_factor)
        small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
        image_bgr[y1:y2, x1:x2] = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

    elif mode == PrivacyMode.SOLID_BLACKOUT:
        image_bgr[y1:y2, x1:x2] = (0, 0, 0)


def render_privacy_video(
    video_provider,
    manager: AnnotationManager,
    out_path: str,
    mode: PrivacyMode = PrivacyMode.GAUSSIAN_BLUR,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    track_ids: Optional[Set[int]] = None,
    class_names: Optional[Set[str]] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Tuple[bool, int]:
    """
    Renders an anonymized MP4 video stream.
    Returns (success_flag, total_frames_processed).
    """
    total_video_frames = video_provider.total_frames
    if total_video_frames <= 0:
        return (False, 0)

    start_f = max(0, min(start_frame, total_video_frames - 1))
    end_f = total_video_frames - 1 if end_frame is None else max(start_f, min(end_frame, total_video_frames - 1))
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

            frame_bgr = video_provider.get_frame_at(f_idx)
            if frame_bgr is None:
                break

            # Fetch frame annotations under thread lock
            with manager.lock:
                boxes = manager.get_annotations(f_idx, visible_only=True, include_outside=False)

            for b in boxes:
                if b.shape_type != ShapeType.BBOX:
                    continue
                if track_ids is not None and b.track_id not in track_ids:
                    continue
                if class_names is not None and b.class_name not in class_names:
                    continue

                apply_privacy_mask(frame_bgr, b.to_xywh(), mode=mode)

            writer.write(frame_bgr)
            processed_count += 1

            if progress_callback:
                elapsed = max(1e-4, time.time() - start_time)
                current_fps = processed_count / elapsed
                progress_callback(processed_count, total_to_process, current_fps)

        return (True, processed_count)
    finally:
        writer.release()