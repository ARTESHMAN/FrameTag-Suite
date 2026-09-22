"""
utils/yolo_io.py

YOLO Detection and YOLO Track Exporter & Importer for DarkLabel Modern:
- Detection format: <class_id> <x_center> <y_center> <width> <height> (normalized [0.0, 1.0])
- Track format: <class_id> <x_center> <y_center> <width> <height> <track_id>
- Handles bidirectional conversion between unnormalized pixel coordinates and normalized YOLO space.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Dict, List, Optional, Tuple

from core.annotation_manager import AnnotationManager
from core.annotation_models import Annotation, ShapeType


def export_yolo_txt(
    manager: AnnotationManager,
    img_width: int,
    img_height: int,
    out_dir: str,
    class_to_id: Dict[str, int],
    include_track_id: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> int:
    """
    Exports each annotated frame into a normalized YOLO text file.
    If include_track_id=True, appends the track_id as a 6th column (YOLO Track format).
    Returns total files exported.
    """
    os.makedirs(out_dir, exist_ok=True)
    frames = manager.get_annotated_frame_indices()
    total = len(frames)
    exported_count = 0

    with manager.lock:
        for idx, frame_idx in enumerate(frames):
            boxes = manager.get_annotations(frame_idx, visible_only=True, include_outside=False)
            if not boxes:
                continue

            file_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                for b in boxes:
                    if b.shape_type != ShapeType.BBOX:
                        continue
                    cls_id = class_to_id.get(b.class_name, b.class_id)
                    xc, yc, w, h = b.to_yolo(img_width, img_height)

                    if include_track_id:
                        f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {b.track_id}\n")
                    else:
                        f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            exported_count += 1
            if progress_callback:
                progress_callback(idx + 1, total)

    return exported_count


def import_yolo_txt(
    manager: AnnotationManager,
    labels_dir: str,
    img_width: int,
    img_height: int,
    id_to_class: Dict[int, str],
    is_track_format: bool = False,
    clear_existing: bool = False
) -> int:
    """
    Imports YOLO format text files from a directory into AnnotationManager.
    Matches frame numbers from file names (e.g., frame_000123.txt -> frame 123).
    Returns total bounding boxes imported.
    """
    if not os.path.exists(labels_dir):
        return 0

    if clear_existing:
        manager.clear()

    txt_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    imported_boxes = 0

    with manager.lock:
        for file_name in txt_files:
            match = re.search(r"(\d+)", file_name)
            if not match:
                continue
            frame_idx = int(match.group(1))

            file_path = os.path.join(labels_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls_id = int(parts[0])
                    norm_xc = float(parts[1])
                    norm_yc = float(parts[2])
                    norm_w = float(parts[3])
                    norm_h = float(parts[4])

                    track_id = int(parts[5]) if (is_track_format and len(parts) >= 6) else manager.track_manager.get_next_track_id()
                    class_name = id_to_class.get(cls_id, f"class_{cls_id}")

                    ann = Annotation.from_yolo(
                        track_id=track_id,
                        class_id=cls_id,
                        class_name=class_name,
                        frame_index=frame_idx,
                        norm_cx=norm_xc,
                        norm_cy=norm_yc,
                        norm_w=norm_w,
                        norm_h=norm_h,
                        img_w=img_width,
                        img_h=img_height,
                        is_keyframe=True,
                        source=AnnotationSource.MANUAL
                    )
                    manager.add_or_update_annotation(ann, auto_create_track=True)
                    imported_boxes += 1

    return imported_boxes