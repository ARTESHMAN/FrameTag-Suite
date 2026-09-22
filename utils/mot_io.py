"""
utils/mot_io.py

MOT Challenge Benchmark Exporter & Importer for DarkLabel Modern:
- Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
- Note: MOT challenge uses 1-based frame indexing (Frame 0 in app = Frame 1 in MOT).
"""

from __future__ import annotations

csv_enabled = True
try:
    import csv
except ImportError:
    csv_enabled = False

import os
from typing import Dict, List, Optional

from core.annotation_manager import AnnotationManager
from core.annotation_models import Annotation, AnnotationSource, ShapeType


def export_mot_challenge(manager: AnnotationManager, out_path: str) -> int:
    """
    Exports annotations into MOT Challenge benchmark gt.txt format.
    Returns total annotation records written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    record_count = 0

    with manager.lock, open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        frames = manager.get_annotated_frame_indices()

        for frame_idx in frames:
            boxes = manager.get_annotations(frame_idx, visible_only=True, include_outside=False)
            for b in boxes:
                if b.shape_type != ShapeType.BBOX:
                    continue
                # MOT is 1-indexed for frames
                writer.writerow([
                    frame_idx + 1,
                    b.track_id,
                    f"{b.x:.2f}",
                    f"{b.y:.2f}",
                    f"{b.width:.2f}",
                    f"{b.height:.2f}",
                    f"{b.confidence:.4f}",
                    -1,
                    -1,
                    -1
                ])
                record_count += 1

    return record_count


def import_mot_challenge(
    manager: AnnotationManager,
    file_path: str,
    default_class_id: int = 0,
    default_class_name: str = "target",
    clear_existing: bool = False
) -> int:
    """
    Imports a MOT challenge gt.txt file into AnnotationManager.
    Converts 1-based MOT frame indices to 0-based application indices.
    """
    if not os.path.exists(file_path):
        return 0

    if clear_existing:
        manager.clear()

    count = 0
    with manager.lock, open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            if len(row) == 1:
                row = row[0].split()

            try:
                frame_idx = int(float(row[0])) - 1  # Convert 1-indexed to 0-indexed
                track_id = int(float(row[1]))
                x = float(row[2])
                y = float(row[3])
                w = float(row[4])
                h = float(row[5])
                conf = float(row[6]) if len(row) > 6 and float(row[6]) >= 0.0 else 1.0
            except (ValueError, IndexError):
                continue

            ann = Annotation(
                track_id=track_id,
                class_id=default_class_id,
                class_name=default_class_name,
                frame_index=max(0, frame_idx),
                x=x,
                y=y,
                width=max(1.0, w),
                height=max(1.0, h),
                shape_type=ShapeType.BBOX,
                is_keyframe=True,
                source=AnnotationSource.MANUAL,
                confidence=conf
            )
            manager.add_or_update_annotation(ann, auto_create_track=True)
            count += 1

    return count