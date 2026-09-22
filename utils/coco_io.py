"""
utils/coco_io.py

MS COCO Instances JSON Exporter & Importer for DarkLabel Modern:
- Schema: {images, annotations, categories}
- Preserves bounding boxes, track IDs, scores, areas, and category mappings.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from core.annotation_manager import AnnotationManager
from core.annotation_models import Annotation, AnnotationSource, ShapeType


def export_coco_json(
    manager: AnnotationManager,
    img_width: int,
    img_height: int,
    out_path: str,
    class_to_id: Dict[str, int]
) -> int:
    """
    Exports project annotations into MS COCO JSON instance format.
    Returns total annotations exported.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    categories = [
        {"id": cid, "name": name, "supercategory": "object"}
        for name, cid in sorted(class_to_id.items(), key=lambda x: x[1])
    ]

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    ann_id_counter = 1

    with manager.lock:
        frames = manager.get_annotated_frame_indices()
        for frame_idx in frames:
            boxes = manager.get_annotations(frame_idx, visible_only=True, include_outside=False)
            if not boxes:
                continue

            images.append({
                "id": frame_idx,
                "file_name": f"frame_{frame_idx:06d}.jpg",
                "width": img_width,
                "height": img_height
            })

            for b in boxes:
                if b.shape_type != ShapeType.BBOX:
                    continue
                cat_id = class_to_id.get(b.class_name, b.class_id)
                annotations.append({
                    "id": ann_id_counter,
                    "image_id": frame_idx,
                    "category_id": cat_id,
                    "track_id": b.track_id,
                    "bbox": [round(b.x, 2), round(b.y, 2), round(b.width, 2), round(b.height, 2)],
                    "area": round(b.area, 2),
                    "iscrowd": 0,
                    "score": round(b.confidence, 4)
                })
                ann_id_counter += 1

    coco_payload = {
        "info": {"description": "DarkLabel Modern Export", "version": "1.0"},
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco_payload, f, indent=2, ensure_ascii=False)

    return len(annotations)


def import_coco_json(manager: AnnotationManager, json_path: str, clear_existing: bool = False) -> int:
    """
    Imports annotations from an MS COCO JSON file into AnnotationManager.
    Returns total annotations imported.
    """
    if not os.path.exists(json_path):
        return 0

    if clear_existing:
        manager.clear()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_map = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    count = 0

    with manager.lock:
        for ann in data.get("annotations", []):
            frame_idx = int(ann.get("image_id", 0))
            bbox = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
            cat_id = int(ann.get("category_id", 0))
            track_id = int(ann.get("track_id", ann.get("id", manager.track_manager.get_next_track_id())))
            score = float(ann.get("score", 1.0))
            class_name = category_map.get(cat_id, f"class_{cat_id}")

            ann_obj = Annotation(
                track_id=track_id,
                class_id=cat_id,
                class_name=class_name,
                frame_index=frame_idx,
                x=float(bbox[0]),
                y=float(bbox[1]),
                width=max(1.0, float(bbox[2])),
                height=max(1.0, float(bbox[3])),
                shape_type=ShapeType.BBOX,
                is_keyframe=True,
                source=AnnotationSource.MANUAL,
                confidence=score
            )
            manager.add_or_update_annotation(ann_obj, auto_create_track=True)
            count += 1

    return count