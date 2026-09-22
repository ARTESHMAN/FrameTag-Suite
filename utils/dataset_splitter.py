"""
utils/dataset_splitter.py

Dataset Sampling and Train/Validation/Test Partitioning Engine:
- Extracts annotated frames from video provider into structured directories:
    dataset/
      images/train/
      images/val/
      labels/train/
      labels/val/
      data.yaml
- Supports configurable split ratios (e.g. 70% train, 20% validation, 10% test).
- Automatically writes Ultralytics-compatible data.yaml.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import cv2

from core.annotation_manager import AnnotationManager
from utils.yolo_io import export_yolo_txt


def export_partitioned_yolo_dataset(
    video_provider,
    manager: AnnotationManager,
    output_dir: str,
    class_to_id: Dict[str, int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    shuffle: bool = True
) -> Dict[str, int]:
    """
    Exports sampled annotated frames partitioned into train/val/test sets
    along with data.yaml for YOLO training.
    """
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    with manager.lock:
        annotated_frames = sorted([f for f in manager.get_annotated_frame_indices() if manager.get_annotations(f)])

    if not annotated_frames:
        return {"train": 0, "val": 0, "test": 0}

    if shuffle:
        random.seed(42)
        random.shuffle(annotated_frames)

    total = len(annotated_frames)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": annotated_frames[:train_end],
        "val": annotated_frames[train_end:val_end],
        "test": annotated_frames[val_end:]
    }

    counts = {"train": 0, "val": 0, "test": 0}
    w_img = video_provider.width
    h_img = video_provider.height

    for split_name, frames in splits.items():
        if not frames:
            continue
        img_split_dir = os.path.join(output_dir, "images", split_name)
        lbl_split_dir = os.path.join(output_dir, "labels", split_name)

        for f_idx in frames:
            frame_bgr = video_provider.get_frame_at(f_idx)
            if frame_bgr is None:
                continue

            base_name = f"frame_{f_idx:06d}"
            # 1. Save Image
            cv2.imwrite(os.path.join(img_split_dir, f"{base_name}.jpg"), frame_bgr)

            # 2. Save Normalized YOLO TXT
            boxes = manager.get_annotations(f_idx, visible_only=True, include_outside=False)
            txt_path = os.path.join(lbl_split_dir, f"{base_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                for b in boxes:
                    if b.shape_type.value != "bbox":
                        continue
                    cid = class_to_id.get(b.class_name, b.class_id)
                    xc, yc, w, h = b.to_yolo(w_img, h_img)
                    f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            counts[split_name] += 1

    # 3. Write data.yaml for Ultralytics YOLO
    sorted_classes = [name for name, _ in sorted(class_to_id.items(), key=lambda x: x[1])]
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        if counts["test"] > 0:
            f.write("test: images/test\n")
        f.write(f"\nnc: {len(sorted_classes)}\n")
        f.write(f"names: {sorted_classes}\n")

    return counts