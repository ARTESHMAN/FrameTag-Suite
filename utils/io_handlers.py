import csv
import json
import os
import re
from typing import Callable, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2
import numpy as np

from core.annotation_manager import AnnotationManager, BoundingBox


# ==============================================================================
# 1. YOLO Annotation Exporter & Importer
# ==============================================================================

def export_yolo_txt(
    manager: AnnotationManager,
    width: int,
    height: int,
    out_dir: str,
    class_to_id: Dict[str, int],
    progress_callback: Optional[Callable[[int, int], None]] = None
):
    """
    Exports bounding boxes to normalized YOLO format text files:
    <class_id> <x_center> <y_center> <width> <height>
    One .txt file per annotated frame: frame_000123.txt
    """
    os.makedirs(out_dir, exist_ok=True)
    frames = sorted(manager.annotations.keys())
    total = len(frames)

    for idx, frame_idx in enumerate(frames):
        boxes = manager.get_boxes(frame_idx)
        if not boxes:
            continue

        file_name = f"frame_{frame_idx:06d}.txt"
        file_path = os.path.join(out_dir, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            for b in boxes:
                cls_id = class_to_id.get(b.label, 0)
                xc, yc, w, h = b.to_yolo(width, height)
                # Clamp coordinates to [0.0, 1.0]
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        if progress_callback:
            progress_callback(idx + 1, total)


def import_yolo_txt(
    manager: AnnotationManager,
    labels_dir: str,
    width: int,
    height: int,
    id_to_class: Dict[int, str],
    clear_existing: bool = False
) -> int:
    """
    Imports YOLO normalized text files into the AnnotationManager.
    Matches frame numbers from file names (e.g. frame_000042.txt -> frame 42).
    """
    if clear_existing:
        manager.clear()

    txt_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    imported_boxes = 0

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
                xc = float(parts[1]) * width
                yc = float(parts[2]) * height
                w = float(parts[3]) * width
                h = float(parts[4]) * height

                x = xc - (w / 2.0)
                y = yc - (h / 2.0)
                label = id_to_class.get(cls_id, str(cls_id))
                track_id = manager.get_next_track_id()

                box = BoundingBox(
                    track_id=track_id,
                    label=label,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    conf=1.0,
                    is_keyframe=True
                )
                manager.add_or_update_box(frame_idx, box, record_history=False)
                imported_boxes += 1

    return imported_boxes


# ==============================================================================
# 2. MOT Challenge Format Exporter & Importer
# ==============================================================================

def export_mot_challenge(manager: AnnotationManager, out_path: str):
    """
    Exports annotations in MOT benchmark format (1-based frame indexing):
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for frame_idx in sorted(manager.annotations.keys()):
            for b in manager.annotations[frame_idx]:
                writer.writerow([
                    frame_idx + 1,  # MOT Challenge is 1-indexed
                    b.track_id,
                    f"{b.x:.2f}",
                    f"{b.y:.2f}",
                    f"{b.w:.2f}",
                    f"{b.h:.2f}",
                    f"{b.conf:.2f}",
                    -1,
                    -1,
                    -1
                ])


def import_mot_challenge(
    manager: AnnotationManager,
    file_path: str,
    default_label: str = "target",
    clear_existing: bool = False
) -> int:
    """Imports a MOT challenge gt.txt file into the AnnotationManager."""
    if clear_existing:
        manager.clear()

    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue

            # Support comma-separated or space-separated rows
            if len(row) == 1:
                row = row[0].split()

            frame_idx = int(float(row[0])) - 1  # Convert 1-indexed to 0-indexed
            track_id = int(float(row[1]))
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            conf = float(row[6]) if len(row) > 6 and float(row[6]) > 0 else 1.0

            box = BoundingBox(
                track_id=track_id,
                label=default_label,
                x=x,
                y=y,
                w=w,
                h=h,
                conf=conf,
                is_keyframe=True
            )
            manager.add_or_update_box(frame_idx, box, record_history=False)
            count += 1

    return count


# ==============================================================================
# 3. COCO JSON Format Exporter & Importer
# ==============================================================================

def export_coco_json(
    manager: AnnotationManager,
    width: int,
    height: int,
    out_path: str,
    class_to_id: Dict[str, int]
):
    """Exports annotations into standard MS COCO instances JSON format."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    categories = [
        {"id": cid, "name": name, "supercategory": "none"}
        for name, cid in class_to_id.items()
    ]

    images = []
    annotations = []
    ann_id = 1

    for frame_idx in sorted(manager.annotations.keys()):
        boxes = manager.get_boxes(frame_idx)
        if not boxes:
            continue

        images.append({
            "id": frame_idx,
            "file_name": f"frame_{frame_idx:06d}.jpg",
            "width": width,
            "height": height
        })

        for b in boxes:
            cat_id = class_to_id.get(b.label, 1)
            annotations.append({
                "id": ann_id,
                "image_id": frame_idx,
                "category_id": cat_id,
                "track_id": b.track_id,
                "bbox": [round(b.x, 2), round(b.y, 2), round(b.w, 2), round(b.h, 2)],
                "area": round(b.w * b.h, 2),
                "iscrowd": 0,
                "score": round(b.conf, 4)
            })
            ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=2)


def import_coco_json(manager: AnnotationManager, json_path: str, clear_existing: bool = False) -> int:
    """Loads annotations from a COCO JSON instances file."""
    if clear_existing:
        manager.clear()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_map = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    count = 0

    for ann in data.get("annotations", []):
        frame_idx = ann.get("image_id", 0)
        bbox = ann.get("bbox", [0, 0, 0, 0])
        cls_id = ann.get("category_id", 0)
        track_id = ann.get("track_id", ann.get("id", manager.get_next_track_id()))
        conf = ann.get("score", 1.0)
        label = category_map.get(cls_id, str(cls_id))

        box = BoundingBox(
            track_id=track_id,
            label=label,
            x=bbox[0],
            y=bbox[1],
            w=bbox[2],
            h=bbox[3],
            conf=conf,
            is_keyframe=True
        )
        manager.add_or_update_box(frame_idx, box, record_history=False)
        count += 1

    return count


# ==============================================================================
# 4. Pascal VOC XML Exporter
# ==============================================================================

def export_pascal_voc(
    manager: AnnotationManager,
    width: int,
    height: int,
    out_dir: str,
    depth: int = 3
):
    """Exports each labeled frame as a Pascal VOC XML document."""
    os.makedirs(out_dir, exist_ok=True)

    for frame_idx, boxes in manager.annotations.items():
        if not boxes:
            continue

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "images"
        ET.SubElement(root, "filename").text = f"frame_{frame_idx:06d}.jpg"

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(depth)

        for b in boxes:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = b.label
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            ET.SubElement(obj, "occluded").text = "1" if b.occluded else "0"
            ET.SubElement(obj, "track_id").text = str(b.track_id)

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(max(0, b.x)))
            ET.SubElement(bndbox, "ymin").text = str(int(max(0, b.y)))
            ET.SubElement(bndbox, "xmax").text = str(int(min(width, b.x + b.w)))
            ET.SubElement(bndbox, "ymax").text = str(int(min(height, b.y + b.h)))

        xml_str = ET.tostring(root, encoding="utf-8")
        parsed = minidom.parseString(xml_str)
        pretty_xml = parsed.toprettyxml(indent="  ")

        out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.xml")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)


# ==============================================================================
# 5. Flat CSV Exporter
# ==============================================================================

def export_csv(manager: AnnotationManager, out_path: str):
    """Exports all annotation metadata into a tabular CSV spreadsheet."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index", "track_id", "label", "confidence",
            "x", "y", "width", "height", "is_keyframe", "occluded"
        ])

        for frame_idx in sorted(manager.annotations.keys()):
            for b in manager.annotations[frame_idx]:
                writer.writerow([
                    frame_idx,
                    b.track_id,
                    b.label,
                    f"{b.conf:.4f}",
                    f"{b.x:.2f}",
                    f"{b.y:.2f}",
                    f"{b.w:.2f}",
                    f"{b.h:.2f}",
                    1 if b.is_keyframe else 0,
                    1 if b.occluded else 0
                ])


# ==============================================================================
# 6. Sampled Dataset Exporter (Images + YOLO TXT)
# ==============================================================================

def export_sampled_dataset(
    video_provider,
    manager: AnnotationManager,
    output_dir: str,
    class_to_id: Optional[Dict[str, int]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
):
    """
    Pulls exact RGB frames from the video provider that contain at least one bounding box.
    Saves matched pairs into:
      output_dir/images/frame_000123.jpg
      output_dir/labels/frame_000123.txt
    """
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    c_map = class_to_id if class_to_id is not None else {}
    w_img = video_provider.width
    h_img = video_provider.height

    labeled_frames = sorted(
        [f for f, boxes in manager.annotations.items() if len(boxes) > 0]
    )
    total = len(labeled_frames)

    for idx, f_idx in enumerate(labeled_frames):
        frame = video_provider.get_frame_at(f_idx)
        if frame is None:
            continue

        base_name = f"frame_{f_idx:06d}"
        cv2.imwrite(os.path.join(img_dir, f"{base_name}.jpg"), frame)

        # Write matching YOLO txt
        boxes = manager.get_boxes(f_idx)
        txt_path = os.path.join(lbl_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for b in boxes:
                cls_id = c_map.get(b.label, 0)
                xc, yc, bw, bh = b.to_yolo(w_img, h_img)
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        if progress_callback:
            progress_callback(idx + 1, total)


# ==============================================================================
# 7. Privacy Blur / Mosaic Exporter
# ==============================================================================

def export_privacy_mosaic(
    video_provider,
    manager: AnnotationManager,
    out_path: str,
    blur_kernel_divisor: int = 4,
    track_id_filter: Optional[List[int]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
):
    """
    Generates a video stream where all annotated targets (or filtered Track IDs)
    are anonymized using adaptive Gaussian blur.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    w, h = video_provider.width, video_provider.height
    fps = video_provider.fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    total = video_provider.total_frames

    for f_idx in range(total):
        frame = video_provider.get_frame_at(f_idx)
        if frame is None:
            break

        boxes = manager.get_boxes(f_idx)
        for b in boxes:
            if track_id_filter is not None and b.track_id not in track_id_filter:
                continue

            x1 = max(0, min(int(b.x), w - 1))
            y1 = max(0, min(int(b.y), h - 1))
            x2 = max(0, min(int(b.x + b.w), w))
            y2 = max(0, min(int(b.y + b.h), h))

            if (x2 - x1) > 2 and (y2 - y1) > 2:
                roi = frame[y1:y2, x1:x2]
                kw = max(3, (roi.shape[1] // blur_kernel_divisor) | 1)
                kh = max(3, (roi.shape[0] // blur_kernel_divisor) | 1)
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kw, kh), 0)

        writer.write(frame)

        if progress_callback:
            progress_callback(f_idx + 1, total)

    writer.release()


# ==============================================================================
# 8. Video Sub-Clip Exporter
# ==============================================================================

def export_video_clip(
    video_provider,
    start_frame: int,
    end_frame: int,
    out_path: str,
    progress_callback: Optional[Callable[[int, int], None]] = None
):
    """Extracts a precise temporal clip between start_frame and end_frame."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    w, h = video_provider.width, video_provider.height
    fps = video_provider.fps

    start_frame = max(0, min(start_frame, video_provider.total_frames - 1))
    end_frame = max(start_frame, min(end_frame, video_provider.total_frames - 1))
    total_steps = end_frame - start_frame + 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for step, f_idx in enumerate(range(start_frame, end_frame + 1)):
        frame = video_provider.get_frame_at(f_idx)
        if frame is None:
            break
        writer.write(frame)

        if progress_callback:
            progress_callback(step + 1, total_steps)

    writer.release()