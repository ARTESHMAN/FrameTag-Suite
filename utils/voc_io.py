"""
utils/voc_io.py

Pascal VOC XML Exporter & Importer for DarkLabel Modern:
- Generates standard Pascal VOC XML annotation trees per frame/image.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom

from core.annotation_manager import AnnotationManager
from core.annotation_models import Annotation, AnnotationSource, ShapeType


def export_pascal_voc(
    manager: AnnotationManager,
    img_width: int,
    img_height: int,
    out_dir: str,
    depth: int = 3
) -> int:
    """
    Exports each annotated frame into a standalone Pascal VOC XML document.
    Returns total XML files generated.
    """
    os.makedirs(out_dir, exist_ok=True)
    exported_count = 0

    with manager.lock:
        frames = manager.get_annotated_frame_indices()
        for frame_idx in frames:
            boxes = manager.get_annotations(frame_idx, visible_only=True, include_outside=False)
            if not boxes:
                continue

            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = "images"
            ET.SubElement(root, "filename").text = f"frame_{frame_idx:06d}.jpg"

            size_elem = ET.SubElement(root, "size")
            ET.SubElement(size_elem, "width").text = str(img_width)
            ET.SubElement(size_elem, "height").text = str(img_height)
            ET.SubElement(size_elem, "depth").text = str(depth)

            for b in boxes:
                if b.shape_type != ShapeType.BBOX:
                    continue
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = b.class_name
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"
                ET.SubElement(obj, "occluded").text = "1" if b.occluded else "0"
                ET.SubElement(obj, "track_id").text = str(b.track_id)

                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(max(0, b.x)))
                ET.SubElement(bndbox, "ymin").text = str(int(max(0, b.y)))
                ET.SubElement(bndbox, "xmax").text = str(int(min(img_width, b.x + b.width)))
                ET.SubElement(bndbox, "ymax").text = str(int(min(img_height, b.y + b.height)))

            xml_bytes = ET.tostring(root, encoding="utf-8")
            parsed_dom = minidom.parseString(xml_bytes)
            pretty_xml = parsed_dom.toprettyxml(indent="  ")

            out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.xml")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(pretty_xml)

            exported_count += 1

    return exported_count