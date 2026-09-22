"""
core/annotation_models.py

Fundamental domain models representing bounding box annotations, identity tracks,
and shape telemetry for DarkLabel Modern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple


class ShapeType(str, Enum):
    BBOX = "bbox"
    POINT = "point"
    POLYGON = "polygon"


class AnnotationSource(str, Enum):
    MANUAL = "manual"
    AI = "ai"
    TRACKER = "tracker"
    INTERPOLATED = "interpolated"


@dataclass
class Annotation:
    """Represents a single annotation instance on a specific video frame."""
    track_id: int
    class_id: int
    class_name: str
    frame_index: int
    x: float
    y: float
    width: float
    height: float
    shape_type: ShapeType = ShapeType.BBOX
    points: List[Tuple[float, float]] = field(default_factory=list)
    is_keyframe: bool = True
    interpolated: bool = False
    source: AnnotationSource = AnnotationSource.MANUAL
    confidence: float = 1.0
    occluded: bool = False
    outside: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def area(self) -> float:
        return max(0.0, float(self.width)) * max(0.0, float(self.height))

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def to_ltrb(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)

    def to_yolo(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        w_img = max(1.0, float(img_w))
        h_img = max(1.0, float(img_h))
        cx, cy = self.center
        return (
            max(0.0, min(1.0, cx / w_img)),
            max(0.0, min(1.0, cy / h_img)),
            max(0.0, min(1.0, self.width / w_img)),
            max(0.0, min(1.0, self.height / h_img)),
        )

    @classmethod
    def from_yolo(
        cls,
        track_id: int,
        class_id: int,
        class_name: str,
        frame_index: int,
        norm_cx: float,
        norm_cy: float,
        norm_w: float,
        norm_h: float,
        img_w: int,
        img_h: int,
        is_keyframe: bool = True,
        source: AnnotationSource = AnnotationSource.MANUAL,
        confidence: float = 1.0,
    ) -> Annotation:
        w = norm_w * img_w
        h = norm_h * img_h
        cx = norm_cx * img_w
        cy = norm_cy * img_h
        return cls(
            track_id=track_id,
            class_id=class_id,
            class_name=class_name,
            frame_index=frame_index,
            x=cx - (w / 2.0),
            y=cy - (h / 2.0),
            width=w,
            height=h,
            is_keyframe=is_keyframe,
            source=source,
            confidence=confidence,
        )

    def copy(self) -> Annotation:
        return Annotation(
            track_id=self.track_id,
            class_id=self.class_id,
            class_name=self.class_name,
            frame_index=self.frame_index,
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            shape_type=self.shape_type,
            points=[(float(px), float(py)) for px, py in self.points],
            is_keyframe=self.is_keyframe,
            interpolated=self.interpolated,
            source=self.source,
            confidence=self.confidence,
            occluded=self.occluded,
            outside=self.outside,
            attributes=dict(self.attributes),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "frame_index": self.frame_index,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "shape_type": self.shape_type.value if hasattr(self.shape_type, "value") else str(self.shape_type),
            "points": self.points,
            "is_keyframe": self.is_keyframe,
            "interpolated": self.interpolated,
            "source": self.source.value if hasattr(self.source, "value") else str(self.source),
            "confidence": self.confidence,
            "occluded": self.occluded,
            "outside": self.outside,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Annotation:
        return cls(
            track_id=int(data["track_id"]),
            class_id=int(data["class_id"]),
            class_name=str(data["class_name"]),
            frame_index=int(data["frame_index"]),
            x=float(data["x"]),
            y=float(data["y"]),
            width=float(data["width"]),
            height=float(data["height"]),
            shape_type=ShapeType(data.get("shape_type", "bbox")),
            points=[(float(p[0]), float(p[1])) for p in data.get("points", [])],
            is_keyframe=bool(data.get("is_keyframe", True)),
            interpolated=bool(data.get("interpolated", False)),
            source=AnnotationSource(data.get("source", "manual")),
            confidence=float(data.get("confidence", 1.0)),
            occluded=bool(data.get("occluded", False)),
            outside=bool(data.get("outside", False)),
            attributes=dict(data.get("attributes", {})),
        )


@dataclass
class Track:
    """Manages full temporal trajectory and state lifecycle for an individual track identity."""
    track_id: int
    class_id: int
    class_name: str
    annotations: Dict[int, Annotation] = field(default_factory=dict)
    locked: bool = False
    visible: bool = True
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not hasattr(self, "locked"):
            self.locked = False
        if not hasattr(self, "visible"):
            self.visible = True

    @property
    def total_frames(self) -> int:
        return len(self.annotations)

    @property
    def start_frame(self) -> Optional[int]:
        return min(self.annotations.keys()) if self.annotations else None

    @property
    def end_frame(self) -> Optional[int]:
        return max(self.annotations.keys()) if self.annotations else None

    @property
    def keyframes(self) -> List[int]:
        return sorted([f for f, ann in self.annotations.items() if ann.is_keyframe and not ann.outside])

    def get_annotation(self, frame_index: int) -> Optional[Annotation]:
        return self.annotations.get(frame_index)

    def set_annotation(self, annotation: Annotation) -> None:
        annotation.track_id = self.track_id
        self.annotations[annotation.frame_index] = annotation

    def remove_annotation(self, frame_index: int) -> Optional[Annotation]:
        return self.annotations.pop(frame_index, None)

    def find_prev_keyframe(self, frame_index: int) -> Optional[int]:
        candidates = [f for f in self.keyframes if f < frame_index]
        return max(candidates) if candidates else None

    def find_next_keyframe(self, frame_index: int) -> Optional[int]:
        candidates = [f for f in self.keyframes if f > frame_index]
        return min(candidates) if candidates else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "locked": getattr(self, "locked", False),
            "visible": getattr(self, "visible", True),
            "attributes": getattr(self, "attributes", {}),
            "annotations": {str(f): ann.to_dict() for f, ann in self.annotations.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Track:
        t = cls(
            track_id=int(data["track_id"]),
            class_id=int(data["class_id"]),
            class_name=str(data["class_name"]),
            locked=bool(data.get("locked", False)),
            visible=bool(data.get("visible", True)),
            attributes=dict(data.get("attributes", {})),
        )
        for f_str, ann_dict in data.get("annotations", {}).items():
            t.annotations[int(f_str)] = Annotation.from_dict(ann_dict)
        return t