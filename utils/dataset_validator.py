"""
utils/dataset_validator.py

Automated Dataset Quality Assurance & Heuristic Auditor for DarkLabel Modern:
- Detects degenerate geometry: zero-width, zero-height, zero-area, or negative boxes.
- Detects spatial bounds violations: boxes extending outside image resolution.
- Detects temporal velocity anomalies: sudden impossible jumps / teleportation between adjacent frames.
- Detects broken interpolation sequences: orphaned interpolated frames missing valid keyframe anchors.
- Detects schema anomalies: invalid class IDs, unregistered categories, and frame index boundaries.
- Produces a structured ValidationReport with categorised severity (ERROR vs. WARNING vs. INFO).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Set, Tuple

from core.annotation_manager import AnnotationManager
from core.annotation_models import Annotation, AnnotationSource, ShapeType


class IssueSeverity(str, Enum):
    ERROR = "error"        # Fatal error that breaks YOLO/COCO training or coordinate loaders
    WARNING = "warning"    # Suspicious anomaly (e.g. extreme velocity jump, tiny box)
    INFO = "info"          # Non-destructive notification


@dataclass
class ValidationIssue:
    """Represents a specific validation anomaly found in the dataset."""
    severity: IssueSeverity
    code: str
    frame_index: int
    track_id: int
    class_name: str
    message: str
    suggested_fix: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "frame_index": str(self.frame_index),
            "track_id": str(self.track_id),
            "class_name": self.class_name,
            "message": self.message,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class ValidationReport:
    """Comprehensive summary produced by the DatasetValidator."""
    issues: List[ValidationIssue] = field(default_factory=list)
    total_annotations_audited: int = 0
    total_tracks_audited: int = 0
    total_frames_audited: int = 0

    @property
    def is_clean(self) -> bool:
        """Returns True if zero errors and zero warnings were detected."""
        return len(self.issues) == 0

    @property
    def has_errors(self) -> bool:
        return any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.INFO)

    def filter_by_severity(self, severity: IssueSeverity) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == severity]


class DatasetValidator:
    """
    Heuristic rule engine auditing annotations against computer vision standards.
    """

    def __init__(
        self,
        min_box_dimension_px: float = 2.0,
        max_box_dimension_ratio: float = 0.98,
        max_velocity_jump_multiplier: float = 4.0,
        min_absolute_velocity_jump_px: float = 200.0,
    ):
        self.min_dim = min_box_dimension_px
        self.max_ratio = max_box_dimension_ratio
        self.jump_mult = max_velocity_jump_multiplier
        self.min_jump_px = min_absolute_velocity_jump_px

    def validate(
        self,
        manager: AnnotationManager,
        img_width: int,
        img_height: int,
        total_frames: int,
        valid_class_ids: Optional[Set[int]] = None
    ) -> ValidationReport:
        """
        Executes full QA audit across all tracks and frames.
        Thread-safe under AnnotationManager lock.
        """
        report = ValidationReport()

        with manager.lock:
            all_tracks = manager.track_manager.get_all_tracks()
            report.total_tracks_audited = len(all_tracks)
            report.total_frames_audited = len(manager.get_annotated_frame_indices())

            for track in all_tracks:
                sorted_frames = sorted(track.annotations.keys())
                prev_ann: Optional[Annotation] = None

                for f in sorted_frames:
                    ann = track.annotations[f]
                    report.total_annotations_audited += 1

                    # 1. Frame Boundary Checks
                    if f < 0 or f >= total_frames:
                        report.issues.append(
                            ValidationIssue(
                                severity=IssueSeverity.ERROR,
                                code="ERR_FRAME_OUT_OF_BOUNDS",
                                frame_index=f,
                                track_id=track.track_id,
                                class_name=ann.class_name,
                                message=f"Frame index {f} is outside video duration [0, {total_frames - 1}].",
                                suggested_fix="Delete annotation or extend video bounds."
                            )
                        )

                    # 2. Category Schema Checks
                    if valid_class_ids is not None and ann.class_id not in valid_class_ids:
                        report.issues.append(
                            ValidationIssue(
                                severity=IssueSeverity.ERROR,
                                code="ERR_INVALID_CLASS_ID",
                                frame_index=f,
                                track_id=track.track_id,
                                class_name=ann.class_name,
                                message=f"Class ID {ann.class_id} ('{ann.class_name}') is not in the project class catalog.",
                                suggested_fix="Reassign to an existing class in Class Panel."
                            )
                        )

                    # Skip spatial checks for OUTSIDE (invisible/departed) frames
                    if ann.outside:
                        prev_ann = ann
                        continue

                    # 3. Degenerate Geometry Checks (Zero/Negative dimensions)
                    if ann.shape_type == ShapeType.BBOX:
                        if ann.width <= 0.0 or ann.height <= 0.0 or ann.area <= 0.0:
                            report.issues.append(
                                ValidationIssue(
                                    severity=IssueSeverity.ERROR,
                                    code="ERR_ZERO_AREA_BOX",
                                    frame_index=f,
                                    track_id=track.track_id,
                                    class_name=ann.class_name,
                                    message=f"Bounding box has zero or negative area ({ann.width:.1f}x{ann.height:.1f} px).",
                                    suggested_fix="Resize box on canvas or delete this frame instance."
                                )
                            )
                        elif ann.width < self.min_dim or ann.height < self.min_dim:
                            report.issues.append(
                                ValidationIssue(
                                    severity=IssueSeverity.WARNING,
                                    code="WARN_TINY_BOX",
                                    frame_index=f,
                                    track_id=track.track_id,
                                    class_name=ann.class_name,
                                    message=f"Tiny bounding box ({ann.width:.1f}x{ann.height:.1f} px).",
                                    suggested_fix="Verify if this target is an annotation artifact."
                                )
                            )

                        # Extreme aspect ratio check
                        ratio = max(ann.width / max(0.1, ann.height), ann.height / max(0.1, ann.width))
                        if ratio > 40.0:
                            report.issues.append(
                                ValidationIssue(
                                    severity=IssueSeverity.WARNING,
                                    code="WARN_EXTREME_ASPECT_RATIO",
                                    frame_index=f,
                                    track_id=track.track_id,
                                    class_name=ann.class_name,
                                    message=f"Unusual aspect ratio ({ratio:.1f}:1).",
                                    suggested_fix="Verify box boundary handles."
                                )
                            )

                        # 4. Out-of-Bounds Canvas Extent Checks
                        r = ann.x + ann.width
                        b = ann.y + ann.height
                        if ann.x < -0.5 or ann.y < -0.5 or r > img_width + 0.5 or b > img_height + 0.5:
                            report.issues.append(
                                ValidationIssue(
                                    severity=IssueSeverity.ERROR,
                                    code="ERR_CANVAS_OUT_OF_BOUNDS",
                                    frame_index=f,
                                    track_id=track.track_id,
                                    class_name=ann.class_name,
                                    message=(
                                        f"Box coordinates [x1:{ann.x:.1f}, y1:{ann.y:.1f}, x2:{r:.1f}, y2:{b:.1f}] "
                                        f"exceed image bounds [{img_width}x{img_height}]."
                                    ),
                                    suggested_fix="Clamp box coordinates to canvas dimensions."
                                )
                            )

                    # 5. Temporal Velocity & Teleportation Anomalies
                    if prev_ann is not None and not prev_ann.outside:
                        frame_gap = f - prev_ann.frame_index
                        if frame_gap == 1:  # Consecutive frames
                            dx = ann.center[0] - prev_ann.center[0]
                            dy = ann.center[1] - prev_ann.center[1]
                            disp_px = math.hypot(dx, dy)

                            prev_diag = math.hypot(prev_ann.width, prev_ann.height)
                            # Flag if box moves > 4x its own diagonal AND > 200px in a single frame
                            if disp_px > (prev_diag * self.jump_mult) and disp_px > self.min_jump_px:
                                report.issues.append(
                                    ValidationIssue(
                                        severity=IssueSeverity.WARNING,
                                        code="WARN_VELOCITY_JUMP",
                                        frame_index=f,
                                        track_id=track.track_id,
                                        class_name=ann.class_name,
                                        message=(
                                            f"Sudden trajectory jump of {disp_px:.1f} px in 1 frame "
                                            f"(from frame {prev_ann.frame_index})."
                                        ),
                                        suggested_fix="Check if track ID was accidentally swapped or split."
                                    )
                                )

                    # 6. Broken Interpolation Sequences
                    if ann.interpolated and not ann.is_keyframe:
                        prev_kf = track.find_prev_keyframe(f)
                        next_kf = track.find_next_keyframe(f)
                        if prev_kf is None or next_kf is None:
                            report.issues.append(
                                ValidationIssue(
                                    severity=IssueSeverity.WARNING,
                                    code="WARN_ORPHAN_INTERPOLATION",
                                    frame_index=f,
                                    track_id=track.track_id,
                                    class_name=ann.class_name,
                                    message="Interpolated frame has missing keyframe endpoints.",
                                    suggested_fix="Re-interpolate interval or convert this frame to keyframe."
                                )
                            )

                    prev_ann = ann

        return report