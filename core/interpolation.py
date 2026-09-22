"""
core/interpolation.py

Linear spatial interpolation engine calculating intermediate bounding boxes
between sparse keyframe endpoints.
"""

from __future__ import annotations

from typing import List, Optional

from core.annotation_manager import AnnotationManager
from core.annotation_models import Annotation, AnnotationSource, ShapeType


class InterpolationEngine:
    """Calculates linear trajectories for bounding box spans."""

    @classmethod
    def interpolate_single_frame(
        cls,
        box_a: Annotation,
        box_b: Annotation,
        target_frame: int
    ) -> Annotation:
        """Computes linearly interpolated bounding box at target_frame between box_a and box_b."""
        f_a = box_a.frame_index
        f_b = box_b.frame_index

        if f_b == f_a:
            t = 0.0
        else:
            t = (target_frame - f_a) / float(f_b - f_a)
            t = max(0.0, min(1.0, t))

        ix = box_a.x + t * (box_b.x - box_a.x)
        iy = box_a.y + t * (box_b.y - box_a.y)
        iw = box_a.width + t * (box_b.width - box_a.width)
        ih = box_a.height + t * (box_b.height - box_a.height)

        # Polygon interpolation if point sets match
        interp_points = []
        if box_a.points and box_b.points and len(box_a.points) == len(box_b.points):
            for (px_a, py_a), (px_b, py_b) in zip(box_a.points, box_b.points):
                interp_points.append((px_a + t * (px_b - px_a), py_a + t * (py_b - py_a)))

        return Annotation(
            track_id=box_a.track_id,
            class_id=box_a.class_id,
            class_name=box_a.class_name,
            frame_index=target_frame,
            x=ix,
            y=iy,
            width=iw,
            height=ih,
            shape_type=box_a.shape_type,
            points=interp_points,
            is_keyframe=False,
            interpolated=True,
            source=AnnotationSource.INTERPOLATED,
            confidence=round(box_a.confidence + t * (box_b.confidence - box_a.confidence), 4),
            occluded=box_a.occluded if t < 0.5 else box_b.occluded,
            outside=box_a.outside if t < 0.5 else box_b.outside,
            attributes=dict(box_a.attributes)
        )

    @classmethod
    def interpolate_range(
        cls,
        manager: AnnotationManager,
        track_id: int,
        start_frame: int,
        end_frame: int
    ) -> List[Annotation]:
        """Interpolates and inserts all frames strictly between start_frame and end_frame."""
        with manager.lock:
            box_a = manager.get_annotation(start_frame, track_id)
            box_b = manager.get_annotation(end_frame, track_id)

            if not box_a or not box_b or abs(end_frame - start_frame) <= 1:
                return []

            generated: List[Annotation] = []
            for f in range(start_frame + 1, end_frame):
                interp_box = cls.interpolate_single_frame(box_a, box_b, f)
                manager.add_or_update_annotation(interp_box, auto_create_track=False)
                generated.append(interp_box)

            return generated

    @classmethod
    def regenerate_surrounding_intervals(
        cls,
        manager: AnnotationManager,
        track_id: int,
        keyframe: int
    ) -> None:
        """Re-interpolates intervals (prev_kf -> keyframe) and (keyframe -> next_kf)."""
        with manager.lock:
            track = manager.track_manager.get_track(track_id)
            if not track:
                return

            prev_kf = track.find_prev_keyframe(keyframe)
            next_kf = track.find_next_keyframe(keyframe)

            if prev_kf is not None:
                cls.interpolate_range(manager, track_id, prev_kf, keyframe)
            if next_kf is not None:
                cls.interpolate_range(manager, track_id, keyframe, next_kf)