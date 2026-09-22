"""
core/cadence.py

5-FPS Cadence, Forward-Hold, and Track Termination Controller for DarkLabel Modern.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional

from core.annotation_manager import AnnotationManager
from core.annotation_models import Annotation, AnnotationSource
from core.history_manager import (
    AddAnnotationCommand,
    BatchAddAnnotationCommand,
    DeleteTailCommand,
    HistoryManager,
    UpdateAnnotationCommand,
)
from core.interpolation import InterpolationEngine


@dataclass
class CadenceStepResult:
    success: bool
    source_frame: int
    target_frame: int
    step_delta: int
    active_track_id: int
    interpolated_count: int
    message: str = ""


class CadenceEngine:
    """Controls cadence navigation, forward-hold box propagation, and track termination."""

    DEFAULT_CADENCE_FPS = 5.0

    @classmethod
    def calculate_step(cls, video_fps: float, target_cadence_fps: float = DEFAULT_CADENCE_FPS) -> int:
        fps = max(1.0, float(video_fps))
        cadence = max(0.1, float(target_cadence_fps))
        return max(1, round(fps / cadence))

    @classmethod
    def step_forward_cadence(
        cls,
        manager: AnnotationManager,
        history: Optional[HistoryManager],
        track_id: int,
        current_frame: int,
        total_frames: int,
        video_fps: float,
        cadence_fps: float = DEFAULT_CADENCE_FPS,
        auto_interpolate: bool = True,
        forward_hold: bool = True
    ) -> CadenceStepResult:
        step = cls.calculate_step(video_fps, cadence_fps)
        target_frame = min(current_frame + step, total_frames - 1)

        if target_frame == current_frame:
            return CadenceStepResult(
                success=False,
                source_frame=current_frame,
                target_frame=current_frame,
                step_delta=0,
                active_track_id=track_id,
                interpolated_count=0,
                message="Already at the final frame of the video."
            )

        with manager.lock:
            cur_box = manager.get_annotation(current_frame, track_id)
            if not cur_box:
                return CadenceStepResult(
                    success=True,
                    source_frame=current_frame,
                    target_frame=target_frame,
                    step_delta=step,
                    active_track_id=track_id,
                    interpolated_count=0,
                    message=f"Stepped forward {step} frames to frame {target_frame}."
                )

            if history:
                history.begin_transaction(f"5-FPS Cadence Step to Frame {target_frame}")

            interpolated_boxes: List[Annotation] = []

            try:
                if forward_hold:
                    carried_box = Annotation(
                        track_id=cur_box.track_id,
                        class_id=cur_box.class_id,
                        class_name=cur_box.class_name,
                        frame_index=target_frame,
                        x=cur_box.x,
                        y=cur_box.y,
                        width=cur_box.width,
                        height=cur_box.height,
                        shape_type=cur_box.shape_type,
                        points=copy.deepcopy(cur_box.points),
                        is_keyframe=True,
                        interpolated=False,
                        source=AnnotationSource.MANUAL,
                        confidence=cur_box.confidence,
                        occluded=cur_box.occluded,
                        outside=cur_box.outside,
                        attributes=copy.deepcopy(cur_box.attributes),
                    )

                    if history:
                        history.execute(AddAnnotationCommand(carried_box))
                    else:
                        manager.add_or_update_annotation(carried_box, auto_create_track=True)

                if auto_interpolate and (target_frame - current_frame) > 1:
                    dest_box = manager.get_annotation(target_frame, track_id)
                    if dest_box:
                        for f in range(current_frame + 1, target_frame):
                            interp_ann = InterpolationEngine.interpolate_single_frame(cur_box, dest_box, f)
                            interpolated_boxes.append(interp_ann)

                        if interpolated_boxes:
                            if history:
                                history.execute(BatchAddAnnotationCommand(interpolated_boxes))
                            else:
                                for ann in interpolated_boxes:
                                    manager.add_or_update_annotation(ann, auto_create_track=False)

                if history:
                    history.commit_transaction()

            except Exception as e:
                if history:
                    history.rollback_transaction()
                raise e

            return CadenceStepResult(
                success=True,
                source_frame=current_frame,
                target_frame=target_frame,
                step_delta=step,
                active_track_id=track_id,
                interpolated_count=len(interpolated_boxes),
                message=(
                    f"Advanced +{step} frames to frame {target_frame}. "
                    f"Held Track #{track_id} and generated {len(interpolated_boxes)} intermediate frames."
                )
            )

    @classmethod
    def step_backward_cadence(
        cls,
        current_frame: int,
        video_fps: float,
        cadence_fps: float = DEFAULT_CADENCE_FPS
    ) -> int:
        step = cls.calculate_step(video_fps, cadence_fps)
        return max(0, current_frame - step)

    @classmethod
    def terminate_track(
        cls,
        manager: AnnotationManager,
        history: Optional[HistoryManager],
        track_id: int,
        current_frame: int
    ) -> List[int]:
        with manager.lock:
            track = manager.track_manager.get_track(track_id)
            if not track:
                return []

            future_frames = [f for f in track.annotations.keys() if f > current_frame]
            if not future_frames:
                return []

            if history:
                cmd = DeleteTailCommand(track_id, current_frame)
                history.execute(cmd)
            else:
                manager.terminate_track(track_id, current_frame, force=True)

            return sorted(future_frames)

    @classmethod
    def toggle_occlusion(
        cls,
        manager: AnnotationManager,
        history: Optional[HistoryManager],
        track_id: int,
        current_frame: int
    ) -> Optional[bool]:
        with manager.lock:
            box = manager.get_annotation(current_frame, track_id)
            if not box:
                return None

            modified_box = box.copy()
            modified_box.occluded = not modified_box.occluded

            if history:
                history.execute(
                    UpdateAnnotationCommand(
                        modified_box,
                        f"Toggle Occlusion #{track_id} on frame {current_frame}"
                    )
                )
            else:
                manager.add_or_update_annotation(modified_box, auto_create_track=False)

            return modified_box.occluded

    @classmethod
    def toggle_outside(
        cls,
        manager: AnnotationManager,
        history: Optional[HistoryManager],
        track_id: int,
        current_frame: int
    ) -> Optional[bool]:
        with manager.lock:
            box = manager.get_annotation(current_frame, track_id)
            if not box:
                return None

            modified_box = box.copy()
            modified_box.outside = not modified_box.outside

            if history:
                history.execute(
                    UpdateAnnotationCommand(
                        modified_box,
                        f"Toggle Outside State #{track_id} on frame {current_frame}"
                    )
                )
            else:
                manager.add_or_update_annotation(modified_box, auto_create_track=False)

            return modified_box.outside