"""
core/annotation_manager.py

Thread-safe in-memory spatial-temporal database for DarkLabel Modern.
Dual indexed:
  1. Temporal Spatial Index: frame_index -> {track_id: Annotation}
  2. Identity Track Index: track_id -> Track
"""

from __future__ import annotations

import copy
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from core.annotation_models import Annotation, AnnotationSource, ShapeType, Track
from core.track_manager import (
    MergeResult,
    MergeStrategy,
    SplitResult,
    TrackLockedError,
    TrackManager,
    TrackNotFoundError,
    TrackOperationError,
)

# Backward-compatibility alias
BoundingBox = Annotation


class AnnotationManager:
    """Central query and update engine for all video annotations."""

    def __init__(self, track_manager: Optional[TrackManager] = None):
        self._lock = threading.RLock()
        self.track_manager: TrackManager = track_manager if track_manager is not None else TrackManager()
        self._frame_index: Dict[int, Dict[int, Annotation]] = {}

        if self.track_manager.tracks:
            self.rebuild_frame_index()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def add_or_update_annotation(
        self,
        annotation: Annotation,
        auto_create_track: bool = True
    ) -> Annotation:
        with self._lock:
            track = self.track_manager.get_track(annotation.track_id)
            if track is None:
                if auto_create_track:
                    track = self.track_manager.create_track(
                        class_id=annotation.class_id,
                        class_name=annotation.class_name,
                        track_id=annotation.track_id
                    )
                else:
                    raise TrackNotFoundError(
                        f"Cannot add annotation: Track #{annotation.track_id} does not exist."
                    )

            if track.locked:
                raise TrackLockedError(f"Cannot modify Track #{annotation.track_id}: track is locked.")

            annotation.track_id = track.track_id
            track.set_annotation(annotation)

            frame_dict = self._frame_index.setdefault(annotation.frame_index, {})
            frame_dict[annotation.track_id] = annotation
            return annotation

    def remove_annotation(self, frame_idx: int, track_id: int, force: bool = False) -> Optional[Annotation]:
        with self._lock:
            track = self.track_manager.get_track(track_id)
            if track and track.locked and not force:
                raise TrackLockedError(f"Cannot remove annotation: Track #{track_id} is locked.")

            removed_track = track.remove_annotation(frame_idx) if track else None
            removed_index: Optional[Annotation] = None
            if frame_idx in self._frame_index:
                removed_index = self._frame_index[frame_idx].pop(track_id, None)
                if not self._frame_index[frame_idx]:
                    del self._frame_index[frame_idx]

            return removed_track or removed_index

    def remove_all_annotations_at_frame(self, frame_idx: int, force: bool = False) -> List[Annotation]:
        with self._lock:
            if frame_idx not in self._frame_index:
                return []

            removed: List[Annotation] = []
            for tid in list(self._frame_index[frame_idx].keys()):
                track = self.track_manager.get_track(tid)
                if track and track.locked and not force:
                    continue
                ann = self.remove_annotation(frame_idx, tid, force=force)
                if ann:
                    removed.append(ann)
            return removed

    def get_annotations(
        self,
        frame_idx: int,
        visible_only: bool = True,
        include_outside: bool = False
    ) -> List[Annotation]:
        with self._lock:
            if frame_idx not in self._frame_index:
                return []

            results: List[Annotation] = []
            for tid, ann in self._frame_index[frame_idx].items():
                track = self.track_manager.get_track(tid)
                if visible_only and track and not track.visible:
                    continue
                if not include_outside and ann.outside:
                    continue
                results.append(ann)
            return results

    def get_annotation(self, frame_idx: int, track_id: int) -> Optional[Annotation]:
        with self._lock:
            return self._frame_index.get(frame_idx, {}).get(track_id)

    def get_annotation_at_pixel(
        self,
        frame_idx: int,
        pixel_x: float,
        pixel_y: float,
        visible_only: bool = True
    ) -> Optional[Annotation]:
        with self._lock:
            candidates = self.get_annotations(frame_idx, visible_only=visible_only, include_outside=False)
            hit_boxes: List[Annotation] = []
            for ann in candidates:
                if ann.shape_type == ShapeType.BBOX:
                    l, t, r, b = ann.to_ltrb()
                    if l <= pixel_x <= r and t <= pixel_y <= b:
                        hit_boxes.append(ann)
            if not hit_boxes:
                return None
            hit_boxes.sort(key=lambda a: a.area)
            return hit_boxes[0]

    def has_annotations(self, frame_idx: int) -> bool:
        with self._lock:
            return frame_idx in self._frame_index and len(self._frame_index[frame_idx]) > 0

    def get_annotated_frame_indices(self) -> List[int]:
        with self._lock:
            return sorted(self._frame_index.keys())

    def get_all_keyframes(self) -> Set[int]:
        with self._lock:
            kfs: Set[int] = set()
            for frame_idx, frame_dict in self._frame_index.items():
                for ann in frame_dict.values():
                    if ann.is_keyframe and not ann.outside:
                        kfs.add(frame_idx)
                        break
            return kfs

    def terminate_track(self, track_id: int, at_frame: int, force: bool = False) -> List[int]:
        with self._lock:
            purged = self.track_manager.terminate_track(track_id, at_frame, force=force)
            for f in purged:
                if f in self._frame_index:
                    self._frame_index[f].pop(track_id, None)
                    if not self._frame_index[f]:
                        del self._frame_index[f]
            return purged

    def split_track(
        self,
        track_id: int,
        split_frame: int,
        new_track_id: Optional[int] = None,
        force: bool = False
    ) -> SplitResult:
        with self._lock:
            res = self.track_manager.split_track(track_id, split_frame, new_track_id, force=force)
            new_track = self.track_manager.get_track_or_raise(res.new_track_id)
            for f, ann in new_track.annotations.items():
                if f in self._frame_index:
                    self._frame_index[f].pop(track_id, None)
                    self._frame_index[f][res.new_track_id] = ann
            return res

    def merge_tracks(
        self,
        source_track_id: int,
        target_track_id: int,
        strategy: MergeStrategy = MergeStrategy.FAIL_ON_CONFLICT,
        force: bool = False
    ) -> MergeResult:
        with self._lock:
            res = self.track_manager.merge_tracks(source_track_id, target_track_id, strategy=strategy, force=force)
            self.rebuild_frame_index()
            return res

    def delete_track(self, track_id: int, force: bool = False) -> None:
        with self._lock:
            track = self.track_manager.remove_track(track_id, force=force)
            for f in track.annotations.keys():
                if f in self._frame_index:
                    self._frame_index[f].pop(track_id, None)
                    if not self._frame_index[f]:
                        del self._frame_index[f]

    def delete_interval(
        self,
        start_frame: int,
        end_frame: int,
        track_id: Optional[int] = None,
        force: bool = False
    ) -> int:
        with self._lock:
            count = 0
            if track_id is not None:
                purged = self.track_manager.delete_track_interval(track_id, start_frame, end_frame, force=force)
                for f in purged:
                    if f in self._frame_index:
                        self._frame_index[f].pop(track_id, None)
                        if not self._frame_index[f]:
                            del self._frame_index[f]
                count = len(purged)
            else:
                for f in range(start_frame, end_frame + 1):
                    if f in self._frame_index:
                        count += len(self._frame_index[f])
                        for tid in list(self._frame_index[f].keys()):
                            track = self.track_manager.get_track(tid)
                            if track and (not track.locked or force):
                                track.remove_annotation(f)
                                del self._frame_index[f][tid]
                        if not self._frame_index[f]:
                            del self._frame_index[f]
            return count

    def rebuild_frame_index(self) -> None:
        with self._lock:
            self._frame_index.clear()
            for track in self.track_manager.tracks.values():
                for f, ann in track.annotations.items():
                    frame_dict = self._frame_index.setdefault(f, {})
                    frame_dict[track.track_id] = ann

    def clear(self) -> None:
        with self._lock:
            self._frame_index.clear()
            self.track_manager.clear()

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "version": "1.0",
                "track_manager": self.track_manager.to_dict(),
            }

    def load_dict(self, data: Dict[str, Any], clear_existing: bool = True) -> None:
        with self._lock:
            if clear_existing:
                self.clear()
            self.track_manager.load_dict(data.get("track_manager", {}), clear_existing=True)
            self.rebuild_frame_index()