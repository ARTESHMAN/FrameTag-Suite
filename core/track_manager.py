"""
core/track_manager.py

Lifecycle, entity repository, and trajectory manipulation for object tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from core.annotation_models import Annotation, Track


class TrackOperationError(Exception):
    pass


class TrackNotFoundError(TrackOperationError):
    pass


class TrackLockedError(TrackOperationError):
    pass


class MergeStrategy(str, Enum):
    FAIL_ON_CONFLICT = "fail"
    PREFER_TARGET = "target"
    PREFER_SOURCE = "source"


@dataclass
class SplitResult:
    original_track_id: int
    new_track_id: int
    split_frame: int
    moved_annotations_count: int
    message: str = ""


@dataclass
class MergeResult:
    source_track_id: int
    target_track_id: int
    merged_annotations_count: int
    conflict_count: int
    message: str = ""


class TrackManager:
    """Maintains tracked entities, ID auto-incrementing, and trajectory merges/splits."""

    def __init__(self):
        self.tracks: Dict[int, Track] = {}
        self._next_id: int = 1

    def get_next_track_id(self) -> int:
        while self._next_id in self.tracks:
            self._next_id += 1
        return self._next_id

    def create_track(
        self,
        class_id: int,
        class_name: str,
        track_id: Optional[int] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Track:
        tid = track_id if track_id is not None else self.get_next_track_id()
        if tid in self.tracks:
            raise TrackOperationError(f"Track #{tid} already exists.")

        track = Track(
            track_id=tid,
            class_id=class_id,
            class_name=class_name,
            attributes=attributes or {}
        )
        self.tracks[tid] = track
        if tid >= self._next_id:
            self._next_id = tid + 1
        return track

    def get_track(self, track_id: int) -> Optional[Track]:
        return self.tracks.get(track_id)

    def get_track_or_raise(self, track_id: int) -> Track:
        track = self.tracks.get(track_id)
        if not track:
            raise TrackNotFoundError(f"Track #{track_id} not found.")
        return track

    def get_all_tracks(self) -> List[Track]:
        return sorted(self.tracks.values(), key=lambda t: t.track_id)

    def remove_track(self, track_id: int, force: bool = False) -> Track:
        track = self.get_track_or_raise(track_id)
        if track.locked and not force:
            raise TrackLockedError(f"Cannot delete Track #{track_id}: track is locked.")
        del self.tracks[track_id]
        return track

    def terminate_track(self, track_id: int, at_frame: int, force: bool = False) -> List[int]:
        track = self.get_track_or_raise(track_id)
        if track.locked and not force:
            raise TrackLockedError(f"Cannot terminate Track #{track_id}: track is locked.")

        purged_frames = [f for f in track.annotations.keys() if f > at_frame]
        for f in purged_frames:
            del track.annotations[f]
        return sorted(purged_frames)

    def delete_track_interval(self, track_id: int, start_frame: int, end_frame: int, force: bool = False) -> List[int]:
        track = self.get_track_or_raise(track_id)
        if track.locked and not force:
            raise TrackLockedError(f"Cannot modify Track #{track_id}: track is locked.")

        purged = [f for f in track.annotations.keys() if start_frame <= f <= end_frame]
        for f in purged:
            del track.annotations[f]
        return sorted(purged)

    def split_track(
        self,
        track_id: int,
        split_frame: int,
        new_track_id: Optional[int] = None,
        force: bool = False
    ) -> SplitResult:
        old_track = self.get_track_or_raise(track_id)
        if old_track.locked and not force:
            raise TrackLockedError(f"Cannot split Track #{track_id}: track is locked.")

        moving_frames = [f for f in old_track.annotations.keys() if f >= split_frame]
        if not moving_frames:
            raise TrackOperationError(f"No annotations exist on Track #{track_id} at or after frame {split_frame}.")

        nid = new_track_id if new_track_id is not None else self.get_next_track_id()
        new_track = self.create_track(
            class_id=old_track.class_id,
            class_name=old_track.class_name,
            track_id=nid,
            attributes=dict(old_track.attributes)
        )

        for f in sorted(moving_frames):
            ann = old_track.annotations.pop(f)
            ann.track_id = nid
            new_track.annotations[f] = ann

        return SplitResult(
            original_track_id=track_id,
            new_track_id=nid,
            split_frame=split_frame,
            moved_annotations_count=len(moving_frames),
            message=f"Split Track #{track_id} into #{nid} at frame {split_frame} ({len(moving_frames)} frames moved)."
        )

    def detect_temporal_overlap(self, track_id_a: int, track_id_b: int) -> List[int]:
        ta = self.get_track_or_raise(track_id_a)
        tb = self.get_track_or_raise(track_id_b)
        return sorted(set(ta.annotations.keys()) & set(tb.annotations.keys()))

    def merge_tracks(
        self,
        source_track_id: int,
        target_track_id: int,
        strategy: MergeStrategy = MergeStrategy.FAIL_ON_CONFLICT,
        force: bool = False
    ) -> MergeResult:
        if source_track_id == target_track_id:
            raise TrackOperationError("Cannot merge a track with itself.")

        s_track = self.get_track_or_raise(source_track_id)
        t_track = self.get_track_or_raise(target_track_id)

        if (s_track.locked or t_track.locked) and not force:
            raise TrackLockedError("Cannot merge locked tracks.")

        overlapping = self.detect_temporal_overlap(source_track_id, target_track_id)
        if overlapping and strategy == MergeStrategy.FAIL_ON_CONFLICT:
            raise TrackOperationError(
                f"Cannot merge Track #{source_track_id} into #{target_track_id}: "
                f"collision on {len(overlapping)} frames: {overlapping[:5]}..."
            )

        merged_count = 0
        for f, s_ann in list(s_track.annotations.items()):
            if f in t_track.annotations:
                if strategy == MergeStrategy.PREFER_SOURCE:
                    s_ann.track_id = target_track_id
                    t_track.annotations[f] = s_ann
                    merged_count += 1
            else:
                s_ann.track_id = target_track_id
                t_track.annotations[f] = s_ann
                merged_count += 1

        del self.tracks[source_track_id]
        return MergeResult(
            source_track_id=source_track_id,
            target_track_id=target_track_id,
            merged_annotations_count=merged_count,
            conflict_count=len(overlapping),
            message=f"Merged #{source_track_id} into #{target_track_id} ({merged_count} frames, {len(overlapping)} conflicts)."
        )

    def clear(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_id": self._next_id,
            "tracks": {str(tid): t.to_dict() for tid, t in self.tracks.items()}
        }

    def load_dict(self, data: Dict[str, Any], clear_existing: bool = True) -> None:
        if clear_existing:
            self.clear()
        self._next_id = int(data.get("next_id", 1))
        for tid_str, t_dict in data.get("tracks", {}).items():
            t = Track.from_dict(t_dict)
            self.tracks[t.track_id] = t