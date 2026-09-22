"""
utils/statistics.py

Comprehensive Annotation & Tracking Analytics Calculator for DarkLabel Modern:
- Calculates dataset density, coverage, and total frame durations.
- Class category distributions (raw counts, frame counts, and percentages).
- Provenance distributions (MANUAL keyframe vs. AI vs. Tracker vs. Interpolated).
- Track duration metrics: shortest track, longest track, and average track length.
- Density maps: identifies most populated frame indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import statistics
from typing import Any, Dict, List, Optional, Tuple

from core.annotation_manager import AnnotationManager
from core.annotation_models import AnnotationSource


@dataclass
class ProjectStatistics:
    """Encapsulates dataset analytics and distribution metrics."""
    total_video_frames: int = 0
    annotated_frames: int = 0
    annotation_coverage_pct: float = 0.0
    total_tracks: int = 0
    total_annotations: int = 0

    fps: float = 30.0
    video_duration_sec: float = 0.0
    annotated_duration_sec: float = 0.0

    # Provenance Breakdown
    manual_keyframes: int = 0
    manual_holds: int = 0
    ai_generated: int = 0
    tracker_generated: int = 0
    interpolated_frames: int = 0

    # Flags Breakdown
    occluded_count: int = 0
    outside_count: int = 0

    # Track Lengths (in frames)
    min_track_len: int = 0
    max_track_len: int = 0
    avg_track_len: float = 0.0
    median_track_len: float = 0.0
    longest_track_id: Optional[int] = None
    shortest_track_id: Optional[int] = None

    # Density
    avg_boxes_per_annotated_frame: float = 0.0
    peak_boxes_in_frame: int = 0
    peak_frame_index: int = 0

    # Category Breakdown: {class_name: count}
    class_counts: Dict[str, int] = field(default_factory=dict)
    class_percentages: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_video_frames": self.total_video_frames,
            "annotated_frames": self.annotated_frames,
            "annotation_coverage_pct": round(self.annotation_coverage_pct, 2),
            "total_tracks": self.total_tracks,
            "total_annotations": self.total_annotations,
            "fps": round(self.fps, 2),
            "video_duration_sec": round(self.video_duration_sec, 2),
            "annotated_duration_sec": round(self.annotated_duration_sec, 2),
            "manual_keyframes": self.manual_keyframes,
            "manual_holds": self.manual_holds,
            "ai_generated": self.ai_generated,
            "tracker_generated": self.tracker_generated,
            "interpolated_frames": self.interpolated_frames,
            "occluded_count": self.occluded_count,
            "outside_count": self.outside_count,
            "min_track_len": self.min_track_len,
            "max_track_len": self.max_track_len,
            "avg_track_len": round(self.avg_track_len, 1),
            "median_track_len": round(self.median_track_len, 1),
            "longest_track_id": self.longest_track_id,
            "shortest_track_id": self.shortest_track_id,
            "avg_boxes_per_frame": round(self.avg_boxes_per_annotated_frame, 2),
            "peak_boxes_in_frame": self.peak_boxes_in_frame,
            "peak_frame_index": self.peak_frame_index,
            "class_counts": self.class_counts,
            "class_percentages": self.class_percentages,
        }


class StatisticsCalculator:
    """Computes project analytical metrics."""

    @classmethod
    def compute(
        cls,
        manager: AnnotationManager,
        total_video_frames: int,
        fps: float
    ) -> ProjectStatistics:
        stats = ProjectStatistics()
        stats.total_video_frames = max(1, total_video_frames)
        stats.fps = max(1.0, fps)
        stats.video_duration_sec = stats.total_video_frames / stats.fps

        with manager.lock:
            annotated_frame_indices = manager.get_annotated_frame_indices()
            stats.annotated_frames = len(annotated_frame_indices)
            stats.annotation_coverage_pct = (stats.annotated_frames / float(stats.total_video_frames)) * 100.0
            stats.annotated_duration_sec = stats.annotated_frames / stats.fps

            all_tracks = manager.track_manager.get_all_tracks()
            stats.total_tracks = len(all_tracks)

            track_lengths: List[int] = []
            longest_len = -1
            shortest_len = 999999999

            for track in all_tracks:
                length = track.total_frames
                track_lengths.append(length)

                if length > longest_len:
                    longest_len = length
                    stats.longest_track_id = track.track_id
                if length < shortest_len and length > 0:
                    shortest_len = length
                    stats.shortest_track_id = track.track_id

            if track_lengths:
                stats.min_track_len = min(track_lengths)
                stats.max_track_len = max(track_lengths)
                stats.avg_track_len = statistics.mean(track_lengths)
                stats.median_track_len = statistics.median(track_lengths)

            # Analyze frame boxes and provenance breakdown
            total_boxes = 0
            peak_boxes = 0
            peak_frame = 0

            for f in annotated_frame_indices:
                boxes = manager.get_annotations(f, visible_only=False, include_outside=True)
                b_count = len(boxes)
                total_boxes += b_count

                if b_count > peak_boxes:
                    peak_boxes = b_count
                    peak_frame = f

                for b in boxes:
                    # Class counts
                    stats.class_counts[b.class_name] = stats.class_counts.get(b.class_name, 0) + 1

                    # Provenance
                    if b.source == AnnotationSource.AI:
                        stats.ai_generated += 1
                    elif b.source == AnnotationSource.TRACKER:
                        stats.tracker_generated += 1
                    elif b.source == AnnotationSource.INTERPOLATED:
                        stats.interpolated_frames += 1
                    elif b.is_keyframe:
                        stats.manual_keyframes += 1
                    else:
                        stats.manual_holds += 1

                    # States
                    if b.occluded:
                        stats.occluded_count += 1
                    if b.outside:
                        stats.outside_count += 1

            stats.total_annotations = total_boxes
            stats.peak_boxes_in_frame = peak_boxes
            stats.peak_frame_index = peak_frame
            stats.avg_boxes_per_annotated_frame = (
                total_boxes / float(stats.annotated_frames) if stats.annotated_frames > 0 else 0.0
            )

            # Calculate class distribution percentages
            if total_boxes > 0:
                for cname, count in stats.class_counts.items():
                    stats.class_percentages[cname] = round((count / float(total_boxes)) * 100.0, 1)

        return stats