"""
core/__init__.py

Core business logic, data models, track management, video threading,
AI detection/tracking, and project session persistence.
"""

from .annotation_manager import AnnotationManager, BoundingBox
from .annotation_models import Annotation, AnnotationSource, ShapeType, Track
from .byte_track import BatchAutoTrackWorker
from .cadence import CadenceEngine, CadenceStepResult
from .detector import HardwareDeviceResolver, YOLODetector
from .frame_cache import FrameCache
from .history_manager import (
    AddAnnotationCommand,
    BatchAddAnnotationCommand,
    Command,
    DeleteTailCommand,
    DeleteTrackCommand,
    HistoryManager,
    MacroCommand,
    RemoveAnnotationCommand,
    UpdateAnnotationCommand,
)
from .interpolation import InterpolationEngine
from .project_manager import ClassCatalogEntry, ProjectManager, RecoveryEntry, RecoveryInfo, VideoMetadata
from .track_manager import (
    MergeResult,
    MergeStrategy,
    SplitResult,
    TrackLockedError,
    TrackManager,
    TrackNotFoundError,
    TrackOperationError,
)
from .tracker import TrackerType, VisualTracker
from .video_thread import VideoDecoderWorker, VideoProvider

__all__ = [
    "Annotation",
    "Track",
    "ShapeType",
    "AnnotationSource",
    "BoundingBox",
    "TrackManager",
    "TrackOperationError",
    "TrackNotFoundError",
    "TrackLockedError",
    "SplitResult",
    "MergeResult",
    "MergeStrategy",
    "AnnotationManager",
    "Command",
    "AddAnnotationCommand",
    "RemoveAnnotationCommand",
    "UpdateAnnotationCommand",
    "BatchAddAnnotationCommand",
    "DeleteTailCommand",
    "DeleteTrackCommand",
    "MacroCommand",
    "HistoryManager",
    "FrameCache",
    "VideoProvider",
    "VideoDecoderWorker",
    "InterpolationEngine",
    "CadenceEngine",
    "CadenceStepResult",
    "HardwareDeviceResolver",
    "YOLODetector",
    "TrackerType",
    "VisualTracker",
    "BatchAutoTrackWorker",
    "VideoMetadata",
    "ClassCatalogEntry",
    "RecoveryEntry",
    "RecoveryInfo",
    "ProjectManager",
]