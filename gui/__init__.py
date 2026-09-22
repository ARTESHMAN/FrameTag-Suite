"""
gui/__init__.py

User interface widgets, canvas items, timeline scrubbers, and dialogs.
"""

from .canvas import CanvasView, ImageAdjustments, OnionSkinMode
from .canvas_items import AnnotationBBoxItem, ResizeHandle, get_track_color
from .class_panel import ClassPanel
from .dialogs import IntervalPurgeDialog, MergeConflictDialog, SettingsDialog
from .export_dialog import ExportDialog, ExportWorker
from .model_panel import ModelPanel
from .object_panel import ObjectPanel
from .properties_panel import PropertiesPanel
from .timeline import TimelineWidget
from .track_lanes import TrackLanesWidget
from .validation_dialog import QualityCheckerTab, StatisticsTab, ValidationDialog

__all__ = [
    "CanvasView",
    "ImageAdjustments",
    "OnionSkinMode",
    "AnnotationBBoxItem",
    "ResizeHandle",
    "get_track_color",
    "TimelineWidget",
    "TrackLanesWidget",
    "ObjectPanel",
    "PropertiesPanel",
    "ClassPanel",
    "ModelPanel",
    "IntervalPurgeDialog",
    "MergeConflictDialog",
    "SettingsDialog",
    "ValidationDialog",
    "QualityCheckerTab",
    "StatisticsTab",
    "ExportDialog",
    "ExportWorker",
]