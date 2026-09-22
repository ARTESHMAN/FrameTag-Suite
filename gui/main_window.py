"""
gui/main_window.py

Master Studio Shell for DarkLabel Modern:
- Houses CanvasView, TimelineWidget, and collapsible TrackLanesWidget.
- Connects ObjectPanel, PropertiesPanel, ClassPanel, and ModelPanel.
- Direct Hotkeys:
    * Key D: Propagates current frame bounding box(es) forward by 1 frame (+1) and advances playhead.
    * Key F: Propagates current frame bounding box(es) forward across 20 frames (+20) and lands on frame +20.
    * Key E: Terminates track at current frame (purges all future frames on this track).
    * Shift+F: Fit Canvas to Window.
- Auto-Refreshes canvas overlays immediately after AI detection and batch tracking finishes.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.annotation_manager import AnnotationManager
from core.annotation_models import Annotation, AnnotationSource, Track
from core.byte_track import BatchAutoTrackWorker
from core.cadence import CadenceEngine
from core.detector import YOLODetector
from core.history_manager import (
    AddAnnotationCommand,
    DeleteTrackCommand,
    HistoryManager,
    RemoveAnnotationCommand,
    UpdateAnnotationCommand,
)
from core.interpolation import InterpolationEngine
from core.project_manager import ClassCatalogEntry, ProjectManager
from core.track_manager import MergeStrategy, TrackManager
from core.tracker import TrackerType, VisualTracker
from core.video_thread import VideoProvider
from gui.canvas import CanvasView, OnionSkinMode
from gui.canvas_items import AnnotationBBoxItem, get_track_color
from gui.class_panel import ClassPanel
from gui.dialogs import IntervalPurgeDialog, MergeConflictDialog, SettingsDialog
from gui.export_dialog import ExportDialog
from gui.model_panel import ModelPanel
from gui.object_panel import ObjectPanel
from gui.properties_panel import PropertiesPanel
from gui.timeline import TimelineWidget
from gui.track_lanes import TrackLanesWidget
from gui.validation_dialog import ValidationDialog


MODERN_WORKSTATION_QSS = """
QMainWindow, QWidget {
    background-color: #121215;
    color: #e4e4e7;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
    font-size: 12px;
    outline: none;
}

QMenuBar {
    background-color: #18181b;
    border-bottom: 1px solid #27272a;
    padding: 3px 6px;
    color: #a1a1aa;
}
QMenuBar::item {
    background: transparent;
    padding: 4px 10px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #27272a;
    color: #ffffff;
}

QMenu {
    background-color: #18181b;
    border: 1px solid #3f3f46;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 22px 6px 12px;
    border-radius: 4px;
    color: #e4e4e7;
}
QMenu::item:selected {
    background-color: #2563eb;
    color: #ffffff;
}
QMenu::separator {
    height: 1px;
    background-color: #27272a;
    margin: 4px 8px;
}

QSplitter::handle {
    background-color: #1e1e22;
}
QSplitter::handle:hover {
    background-color: #3b82f6;
}

QTabWidget::pane {
    border: 1px solid #27272a;
    background: #18181b;
    border-radius: 4px;
}
QTabBar::tab {
    background: #141417;
    color: #a1a1aa;
    padding: 7px 14px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    margin-right: 2px;
    font-weight: 500;
}
QTabBar::tab:hover {
    background: #1f1f23;
    color: #e4e4e7;
}
QTabBar::tab:selected {
    background: #18181b;
    color: #60a5fa;
    border-bottom: 2px solid #3b82f6;
}

QPushButton {
    background-color: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 5px 12px;
    color: #fafafa;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #323238;
    border-color: #52525b;
}
QPushButton:pressed {
    background-color: #18181b;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #18181b;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 4px 8px;
    color: #f4f4f5;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #3b82f6;
}

QStatusBar {
    background-color: #101012;
    border-top: 1px solid #27272a;
    color: #82828e;
    font-size: 11px;
}
"""


class DarkLabelMainWindow(QMainWindow):
    """Master Application Window orchestrating all modules and interactions."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DarkLabel Modern - Professional Video Annotation & Tracking Studio")
        self.resize(1720, 980)

        self._apply_application_theme()

        # 1. Core Engines
        self.track_manager = TrackManager()
        self.annotation_manager = AnnotationManager(self.track_manager)
        self.history_manager = HistoryManager(self.annotation_manager, max_depth=150)
        self.project_manager = ProjectManager(self.annotation_manager, self.history_manager)
        self.video_provider = VideoProvider(cache_size_mb=512)
        self.detector = YOLODetector()
        self.manual_tracker = VisualTracker(tracker_type=TrackerType.CSRT)

        # 2. State Variables
        self.current_frame: int = 0
        self.current_frame_bgr: Optional[np.ndarray] = None
        self.active_track_id: int = 1
        self.active_class_id: int = 0
        self.active_class_name: str = "fire"

        self.settings: Dict[str, Any] = {
            "cadence_fps": 5.0,
            "auto_interpolate": True,
            "forward_hold": True,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "preferred_device": "Auto Detect",
            "cache_size_mb": 512,
            "autosave_interval_sec": 60,
        }

        self._init_default_classes()

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.step_forward)

        self.batch_worker: Optional[BatchAutoTrackWorker] = None

        self._init_ui()
        self._setup_menus()
        self._setup_shortcuts()
        self._bind_signals()

        self.project_manager.start_autosave_daemon(60)
        self._check_for_crash_recovery()

    def _apply_application_theme(self) -> None:
        qss_file = Path(__file__).resolve().parent.parent / "assets" / "style.qss"
        if qss_file.exists():
            with open(qss_file, "r", encoding="utf-8") as f:
                theme_qss = f.read()
        else:
            theme_qss = MODERN_WORKSTATION_QSS
        self.setStyleSheet(theme_qss)

    def _init_default_classes(self) -> None:
        defaults = ["fire", "smoke", "person", "vehicle", "target"]
        for idx, name in enumerate(defaults):
            color = get_track_color(idx + 1).name()
            entry = ClassCatalogEntry(class_id=idx, name=name, color=color)
            self.project_manager.classes[idx] = entry

    def _init_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        self.main_splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(self.main_splitter)

        # Left Column: Canvas, Track Lanes, and Timeline
        left_box = QWidget()
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)

        self.canvas = CanvasView()
        self.track_lanes = TrackLanesWidget()
        self.track_lanes.set_manager(self.annotation_manager)
        self.track_lanes.setVisible(False)

        self.timeline = TimelineWidget()
        self.timeline.set_manager(self.annotation_manager)

        left_layout.addWidget(self.canvas, stretch=1)
        left_layout.addWidget(self.track_lanes, stretch=0)
        left_layout.addWidget(self.timeline, stretch=0)
        self.main_splitter.addWidget(left_box)

        # Right Column: Sidebar Panels
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setMinimumWidth(380)
        self.sidebar_tabs.setMaximumWidth(460)

        self.panel_objects = ObjectPanel()
        self.panel_objects.set_manager(self.annotation_manager)

        self.panel_properties = PropertiesPanel()
        self.panel_properties.set_manager(self.annotation_manager)
        self._sync_class_combos()

        self.panel_classes = ClassPanel()
        self.panel_classes.set_manager(self.annotation_manager)
        self.panel_classes.set_classes(self.project_manager.classes)

        self.panel_models = ModelPanel(self.detector)

        self.sidebar_tabs.addTab(self.panel_objects, "Objects")
        self.sidebar_tabs.addTab(self.panel_properties, "Properties")
        self.sidebar_tabs.addTab(self.panel_classes, "Classes")
        self.sidebar_tabs.addTab(self.panel_models, "AI & Tracker")

        self.main_splitter.addWidget(self.sidebar_tabs)
        self.main_splitter.setSizes([1260, 420])

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_status_device = QLabel(f"Backend: {self.detector.device_name}")
        self.lbl_status_device.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; padding: 2px 8px; "
            "border-radius: 4px; font-weight: 600; font-size: 11px; margin-right: 8px;"
        )
        self.status_bar.addPermanentWidget(self.lbl_status_device)
        self.status_bar.showMessage("Ready. Open a video (Ctrl+O) to begin.")

    def _setup_menus(self) -> None:
        mb = self.menuBar()

        # File Menu
        m_file = mb.addMenu("File")
        self._act(m_file, "Open Media / Sequence...", "Ctrl+O", self.open_media)
        self._act(m_file, "Open Project (.dlm)...", None, self.open_project)
        self._act(m_file, "Save Project", "Ctrl+S", self.save_project)
        self._act(m_file, "Save Project As...", "Ctrl+Shift+S", self.save_project_as)
        m_file.addSeparator()
        self._act(m_file, "Export & Media Studio...", "Ctrl+E", self.open_export_dialog)
        m_file.addSeparator()
        self._act(m_file, "Exit", "Ctrl+Q", self.close)

        # Edit Menu
        m_edit = mb.addMenu("Edit")
        self._act(m_edit, "Undo", "Ctrl+Z", self.undo)
        self._act(m_edit, "Redo", "Ctrl+Y", self.redo)
        m_edit.addSeparator()
        self._act(m_edit, "Delete Active Box", "Delete", self.delete_active_box)
        self._act(m_edit, "Delete Entire Track", "Shift+Delete", self.delete_entire_track)
        self._act(m_edit, "Purge Frame Interval...", None, self.purge_interval)

        # View Menu
        m_view = mb.addMenu("View")
        self._act(m_view, "Fit Canvas to Window", "Shift+F", self.canvas.fit_to_window)
        self._act(m_view, "1:1 Pixel Scale", "1", self.canvas.zoom_100)
        m_view.addSeparator()
        act_lanes = m_view.addAction("Show Track Activity Lanes")
        act_lanes.setCheckable(True)
        act_lanes.toggled.connect(self.track_lanes.setVisible)

        # Annotation Menu
        m_ann = mb.addMenu("Annotation")
        self._act(m_ann, "Propagate Next Frame (1 Frame)", "D", lambda: self.propagate_boxes_forward(1))
        self._act(m_ann, "Propagate 20 Frames Forward", "F", lambda: self.propagate_boxes_forward(20))
        self._act(m_ann, "Terminate Track at Frame", "E", self.terminate_active_track)
        m_ann.addSeparator()
        self._act(m_ann, "Interpolate Between Keyframes", "I", self.interpolate_active_track)
        self._act(m_ann, "Step-Track with CSRT", "Return", self.step_visual_tracker)
        self._act(m_ann, "Toggle Occluded State", "X", self.toggle_occluded)
        self._act(m_ann, "Toggle Outside State", "O", self.toggle_outside)

        # AI Menu
        m_ai = mb.addMenu("AI")
        self._act(m_ai, "Load YOLO Model...", "Ctrl+M", self.panel_models._on_load_weights_clicked)
        self._act(m_ai, "Auto-Tag Current Frame", "Ctrl+F", self.auto_tag_current_frame)
        self._act(m_ai, "Batch Auto-Track Video...", "Ctrl+B", self.open_batch_tracking)

        # QA Menu
        m_qa = mb.addMenu("QA & Analytics")
        self._act(m_qa, "Run Quality Assurance Audit...", "Ctrl+T", self.open_qa_dialog)

        # Settings Menu
        m_set = mb.addMenu("Settings")
        self._act(m_set, "Preferences...", None, self.open_settings_dialog)

    def _act(self, menu, title: str, shortcut: Optional[str], slot) -> QAction:
        act = QAction(title, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
            act.setShortcutContext(Qt.WindowShortcut)
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    def _setup_shortcuts(self) -> None:
        """Global navigation shortcuts without duplicates from the menu bar."""
        shortcuts = [
            ("Space", self.toggle_play_pause),
            ("Right", self.step_forward),
            ("Left", self.step_backward),
            ("Shift+Right", lambda: self.seek_frame(self.current_frame + 10)),
            ("Shift+Left", lambda: self.seek_frame(self.current_frame - 10)),
            ("Alt+Right", self.jump_next_keyframe),
            ("Alt+Left", self.jump_prev_keyframe),
            ("Home", lambda: self.seek_frame(0)),
            ("End", lambda: self.seek_frame(self.video_provider.total_frames - 1)),
        ]
        for key, slot in shortcuts:
            act = QAction(self)
            act.setShortcut(QKeySequence(key))
            act.setShortcutContext(Qt.WindowShortcut)
            act.triggered.connect(slot)
            self.addAction(act)

    def _bind_signals(self) -> None:
        self.video_provider.frame_ready.connect(self._on_frame_ready)
        self.video_provider.video_loaded.connect(self._on_video_loaded)
        self.video_provider.error.connect(lambda e: QMessageBox.critical(self, "Video Error", e))

        self.canvas.box_created.connect(self._on_canvas_box_created)
        self.canvas.box_modified.connect(self._on_canvas_box_modified)
        self.canvas.box_selected.connect(self._on_canvas_box_selected)
        self.canvas.box_deleted.connect(lambda *args: self.delete_active_box())

        self.timeline.frame_changed.connect(self.seek_frame)
        self.timeline.play_pause_toggled.connect(self.toggle_play_pause)
        self.timeline.step_forward_requested.connect(self.step_forward)
        self.timeline.step_backward_requested.connect(self.step_backward)
        self.timeline.jump_delta_requested.connect(lambda d: self.seek_frame(self.current_frame + d))
        self.timeline.prev_keyframe_requested.connect(self.jump_prev_keyframe)
        self.timeline.next_keyframe_requested.connect(self.jump_next_keyframe)

        self.track_lanes.track_selected.connect(self._select_track)
        self.track_lanes.frame_seek_requested.connect(self.seek_frame)

        self.panel_objects.track_selected.connect(self._select_track)
        self.panel_objects.track_visibility_toggled.connect(lambda tid, v: self.redraw_annotations())
        self.panel_objects.track_lock_toggled.connect(lambda tid, l: self.redraw_annotations())
        self.panel_objects.track_delete_requested.connect(self._delete_track_by_id)
        self.panel_objects.tracks_merge_requested.connect(self._merge_tracks_dialog)
        self.panel_objects.track_split_requested.connect(self._split_track_dialog)

        self.panel_properties.class_changed.connect(self._on_property_class_changed)
        self.panel_properties.lock_toggled.connect(self._on_property_lock_changed)
        self.panel_properties.occlusion_toggled.connect(lambda tid, o: self.redraw_annotations())
        self.panel_properties.outside_toggled.connect(lambda tid, o: self.redraw_annotations())
        self.panel_properties.jump_frame_requested.connect(self.seek_frame)
        self.panel_properties.split_track_requested.connect(self._split_track_dialog)
        self.panel_properties.terminate_tail_requested.connect(self.terminate_active_track)
        self.panel_properties.delete_track_requested.connect(self._delete_track_by_id)

        self.panel_classes.class_created.connect(self._on_class_catalog_changed)
        self.panel_classes.class_renamed.connect(self._on_class_catalog_changed)
        self.panel_classes.class_color_changed.connect(self._on_class_catalog_changed)
        self.panel_classes.class_deleted.connect(self._on_class_catalog_changed)

        if hasattr(self.panel_classes, "class_selected"):
            self.panel_classes.class_selected.connect(self._on_class_selected)

        self.panel_models.auto_tag_single_requested.connect(self._on_model_auto_tag)
        self.panel_models.batch_track_requested.connect(self._on_model_batch_track)
        self.panel_models.batch_cancel_requested.connect(self._cancel_batch_tracking)

    def _on_class_selected(self, class_id: int, class_name: str) -> None:
        """Updates the default active class for newly drawn bounding boxes without modifying existing tracks."""
        self.active_class_id = class_id
        self.active_class_name = class_name
        self.canvas.set_active_class(class_id, class_name, get_track_color(class_id + 1))
        self.status_bar.showMessage(f"Selected active drawing class: {class_name} (ID #{class_id})")

    def open_media(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File or Select Folder", "", "Videos (*.mp4 *.avi *.mov *.mkv *.png *.jpg);;All Files (*)"
        )
        if path:
            self.play_timer.stop()
            self.timeline.set_playback_active(False)
            self.video_provider.open_source(path)
            self.project_manager.video_metadata.source_path = path

    def _on_video_loaded(self, total: int, fps: float, w: int, h: int, fmt: str) -> None:
        self.timeline.set_video_metadata(total, fps)
        self.track_lanes.set_video_meta(total)

        ms = max(10, int(1000.0 / (fps or 30.0)))
        self.play_timer.setInterval(ms)
        self.canvas.fit_to_window()

        self.project_manager.video_metadata.total_frames = total
        self.project_manager.video_metadata.fps = fps
        self.project_manager.video_metadata.width = w
        self.project_manager.video_metadata.height = h
        self.project_manager.video_metadata.codec = fmt

        self.status_bar.showMessage(f"Loaded: {os.path.basename(self.video_provider.source_path)} [{w}x{h} @ {fps:.2f} FPS]")

    def _on_frame_ready(self, frame_idx: int, frame_bgr: np.ndarray) -> None:
        self.current_frame = frame_idx
        self.current_frame_bgr = frame_bgr

        prev_bgr = self.video_provider.get_frame_at(frame_idx - 1) if self.canvas.onion_mode != OnionSkinMode.DISABLED else None
        next_bgr = self.video_provider.get_frame_at(frame_idx + 1) if self.canvas.onion_mode != OnionSkinMode.DISABLED else None

        self.canvas.set_frames(frame_bgr, prev_bgr, next_bgr)
        self.canvas.set_frame_index(frame_idx)
        self.timeline.set_frame(frame_idx)
        self.track_lanes.set_current_frame(frame_idx)
        self.panel_properties.set_target(self.active_track_id, frame_idx)

        self.redraw_annotations()

    def seek_frame(self, frame_idx: int) -> None:
        self.video_provider.seek(frame_idx)

    def step_forward(self) -> None:
        if self.current_frame < self.video_provider.total_frames - 1:
            self.seek_frame(self.current_frame + 1)
        else:
            self.play_timer.stop()
            self.timeline.set_playback_active(False)

    def step_backward(self) -> None:
        if self.current_frame > 0:
            self.seek_frame(self.current_frame - 1)

    def toggle_play_pause(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.timeline.set_playback_active(False)
        else:
            self.play_timer.start()
            self.timeline.set_playback_active(True)

    def jump_prev_keyframe(self) -> None:
        with self.annotation_manager.lock:
            kfs = [f for f in self.annotation_manager.get_all_keyframes() if f < self.current_frame]
        if kfs:
            self.seek_frame(max(kfs))

    def jump_next_keyframe(self) -> None:
        with self.annotation_manager.lock:
            kfs = [f for f in self.annotation_manager.get_all_keyframes() if f > self.current_frame]
        if kfs:
            self.seek_frame(min(kfs))

    def redraw_annotations(self) -> None:
        """Clears and re-adds all persistent annotations for the current frame."""
        for item in list(self.canvas.scene.items()):
            if isinstance(item, AnnotationBBoxItem):
                self.canvas.scene.removeItem(item)

        with self.annotation_manager.lock:
            boxes = self.annotation_manager.get_annotations(
                self.current_frame, visible_only=True, include_outside=False
            )

        # Derive valid scene bounds from image or video provider
        w = float(self.video_provider.width if getattr(self.video_provider, "width", 0) > 0 else self.canvas.image_width)
        h = float(self.video_provider.height if getattr(self.video_provider, "height", 0) > 0 else self.canvas.image_height)
        scene_bounds = QRectF(0.0, 0.0, max(1.0, w), max(1.0, h)) if (w > 0 and h > 0) else None

        for b in boxes:
            track = self.track_manager.get_track(b.track_id)
            is_locked = track.locked if track else False

            # Extract coordinates safely regardless of model attribute naming
            if hasattr(b, "bbox") and b.bbox is not None and len(b.bbox) == 4:
                bx, by, bw, bh = b.bbox
            elif hasattr(b, "rect") and b.rect is not None:
                if isinstance(b.rect, QRectF):
                    bx, by, bw, bh = b.rect.x(), b.rect.y(), b.rect.width(), b.rect.height()
                else:
                    bx, by, bw, bh = b.rect[0], b.rect[1], b.rect[2], b.rect[3]
            else:
                bx = getattr(b, "x", 0.0)
                by = getattr(b, "y", 0.0)
                bw = getattr(b, "width", getattr(b, "w", 50.0))
                bh = getattr(b, "height", getattr(b, "h", 50.0))

            bx, by, bw, bh = float(bx), float(by), float(bw), float(bh)
            box_rect = QRectF(bx, by, bw, bh)

            cid = getattr(b, "class_id", 0)
            cname = getattr(b, "class_name", getattr(b, "label", f"Class {cid}"))

            item = AnnotationBBoxItem(
                rect=box_rect,
                class_id=cid,
                label=cname,
                track_id=b.track_id,
                confidence=getattr(b, "confidence", 1.0),
                occluded=getattr(b, "occluded", False),
                keyframe=getattr(b, "is_keyframe", True),
                annotation=b,
                frame_index=self.current_frame,
                is_locked=is_locked,
            )
            # Ensure annotations render in front of video background
            item.setZValue(20.0)

            if scene_bounds:
                item.set_bounds_limit(scene_bounds)

            self.canvas.scene.addItem(item)
            if b.track_id == self.active_track_id:
                item.setSelected(True)

        self.timeline.set_keyframes(self.annotation_manager.get_all_keyframes())
        self.track_lanes.refresh()
        self.panel_objects.refresh()
        self.panel_properties.refresh()

    def _on_canvas_box_created(self, *args) -> None:
        """
        Creates an annotation on the current frame with a new unique track ID.
        Accepts (x, y, w, h) coordinates or an item/rect object.
        """
        if len(args) == 4:
            x, y, w, h = [float(v) for v in args]
        elif len(args) == 1:
            item = args[0]
            if hasattr(item, "get_coordinates"):
                x, y, w, h = item.get_coordinates()
            elif hasattr(item, "rect") and callable(item.rect):
                r = item.rect()
                p = item.pos() if hasattr(item, "pos") else QPointF(0, 0)
                x, y, w, h = float(p.x() + r.x()), float(p.y() + r.y()), float(r.width()), float(r.height())
            elif hasattr(item, "rect") and isinstance(item.rect, QRectF):
                r = item.rect
                x, y, w, h = float(r.x()), float(r.y()), float(r.width()), float(r.height())
            elif isinstance(item, QRectF):
                x, y, w, h = float(item.x()), float(item.y()), float(item.width()), float(item.height())
            elif isinstance(item, (list, tuple)) and len(item) == 4:
                x, y, w, h = [float(v) for v in item]
            else:
                return
        else:
            return

        if w <= 3.0 or h <= 3.0:
            return

        new_track_id = self.track_manager.get_next_track_id()
        self.active_track_id = new_track_id

        new_ann = Annotation(
            track_id=new_track_id,
            class_id=self.active_class_id,
            class_name=self.active_class_name,
            frame_index=self.current_frame,
            x=float(x),
            y=float(y),
            width=float(w),
            height=float(h),
            is_keyframe=True,
            source=AnnotationSource.MANUAL,
        )
        if hasattr(new_ann, "bbox"):
            new_ann.bbox = (float(x), float(y), float(w), float(h))
        if hasattr(new_ann, "label"):
            new_ann.label = self.active_class_name

        # Store to annotation manager and command history
        self.history_manager.execute(AddAnnotationCommand(new_ann))

        track = self.track_manager.get_track(new_track_id)
        if track:
            track.class_id = self.active_class_id
            track.class_name = self.active_class_name
            track.visible = True

        self._select_track(new_track_id)
        self.redraw_annotations()

    def _on_canvas_box_modified(self, ann) -> None:
        """Called when an annotation box is modified on the canvas."""
        if ann is None:
            return

        if hasattr(ann, "annotation") and ann.annotation is not None:
            old_ann = copy.deepcopy(ann.annotation)

            if hasattr(ann, "get_coordinates"):
                coords = ann.get_coordinates()
                new_bbox = [float(c) for c in coords]
            else:
                rect = ann.rect()
                pos = ann.pos()
                new_bbox = [
                    float(pos.x() + rect.x()),
                    float(pos.y() + rect.y()),
                    float(rect.width()),
                    float(rect.height()),
                ]

            new_ann = copy.deepcopy(old_ann)
            new_ann.bbox = new_bbox
            if hasattr(new_ann, "x"):
                new_ann.x, new_ann.y, new_ann.width, new_ann.height = new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3]

            if not hasattr(new_ann, "frame_index") or new_ann.frame_index is None:
                new_ann.frame_index = self.current_frame

            ann.annotation = new_ann

            try:
                self.history_manager.execute(
                    UpdateAnnotationCommand(
                        new_annotation=new_ann,
                        old_annotation=old_ann,
                        annotation_manager=self.annotation_manager,
                    )
                )
            except TypeError:
                self.history_manager.execute(UpdateAnnotationCommand(new_ann))
        elif hasattr(ann, "track_id") and not hasattr(ann, "frame_index"):
            setattr(ann, "frame_index", self.current_frame)
            self.history_manager.execute(UpdateAnnotationCommand(ann))
        else:
            self.history_manager.execute(UpdateAnnotationCommand(ann))

    def _on_canvas_box_selected(self, target: Any) -> None:
        """Bridges canvas box selection to track selection."""
        self._select_track(target)

    def propagate_boxes_forward(self, num_frames: int) -> None:
        total_f = self.video_provider.total_frames
        if total_f <= 0:
            return

        cur_f = self.current_frame
        target_f = min(cur_f + num_frames, total_f - 1)
        if target_f == cur_f:
            self.status_bar.showMessage("Already at the final frame of the video.")
            return

        with self.annotation_manager.lock:
            current_boxes = self.annotation_manager.get_annotations(cur_f, visible_only=True, include_outside=False)

        if not current_boxes:
            self.seek_frame(target_f)
            self.status_bar.showMessage(f"Advanced {num_frames} frame(s) to Frame {target_f}.")
            return

        self.history_manager.begin_transaction(f"Propagate {len(current_boxes)} box(es) forward {num_frames} frames")
        try:
            with self.annotation_manager.lock:
                for f in range(cur_f + 1, target_f + 1):
                    is_dest = (f == target_f)
                    for box in current_boxes:
                        carried = box.copy()
                        carried.frame_index = f
                        carried.is_keyframe = is_dest
                        carried.interpolated = not is_dest
                        carried.source = AnnotationSource.MANUAL if is_dest else AnnotationSource.INTERPOLATED
                        self.history_manager.execute(AddAnnotationCommand(carried))
            self.history_manager.commit_transaction()
        except Exception as e:
            self.history_manager.rollback_transaction()
            raise e

        self.redraw_annotations()
        self.seek_frame(target_f)
        self.status_bar.showMessage(f"Propagated {len(current_boxes)} box(es) forward {num_frames} frame(s) -> Frame {target_f}.")

    def _select_track(self, track_id: int | Any) -> None:
        if hasattr(track_id, "track_id"):
            track_id = track_id.track_id
        elif hasattr(track_id, "annotation") and hasattr(track_id.annotation, "track_id"):
            track_id = track_id.annotation.track_id

        if track_id is None:
            return

        track_id = int(track_id)
        self.active_track_id = track_id
        track = self.track_manager.get_track(track_id)
        if track:
            self.active_class_id = track.class_id
            self.active_class_name = track.class_name

        self.panel_objects.set_active_track(track_id)
        self.panel_properties.set_target(track_id, self.current_frame)
        self.track_lanes.set_active_track(track_id)

        for it in self.canvas.scene.items():
            if isinstance(it, AnnotationBBoxItem):
                tid = getattr(it, "track_id", None)
                if tid is None and hasattr(it, "annotation") and it.annotation is not None:
                    tid = getattr(it.annotation, "track_id", None)
                it.setSelected(tid == track_id)

    def terminate_active_track(self) -> None:
        """Hotkey 'E': terminates track at current frame, deleting all future frames."""
        purged = CadenceEngine.terminate_track(
            manager=self.annotation_manager,
            history=self.history_manager,
            track_id=self.active_track_id,
            current_frame=self.current_frame,
        )
        self.redraw_annotations()
        self.status_bar.showMessage(f"Terminated #{self.active_track_id} at frame {self.current_frame} (purged {len(purged)} future frames).")

    def toggle_occluded(self) -> None:
        occ = CadenceEngine.toggle_occlusion(self.annotation_manager, self.history_manager, self.active_track_id, self.current_frame)
        self.redraw_annotations()
        self.status_bar.showMessage(f"Track #{self.active_track_id} Occluded: {occ}")

    def toggle_outside(self) -> None:
        out = CadenceEngine.toggle_outside(self.annotation_manager, self.history_manager, self.active_track_id, self.current_frame)
        self.redraw_annotations()
        self.status_bar.showMessage(f"Track #{self.active_track_id} Outside: {out}")

    def interpolate_active_track(self) -> None:
        with self.annotation_manager.lock:
            track = self.track_manager.get_track(self.active_track_id)
            if not track:
                return
            prev_kf = track.find_prev_keyframe(self.current_frame)
            if prev_kf is None:
                QMessageBox.warning(self, "Interpolation", f"No preceding keyframe for Track #{self.active_track_id}.")
                return
            generated = InterpolationEngine.interpolate_range(self.annotation_manager, self.active_track_id, prev_kf, self.current_frame)

        self.redraw_annotations()
        self.status_bar.showMessage(f"Interpolated {len(generated)} frames between {prev_kf} and {self.current_frame}.")

    def step_visual_tracker(self) -> None:
        next_idx = self.current_frame + 1
        if next_idx >= self.video_provider.total_frames or self.current_frame_bgr is None:
            return

        next_img = self.video_provider.get_frame_at(next_idx)
        if next_img is None:
            return

        success, ann, msg = VisualTracker.step_forward_track(
            manager=self.annotation_manager,
            track_id=self.active_track_id,
            source_frame=self.current_frame,
            target_frame=next_idx,
            source_img=self.current_frame_bgr,
            target_img=next_img,
            tracker_type=TrackerType.CSRT,
        )
        if success:
            self.seek_frame(next_idx)
            self.status_bar.showMessage(msg)
        else:
            QMessageBox.warning(self, "Tracker Lost Lock", msg)

    def auto_tag_current_frame(self) -> None:
        if self.current_frame_bgr is None or not self.detector.is_loaded:
            return
        conf = float(self.settings.get("confidence_threshold", 0.25))
        iou = float(self.settings.get("iou_threshold", 0.45))
        self._on_model_auto_tag(conf, iou, None)

    def _on_model_auto_tag(self, conf: float, iou: float, allowed: Optional[list]) -> None:
        if self.current_frame_bgr is None or not self.detector.is_loaded:
            return
        boxes = self.detector.detect_frame(self.current_frame_bgr, self.current_frame, conf, iou, allowed)
        for b in boxes:
            b.track_id = self.track_manager.get_next_track_id()
            self.history_manager.execute(AddAnnotationCommand(b))

        self.redraw_annotations()
        self.status_bar.showMessage(f"AI Auto-Tagged {len(boxes)} targets on frame {self.current_frame}.")

    def open_batch_tracking(self) -> None:
        self.sidebar_tabs.setCurrentWidget(self.panel_models)
        self.panel_models._on_batch_track_clicked()

    def _on_model_batch_track(self, conf: float, iou: float, allowed: Optional[list]) -> None:
        if not self.video_provider.source_path or self.video_provider.is_sequence:
            QMessageBox.warning(self, "Batch Track", "Batch tracking requires an open video container file.")
            return

        s, ok1 = QInputDialog.getInt(self, "Batch Range", "Start Frame:", self.current_frame, 0, self.video_provider.total_frames - 1)
        if not ok1:
            return
        e, ok2 = QInputDialog.getInt(self, "Batch Range", "End Frame:", self.video_provider.total_frames - 1, s, self.video_provider.total_frames - 1)
        if not ok2:
            return

        self.panel_models.start_batch_hud()
        self.batch_worker = BatchAutoTrackWorker(
            weights_path=self.detector.weights_path,
            video_path=self.video_provider.source_path,
            annotation_manager=self.annotation_manager,
            start_frame=s,
            end_frame=e,
            conf_thresh=conf,
            iou_thresh=iou,
            allowed_classes=allowed,
            preferred_device=self.settings.get("preferred_device"),
        )
        self.batch_worker.progress.connect(self.panel_models.update_batch_hud)
        self.batch_worker.finished.connect(self._on_batch_finished)
        self.batch_worker.error.connect(lambda err: QMessageBox.critical(self, "Batch Error", err))
        self.batch_worker.start()

    def _cancel_batch_tracking(self) -> None:
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.cancel()

    def _on_batch_finished(self, frames_done: int, tracks_done: int) -> None:
        self.panel_models.finish_batch_hud()
        self.redraw_annotations()
        QMessageBox.information(self, "Batch Tracking Complete", f"Processed {frames_done} frames. Created/tracked {tracks_done} unique objects.")

    def open_qa_dialog(self) -> None:
        dlg = ValidationDialog(
            manager=self.annotation_manager,
            img_width=self.video_provider.width,
            img_height=self.video_provider.height,
            total_frames=self.video_provider.total_frames,
            fps=self.video_provider.fps,
            parent=self,
        )
        dlg.jump_to_frame_requested.connect(lambda f, tid: [self.seek_frame(f), self._select_track(tid)])
        dlg.exec()

    def open_export_dialog(self) -> None:
        class_catalog = {entry.name: cid for cid, entry in self.project_manager.classes.items()}
        dlg = ExportDialog(self.annotation_manager, self.video_provider, class_catalog, self)
        dlg.exec()

    def open_settings_dialog(self) -> None:
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec():
            self.settings = dlg.get_settings()
            self.video_provider.set_cache_limit_mb(self.settings["cache_size_mb"])
            self.status_bar.showMessage("Preferences saved.")

    def purge_interval(self) -> None:
        dlg = IntervalPurgeDialog(
            self.current_frame,
            self.video_provider.total_frames,
            self.active_track_id,
            self.active_class_name,
            self,
        )
        if dlg.exec():
            s, e, scope = dlg.get_results()
            tid = self.active_track_id if scope == "track" else None
            cnt = self.annotation_manager.delete_interval(s, e, tid)
            self.redraw_annotations()
            self.status_bar.showMessage(f"Purged {cnt} annotations across interval [{s}, {e}].")

    def delete_active_box(self) -> None:
        self.history_manager.execute(RemoveAnnotationCommand(self.current_frame, self.active_track_id))
        self.redraw_annotations()

    def delete_entire_track(self) -> None:
        self._delete_track_by_id(self.active_track_id)

    def _delete_track_by_id(self, track_id: int) -> None:
        reply = QMessageBox.question(self, "Delete Track", f"Permanently delete Track #{track_id} across all frames?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.history_manager.execute(DeleteTrackCommand(track_id))
            self.redraw_annotations()

    def _split_track_dialog(self, track_id: int) -> None:
        try:
            res = self.annotation_manager.split_track(track_id, self.current_frame)
            self.redraw_annotations()
            QMessageBox.information(self, "Track Split", res.message)
        except Exception as e:
            QMessageBox.critical(self, "Split Error", str(e))

    def _merge_tracks_dialog(self, s_id: int, t_id: int) -> None:
        overlap = self.track_manager.detect_temporal_overlap(s_id, t_id)
        strategy = MergeStrategy.FAIL_ON_CONFLICT
        if overlap:
            dlg = MergeConflictDialog(s_id, t_id, overlap, self)
            if not dlg.exec():
                return
            res_strat = dlg.get_resolution_strategy()
            if res_strat is None:
                return
            strategy = res_strat

        try:
            res = self.annotation_manager.merge_tracks(s_id, t_id, strategy=strategy)
            self.redraw_annotations()
            QMessageBox.information(self, "Track Merge", res.message)
        except Exception as e:
            QMessageBox.critical(self, "Merge Error", str(e))

    def undo(self) -> None:
        cmd = self.history_manager.undo()
        if cmd:
            self.redraw_annotations()
            self.status_bar.showMessage(f"Undo: {cmd.description}")

    def redo(self) -> None:
        cmd = self.history_manager.redo()
        if cmd:
            self.redraw_annotations()
            self.status_bar.showMessage(f"Redo: {cmd.description}")

    def save_project(self) -> None:
        if not self.project_manager.project_path:
            self.save_project_as()
            return
        p = self.project_manager.save_project()
        self.status_bar.showMessage(f"Project saved: {p}")

    def save_project_as(self) -> None:
        f, _ = QFileDialog.getSaveFileName(self, "Save DLM Project", "project.dlm", "DarkLabel Project (*.dlm)")
        if f:
            p = self.project_manager.save_project(f)
            self.status_bar.showMessage(f"Project saved: {p}")

    def open_project(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Open DLM Project", "", "DarkLabel Project (*.dlm)")
        if f:
            self.project_manager.load_project(f)
            if self.project_manager.video_metadata.source_path:
                self.video_provider.open_source(self.project_manager.video_metadata.source_path)
            self.panel_classes.set_classes(self.project_manager.classes)
            self._sync_class_combos()
            self.redraw_annotations()
            self.status_bar.showMessage(f"Loaded project: {f}")

    def _check_for_crash_recovery(self) -> None:
        recoveries = self.project_manager.detect_available_recoveries()
        if recoveries:
            rec = recoveries[0]
            reply = QMessageBox.question(
                self,
                "Crash Recovery Available",
                f"An autosaved session from {rec.formatted_time} was recovered "
                f"({rec.track_count} tracks, {rec.frame_count} frames).\n\nRestore this session?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.project_manager.restore_recovery(rec)
                if self.project_manager.video_metadata.source_path:
                    self.video_provider.open_source(self.project_manager.video_metadata.source_path)
                self.redraw_annotations()
                self.status_bar.showMessage("Restored session from autosave recovery.")

    def _sync_class_combos(self) -> None:
        classes_map = {cid: entry.name for cid, entry in self.project_manager.classes.items()}
        self.panel_properties.update_classes(classes_map)

    def _on_class_catalog_changed(self) -> None:
        self._sync_class_combos()
        self.redraw_annotations()

    def _on_property_class_changed(self, track_id: int, cid: int, cname: str) -> None:
        self.active_class_id = cid
        self.active_class_name = cname
        with self.annotation_manager.lock:
            track = self.track_manager.get_track(track_id)
            if track:
                track.class_id = cid
                track.class_name = cname
                for b in track.annotations.values():
                    b.class_id = cid
                    b.class_name = cname
        self.redraw_annotations()

    def _on_property_lock_changed(self, track_id: int, locked: bool) -> None:
        with self.annotation_manager.lock:
            track = self.track_manager.get_track(track_id)
            if track:
                track.locked = locked
        self.redraw_annotations()

    def closeEvent(self, event) -> None:
        self.play_timer.stop()
        self.project_manager.stop_autosave_daemon()
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.cancel()
            self.batch_worker.wait()
        self.video_provider.release()
        event.accept()