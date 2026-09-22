"""
gui/properties_panel.py

Detailed object properties inspector for DarkLabel Modern:
- Inspects identity, class assignment, bounding box geometry, and confidence.
- Interactive state toggles: Occluded checkbox, Outside checkbox, Locked checkbox.
- Dynamic attributes sub-form generating appropriate controls (QCheckBox, QSpinBox,
  QDoubleSpinBox, QLineEdit, QComboBox) based on catalog schemas and object metadata.
- Temporal navigation and trajectory management triggers (Jump to Start/End, Prev/Next KF,
  Split Track, Terminate Tail, Delete Track).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.annotation_manager import AnnotationManager
from core.annotation_models import Annotation, AnnotationSource, Track


class PropertiesPanel(QWidget):
    """
    Dynamic properties inspector editing attributes and states for the active track.
    """

    class_changed = Signal(int, int, str)              # track_id, new_class_id, new_class_name
    occlusion_toggled = Signal(int, bool)             # track_id, is_occluded
    outside_toggled = Signal(int, bool)               # track_id, is_outside
    lock_toggled = Signal(int, bool)                  # track_id, is_locked
    attribute_changed = Signal(int, str, object)       # track_id, key, val
    jump_frame_requested = Signal(int)                # target_frame
    split_track_requested = Signal(int)               # track_id
    terminate_tail_requested = Signal(int)            # track_id
    delete_track_requested = Signal(int)              # track_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.annotation_manager: Optional[AnnotationManager] = None
        self.active_track_id: Optional[int] = None
        self.current_frame: int = 0
        self._attribute_widgets: Dict[str, QWidget] = {}

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.content_layout = QVBoxLayout(container)
        self.content_layout.setContentsMargins(4, 4, 4, 4)
        self.content_layout.setSpacing(8)

        # 1. Identity & Classification Box
        ident_group = QGroupBox("Target Identity & State")
        self.id_form = QFormLayout(ident_group)

        self.lbl_id = QLabel("None")
        self.lbl_id.setStyleSheet("font-weight: bold; color: #00ADB5;")

        self.combo_class = QComboBox()
        self.combo_class.currentIndexChanged.connect(self._on_class_combo_changed)

        self.chk_locked = QCheckBox("Locked (Prevent edits)")
        self.chk_locked.stateChanged.connect(self._on_lock_changed)

        self.chk_occluded = QCheckBox("Occluded (Partially hidden)")
        self.chk_occluded.stateChanged.connect(self._on_occluded_changed)

        self.chk_outside = QCheckBox("Outside (Invisible / Departed)")
        self.chk_outside.stateChanged.connect(self._on_outside_changed)

        self.id_form.addRow("Track ID:", self.lbl_id)
        self.id_form.addRow("Class:", self.combo_class)
        self.id_form.addRow("", self.chk_locked)
        self.id_form.addRow("", self.chk_occluded)
        self.id_form.addRow("", self.chk_outside)
        self.content_layout.addWidget(ident_group)

        # 2. Geometry & Temporal Telemetry
        geom_group = QGroupBox("Frame Metrics & Source")
        self.geom_form = QFormLayout(geom_group)

        self.lbl_coords = QLabel("x: 0, y: 0, w: 0, h: 0")
        self.lbl_conf = QLabel("1.00")
        self.lbl_source = QLabel("MANUAL")
        self.lbl_span = QLabel("Frame 0 → 0 (0 frames)")

        self.geom_form.addRow("Box Bounding:", self.lbl_coords)
        self.geom_form.addRow("Confidence:", self.lbl_conf)
        self.geom_form.addRow("Provenance:", self.lbl_source)
        self.geom_form.addRow("Track Span:", self.lbl_span)
        self.content_layout.addWidget(geom_group)

        # 3. Dynamic Attributes Sub-Form
        self.attr_group = QGroupBox("Custom Attributes")
        self.attr_form = QFormLayout(self.attr_group)
        self.lbl_no_attrs = QLabel("No attributes defined.")
        self.lbl_no_attrs.setStyleSheet("color: #6C757D;")
        self.attr_form.addRow(self.lbl_no_attrs)
        self.content_layout.addWidget(self.attr_group)

        # 4. Temporal Navigation Buttons
        nav_group = QGroupBox("Temporal Navigation")
        nav_layout = QVBoxLayout(nav_group)

        nav_row1 = QHBoxLayout()
        self.btn_start = QPushButton("⏮ Jump to Start")
        self.btn_start.clicked.connect(self._jump_to_start)
        self.btn_end = QPushButton("Jump to End ⏭")
        self.btn_end.clicked.connect(self._jump_to_end)
        nav_row1.addWidget(self.btn_start)
        nav_row1.addWidget(self.btn_end)

        nav_row2 = QHBoxLayout()
        self.btn_prev_kf = QPushButton("◀ Prev KF")
        self.btn_prev_kf.clicked.connect(self._jump_prev_kf)
        self.btn_next_kf = QPushButton("Next KF ▶")
        self.btn_next_kf.clicked.connect(self._jump_next_kf)
        nav_row2.addWidget(self.btn_prev_kf)
        nav_row2.addWidget(self.btn_next_kf)

        nav_layout.addLayout(nav_row1)
        nav_layout.addLayout(nav_row2)
        self.content_layout.addWidget(nav_group)

        # 5. Trajectory Actions
        act_group = QGroupBox("Trajectory Operations")
        act_layout = QVBoxLayout(act_group)

        self.btn_split = QPushButton("Split Track at Current Frame")
        self.btn_split.clicked.connect(self._split_track)

        self.btn_del_tail = QPushButton("Delete Tail (N+1 → End)")
        self.btn_del_tail.clicked.connect(self._delete_tail)

        self.btn_del_track = QPushButton("Delete Entire Track")
        self.btn_del_track.setStyleSheet("color: #FF5252;")
        self.btn_del_track.clicked.connect(self._delete_track)

        act_layout.addWidget(self.btn_split)
        act_layout.addWidget(self.btn_del_tail)
        act_layout.addWidget(self.btn_del_track)
        self.content_layout.addWidget(act_group)

        self.content_layout.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def set_manager(self, manager: AnnotationManager) -> None:
        self.annotation_manager = manager
        self.refresh()

    def update_classes(self, class_list: Dict[int, str]) -> None:
        """Populates class selector combo box with available categories."""
        self.combo_class.blockSignals(True)
        self.combo_class.clear()
        for cid, name in sorted(class_list.items()):
            self.combo_class.addItem(name, cid)
        self.combo_class.blockSignals(False)

    def set_target(self, track_id: Optional[int], current_frame: int) -> None:
        self.active_track_id = track_id
        self.current_frame = current_frame
        self.refresh()

    def refresh(self) -> None:
        """Synchronizes controls with the active track and frame annotation."""
        if not self.annotation_manager or self.active_track_id is None:
            self._set_empty_state()
            return

        with self.annotation_manager.lock:
            track = self.annotation_manager.track_manager.get_track(self.active_track_id)
            box = self.annotation_manager.get_annotation(self.current_frame, self.active_track_id)

        if not track:
            self._set_empty_state()
            return

        self.setEnabled(True)
        self.lbl_id.setText(f"Track #{track.track_id}")

        # Update Class combo
        self.combo_class.blockSignals(True)
        idx = self.combo_class.findData(track.class_id)
        if idx >= 0:
            self.combo_class.setCurrentIndex(idx)
        else:
            self.combo_class.setCurrentText(track.class_name)
        self.combo_class.blockSignals(False)

        # Update Lock
        self.chk_locked.blockSignals(True)
        self.chk_locked.setChecked(track.locked)
        self.chk_locked.blockSignals(False)

        # Update Track Span
        s_f = track.start_frame if track.start_frame is not None else 0
        e_f = track.end_frame if track.end_frame is not None else 0
        self.lbl_span.setText(f"Frame {s_f:,} → {e_f:,} ({track.total_frames:,} frames)")

        # Update Frame Specific metrics
        if box:
            self.lbl_coords.setText(f"x:{box.x:.1f}, y:{box.y:.1f}, w:{box.width:.1f}, h:{box.height:.1f}")
            self.lbl_conf.setText(f"{box.confidence:.2f}")
            prov = {
                AnnotationSource.MANUAL: "Manual Keyframe" if box.is_keyframe else "Manual Hold",
                AnnotationSource.AI: "YOLO Inference",
                AnnotationSource.TRACKER: "Visual Tracker (CSRT)",
                AnnotationSource.INTERPOLATED: "Linear Interpolated",
            }.get(box.source, str(box.source.value))
            self.lbl_source.setText(prov)

            self.chk_occluded.blockSignals(True)
            self.chk_occluded.setChecked(box.occluded)
            self.chk_occluded.blockSignals(False)

            self.chk_outside.blockSignals(True)
            self.chk_outside.setChecked(box.outside)
            self.chk_outside.blockSignals(False)

            self._build_dynamic_attributes(box.attributes)
        else:
            self.lbl_coords.setText("No box on current frame")
            self.lbl_conf.setText("–")
            self.lbl_source.setText("–")
            self._build_dynamic_attributes(track.attributes)

    def _set_empty_state(self) -> None:
        self.lbl_id.setText("None")
        self.lbl_coords.setText("–")
        self.lbl_conf.setText("–")
        self.lbl_source.setText("–")
        self.lbl_span.setText("–")
        self.setEnabled(False)

    def _build_dynamic_attributes(self, attributes: Dict[str, Any]) -> None:
        """Constructs appropriate input controls for custom key-value attributes."""
        # Clear existing dynamic widgets
        while self.attr_form.rowCount() > 0:
            self.attr_form.removeRow(0)
        self._attribute_widgets.clear()

        if not attributes:
            self.attr_form.addRow(QLabel("No attributes defined."))
            return

        for key, val in attributes.items():
            if isinstance(val, bool):
                widget = QCheckBox()
                widget.setChecked(val)
                widget.stateChanged.connect(lambda s, k=key: self._on_attr_changed(k, s == Qt.Checked))
            elif isinstance(val, int):
                widget = QSpinBox()
                widget.setRange(-999999, 999999)
                widget.setValue(val)
                widget.valueChanged.connect(lambda v, k=key: self._on_attr_changed(k, v))
            elif isinstance(val, float):
                widget = QDoubleSpinBox()
                widget.setRange(-999999.0, 999999.0)
                widget.setValue(val)
                widget.valueChanged.connect(lambda v, k=key: self._on_attr_changed(k, v))
            else:
                widget = QLineEdit(str(val))
                widget.textChanged.connect(lambda t, k=key: self._on_attr_changed(k, t))

            self._attribute_widgets[key] = widget
            self.attr_form.addRow(f"{key}:", widget)

    def _on_class_combo_changed(self, index: int) -> None:
        if self.active_track_id is None:
            return
        cid = self.combo_class.currentData()
        name = self.combo_class.currentText()
        self.class_changed.emit(self.active_track_id, cid if cid is not None else 0, name)

    def _on_lock_changed(self, state: int) -> None:
        if self.active_track_id is not None:
            self.lock_toggled.emit(self.active_track_id, state == Qt.Checked)

    def _on_occluded_changed(self, state: int) -> None:
        if self.active_track_id is not None:
            self.occlusion_toggled.emit(self.active_track_id, state == Qt.Checked)

    def _on_outside_changed(self, state: int) -> None:
        if self.active_track_id is not None:
            self.outside_toggled.emit(self.active_track_id, state == Qt.Checked)

    def _on_attr_changed(self, key: str, val: Any) -> None:
        if self.active_track_id is not None:
            self.attribute_changed.emit(self.active_track_id, key, val)

    def _jump_to_start(self) -> None:
        if not self.annotation_manager or self.active_track_id is None:
            return
        track = self.annotation_manager.track_manager.get_track(self.active_track_id)
        if track and track.start_frame is not None:
            self.jump_frame_requested.emit(track.start_frame)

    def _jump_to_end(self) -> None:
        if not self.annotation_manager or self.active_track_id is None:
            return
        track = self.annotation_manager.track_manager.get_track(self.active_track_id)
        if track and track.end_frame is not None:
            self.jump_frame_requested.emit(track.end_frame)

    def _jump_prev_kf(self) -> None:
        if not self.annotation_manager or self.active_track_id is None:
            return
        track = self.annotation_manager.track_manager.get_track(self.active_track_id)
        if track:
            pkf = track.find_prev_keyframe(self.current_frame)
            if pkf is not None:
                self.jump_frame_requested.emit(pkf)

    def _jump_next_kf(self) -> None:
        if not self.annotation_manager or self.active_track_id is None:
            return
        track = self.annotation_manager.track_manager.get_track(self.active_track_id)
        if track:
            nkf = track.find_next_keyframe(self.current_frame)
            if nkf is not None:
                self.jump_frame_requested.emit(nkf)

    def _split_track(self) -> None:
        if self.active_track_id is not None:
            self.split_track_requested.emit(self.active_track_id)

    def _delete_tail(self) -> None:
        if self.active_track_id is not None:
            self.terminate_tail_requested.emit(self.active_track_id)

    def _delete_track(self) -> None:
        if self.active_track_id is not None:
            self.delete_track_requested.emit(self.active_track_id)