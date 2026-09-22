"""
gui/dialogs.py

Specialized Modal Dialogs for DarkLabel Modern:
- IntervalPurgeDialog: Precise frame range purging scoped to Selected Track, Class, or All Objects.
- MergeConflictDialog: Interactive resolution for temporal track collisions (Fail, Prefer Target, Prefer Source).
- SettingsDialog: Global studio preferences (Theme, Cadence, YOLO thresholds, RAM cache, Autosave).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.track_manager import MergeStrategy


class IntervalPurgeDialog(QDialog):
    """
    Dialog allowing surgical deletion of annotations within [start_frame, end_frame]
    scoped by active track, active class, or all objects.
    """

    def __init__(
        self,
        current_frame: int,
        total_frames: int,
        active_track_id: Optional[int],
        active_class_name: str,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Purge Annotation Interval")
        self.setFixedWidth(380)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, max(0, total_frames - 1))
        self.spin_start.setValue(current_frame)

        self.spin_end = QSpinBox()
        self.spin_end.setRange(0, max(0, total_frames - 1))
        self.spin_end.setValue(min(current_frame + 30, max(0, total_frames - 1)))

        form.addRow("Start Frame:", self.spin_start)
        form.addRow("End Frame:", self.spin_end)
        layout.addLayout(form)

        # Scope Selection
        scope_group = QGroupBox("Purge Scope")
        scope_layout = QVBoxLayout(scope_group)

        self.rad_track = QRadioButton(f"Selected Track only (#{active_track_id or 'None'})")
        self.rad_class = QRadioButton(f"All objects of Class '{active_class_name}'")
        self.rad_all = QRadioButton("All objects across all classes")

        self.rad_track.setChecked(True)
        if active_track_id is None:
            self.rad_track.setEnabled(False)
            self.rad_all.setChecked(True)

        scope_layout.addWidget(self.rad_track)
        scope_layout.addWidget(self.rad_class)
        scope_layout.addWidget(self.rad_all)
        layout.addWidget(scope_group)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_results(self) -> Tuple[int, int, str]:
        """Returns (start_frame, end_frame, scope: 'track' | 'class' | 'all')."""
        scope = "all"
        if self.rad_track.isChecked():
            scope = "track"
        elif self.rad_class.isChecked():
            scope = "class"
        return (self.spin_start.value(), self.spin_end.value(), scope)


class MergeConflictDialog(QDialog):
    """
    Modal conflict resolver displayed when two tracks contain overlapping annotations.
    """

    def __init__(
        self,
        source_id: int,
        target_id: int,
        conflicting_frames: List[int],
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Track Merge Conflict Detected")
        self.setFixedWidth(440)

        layout = QVBoxLayout(self)

        lbl_desc = QLabel(
            f"<b>Temporal collision detected!</b><br>"
            f"Track #{source_id} and Track #{target_id} both contain annotations on "
            f"<b>{len(conflicting_frames)}</b> identical frame(s).<br><br>"
            f"<i>Conflicting frame range: {min(conflicting_frames)} → {max(conflicting_frames)}</i><br>"
            f"Select a resolution policy:"
        )
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        policy_group = QGroupBox("Conflict Policy")
        p_layout = QVBoxLayout(policy_group)

        self.rad_target = QRadioButton(f"Prefer Target (#{target_id}) - Keep target boxes, drop source conflicts")
        self.rad_source = QRadioButton(f"Prefer Source (#{source_id}) - Overwrite target boxes with source")
        self.rad_abort = QRadioButton("Abort Merge (Cancel operation)")

        self.rad_target.setChecked(True)
        p_layout.addWidget(self.rad_target)
        p_layout.addWidget(self.rad_source)
        p_layout.addWidget(self.rad_abort)
        layout.addWidget(policy_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_resolution_strategy(self) -> Optional[MergeStrategy]:
        """Returns selected MergeStrategy or None if aborted."""
        if self.rad_abort.isChecked():
            return None
        if self.rad_source.isChecked():
            return MergeStrategy.PREFER_SOURCE
        return MergeStrategy.PREFER_TARGET


class SettingsDialog(QDialog):
    """
    Application-wide studio preferences and performance configuration.
    """

    def __init__(self, current_settings: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Studio Settings & Preferences")
        self.resize(520, 420)
        self.settings = dict(current_settings)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # Tab 1: Workflow & Cadence
        tab_wf = QWidget()
        wf_form = QFormLayout(tab_wf)
        self.spin_cadence = QSpinBox()
        self.spin_cadence.setRange(1, 60)
        self.spin_cadence.setValue(int(self.settings.get("cadence_fps", 5)))
        self.spin_cadence.setSuffix(" FPS")

        self.chk_auto_interp = QCheckBox("Auto-interpolate between cadence steps")
        self.chk_auto_interp.setChecked(bool(self.settings.get("auto_interpolate", True)))

        self.chk_forward_hold = QCheckBox("Enable forward-hold box persistence")
        self.chk_forward_hold.setChecked(bool(self.settings.get("forward_hold", True)))

        wf_form.addRow("Default Cadence Rate:", self.spin_cadence)
        wf_form.addRow("", self.chk_auto_interp)
        wf_form.addRow("", self.chk_forward_hold)
        tabs.addTab(tab_wf, "Annotation & Cadence")

        # Tab 2: AI & Inference
        tab_ai = QWidget()
        ai_form = QFormLayout(tab_ai)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setValue(float(self.settings.get("confidence_threshold", 0.25)))
        self.spin_conf.setSingleStep(0.05)

        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.01, 1.0)
        self.spin_iou.setValue(float(self.settings.get("iou_threshold", 0.45)))
        self.spin_iou.setSingleStep(0.05)

        self.combo_device = QComboBox()
        self.combo_device.addItems(["Auto Detect", "CUDA (NVIDIA)", "Apple MPS", "CPU"])
        cur_dev = self.settings.get("preferred_device", "Auto Detect")
        self.combo_device.setCurrentText(cur_dev)

        ai_form.addRow("Default Confidence:", self.spin_conf)
        ai_form.addRow("Default NMS IoU:", self.spin_iou)
        ai_form.addRow("Preferred Device:", self.combo_device)
        tabs.addTab(tab_ai, "AI & Tracker")

        # Tab 3: Memory & Autosave
        tab_mem = QWidget()
        mem_form = QFormLayout(tab_mem)
        self.spin_cache = QSpinBox()
        self.spin_cache.setRange(128, 4096)
        self.spin_cache.setValue(int(self.settings.get("cache_size_mb", 512)))
        self.spin_cache.setSuffix(" MB")

        self.spin_autosave = QSpinBox()
        self.spin_autosave.setRange(10, 600)
        self.spin_autosave.setValue(int(self.settings.get("autosave_interval_sec", 60)))
        self.spin_autosave.setSuffix(" sec")

        mem_form.addRow("RAM Video Cache Size:", self.spin_cache)
        mem_form.addRow("Autosave Frequency:", self.spin_autosave)
        tabs.addTab(tab_mem, "System & Cache")

        layout.addWidget(tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self) -> Dict[str, Any]:
        """Returns the updated preferences dictionary."""
        return {
            "cadence_fps": self.spin_cadence.value(),
            "auto_interpolate": self.chk_auto_interp.isChecked(),
            "forward_hold": self.chk_forward_hold.isChecked(),
            "confidence_threshold": self.spin_conf.value(),
            "iou_threshold": self.spin_iou.value(),
            "preferred_device": self.combo_device.currentText(),
            "cache_size_mb": self.spin_cache.value(),
            "autosave_interval_sec": self.spin_autosave.value(),
        }