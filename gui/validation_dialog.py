"""
gui/validation_dialog.py

Dataset Quality Assurance & Analytics Modal Window for DarkLabel Modern:
- Tab 1 (Quality Checker): Live issue auditor displaying Errors and Warnings.
  Double-clicking any issue row instantly seeks the canvas playhead to that frame and selects the track.
- Tab 2 (Dataset Analytics): Visual KPI cards and class distribution percentage bars.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.annotation_manager import AnnotationManager
from utils.dataset_validator import DatasetValidator, IssueSeverity, ValidationIssue, ValidationReport
from utils.statistics import ProjectStatistics, StatisticsCalculator


class QualityCheckerTab(QWidget):
    """Quality Checker tab displaying errors, warnings, and double-click seek."""

    jump_to_frame_requested = Signal(int, int)  # frame_index, track_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.report: Optional[ValidationReport] = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 1. Summary Cards Header
        hdr_layout = QHBoxLayout()

        self.lbl_status = QLabel("Status: Not Audited")
        self.lbl_status.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.lbl_status.setStyleSheet("color: #00ADB5;")

        self.lbl_errors = QLabel("Errors: 0")
        self.lbl_errors.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.lbl_errors.setStyleSheet("color: #FF5252;")

        self.lbl_warnings = QLabel("Warnings: 0")
        self.lbl_warnings.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.lbl_warnings.setStyleSheet("color: #FFA726;")

        hdr_layout.addWidget(self.lbl_status)
        hdr_layout.addSpacing(20)
        hdr_layout.addWidget(self.lbl_errors)
        hdr_layout.addSpacing(15)
        hdr_layout.addWidget(self.lbl_warnings)
        hdr_layout.addStretch()

        # Severity Filter Dropdown
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["All Issues", "Errors Only", "Warnings Only"])
        self.combo_filter.currentIndexChanged.connect(self._apply_filter)
        hdr_layout.addWidget(self.combo_filter)

        layout.addLayout(hdr_layout)

        # 2. Issues Table Widget
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Sev", "Frame", "Track", "Class", "Message", "Suggested Fix"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 45)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)

        layout.addWidget(self.table, stretch=1)

        hint = QLabel("<i>Tip: Double-click any row to jump canvas to that frame and select the track.</i>")
        hint.setStyleSheet("color: #909296;")
        layout.addWidget(hint)

    def display_report(self, report: ValidationReport) -> None:
        self.report = report
        self.lbl_errors.setText(f"Errors: {report.error_count}")
        self.lbl_warnings.setText(f"Warnings: {report.warning_count}")

        if report.error_count > 0:
            self.lbl_status.setText("Status: Issues Detected (Fix errors before export)")
            self.lbl_status.setStyleSheet("color: #FF5252; font-weight: bold;")
        elif report.warning_count > 0:
            self.lbl_status.setText("Status: Clean with Warnings")
            self.lbl_status.setStyleSheet("color: #FFA726; font-weight: bold;")
        else:
            self.lbl_status.setText("Status: Dataset 100% Clean")
            self.lbl_status.setStyleSheet("color: #00E676; font-weight: bold;")

        self._apply_filter()

    def _apply_filter(self) -> None:
        if not self.report:
            return

        filter_idx = self.combo_filter.currentIndex()
        if filter_idx == 1:
            issues = self.report.filter_by_severity(IssueSeverity.ERROR)
        elif filter_idx == 2:
            issues = self.report.filter_by_severity(IssueSeverity.WARNING)
        else:
            issues = self.report.issues

        self.table.setRowCount(len(issues))
        for r, iss in enumerate(issues):
            # Severity icon
            sev_item = QTableWidgetItem("ERR" if iss.severity == IssueSeverity.ERROR else "WARN")
            sev_item.setTextAlignment(Qt.AlignCenter)
            sev_item.setFont(QFont("Monospace", 8, QFont.Bold))
            sev_item.setForeground(QColor("#FF5252") if iss.severity == IssueSeverity.ERROR else QColor("#FFA726"))
            self.table.setItem(r, 0, sev_item)

            # Frame
            f_item = QTableWidgetItem(f"{iss.frame_index:,}")
            f_item.setTextAlignment(Qt.AlignCenter)
            f_item.setFont(QFont("Monospace", 9))
            self.table.setItem(r, 1, f_item)

            # Track
            t_item = QTableWidgetItem(f"#{iss.track_id}")
            t_item.setTextAlignment(Qt.AlignCenter)
            t_item.setFont(QFont("Monospace", 9))
            self.table.setItem(r, 2, t_item)

            # Class
            self.table.setItem(r, 3, QTableWidgetItem(iss.class_name))

            # Message & Fix
            self.table.setItem(r, 4, QTableWidgetItem(iss.message))
            self.table.setItem(r, 5, QTableWidgetItem(iss.suggested_fix))

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        frame_text = self.table.item(row, 1).text().replace(",", "")
        track_text = self.table.item(row, 2).text().replace("#", "")
        if frame_text.isdigit() and track_text.isdigit():
            self.jump_to_frame_requested.emit(int(frame_text), int(track_text))


class StatisticsTab(QWidget):
    """Dataset Analytics and Distribution KPI tab."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # 1. High-Level KPI Grid
        kpi_group = QGroupBox("Project Telemetry")
        kpi_grid = QGridLayout(kpi_group)

        self.kpi_boxes = self._create_card("Total Annotations", "0")
        self.kpi_tracks = self._create_card("Total Tracks", "0")
        self.kpi_frames = self._create_card("Annotated Frames", "0 / 0 (0%)")
        self.kpi_track_len = self._create_card("Avg Track Length", "0 frames")

        kpi_grid.addWidget(self.kpi_boxes, 0, 0)
        kpi_grid.addWidget(self.kpi_tracks, 0, 1)
        kpi_grid.addWidget(self.kpi_frames, 1, 0)
        kpi_grid.addWidget(self.kpi_track_len, 1, 1)
        layout.addWidget(kpi_group)

        # 2. Provenance Distribution
        prov_group = QGroupBox("Annotation Provenance & States")
        self.prov_form = QFormLayout(prov_group)
        self.lbl_manual_kf = QLabel("0")
        self.lbl_interpolated = QLabel("0")
        self.lbl_ai = QLabel("0")
        self.lbl_tracker = QLabel("0")
        self.lbl_occluded = QLabel("0")

        self.prov_form.addRow("Manual Keyframes:", self.lbl_manual_kf)
        self.prov_form.addRow("Linear Interpolated:", self.lbl_interpolated)
        self.prov_form.addRow("AI (YOLO / ByteTrack):", self.lbl_ai)
        self.prov_form.addRow("Visual Tracker (CSRT):", self.lbl_tracker)
        self.prov_form.addRow("Occluded Frames:", self.lbl_occluded)
        layout.addWidget(prov_group)

        # 3. Class Distribution Bars
        self.class_group = QGroupBox("Class Distribution")
        self.class_layout = QVBoxLayout(self.class_group)
        layout.addWidget(self.class_group)

        layout.addStretch()
        scroll.setWidget(container)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _create_card(self, title: str, default_val: str) -> QWidget:
        box = QWidget()
        vl = QVBoxLayout(box)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.setSpacing(2)

        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("color: #909296; font-size: 11px;")
        lbl_v = QLabel(default_val)
        lbl_v.setFont(QFont("Segoe UI", 12, QFont.Bold))
        lbl_v.setStyleSheet("color: #00ADB5;")

        vl.addWidget(lbl_t)
        vl.addWidget(lbl_v)
        box.val_label = lbl_v  # Dynamic reference
        return box

    def display_statistics(self, stats: ProjectStatistics) -> None:
        self.kpi_boxes.val_label.setText(f"{stats.total_annotations:,}")
        self.kpi_tracks.val_label.setText(f"{stats.total_tracks:,}")
        self.kpi_frames.val_label.setText(
            f"{stats.annotated_frames:,} / {stats.total_video_frames:,} ({stats.annotation_coverage_pct:.1f}%)"
        )
        self.kpi_track_len.val_label.setText(
            f"{stats.avg_track_len:.1f} frames (Max: {stats.max_track_len:,})"
        )

        self.lbl_manual_kf.setText(f"{stats.manual_keyframes:,}")
        self.lbl_interpolated.setText(f"{stats.interpolated_frames:,}")
        self.lbl_ai.setText(f"{stats.ai_generated:,}")
        self.lbl_tracker.setText(f"{stats.tracker_generated:,}")
        self.lbl_occluded.setText(f"{stats.occluded_count:,}")

        # Build Class Progress Bars
        while self.class_layout.count():
            item = self.class_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for cname, count in sorted(stats.class_counts.items(), key=lambda x: x[1], reverse=True):
            pct = stats.class_percentages.get(cname, 0.0)
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 1, 0, 1)

            lbl_name = QLabel(f"{cname} ({count:,})")
            lbl_name.setFixedWidth(130)

            pbar = QProgressBar()
            pbar.setRange(0, 100)
            pbar.setValue(int(pct))
            pbar.setFormat(f"{pct:.1f}%")
            pbar.setFixedHeight(14)

            rl.addWidget(lbl_name)
            rl.addWidget(pbar)
            self.class_layout.addWidget(row)


class ValidationDialog(QDialog):
    """
    Combined Quality Checker & Analytics Dialog.
    """

    jump_to_frame_requested = Signal(int, int)  # frame_idx, track_id

    def __init__(
        self,
        manager: AnnotationManager,
        img_width: int,
        img_height: int,
        total_frames: int,
        fps: float,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Dataset Quality Assurance & Analytics Studio")
        self.resize(880, 560)

        self.manager = manager
        self.width = img_width
        self.height = img_height
        self.total_frames = total_frames
        self.fps = fps

        self.validator = DatasetValidator()

        self._init_ui()
        self.run_audit()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.tab_qc = QualityCheckerTab()
        self.tab_qc.jump_to_frame_requested.connect(self._on_jump_requested)

        self.tab_stats = StatisticsTab()

        self.tabs.addTab(self.tab_qc, "Dataset Quality Checker")
        self.tabs.addTab(self.tab_stats, "Dataset Statistics & Analytics")
        layout.addWidget(self.tabs)

        # Bottom Buttons
        btn_bar = QHBoxLayout()
        btn_recheck = QPushButton("Re-Run Audit")
        btn_recheck.clicked.connect(self.run_audit)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)

        btn_bar.addWidget(btn_recheck)
        btn_bar.addStretch()
        btn_bar.addWidget(btn_close)
        layout.addLayout(btn_bar)

    def run_audit(self) -> None:
        """Executes validator and statistics calculator in memory."""
        # 1. Run Validation
        valid_classes = set(self.manager.track_manager.tracks[t].class_id for t in self.manager.track_manager.tracks)
        report = self.validator.validate(
            manager=self.manager,
            img_width=self.width,
            img_height=self.height,
            total_frames=self.total_frames,
            valid_class_ids=valid_classes or None
        )
        self.tab_qc.display_report(report)

        # 2. Run Analytics
        stats = StatisticsCalculator.compute(
            manager=self.manager,
            total_video_frames=self.total_frames,
            fps=self.fps
        )
        self.tab_stats.display_statistics(stats)

    def _on_jump_requested(self, frame_idx: int, track_id: int) -> None:
        self.jump_to_frame_requested.emit(frame_idx, track_id)
        # Note: Keeps the dialog open or closes based on preference.