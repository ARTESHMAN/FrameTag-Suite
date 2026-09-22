"""
gui/export_dialog.py

Integrated Export Studio Modal for DarkLabel Modern:
- Unified export interface managing:
    * YOLO Detection (*.txt) & YOLO Track (*.txt)
    * MOT Challenge (gt.txt)
    * MS COCO Instances (*.json)
    * Pascal VOC (*.xml)
    * CSV Table (*.csv)
    * Sampled Dataset with Train/Val/Test Splits & data.yaml
    * Privacy-Blurred Video (*.mp4)
    * Annotated Overlay / Verification Video (*.mp4)
    * Temporal Sub-Clip (*.mp4)
- Pre-Export Quality Assurance Check: runs DatasetValidator, displaying error/warning tallies.
- Asynchronous QThread ExportWorker keeping GUI responsive during heavy serialization.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Set

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.annotation_manager import AnnotationManager
from utils.coco_io import export_coco_json
from utils.dataset_splitter import export_partitioned_yolo_dataset
from utils.dataset_validator import DatasetValidator, ValidationReport
from utils.io_handlers import export_csv
from utils.mot_io import export_mot_challenge
from utils.privacy import PrivacyMode, render_privacy_video
from utils.video_utils import extract_subclip, render_annotated_video
from utils.voc_io import export_pascal_voc
from utils.yolo_io import export_yolo_txt


class ExportWorker(QThread):
    """
    Dedicated background thread executing heavy serialization and video encoding.
    """

    progress = Signal(int, int, float)  # current, total, processing_fps
    finished = Signal(bool, str)        # success, message
    error = Signal(str)

    def __init__(
        self,
        task_type: str,
        params: Dict[str, Any],
        manager: AnnotationManager,
        video_provider
    ):
        super().__init__()
        self.task_type = task_type
        self.params = params
        self.manager = manager
        self.video_provider = video_provider
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def _cancel_check(self) -> bool:
        return self._is_cancelled

    def _progress_cb(self, cur: int, tot: int, fps: float = 0.0) -> None:
        self.progress.emit(cur, tot, fps)

    def run(self) -> None:
        try:
            p = self.params
            tt = self.task_type

            if tt == "yolo":
                cnt = export_yolo_txt(
                    manager=self.manager,
                    img_width=self.video_provider.width,
                    img_height=self.video_provider.height,
                    out_dir=p["out_path"],
                    class_to_id=p["class_to_id"],
                    include_track_id=p.get("include_track_id", False),
                    progress_callback=self._progress_cb
                )
                self.finished.emit(True, f"Successfully exported {cnt} YOLO label files.")

            elif tt == "mot":
                cnt = export_mot_challenge(self.manager, p["out_path"])
                self.finished.emit(True, f"Successfully exported {cnt} MOT annotations to gt.txt.")

            elif tt == "coco":
                cnt = export_coco_json(
                    manager=self.manager,
                    img_width=self.video_provider.width,
                    img_height=self.video_provider.height,
                    out_path=p["out_path"],
                    class_to_id=p["class_to_id"]
                )
                self.finished.emit(True, f"Successfully exported {cnt} COCO instances to JSON.")

            elif tt == "voc":
                cnt = export_pascal_voc(
                    manager=self.manager,
                    img_width=self.video_provider.width,
                    img_height=self.video_provider.height,
                    out_dir=p["out_path"]
                )
                self.finished.emit(True, f"Successfully exported {cnt} Pascal VOC XML files.")

            elif tt == "csv":
                export_csv(self.manager, p["out_path"])
                self.finished.emit(True, f"Exported annotations spreadsheet to {p['out_path']}.")

            elif tt == "sampled":
                counts = export_partitioned_yolo_dataset(
                    video_provider=self.video_provider,
                    manager=self.manager,
                    output_dir=p["out_path"],
                    class_to_id=p["class_to_id"],
                    train_ratio=p["train_ratio"],
                    val_ratio=p["val_ratio"],
                    test_ratio=p["test_ratio"]
                )
                tot = sum(counts.values())
                msg = f"Exported {tot} frames (Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']})."
                self.finished.emit(True, msg)

            elif tt == "privacy":
                success, cnt = render_privacy_video(
                    video_provider=self.video_provider,
                    manager=self.manager,
                    out_path=p["out_path"],
                    mode=p["mode"],
                    start_frame=p["start_frame"],
                    end_frame=p["end_frame"],
                    progress_callback=self._progress_cb,
                    cancel_check=self._cancel_check
                )
                if success:
                    self.finished.emit(True, f"Anonymized video exported ({cnt} frames processed).")
                else:
                    self.finished.emit(False, "Privacy export was canceled or failed.")

            elif tt == "overlay":
                success, cnt = render_annotated_video(
                    video_provider=self.video_provider,
                    manager=self.manager,
                    out_path=p["out_path"],
                    start_frame=p["start_frame"],
                    end_frame=p["end_frame"],
                    show_ids=p.get("show_ids", True),
                    show_labels=p.get("show_labels", True),
                    show_conf=p.get("show_conf", True),
                    show_trajectory=p.get("show_trajectory", True),
                    progress_callback=self._progress_cb,
                    cancel_check=self._cancel_check
                )
                if success:
                    self.finished.emit(True, f"Overlay verification video exported ({cnt} frames).")
                else:
                    self.finished.emit(False, "Overlay video rendering was canceled or failed.")

            elif tt == "subclip":
                success, cnt = extract_subclip(
                    video_provider=self.video_provider,
                    start_frame=p["start_frame"],
                    end_frame=p["end_frame"],
                    out_path=p["out_path"],
                    progress_callback=self._progress_cb,
                    cancel_check=self._cancel_check
                )
                if success:
                    self.finished.emit(True, f"Extracted {cnt} frames to clip.")
                else:
                    self.finished.emit(False, "Clip extraction was canceled or failed.")

        except Exception as e:
            self.error.emit(str(e))


class ExportDialog(QDialog):
    """
    Master Export Studio Dialog with built-in QA check and progress tracking.
    """

    def __init__(
        self,
        manager: AnnotationManager,
        video_provider,
        class_catalog: Dict[str, int],
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Export & Media Rendering Studio")
        self.resize(680, 520)

        self.manager = manager
        self.video_provider = video_provider
        self.class_catalog = class_catalog

        self.worker: Optional[ExportWorker] = None

        self._init_ui()
        self._run_pre_export_qa()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # 1. Pre-Export QA Banner
        self.qa_box = QGroupBox("Pre-Export Dataset Health Check")
        qa_layout = QHBoxLayout(self.qa_box)
        self.lbl_qa_status = QLabel("Auditing dataset...")
        self.lbl_qa_status.setStyleSheet("color: #00ADB5; font-weight: bold;")
        btn_recheck = QPushButton("Re-Check")
        btn_recheck.setFixedHeight(22)
        btn_recheck.clicked.connect(self._run_pre_export_qa)

        qa_layout.addWidget(self.lbl_qa_status)
        qa_layout.addStretch()
        qa_layout.addWidget(btn_recheck)
        layout.addWidget(self.qa_box)

        # 2. Export Categories Tabs
        self.tabs = QTabWidget()

        # Tab A: Computer Vision & ML Datasets
        tab_ml = QWidget()
        ml_layout = QVBoxLayout(tab_ml)

        self.btn_group_fmt = QButtonGroup(self)
        self.rad_yolo = QRadioButton("YOLO Detection (*.txt)")
        self.rad_yolo_track = QRadioButton("YOLO Track (*.txt with track IDs)")
        self.rad_mot = QRadioButton("MOT Challenge (gt.txt)")
        self.rad_coco = QRadioButton("MS COCO Instances (*.json)")
        self.rad_voc = QRadioButton("Pascal VOC (*.xml)")
        self.rad_csv = QRadioButton("Tabular CSV (*.csv)")
        self.rad_sampled = QRadioButton("Sampled Dataset (Images + Labels with Train/Val/Test Split)")

        self.rad_yolo.setChecked(True)
        for r in [self.rad_yolo, self.rad_yolo_track, self.rad_mot, self.rad_coco, self.rad_voc, self.rad_csv, self.rad_sampled]:
            self.btn_group_fmt.addButton(r)
            ml_layout.addWidget(r)

        # Sampling split controls
        self.grp_split = QGroupBox("Dataset Splits (Sampled Dataset only)")
        sp_form = QFormLayout(self.grp_split)
        self.spin_train = QSpinBox()
        self.spin_train.setRange(10, 100)
        self.spin_train.setValue(70)
        self.spin_train.setSuffix("%")

        self.spin_val = QSpinBox()
        self.spin_val.setRange(0, 100)
        self.spin_val.setValue(20)
        self.spin_val.setSuffix("%")

        self.spin_test = QSpinBox()
        self.spin_test.setRange(0, 100)
        self.spin_test.setValue(10)
        self.spin_test.setSuffix("%")

        sp_form.addRow("Train:", self.spin_train)
        sp_form.addRow("Validation:", self.spin_val)
        sp_form.addRow("Test:", self.spin_test)
        ml_layout.addWidget(self.grp_split)

        self.tabs.addTab(tab_ml, "ML Training Formats")

        # Tab B: Video & Media Processing
        tab_media = QWidget()
        m_layout = QVBoxLayout(tab_media)

        self.btn_group_media = QButtonGroup(self)
        self.rad_privacy = QRadioButton("Privacy Anonymization Video")
        self.rad_overlay = QRadioButton("Burned-In Verification Overlay Video")
        self.rad_subclip = QRadioButton("Extract Sub-Clip (Clean Video)")

        self.rad_privacy.setChecked(True)
        for r in [self.rad_privacy, self.rad_overlay, self.rad_subclip]:
            self.btn_group_media.addButton(r)
            m_layout.addWidget(r)

        # Privacy options
        p_grp = QGroupBox("Privacy Mode")
        p_form = QFormLayout(p_grp)
        self.combo_priv_mode = QComboBox()
        self.combo_priv_mode.addItems(["Gaussian Blur", "Pixelated Mosaic", "Solid Blackout"])
        p_form.addRow("Method:", self.combo_priv_mode)
        m_layout.addWidget(p_grp)

        # Frame Range
        range_grp = QGroupBox("Temporal Frame Range")
        r_form = QFormLayout(range_grp)
        self.spin_start_f = QSpinBox()
        self.spin_start_f.setRange(0, max(0, self.video_provider.total_frames - 1))
        self.spin_start_f.setValue(0)

        self.spin_end_f = QSpinBox()
        self.spin_end_f.setRange(0, max(0, self.video_provider.total_frames - 1))
        self.spin_end_f.setValue(max(0, self.video_provider.total_frames - 1))

        r_form.addRow("Start Frame:", self.spin_start_f)
        r_form.addRow("End Frame:", self.spin_end_f)
        m_layout.addWidget(range_grp)

        self.tabs.addTab(tab_media, "Video & Privacy Rendering")
        layout.addWidget(self.tabs)

        # 3. Destination File/Folder Picker
        dest_box = QGroupBox("Output Destination")
        d_layout = QHBoxLayout(dest_box)
        self.txt_dest = QLineEdit()
        self.txt_dest.setPlaceholderText("Select file or folder path...")
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_destination)
        d_layout.addWidget(self.txt_dest)
        d_layout.addWidget(btn_browse)
        layout.addWidget(dest_box)

        # 4. Progress Bar HUD
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        self.prog_bar.setVisible(False)
        layout.addWidget(self.prog_bar)

        # 5. Bottom Action Buttons
        btn_bar = QHBoxLayout()
        self.btn_start = QPushButton("Start Export")
        self.btn_start.setStyleSheet("background-color: #00ADB5; color: white; font-weight: bold;")
        self.btn_start.clicked.connect(self._start_export)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel_or_close)

        btn_bar.addStretch()
        btn_bar.addWidget(self.btn_start)
        btn_bar.addWidget(self.btn_cancel)
        layout.addLayout(btn_bar)

    def _run_pre_export_qa(self) -> None:
        validator = DatasetValidator()
        valid_classes = set(self.class_catalog.values())
        report: ValidationReport = validator.validate(
            manager=self.manager,
            img_width=self.video_provider.width,
            img_height=self.video_provider.height,
            total_frames=self.video_provider.total_frames,
            valid_class_ids=valid_classes or None
        )

        if report.error_count > 0:
            self.lbl_qa_status.setText(f"❌ {report.error_count} Fatal Errors detected! (Export blocked)")
            self.lbl_qa_status.setStyleSheet("color: #FF5252; font-weight: bold;")
            self.btn_start.setEnabled(False)
        elif report.warning_count > 0:
            self.lbl_qa_status.setText(f"⚠️ Clean with {report.warning_count} Warnings (Ready to export)")
            self.lbl_qa_status.setStyleSheet("color: #FFA726; font-weight: bold;")
            self.btn_start.setEnabled(True)
        else:
            self.lbl_qa_status.setText("✅ Dataset 100% Validated & Clean")
            self.lbl_qa_status.setStyleSheet("color: #00E676; font-weight: bold;")
            self.btn_start.setEnabled(True)

    def _browse_destination(self) -> None:
        is_ml_tab = (self.tabs.currentIndex() == 0)
        if is_ml_tab:
            if self.rad_yolo.isChecked() or self.rad_yolo_track.isChecked() or self.rad_voc.isChecked() or self.rad_sampled.isChecked():
                d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
                if d:
                    self.txt_dest.setText(d)
            elif self.rad_mot.isChecked():
                f, _ = QFileDialog.getSaveFileName(self, "Save MOT Benchmark", "gt.txt", "Text Files (*.txt)")
                if f:
                    self.txt_dest.setText(f)
            elif self.rad_coco.isChecked():
                f, _ = QFileDialog.getSaveFileName(self, "Save COCO Instances", "instances.json", "JSON (*.json)")
                if f:
                    self.txt_dest.setText(f)
            elif self.rad_csv.isChecked():
                f, _ = QFileDialog.getSaveFileName(self, "Save Spreadsheet", "annotations.csv", "CSV (*.csv)")
                if f:
                    self.txt_dest.setText(f)
        else:
            # Video tab
            f, _ = QFileDialog.getSaveFileName(self, "Save Output Video", "exported_video.mp4", "MP4 Video (*.mp4)")
            if f:
                self.txt_dest.setText(f)

    def _start_export(self) -> None:
        dest = self.txt_dest.text().strip()
        if not dest:
            QMessageBox.warning(self, "Missing Path", "Please specify an output destination path.")
            return

        is_ml_tab = (self.tabs.currentIndex() == 0)
        task_type = ""
        params: Dict[str, Any] = {"out_path": dest, "class_to_id": self.class_catalog}

        if is_ml_tab:
            if self.rad_yolo.isChecked():
                task_type = "yolo"
                params["include_track_id"] = False
            elif self.rad_yolo_track.isChecked():
                task_type = "yolo"
                params["include_track_id"] = True
            elif self.rad_mot.isChecked():
                task_type = "mot"
            elif self.rad_coco.isChecked():
                task_type = "coco"
            elif self.rad_voc.isChecked():
                task_type = "voc"
            elif self.rad_csv.isChecked():
                task_type = "csv"
            elif self.rad_sampled.isChecked():
                task_type = "sampled"
                tot = self.spin_train.value() + self.spin_val.value() + self.spin_test.value()
                if tot != 100:
                    QMessageBox.warning(self, "Invalid Split", "Train + Val + Test percentages must sum to 100%.")
                    return
                params["train_ratio"] = self.spin_train.value() / 100.0
                params["val_ratio"] = self.spin_val.value() / 100.0
                params["test_ratio"] = self.spin_test.value() / 100.0
        else:
            params["start_frame"] = self.spin_start_f.value()
            params["end_frame"] = self.spin_end_f.value()
            if self.rad_privacy.isChecked():
                task_type = "privacy"
                idx = self.combo_priv_mode.currentIndex()
                params["mode"] = [PrivacyMode.GAUSSIAN_BLUR, PrivacyMode.PIXELATED_MOSAIC, PrivacyMode.SOLID_BLACKOUT][idx]
            elif self.rad_overlay.isChecked():
                task_type = "overlay"
                params["show_ids"] = True
                params["show_labels"] = True
                params["show_conf"] = True
                params["show_trajectory"] = True
            elif self.rad_subclip.isChecked():
                task_type = "subclip"

        self.prog_bar.setVisible(True)
        self.prog_bar.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_cancel.setText("Cancel Job")

        self.worker = ExportWorker(task_type, params, self.manager, self.video_provider)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, current: int, total: int, fps: float) -> None:
        pct = int((current / max(1, total)) * 100)
        self.prog_bar.setValue(pct)
        if fps > 0:
            self.prog_bar.setFormat(f"{pct}% ({fps:.1f} FPS)")

    def _on_finished(self, success: bool, message: str) -> None:
        self.prog_bar.setVisible(False)
        self.btn_start.setEnabled(True)
        self.btn_cancel.setText("Close")
        if success:
            QMessageBox.information(self, "Export Complete", message)
            self.accept()
        else:
            QMessageBox.warning(self, "Export Halted", message)

    def _on_error(self, err_msg: str) -> None:
        self.prog_bar.setVisible(False)
        self.btn_start.setEnabled(True)
        self.btn_cancel.setText("Close")
        QMessageBox.critical(self, "Export Error", f"An error occurred during export:\n{err_msg}")

    def _cancel_or_close(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        self.reject()