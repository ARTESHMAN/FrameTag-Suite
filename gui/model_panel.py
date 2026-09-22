"""
gui/model_panel.py

AI & Tracker Control Panel for DarkLabel Modern:
- Manages YOLO model weight file selection (`best.pt`), hardware device selection,
  and confidence/IoU threshold sliders.
- Controls single-frame auto-tagging and batch tracking execution pipelines.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.detector import HardwareDeviceResolver, YOLODetector


class ModelPanel(QWidget):
    """Sidebar tab controlling AI model weights, hyperparameters, and batch operations."""

    auto_tag_single_requested = Signal(float, float, list)  # conf, iou, allowed_classes
    batch_track_requested = Signal(float, float, list)      # conf, iou, allowed_classes
    batch_cancel_requested = Signal()

    def __init__(self, detector: YOLODetector, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.detector = detector
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        # 1. Model Weights Box
        grp_weights = QGroupBox("YOLO Weights & Device")
        v_weights = QVBoxLayout(grp_weights)

        self.btn_load_weights = QPushButton("Load Model Weights (.pt)...")
        self.btn_load_weights.clicked.connect(self._on_load_weights_clicked)
        v_weights.addWidget(self.btn_load_weights)

        self.lbl_model_info = QLabel("Status: No model loaded")
        self.lbl_model_info.setStyleSheet("color: #e06c75; font-weight: bold;")
        v_weights.addWidget(self.lbl_model_info)

        h_dev = QHBoxLayout()
        h_dev.addWidget(QLabel("Device:"))
        self.cmb_device = QComboBox()
        self.cmb_device.addItems(["Auto Detect", "CUDA (GPU)", "MPS (Apple Silicon)", "CPU"])
        h_dev.addWidget(self.cmb_device)
        v_weights.addLayout(h_dev)
        layout.addWidget(grp_weights)

        # 2. Hyperparameters Box
        grp_params = QGroupBox("Inference Hyperparameters")
        v_params = QVBoxLayout(grp_params)

        # Confidence Slider
        h_conf = QHBoxLayout()
        h_conf.addWidget(QLabel("Confidence:"))
        self.lbl_conf_val = QLabel("0.25")
        h_conf.addWidget(self.lbl_conf_val)
        v_params.addLayout(h_conf)
        self.sld_conf = QSlider(Qt.Horizontal)
        self.sld_conf.setRange(1, 99)
        self.sld_conf.setValue(25)
        self.sld_conf.valueChanged.connect(lambda v: self.lbl_conf_val.setText(f"{v/100.0:.2f}"))
        v_params.addWidget(self.sld_conf)

        # IoU Threshold Slider
        h_iou = QHBoxLayout()
        h_iou.addWidget(QLabel("IoU Threshold:"))
        self.lbl_iou_val = QLabel("0.45")
        h_iou.addWidget(self.lbl_iou_val)
        v_params.addLayout(h_iou)
        self.sld_iou = QSlider(Qt.Horizontal)
        self.sld_iou.setRange(1, 99)
        self.sld_iou.setValue(45)
        self.sld_iou.valueChanged.connect(lambda v: self.lbl_iou_val.setText(f"{v/100.0:.2f}"))
        v_params.addWidget(self.sld_iou)
        layout.addWidget(grp_params)

        # 3. Actions Box
        grp_actions = QGroupBox("Execution")
        v_actions = QVBoxLayout(grp_actions)

        self.btn_auto_tag = QPushButton("Auto-Tag Current Frame (Ctrl+F)")
        self.btn_auto_tag.clicked.connect(self._on_auto_tag_clicked)
        v_actions.addWidget(self.btn_auto_tag)

        self.btn_batch_track = QPushButton("Batch Auto-Track Video... (Ctrl+B)")
        self.btn_batch_track.clicked.connect(self._on_batch_track_clicked)
        v_actions.addWidget(self.btn_batch_track)

        self.btn_cancel_batch = QPushButton("Cancel Batch Processing")
        self.btn_cancel_batch.setStyleSheet("background-color: #3b2225; color: #e06c75;")
        self.btn_cancel_batch.setEnabled(False)
        self.btn_cancel_batch.clicked.connect(self.batch_cancel_requested.emit)
        v_actions.addWidget(self.btn_cancel_batch)

        # Progress HUD
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        v_actions.addWidget(self.progress_bar)

        self.lbl_batch_status = QLabel("")
        v_actions.addWidget(self.lbl_batch_status)

        layout.addWidget(grp_actions)
        layout.addStretch(1)

    def _on_load_weights_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load YOLO Weights File", "", "PyTorch Weights (*.pt);;All Files (*)"
        )
        if path:
            device_pref = self.cmb_device.currentText()
            success = self.detector.load_model(path, device_pref)
            if success:
                self.lbl_model_info.setText(f"Loaded: {os.path.basename(path)} [{self.detector.device_name}]")
                self.lbl_model_info.setStyleSheet("color: #98c379; font-weight: bold;")
            else:
                self.lbl_model_info.setText("Error loading weights file.")
                self.lbl_model_info.setStyleSheet("color: #e06c75; font-weight: bold;")

    def _on_auto_tag_clicked(self) -> None:
        if not self.detector.is_loaded:
            return
        conf = self.sld_conf.value() / 100.0
        iou = self.sld_iou.value() / 100.0
        self.auto_tag_single_requested.emit(conf, iou, [])

    def _on_batch_track_clicked(self) -> None:
        if not self.detector.is_loaded:
            return
        conf = self.sld_conf.value() / 100.0
        iou = self.sld_iou.value() / 100.0
        self.batch_track_requested.emit(conf, iou, [])

    def start_batch_hud(self) -> None:
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_batch_track.setEnabled(False)
        self.btn_cancel_batch.setEnabled(True)

    def update_batch_hud(self, current: int, total: int, message: str) -> None:
        if total > 0:
            pct = min(100, max(0, int((current / total) * 100)))
            self.progress_bar.setValue(pct)
        self.lbl_batch_status.setText(message)

    def finish_batch_hud(self) -> None:
        self.progress_bar.setValue(100)
        self.lbl_batch_status.setText("Batch tracking complete.")
        self.btn_batch_track.setEnabled(True)
        self.btn_cancel_batch.setEnabled(False)