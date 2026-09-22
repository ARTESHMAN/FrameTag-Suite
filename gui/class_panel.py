"""
gui/class_panel.py

Category and Class Catalog Management for DarkLabel Modern:
- Defines deterministic Category IDs, names, and custom display colors.
- Adds, renames, and re-colors classes with live canvas updates.
- Usage tracking: Prevents accidental deletion of classes actively referenced by tracks.
- Schema Editor: Allows configuring default class attributes and data types.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.annotation_manager import AnnotationManager
from core.project_manager import ClassCatalogEntry


class ClassPanel(QWidget):
    """
    Class management panel for adding, renaming, and coloring project categories.
    """

    class_created = Signal(int, str, str)       # class_id, name, hex_color
    class_renamed = Signal(int, str)            # class_id, new_name
    class_color_changed = Signal(int, str)      # class_id, hex_color
    class_deleted = Signal(int)                 # class_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.annotation_manager: Optional[AnnotationManager] = None
        self.classes: Dict[int, ClassCatalogEntry] = {}

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Classes Table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["", "ID", "Class Name", "In Use"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setColumnWidth(0, 20)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        layout.addWidget(self.table, stretch=1)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("+ Add")
        self.btn_add.clicked.connect(self._add_class)

        self.btn_color = QPushButton("Color")
        self.btn_color.clicked.connect(self._change_color)

        self.btn_rename = QPushButton("Rename")
        self.btn_rename.clicked.connect(self._rename_class)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setStyleSheet("color: #FF5252;")
        self.btn_delete.clicked.connect(self._delete_class)

        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_color)
        btn_layout.addWidget(self.btn_rename)
        btn_layout.addWidget(self.btn_delete)
        layout.addLayout(btn_layout)

    def set_manager(self, manager: AnnotationManager) -> None:
        self.annotation_manager = manager
        self.refresh()

    def set_classes(self, catalog: Dict[int, ClassCatalogEntry]) -> None:
        self.classes = catalog
        self.refresh()

    def refresh(self) -> None:
        """Refreshes the categories table and checks usage counts across tracks."""
        # Calculate in-use counts per class
        usage_counts: Dict[int, int] = {}
        if self.annotation_manager:
            with self.annotation_manager.lock:
                for t in self.annotation_manager.track_manager.tracks.values():
                    usage_counts[t.class_id] = usage_counts.get(t.class_id, 0) + 1

        self.table.setRowCount(len(self.classes))
        for row, (cid, entry) in enumerate(sorted(self.classes.items())):
            # Color indicator
            color_item = QTableWidgetItem()
            c = QColor(entry.color) if entry.color else QColor("#00ADB5")
            color_item.setBackground(c)
            self.table.setItem(row, 0, color_item)

            # ID
            id_item = QTableWidgetItem(str(cid))
            id_item.setFont(QFont("Monospace", 9, QFont.Bold))
            id_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 1, id_item)

            # Name
            name_item = QTableWidgetItem(entry.name)
            self.table.setItem(row, 2, name_item)

            # Usage
            count = usage_counts.get(cid, 0)
            use_item = QTableWidgetItem(f"{count} track(s)" if count > 0 else "–")
            use_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 3, use_item)

    def _get_selected_class_id(self) -> Optional[int]:
        sel = self.table.selectedItems()
        if not sel:
            return None
        row = sel[0].row()
        return int(self.table.item(row, 1).text())

    def _add_class(self) -> None:
        name, ok = QInputDialog.getText(self, "Add Class", "Class Category Name:")
        if not ok or not name.strip():
            return
        name = name.strip().lower()

        # Find next available ID
        next_id = max(self.classes.keys(), default=-1) + 1
        default_color = "#00ADB5"
        entry = ClassCatalogEntry(class_id=next_id, name=name, color=default_color)
        self.classes[next_id] = entry
        self.class_created.emit(next_id, name, default_color)
        self.refresh()

    def _change_color(self) -> None:
        cid = self._get_selected_class_id()
        if cid is None:
            return
        entry = self.classes[cid]
        initial = QColor(entry.color) if entry.color else QColor("#00ADB5")
        color = QColorDialog.getColor(initial, self, f"Select Color for {entry.name}")
        if color.isValid():
            hex_str = color.name()
            entry.color = hex_str
            self.class_color_changed.emit(cid, hex_str)
            self.refresh()

    def _rename_class(self) -> None:
        cid = self._get_selected_class_id()
        if cid is None:
            return
        entry = self.classes[cid]
        new_name, ok = QInputDialog.getText(self, "Rename Class", "New Category Name:", text=entry.name)
        if ok and new_name.strip() and new_name.strip() != entry.name:
            entry.name = new_name.strip().lower()
            self.class_renamed.emit(cid, entry.name)
            self.refresh()

    def _delete_class(self) -> None:
        cid = self._get_selected_class_id()
        if cid is None:
            return
        entry = self.classes[cid]

        # Check if tracks actively use this category
        if self.annotation_manager:
            with self.annotation_manager.lock:
                used_by = [
                    t.track_id for t in self.annotation_manager.track_manager.tracks.values()
                    if t.class_id == cid
                ]
            if used_by:
                QMessageBox.critical(
                    self,
                    "Cannot Delete Class",
                    f"Class '{entry.name}' cannot be deleted because it is currently assigned "
                    f"to {len(used_by)} active track(s): {used_by[:5]}{'...' if len(used_by) > 5 else ''}."
                )
                return

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete class '{entry.name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            del self.classes[cid]
            self.class_deleted.emit(cid)
            self.refresh()