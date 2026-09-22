"""
gui/object_panel.py

Right-hand object management panel for DarkLabel Modern:
- Tabular list of all active/historical tracks in the project.
- Real-time search/filtering across Track ID, Class Name, and custom attributes.
- Inline controls: Visibility toggles (👁), Lock state toggles (🔒), and deterministic color chips.
- Context actions: Single-track delete, multi-track merge, and trajectory split.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.annotation_manager import AnnotationManager
from core.annotation_models import Track
from gui.canvas_items import get_track_color


class ObjectPanel(QWidget):
    """
    Object tracking hierarchy panel displaying all tracks, their visibility,
    lock state, and temporal spans with real-time search.
    """

    track_selected = Signal(int)                   # track_id
    track_visibility_toggled = Signal(int, bool)   # track_id, is_visible
    track_lock_toggled = Signal(int, bool)         # track_id, is_locked
    track_delete_requested = Signal(int)           # track_id
    tracks_merge_requested = Signal(int, int)      # source_id, target_id
    track_split_requested = Signal(int)            # track_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.annotation_manager: Optional[AnnotationManager] = None
        self.active_track_id: Optional[int] = None
        self._filter_text: str = ""

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # 1. Search Bar
        search_layout = QHBoxLayout()
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Search tracks (ID, class, attributes)...")
        self.txt_search.setClearButtonEnabled(True)
        self.txt_search.textChanged.connect(self._on_search_changed)
        search_layout.addWidget(self.txt_search)
        layout.addLayout(search_layout)

        # 2. Tracks Table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "ID", "Class", "Vis", "Lock"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 16)
        self.table.setColumnWidth(3, 38)
        self.table.setColumnWidth(4, 38)

        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self.table, stretch=1)

        # 3. Action Toolbar Row
        action_layout = QHBoxLayout()
        self.btn_hide_all = QPushButton("Hide All")
        self.btn_hide_all.setFixedHeight(24)
        self.btn_hide_all.clicked.connect(lambda: self._set_all_visibility(False))

        self.btn_show_all = QPushButton("Show All")
        self.btn_show_all.setFixedHeight(24)
        self.btn_show_all.clicked.connect(lambda: self._set_all_visibility(True))

        self.btn_lock_all = QPushButton("Lock All")
        self.btn_lock_all.setFixedHeight(24)
        self.btn_lock_all.clicked.connect(lambda: self._set_all_locks(True))

        action_layout.addWidget(self.btn_hide_all)
        action_layout.addWidget(self.btn_show_all)
        action_layout.addWidget(self.btn_lock_all)
        layout.addLayout(action_layout)

    def set_manager(self, manager: AnnotationManager) -> None:
        self.annotation_manager = manager
        self.refresh()

    def set_active_track(self, track_id: int) -> None:
        self.active_track_id = track_id
        self._highlight_active_row(track_id)

    def refresh(self) -> None:
        """Populates the table with filtered track instances."""
        if not self.annotation_manager:
            self.table.setRowCount(0)
            return

        with self.annotation_manager.lock:
            tracks = self.annotation_manager.track_manager.get_all_tracks()

        # Apply search filter
        filtered_tracks: List[Track] = []
        query = self._filter_text.strip().lower()

        for t in tracks:
            if not query:
                filtered_tracks.append(t)
                continue

            # Match ID, class name, or attributes
            id_match = query in str(t.track_id)
            class_match = query in t.class_name.lower()
            attr_match = any(
                query in str(k).lower() or query in str(v).lower()
                for k, v in t.attributes.items()
            )
            if id_match or class_match or attr_match:
                filtered_tracks.append(t)

        self.table.blockSignals(True)
        self.table.setRowCount(len(filtered_tracks))

        for row, t in enumerate(filtered_tracks):
            # Column 0: Color chip
            color_item = QTableWidgetItem()
            chip_color = get_track_color(t.track_id)
            color_item.setBackground(chip_color)
            self.table.setItem(row, 0, color_item)

            # Column 1: Track ID
            id_item = QTableWidgetItem(f"#{t.track_id}")
            id_item.setFont(QFont("Monospace", 9, QFont.Bold))
            id_item.setData(Qt.UserRole, t.track_id)
            self.table.setItem(row, 1, id_item)

            # Column 2: Class Name
            class_item = QTableWidgetItem(t.class_name)
            self.table.setItem(row, 2, class_item)

            # Column 3: Visibility Checkbox (👁)
            vis_item = QTableWidgetItem("👁" if t.visible else "–")
            vis_item.setTextAlignment(Qt.AlignCenter)
            vis_item.setForeground(QColor("#00ADB5") if t.visible else QColor("#6C757D"))
            self.table.setItem(row, 3, vis_item)

            # Column 4: Lock Checkbox (🔒)
            lock_item = QTableWidgetItem("🔒" if t.locked else "–")
            lock_item.setTextAlignment(Qt.AlignCenter)
            lock_item.setForeground(QColor("#FF5252") if t.locked else QColor("#6C757D"))
            self.table.setItem(row, 4, lock_item)

            if t.track_id == self.active_track_id:
                for c in range(5):
                    self.table.item(row, c).setSelected(True)

        self.table.blockSignals(False)

    def _highlight_active_row(self, track_id: int) -> None:
        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 1)
            if item and item.data(Qt.UserRole) == track_id:
                self.table.selectRow(r)
                break
        self.table.blockSignals(False)

    def _on_search_changed(self, text: str) -> None:
        self._filter_text = text
        self.refresh()

    def _on_table_selection_changed(self) -> None:
        sel = self.table.selectedItems()
        if not sel:
            return
        row = sel[0].row()
        item = self.table.item(row, 1)
        if item:
            tid = int(item.data(Qt.UserRole))
            self.active_track_id = tid
            self.track_selected.emit(tid)

    def _set_all_visibility(self, visible: bool) -> None:
        if not self.annotation_manager:
            return
        with self.annotation_manager.lock:
            for t in self.annotation_manager.track_manager.tracks.values():
                t.visible = visible
        self.refresh()
        if self.active_track_id:
            self.track_visibility_toggled.emit(self.active_track_id, visible)

    def _set_all_locks(self, locked: bool) -> None:
        if not self.annotation_manager:
            return
        with self.annotation_manager.lock:
            for t in self.annotation_manager.track_manager.tracks.values():
                t.locked = locked
        self.refresh()
        if self.active_track_id:
            self.track_lock_toggled.emit(self.active_track_id, locked)

    def _show_context_menu(self, pos: QPoint) -> None:
        item = self.table.itemAt(pos)
        if not item:
            return

        row = item.row()
        tid_item = self.table.item(row, 1)
        if not tid_item:
            return
        track_id = int(tid_item.data(Qt.UserRole))

        menu = QMenu(self)

        act_vis = menu.addAction("Toggle Visibility")
        act_lock = menu.addAction("Toggle Lock")
        menu.addSeparator()

        act_split = menu.addAction("Split Track at Current Frame")
        act_del = menu.addAction("Delete Entire Track")

        # Multi-selection merge action
        selected_rows = list({it.row() for it in self.table.selectedItems()})
        act_merge = None
        if len(selected_rows) == 2:
            menu.addSeparator()
            act_merge = menu.addAction("Merge Selected Tracks")

        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if not action or not self.annotation_manager:
            return

        track = self.annotation_manager.track_manager.get_track(track_id)
        if not track:
            return

        if action == act_vis:
            track.visible = not track.visible
            self.track_visibility_toggled.emit(track_id, track.visible)
            self.refresh()
        elif action == act_lock:
            track.locked = not track.locked
            self.track_lock_toggled.emit(track_id, track.locked)
            self.refresh()
        elif action == act_split:
            self.track_split_requested.emit(track_id)
        elif action == act_del:
            self.track_delete_requested.emit(track_id)
        elif act_merge and action == act_merge:
            id1 = int(self.table.item(selected_rows[0], 1).data(Qt.UserRole))
            id2 = int(self.table.item(selected_rows[1], 1).data(Qt.UserRole))
            self.tracks_merge_requested.emit(id1, id2)