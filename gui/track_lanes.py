"""
gui/track_lanes.py

Multi-Track Lane Visualizer for DarkLabel Modern:
- Renders individual horizontal temporal lanes for every registered Track ID.
- Displays continuous activity spans [start_frame -> end_frame].
- Visualizes keyframe ticks, interpolated spans, and OUTSIDE gaps.
- Click to select active track; double-click a keyframe to seek the playhead directly.
- Fully synchronized horizontal scale with the main timeline ruler.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.annotation_manager import AnnotationManager
from core.annotation_models import Track
from gui.canvas_items import get_track_color


class TrackLanesCanvas(QWidget):
    """
    Renders horizontal activity lanes for all tracks.
    """

    track_selected = Signal(int)             # track_id
    frame_seek_requested = Signal(int)       # frame_idx
    track_double_clicked = Signal(int, int)  # track_id, nearest_keyframe

    LANE_HEIGHT = 22
    LANE_GAP = 3
    HEADER_WIDTH = 110

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMouseTracking(True)

        self.annotation_manager: Optional[AnnotationManager] = None
        self.total_frames: int = 0
        self.current_frame: int = 0
        self.zoom_factor: float = 1.0
        self.scroll_offset_ratio: float = 0.0

        self.active_track_id: int = 1

    def set_manager(self, manager: AnnotationManager) -> None:
        self.annotation_manager = manager
        self.update_geometry_and_repaint()

    def set_video_meta(self, total_frames: int) -> None:
        self.total_frames = max(1, total_frames)
        self.update()

    def set_current_frame(self, frame_idx: int) -> None:
        self.current_frame = frame_idx
        self.update()

    def set_active_track(self, track_id: int) -> None:
        self.active_track_id = track_id
        self.update()

    def set_zoom(self, zoom: float, offset_ratio: float) -> None:
        self.zoom_factor = zoom
        self.scroll_offset_ratio = offset_ratio
        self.update()

    def update_geometry_and_repaint(self) -> None:
        if not self.annotation_manager:
            return
        track_count = len(self.annotation_manager.track_manager.tracks)
        required_height = max(80, track_count * (self.LANE_HEIGHT + self.LANE_GAP) + 10)
        self.setMinimumHeight(required_height)
        self.update()

    # --------------------------------------------------------------------------
    # Coordinate Mappings
    # --------------------------------------------------------------------------
    def get_timeline_bounds(self) -> QRectF:
        """Returns the sub-rectangle allocated for horizontal activity bars."""
        w = max(10, self.width() - self.HEADER_WIDTH)
        return QRectF(self.HEADER_WIDTH, 0, w, self.height())

    def frame_to_x(self, frame_idx: int) -> float:
        bounds = self.get_timeline_bounds()
        if self.total_frames <= 1:
            return bounds.left()

        visible_frame_span = self.total_frames / self.zoom_factor
        max_start = self.total_frames - visible_frame_span
        start_frame = self.scroll_offset_ratio * max_start

        ratio = (frame_idx - start_frame) / float(visible_frame_span)
        return bounds.left() + (ratio * bounds.width())

    def x_to_frame(self, x: float) -> int:
        bounds = self.get_timeline_bounds()
        if x < bounds.left():
            return 0
        ratio = (x - bounds.left()) / float(bounds.width())
        visible_frame_span = self.total_frames / self.zoom_factor
        max_start = self.total_frames - visible_frame_span
        start_frame = self.scroll_offset_ratio * max_start

        frame = int(start_frame + ratio * visible_frame_span)
        return max(0, min(frame, self.total_frames - 1))

    # --------------------------------------------------------------------------
    # Paint Engine
    # --------------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(self.rect(), QColor(20, 21, 24))

        # Left Header Background
        painter.fillRect(QRectF(0, 0, self.HEADER_WIDTH, h), QColor(24, 25, 28))
        painter.setPen(QPen(QColor(45, 48, 54), 1.0))
        painter.drawLine(self.HEADER_WIDTH, 0, self.HEADER_WIDTH, h)

        if not self.annotation_manager or self.total_frames <= 0:
            return

        tracks = self.annotation_manager.track_manager.get_all_tracks()
        t_bounds = self.get_timeline_bounds()

        y_offset = 6

        for track in tracks:
            lane_rect = QRectF(self.HEADER_WIDTH, y_offset, t_bounds.width(), self.LANE_HEIGHT)
            is_active = (track.track_id == self.active_track_id)

            # Row highlight
            if is_active:
                painter.fillRect(QRectF(0, y_offset, w, self.LANE_HEIGHT), QColor(0, 173, 181, 25))

            # Draw Left Header Tag
            track_color = get_track_color(track.track_id)
            painter.fillRect(QRectF(4, y_offset + 3, 4, self.LANE_HEIGHT - 6), track_color)

            font = QFont("Monospace", 8, QFont.Bold if is_active else QFont.Normal)
            painter.setFont(font)
            painter.setPen(QColor(240, 240, 240) if is_active else QColor(160, 165, 175))

            lock_flag = "🔒" if track.locked else ""
            header_text = f"{lock_flag}#{track.track_id} {track.class_name}"
            painter.drawText(QRectF(14, y_offset, self.HEADER_WIDTH - 18, self.LANE_HEIGHT), Qt.AlignVCenter, header_text)

            # Draw Timeline Activity Bar
            if track.start_frame is not None and track.end_frame is not None:
                x_start = self.frame_to_x(track.start_frame)
                x_end = self.frame_to_x(track.end_frame)

                # Clamp bar horizontally to timeline bounds
                clamped_left = max(t_bounds.left(), min(x_start, t_bounds.right()))
                clamped_right = max(t_bounds.left(), min(x_end, t_bounds.right()))

                bar_w = max(2.0, clamped_right - clamped_left)
                bar_y = y_offset + 4
                bar_h = self.LANE_HEIGHT - 8

                if clamped_right > t_bounds.left() and clamped_left < t_bounds.right():
                    # Base Track Activity Span
                    bar_bg = QColor(track_color.red(), track_color.green(), track_color.blue(), 90)
                    painter.fillRect(QRectF(clamped_left, bar_y, bar_w, bar_h), bar_bg)
                    painter.setPen(QPen(track_color, 1.0))
                    painter.drawRect(QRectF(clamped_left, bar_y, bar_w, bar_h))

                    # Render Keyframes & Out-of-Frame intervals
                    for kf in track.keyframes:
                        kf_x = self.frame_to_x(kf)
                        if t_bounds.left() <= kf_x <= t_bounds.right():
                            diamond_r = 3.5
                            d_poly = QPolygonF([
                                QPointF(kf_x, bar_y + bar_h / 2.0 - diamond_r),
                                QPointF(kf_x + diamond_r, bar_y + bar_h / 2.0),
                                QPointF(kf_x, bar_y + bar_h / 2.0 + diamond_r),
                                QPointF(kf_x - diamond_r, bar_y + bar_h / 2.0),
                            ])
                            painter.setPen(Qt.NoPen)
                            painter.setBrush(QBrush(QColor(255, 255, 255, 230)))
                            painter.drawPolygon(d_poly)

            y_offset += self.LANE_HEIGHT + self.LANE_GAP

        # Draw Global Playhead Needle across all lanes
        cur_x = self.frame_to_x(self.current_frame)
        if t_bounds.left() <= cur_x <= t_bounds.right():
            painter.setPen(QPen(QColor(255, 59, 48, 200), 1.0, Qt.DashLine))
            painter.drawLine(int(cur_x), 0, int(cur_x), h)

    # --------------------------------------------------------------------------
    # Mouse Interaction
    # --------------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self.annotation_manager:
            return

        tracks = self.annotation_manager.track_manager.get_all_tracks()
        row_idx = (event.position().y() - 6) // (self.LANE_HEIGHT + self.LANE_GAP)

        if 0 <= int(row_idx) < len(tracks):
            selected_track = tracks[int(row_idx)]
            self.track_selected.emit(selected_track.track_id)

            if event.position().x() >= self.HEADER_WIDTH:
                f = self.x_to_frame(event.position().x())
                self.frame_seek_requested.emit(f)

        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if not self.annotation_manager:
            return

        tracks = self.annotation_manager.track_manager.get_all_tracks()
        row_idx = (event.position().y() - 6) // (self.LANE_HEIGHT + self.LANE_GAP)

        if 0 <= int(row_idx) < len(tracks):
            track = tracks[int(row_idx)]
            click_f = self.x_to_frame(event.position().x())

            # Find nearest keyframe to double click
            kfs = track.keyframes
            if kfs:
                nearest_kf = min(kfs, key=lambda k: abs(k - click_f))
                self.track_double_clicked.emit(track.track_id, nearest_kf)
                self.frame_seek_requested.emit(nearest_kf)

        event.accept()


class TrackLanesWidget(QWidget):
    """
    Scrollable Container for TrackLanesCanvas.
    Synchronizes zoom scale and playhead position with the main timeline.
    """

    track_selected = Signal(int)
    frame_seek_requested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #141518; }")

        self.canvas = TrackLanesCanvas()
        self.canvas.track_selected.connect(self.track_selected.emit)
        self.canvas.frame_seek_requested.connect(self.frame_seek_requested.emit)

        self.scroll_area.setWidget(self.canvas)
        layout.addWidget(self.scroll_area)

    def set_manager(self, manager: AnnotationManager) -> None:
        self.canvas.set_manager(manager)

    def set_video_meta(self, total_frames: int) -> None:
        self.canvas.set_video_meta(total_frames)

    def set_current_frame(self, frame_idx: int) -> None:
        self.canvas.set_current_frame(frame_idx)

    def set_active_track(self, track_id: int) -> None:
        self.canvas.set_active_track(track_id)

    def sync_timeline_zoom(self, zoom: float, offset_ratio: float) -> None:
        """Synchronizes horizontal zoom with the main timeline scrubber."""
        self.canvas.set_zoom(zoom, offset_ratio)

    def refresh(self) -> None:
        self.canvas.update_geometry_and_repaint()