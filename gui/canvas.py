"""
gui/canvas.py
Interactive CanvasView implementation for video annotation.
Supports direct Left-Click bounding box creation that persists on screen.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Tuple, Any
import numpy as np

from PySide6.QtCore import Qt, QRectF, QPointF, Signal, Slot
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontMetrics,
    QWheelEvent,
    QKeyEvent,
    QMouseEvent,
)
from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsItem,
    QWidget,
)

from .canvas_items import (
    AnnotationBBoxItem,
    BoundingBoxItem,
    BBoxItem,
    ResizeHandle,
    HandleType,
    get_track_color,
)


class CanvasMode(Enum):
    SELECT = auto()
    DRAW = auto()
    PAN = auto()


class OnionSkinMode(Enum):
    DISABLED = 0
    OFF = 0
    PREVIOUS = 1
    PREV = 1
    NEXT = 2
    BOTH = 3


@dataclass
class ImageAdjustments:
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    gamma: float = 1.0
    invert: bool = False
    grayscale: bool = False

    def is_default(self) -> bool:
        return (
            self.brightness == 0.0
            and self.contrast == 1.0
            and self.saturation == 1.0
            and self.gamma == 1.0
            and not self.invert
            and not self.grayscale
        )

    def apply(self, qimage: QImage) -> QImage:
        if self.is_default() or qimage.isNull():
            return qimage

        img = qimage.convertToFormat(QImage.Format.Format_RGB888)
        width = img.width()
        height = img.height()
        bytes_per_line = img.bytesPerLine()

        ptr = img.bits()
        arr = np.frombuffer(ptr, np.uint8).reshape((height, bytes_per_line))[:, : width * 3]
        arr = arr.reshape((height, width, 3)).astype(np.float32)

        if self.contrast != 1.0:
            arr = (arr - 128.0) * self.contrast + 128.0
        if self.brightness != 0.0:
            arr += self.brightness

        if self.gamma != 1.0 and self.gamma > 0.0:
            arr = np.clip(arr, 0.0, 255.0) / 255.0
            arr = np.power(arr, 1.0 / self.gamma) * 255.0

        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

        if self.invert:
            arr = 255 - arr

        if self.grayscale:
            gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            arr[..., 0] = gray
            arr[..., 1] = gray
            arr[..., 2] = gray

        return QImage(arr.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()


class CanvasScene(QGraphicsScene):
    box_created = Signal(object)
    box_selected = Signal(object)
    box_modified = Signal(object)
    box_deleted = Signal(object)
    annotation_created = Signal(object)
    annotation_selected = Signal(object)
    annotation_modified = Signal(object)
    annotation_deleted = Signal(object)
    selection_cleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.onion_prev_item = QGraphicsPixmapItem()
        self.onion_prev_item.setZValue(1)
        self.onion_prev_item.setOpacity(0.35)
        self.addItem(self.onion_prev_item)

        self.onion_next_item = QGraphicsPixmapItem()
        self.onion_next_item.setZValue(2)
        self.onion_next_item.setOpacity(0.35)
        self.addItem(self.onion_next_item)

        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setZValue(3)
        self.addItem(self.pixmap_item)

    def __call__(self):
        return self

    def on_box_modified(self, box: AnnotationBBoxItem):
        self.box_modified.emit(box)
        self.annotation_modified.emit(box)


class CanvasView(QGraphicsView):
    mouse_pos_changed = Signal(int, int)
    zoom_changed = Signal(float)
    mode_changed = Signal(object)

    # box_created emits (x, y, w, h) coordinates matching MainWindow's slot
    box_created = Signal(float, float, float, float)
    box_selected = Signal(object)
    box_modified = Signal(object)
    box_deleted = Signal(object)
    annotation_created = Signal(object)
    annotation_selected = Signal(object)
    annotation_modified = Signal(object)
    annotation_deleted = Signal(object)
    selection_cleared = Signal()

    ZOOM_MIN = 0.05
    ZOOM_MAX = 40.0
    ZOOM_FACTOR = 1.15

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene_obj = CanvasScene()
        self.setScene(self._scene_obj)
        self.scene = self._scene_obj

        self._mode: CanvasMode = CanvasMode.DRAW
        self._current_class_id: int = 0
        self._current_label: str = "fire"
        self._current_color: QColor = QColor(0, 255, 128)
        self._current_track_id: Optional[int] = None

        self._space_pressed: bool = False
        self._raw_frame_image: Optional[QImage] = None
        self._adjustments = ImageAdjustments()
        self._image_width: int = 0
        self._image_height: int = 0
        self._current_frame_index: int = 0

        self._onion_mode: OnionSkinMode = OnionSkinMode.DISABLED
        self._onion_opacity: float = 0.35

        self._is_panning: bool = False
        self._pan_start: QPointF = QPointF()
        self._current_zoom: float = 1.0

        # Drawing state
        self._is_drawing: bool = False
        self._draw_start_point: QPointF = QPointF()
        self._rubber_band: Optional[QGraphicsRectItem] = None

        self._show_crosshair: bool = True
        self._mouse_scene_pos: QPointF = QPointF()

        self._init_view()

    def _init_view(self):
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, False)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._rubber_band = QGraphicsRectItem()
        self._rubber_band.setPen(QPen(Qt.GlobalColor.yellow, 1.5, Qt.PenStyle.DashLine))
        self._rubber_band.setBrush(QBrush(QColor(255, 255, 0, 40)))
        self._rubber_band.setZValue(9999)
        self._rubber_band.hide()
        self._scene_obj.addItem(self._rubber_band)

        self._scene_obj.box_selected.connect(self.box_selected)
        self._scene_obj.box_selected.connect(self.annotation_selected)
        self._scene_obj.box_modified.connect(self.box_modified)
        self._scene_obj.box_modified.connect(self.annotation_modified)
        self._scene_obj.box_deleted.connect(self.box_deleted)
        self._scene_obj.box_deleted.connect(self.annotation_deleted)
        self._scene_obj.selection_cleared.connect(self.selection_cleared)
        self._scene_obj.selectionChanged.connect(self._on_scene_selection_changed)

    # -------------------------------------------------------------------------
    # Frame Index Properties
    # -------------------------------------------------------------------------
    @property
    def current_frame_index(self) -> int:
        return self._current_frame_index

    @current_frame_index.setter
    def current_frame_index(self, val: int):
        self._current_frame_index = val

    @property
    def frame_index(self) -> int:
        return self._current_frame_index

    @frame_index.setter
    def frame_index(self, val: int):
        self._current_frame_index = val

    def set_frame_index(self, frame_idx: int):
        self._current_frame_index = frame_idx

    def set_current_frame_index(self, frame_idx: int):
        self._current_frame_index = frame_idx

    def set_current_frame(self, frame_idx: int):
        self._current_frame_index = frame_idx

    # -------------------------------------------------------------------------
    # Onion Mode & Adjustments Properties
    # -------------------------------------------------------------------------
    @property
    def onion_mode(self) -> OnionSkinMode:
        return self._onion_mode

    @onion_mode.setter
    def onion_mode(self, mode: Union[OnionSkinMode, str, int]):
        self.set_onion_skin_mode(mode)

    @property
    def onion_skin_mode(self) -> OnionSkinMode:
        return self._onion_mode

    @onion_skin_mode.setter
    def onion_skin_mode(self, mode: Union[OnionSkinMode, str, int]):
        self.set_onion_skin_mode(mode)

    @property
    def onion_opacity(self) -> float:
        return self._onion_opacity

    @onion_opacity.setter
    def onion_opacity(self, opacity: float):
        self._onion_opacity = opacity
        self._scene_obj.onion_prev_item.setOpacity(opacity)
        self._scene_obj.onion_next_item.setOpacity(opacity)

    @property
    def adjustments(self) -> ImageAdjustments:
        return self._adjustments

    @adjustments.setter
    def adjustments(self, adj: ImageAdjustments):
        self.set_image_adjustments(adj)

    @property
    def image_adjustments(self) -> ImageAdjustments:
        return self._adjustments

    @image_adjustments.setter
    def image_adjustments(self, adj: ImageAdjustments):
        self.set_image_adjustments(adj)

    # -------------------------------------------------------------------------
    # Mode Settings
    # -------------------------------------------------------------------------
    @property
    def mode(self) -> CanvasMode:
        return self._mode

    def set_mode(self, mode: Union[CanvasMode, str]):
        if isinstance(mode, str):
            mode_map = {
                "select": CanvasMode.SELECT,
                "draw": CanvasMode.DRAW,
                "pan": CanvasMode.PAN,
            }
            mode = mode_map.get(mode.lower(), CanvasMode.DRAW)

        if self._mode == mode:
            return

        self._mode = mode
        if self._mode == CanvasMode.PAN:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self._mode == CanvasMode.DRAW:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        self.mode_changed.emit(self._mode)

    def set_active_class(self, class_id: int, label: str, color: QColor):
        self._current_class_id = class_id
        self._current_label = label
        self._current_color = color

    def set_current_class(self, class_id: int, label: str, color: QColor):
        self.set_active_class(class_id, label, color)

    def set_active_track_id(self, track_id: Optional[int]):
        self._current_track_id = track_id

    def set_current_track_id(self, track_id: Optional[int]):
        self.set_active_track_id(track_id)

    @property
    def current_class_id(self) -> int:
        return self._current_class_id

    @current_class_id.setter
    def current_class_id(self, val: int):
        self._current_class_id = val

    @property
    def active_class_id(self) -> int:
        return self._current_class_id

    @active_class_id.setter
    def active_class_id(self, val: int):
        self._current_class_id = val

    @property
    def current_label(self) -> str:
        return self._current_label

    @current_label.setter
    def current_label(self, val: str):
        self._current_label = val

    @property
    def current_color(self) -> QColor:
        return self._current_color

    @current_color.setter
    def current_color(self, val: QColor):
        self._current_color = val

    @property
    def current_track_id(self) -> Optional[int]:
        return self._current_track_id

    @current_track_id.setter
    def current_track_id(self, val: Optional[int]):
        self._current_track_id = val

    @property
    def active_track_id(self) -> Optional[int]:
        return self._current_track_id

    @active_track_id.setter
    def active_track_id(self, val: Optional[int]):
        self._current_track_id = val

    # -------------------------------------------------------------------------
    # Frame Management
    # -------------------------------------------------------------------------
    def set_frame(self, frame: Union[QPixmap, QImage, np.ndarray]):
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3:
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                self._raw_frame_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            else:
                h, w = frame.shape
                self._raw_frame_image = QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        elif isinstance(frame, QPixmap):
            self._raw_frame_image = frame.toImage()
        elif isinstance(frame, QImage):
            self._raw_frame_image = frame
        else:
            return

        self._image_width = self._raw_frame_image.width()
        self._image_height = self._raw_frame_image.height()
        self._scene_obj.setSceneRect(0, 0, self._image_width, self._image_height)

        self._update_display_pixmap()

        frame_rect = QRectF(0, 0, self._image_width, self._image_height)
        for box in self.get_all_boxes():
            box.set_bounds_limit(frame_rect)

    def set_frames(self, *args, **kwargs):
        frame_idx = kwargs.get("frame_index", kwargs.get("frame_idx", kwargs.get("index", None)))
        current_frame = kwargs.get("current_frame", kwargs.get("frame", kwargs.get("current", None)))
        prev_frame = kwargs.get("prev_frame", kwargs.get("prev", None))
        next_frame = kwargs.get("next_frame", kwargs.get("next", None))

        pos_args = list(args)
        if pos_args:
            if isinstance(pos_args[0], (int, np.integer)):
                frame_idx = int(pos_args.pop(0))
            if pos_args:
                current_frame = pos_args.pop(0)
            if pos_args:
                prev_frame = pos_args.pop(0)
            if pos_args:
                next_frame = pos_args.pop(0)

        if frame_idx is not None:
            self._current_frame_index = frame_idx

        if current_frame is not None:
            self.set_frame(current_frame)

        if prev_frame is not None or next_frame is not None or self._onion_mode != OnionSkinMode.DISABLED:
            self.set_onion_skin(prev_frame, next_frame, mode=self._onion_mode, opacity=self._onion_opacity)

    def set_image(self, image: Union[QPixmap, QImage, np.ndarray]):
        self.set_frame(image)

    def set_pixmap(self, pixmap: QPixmap):
        self.set_frame(pixmap)

    def load_frame(self, frame: Union[QPixmap, QImage, np.ndarray]):
        self.set_frame(frame)

    @property
    def image_width(self) -> int:
        return self._image_width

    @property
    def image_height(self) -> int:
        return self._image_height

    @property
    def frame_width(self) -> int:
        return self._image_width

    @property
    def frame_height(self) -> int:
        return self._image_height

    @property
    def pixmap(self) -> Optional[QPixmap]:
        return self._scene_obj.pixmap_item.pixmap()

    @property
    def current_pixmap(self) -> Optional[QPixmap]:
        return self._scene_obj.pixmap_item.pixmap()

    @property
    def frame(self) -> Optional[QImage]:
        return self._raw_frame_image

    def set_image_adjustments(self, adjustments: ImageAdjustments):
        self._adjustments = adjustments
        self._update_display_pixmap()

    def get_image_adjustments(self) -> ImageAdjustments:
        return self._adjustments

    def _update_display_pixmap(self):
        if self._raw_frame_image is None or self._raw_frame_image.isNull():
            return
        processed = self._adjustments.apply(self._raw_frame_image)
        self._scene_obj.pixmap_item.setPixmap(QPixmap.fromImage(processed))

    # -------------------------------------------------------------------------
    # Onion Skinning
    # -------------------------------------------------------------------------
    def set_onion_skin(
        self,
        prev_frame: Optional[Union[QPixmap, QImage]] = None,
        next_frame: Optional[Union[QPixmap, QImage]] = None,
        mode: Optional[OnionSkinMode] = None,
        opacity: Optional[float] = None,
    ):
        if mode is not None:
            self._onion_mode = mode
        if opacity is not None:
            self._onion_opacity = opacity

        self._scene_obj.onion_prev_item.setOpacity(self._onion_opacity)
        self._scene_obj.onion_next_item.setOpacity(self._onion_opacity)

        show_prev = self._onion_mode in (OnionSkinMode.PREVIOUS, OnionSkinMode.PREV, OnionSkinMode.BOTH)
        show_next = self._onion_mode in (OnionSkinMode.NEXT, OnionSkinMode.BOTH)

        if show_prev and prev_frame is not None:
            pix = prev_frame if isinstance(prev_frame, QPixmap) else QPixmap.fromImage(prev_frame)
            self._scene_obj.onion_prev_item.setPixmap(pix)
            self._scene_obj.onion_prev_item.show()
        elif not show_prev:
            self._scene_obj.onion_prev_item.hide()

        if show_next and next_frame is not None:
            pix = next_frame if isinstance(next_frame, QPixmap) else QPixmap.fromImage(next_frame)
            self._scene_obj.onion_next_item.setPixmap(pix)
            self._scene_obj.onion_next_item.show()
        elif not show_next:
            self._scene_obj.onion_next_item.hide()

    def set_onion_skin_mode(self, mode: Union[OnionSkinMode, str, int]):
        if isinstance(mode, str):
            mode_map = {
                "disabled": OnionSkinMode.DISABLED,
                "off": OnionSkinMode.DISABLED,
                "prev": OnionSkinMode.PREVIOUS,
                "previous": OnionSkinMode.PREVIOUS,
                "next": OnionSkinMode.NEXT,
                "both": OnionSkinMode.BOTH,
            }
            mode = mode_map.get(mode.lower(), OnionSkinMode.DISABLED)
        elif isinstance(mode, int):
            try:
                mode = OnionSkinMode(mode)
            except ValueError:
                mode = OnionSkinMode.DISABLED

        self._onion_mode = mode
        self.set_onion_skin(
            self._scene_obj.onion_prev_item.pixmap(),
            self._scene_obj.onion_next_item.pixmap(),
            mode=self._onion_mode,
            opacity=self._onion_opacity,
        )

    # -------------------------------------------------------------------------
    # Box Operations
    # -------------------------------------------------------------------------
    def add_box(self, box_item: AnnotationBBoxItem):
        if self._image_width > 0 and self._image_height > 0:
            box_item.set_bounds_limit(QRectF(0, 0, self._image_width, self._image_height))
        box_item.setZValue(10)
        self._scene_obj.addItem(box_item)

    def create_box(
        self,
        rect: QRectF,
        class_id: Optional[int] = None,
        label: Optional[str] = None,
        track_id: Optional[int] = None,
        color: Optional[QColor] = None,
    ) -> AnnotationBBoxItem:
        cid = class_id if class_id is not None else self._current_class_id
        lbl = label if label is not None else self._current_label
        tid = track_id if track_id is not None else self._current_track_id
        clr = color if color is not None else self._current_color

        box = AnnotationBBoxItem(
            rect=rect,
            class_id=cid,
            label=lbl,
            track_id=tid,
            color=clr,
            frame_index=self._current_frame_index,
        )
        self.add_box(box)
        return box

    def remove_box(self, box_item: AnnotationBBoxItem):
        if box_item in self._scene_obj.items():
            self._scene_obj.removeItem(box_item)
            self._scene_obj.box_deleted.emit(box_item)
            self._scene_obj.annotation_deleted.emit(box_item)

    def delete_box(self, box_item: AnnotationBBoxItem):
        self.remove_box(box_item)

    def clear_boxes(self):
        for box in self.get_all_boxes():
            self._scene_obj.removeItem(box)
        self._scene_obj.clearSelection()

    def clear_annotations(self):
        self.clear_boxes()

    def get_all_boxes(self) -> List[AnnotationBBoxItem]:
        return [item for item in self._scene_obj.items() if isinstance(item, AnnotationBBoxItem)]

    def get_boxes(self) -> List[AnnotationBBoxItem]:
        return self.get_all_boxes()

    def get_annotations(self) -> List[Any]:
        return [box.annotation for box in self.get_all_boxes()]

    def set_annotations(self, annotations: List[Any], frame_index: Optional[int] = None):
        if frame_index is not None:
            self._current_frame_index = frame_index
        self.clear_boxes()
        for ann in annotations:
            if isinstance(ann, AnnotationBBoxItem):
                self.add_box(ann)
            elif hasattr(ann, "rect") or hasattr(ann, "bbox") or hasattr(ann, "x"):
                if hasattr(ann, "bbox") and ann.bbox:
                    b = ann.bbox
                    r = QRectF(b[0], b[1], b[2], b[3])
                elif hasattr(ann, "rect") and isinstance(ann.rect, QRectF):
                    r = ann.rect
                else:
                    r = QRectF(getattr(ann, "x", 0), getattr(ann, "y", 0), getattr(ann, "width", 50), getattr(ann, "height", 50))

                box = AnnotationBBoxItem(
                    rect=r,
                    class_id=getattr(ann, "class_id", 0),
                    label=getattr(ann, "class_name", getattr(ann, "label", "")),
                    track_id=getattr(ann, "track_id", None),
                    confidence=getattr(ann, "confidence", 1.0),
                    occluded=getattr(ann, "occluded", False),
                    keyframe=getattr(ann, "is_keyframe", True),
                    annotation=ann,
                    frame_index=getattr(ann, "frame_index", self._current_frame_index),
                )
                self.add_box(box)

    def load_annotations(self, annotations: List[Any]):
        self.set_annotations(annotations)

    def get_selected_boxes(self) -> List[AnnotationBBoxItem]:
        return [item for item in self._scene_obj.selectedItems() if isinstance(item, AnnotationBBoxItem)]

    def get_selected_box(self) -> Optional[AnnotationBBoxItem]:
        boxes = self.get_selected_boxes()
        return boxes[0] if boxes else None

    def select_box(self, target_box: Optional[AnnotationBBoxItem]):
        self._scene_obj.clearSelection()
        if target_box is not None:
            target_box.setSelected(True)

    def deselect_all(self):
        self._scene_obj.clearSelection()

    def delete_selected(self):
        for box in self.get_selected_boxes():
            self.remove_box(box)

    def add_annotation(self, item: AnnotationBBoxItem):
        self.add_box(item)

    def remove_annotation(self, item: AnnotationBBoxItem):
        self.remove_box(item)

    def delete_annotation(self, item: AnnotationBBoxItem):
        self.remove_box(item)

    def get_selected_annotation(self) -> Optional[AnnotationBBoxItem]:
        return self.get_selected_box()

    def select_annotation(self, target: Optional[AnnotationBBoxItem]):
        self.select_box(target)

    def select_track(self, track_id: Optional[int]):
        for b in self.get_all_boxes():
            b.setSelected(b.track_id == track_id)

    def clear_selection(self):
        self.deselect_all()

    def delete_selected_box(self):
        self.delete_selected()

    def delete_selected_boxes(self):
        self.delete_selected()

    def delete_selected_annotations(self):
        self.delete_selected()

    def set_labels_visible(self, visible: bool):
        for b in self.get_all_boxes():
            b.show_label = visible
            b.update()

    def set_track_ids_visible(self, visible: bool):
        for b in self.get_all_boxes():
            b.show_track_id = visible
            b.update()

    def set_confidences_visible(self, visible: bool):
        for b in self.get_all_boxes():
            b.show_confidence = visible
            b.update()

    def _on_scene_selection_changed(self):
        selected = self.get_selected_boxes()
        if selected:
            self._scene_obj.box_selected.emit(selected[0])
            self._scene_obj.annotation_selected.emit(selected[0])
        else:
            self._scene_obj.selection_cleared.emit()

    # -------------------------------------------------------------------------
    # Viewport Zoom & Pan
    # -------------------------------------------------------------------------
    def zoom_in(self):
        self._apply_zoom(self.ZOOM_FACTOR, QPointF(self.viewport().width() / 2, self.viewport().height() / 2))

    def zoom_out(self):
        self._apply_zoom(1.0 / self.ZOOM_FACTOR, QPointF(self.viewport().width() / 2, self.viewport().height() / 2))

    def fit_to_window(self):
        self.fit_to_view()

    def fit_to_view(self):
        if self._image_width <= 0 or self._image_height <= 0:
            return
        self.resetTransform()
        self.setSceneRect(0, 0, self._image_width, self._image_height)
        self.fitInView(self._scene_obj.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._current_zoom = self.transform().m11()
        self.zoom_changed.emit(self._current_zoom)

    def fit_in_view(self):
        self.fit_to_view()

    def fit_window(self):
        self.fit_to_view()

    def zoom_fit(self):
        self.fit_to_view()

    def reset_zoom(self):
        self.resetTransform()
        self._current_zoom = 1.0
        self.zoom_changed.emit(self._current_zoom)

    def actual_size(self):
        self.reset_zoom()

    def zoom_100(self):
        self.reset_zoom()

    def set_zoom(self, zoom: float):
        if self.ZOOM_MIN <= zoom <= self.ZOOM_MAX:
            factor = zoom / self._current_zoom
            self._apply_zoom(factor, QPointF(self.viewport().width() / 2, self.viewport().height() / 2))

    def get_zoom(self) -> float:
        return self._current_zoom

    @property
    def zoom(self) -> float:
        return self._current_zoom

    @zoom.setter
    def zoom(self, val: float):
        self.set_zoom(val)

    @property
    def current_zoom(self) -> float:
        return self._current_zoom

    def set_crosshair_visible(self, visible: bool):
        self._show_crosshair = visible
        self.viewport().update()

    def toggle_crosshair(self):
        self.set_crosshair_visible(not self._show_crosshair)

    def _apply_zoom(self, factor: float, anchor_pos: QPointF):
        new_zoom = self._current_zoom * factor
        if not (self.ZOOM_MIN <= new_zoom <= self.ZOOM_MAX):
            return

        old_scene_pos = self.mapToScene(anchor_pos.toPoint())
        self.scale(factor, factor)
        new_scene_pos = self.mapToScene(anchor_pos.toPoint())

        delta = new_scene_pos - old_scene_pos
        self.translate(delta.x(), delta.y())

        self._current_zoom = new_zoom
        self.zoom_changed.emit(self._current_zoom)

    # -------------------------------------------------------------------------
    # Event Forwarding
    # -------------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent):
        angle = event.angleDelta().y()
        if angle != 0:
            factor = self.ZOOM_FACTOR if angle > 0 else (1.0 / self.ZOOM_FACTOR)
            self._apply_zoom(factor, event.position())
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        scene_pos = self.mapToScene(event.pos())
        clicked_item = self._scene_obj.itemAt(scene_pos, self.transform())

        # Middle click, Space bar, or explicit PAN mode: Pan the view
        is_pan_button = event.button() == Qt.MouseButton.MiddleButton
        is_pan_mode = (event.button() == Qt.MouseButton.LeftButton and self._mode == CanvasMode.PAN)
        is_space_pan = (event.button() == Qt.MouseButton.LeftButton and self._space_pressed)

        if is_pan_button or is_pan_mode or is_space_pan:
            self._is_panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        # Check if click landed on an existing box resize handle or box body
        is_handle = False
        is_box = False
        item = clicked_item
        while item:
            if isinstance(item, ResizeHandle):
                is_handle = True
                break
            if isinstance(item, AnnotationBBoxItem):
                is_box = True
            item = item.parentItem()

        # If user left-clicked on a resize handle, allow handle resizing
        if is_handle and event.button() == Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        # If user left-clicks on an existing box in SELECT mode, allow dragging/moving
        if is_box and self._mode == CanvasMode.SELECT and event.button() == Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        # Left-Click anywhere else starts drawing a new bounding box
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_drawing = True
            self._draw_start_point = scene_pos
            self._rubber_band.setRect(QRectF(self._draw_start_point, self._draw_start_point))
            self._rubber_band.show()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        scene_pos = self.mapToScene(event.pos())
        self._mouse_scene_pos = scene_pos
        self.mouse_pos_changed.emit(int(scene_pos.x()), int(scene_pos.y()))

        if self._is_panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        if self._is_drawing:
            cur_rect = QRectF(self._draw_start_point, scene_pos).normalized()
            if self._image_width > 0 and self._image_height > 0:
                frame_rect = QRectF(0, 0, self._image_width, self._image_height)
                cur_rect = cur_rect.intersected(frame_rect)
            self._rubber_band.setRect(cur_rect)
            if self._show_crosshair:
                self.viewport().update()
            event.accept()
            return

        super().mouseMoveEvent(event)

        if self._show_crosshair:
            self.viewport().update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._is_panning and (event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton)):
            self._is_panning = False
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._mode == CanvasMode.PAN else Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        if self._is_drawing and event.button() == Qt.MouseButton.LeftButton:
            self._is_drawing = False
            self._rubber_band.hide()
            created_rect = self._rubber_band.rect().normalized()

            if created_rect.width() >= AnnotationBBoxItem.MIN_SIZE and created_rect.height() >= AnnotationBBoxItem.MIN_SIZE:
                # Emit pure scene coordinates (x, y, w, h)
                x = float(created_rect.x())
                y = float(created_rect.y())
                w = float(created_rect.width())
                h = float(created_rect.height())
                self.box_created.emit(x, y, w, h)

            event.accept()
            return

        super().mouseReleaseEvent(event)

    # -------------------------------------------------------------------------
    # Visual Crosshair Overlay
    # -------------------------------------------------------------------------
    def drawForeground(self, painter: QPainter, rect: QRectF):
        super().drawForeground(painter, rect)

        if self._show_crosshair and (self._mode == CanvasMode.DRAW or self._is_drawing):
            painter.save()
            pen = QPen(QColor(255, 255, 255, 120), 1.0, Qt.PenStyle.DashLine)
            painter.setPen(pen)

            painter.drawLine(
                QPointF(rect.left(), self._mouse_scene_pos.y()),
                QPointF(rect.right(), self._mouse_scene_pos.y()),
            )
            painter.drawLine(
                QPointF(self._mouse_scene_pos.x(), rect.top()),
                QPointF(self._mouse_scene_pos.x(), rect.bottom()),
            )
            painter.restore()

    # -------------------------------------------------------------------------
    # Keyboard Navigation
    # -------------------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()

        if key == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            if self._mode != CanvasMode.PAN:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return

        step = 5.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
        selected_boxes = self.get_selected_boxes()

        if selected_boxes and key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            dx = -step if key == Qt.Key.Key_Left else (step if key == Qt.Key.Key_Right else 0.0)
            dy = -step if key == Qt.Key.Key_Up else (step if key == Qt.Key.Key_Down else 0.0)

            for box in selected_boxes:
                new_pos = box.pos() + QPointF(dx, dy)
                if box._bounds_limit:
                    new_pos.setX(min(max(new_pos.x(), box._bounds_limit.left()), box._bounds_limit.right() - box._rect.width()))
                    new_pos.setY(min(max(new_pos.y(), box._bounds_limit.top()), box._bounds_limit.bottom() - box._rect.height()))
                box.setPos(new_pos)
                self._scene_obj.on_box_modified(box)

            event.accept()
            return

        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            for box in selected_boxes:
                self.remove_box(box)
            event.accept()
            return

        if key == Qt.Key.Key_Escape:
            if self._is_drawing:
                self._is_drawing = False
                self._rubber_band.hide()
            else:
                self._scene_obj.clearSelection()
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            if not self._is_panning:
                if self._mode == CanvasMode.DRAW:
                    self.setCursor(Qt.CursorShape.CrossCursor)
                elif self._mode == CanvasMode.PAN:
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)


AnnotationCanvas = CanvasView
AnnotationScene = CanvasScene

__all__ = [
    "CanvasView",
    "AnnotationCanvas",
    "CanvasScene",
    "AnnotationScene",
    "CanvasMode",
    "OnionSkinMode",
    "ImageAdjustments",
]