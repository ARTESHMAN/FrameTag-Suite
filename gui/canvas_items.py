"""
gui/canvas_items.py
Bounding box annotation item with 8-directional interactive resize handles,
label badges, boundary clamping, and hover/drag event handling.
"""

from typing import Optional, Dict, Tuple, Any
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QPen,
    QBrush,
    QColor,
    QPainter,
    QFont,
    QCursor,
    QFontMetrics,
    QPainterPath,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsSceneMouseEvent,
    QGraphicsSceneContextMenuEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

try:
    from core.annotation_models import Annotation
except Exception:
    Annotation = None


def get_track_color(track_id: Optional[int]) -> QColor:
    """Generates a distinct, deterministic QColor for a given track ID."""
    if track_id is None or track_id < 0:
        return QColor(0, 255, 128)

    palette = [
        QColor(230, 25, 75),
        QColor(60, 180, 75),
        QColor(255, 225, 25),
        QColor(0, 130, 200),
        QColor(245, 130, 48),
        QColor(145, 30, 180),
        QColor(70, 240, 240),
        QColor(240, 50, 230),
        QColor(210, 245, 60),
        QColor(250, 190, 212),
        QColor(0, 128, 128),
        QColor(220, 190, 255),
        QColor(170, 110, 40),
        QColor(255, 250, 200),
        QColor(128, 0, 0),
        QColor(170, 255, 195),
        QColor(128, 128, 0),
        QColor(255, 215, 180),
        QColor(0, 0, 128),
        QColor(128, 128, 128),
    ]
    if track_id < len(palette):
        return palette[track_id]

    hue = int((track_id * 137.508) % 360)
    color = QColor()
    color.setHsv(hue, 210, 240)
    return color


class ResizeHandle(QGraphicsRectItem):
    """
    Child handle item attached to an AnnotationBBoxItem.
    Receives mouse drag events and forwards them to the parent box for resizing.
    """

    NONE = 0
    TOP_LEFT = 1
    TOP = 2
    TOP_RIGHT = 3
    RIGHT = 4
    BOTTOM_RIGHT = 5
    BOTTOM = 6
    BOTTOM_LEFT = 7
    LEFT = 8

    def __init__(
        self,
        handle_type: int,
        parent: "AnnotationBBoxItem",
        size: float = 8.0,
    ):
        super().__init__(-size / 2.0, -size / 2.0, size, size, parent)
        self.handle_type = handle_type
        self.parent_box = parent
        self.size = size

        self.setPen(QPen(Qt.GlobalColor.black, 1.0))
        self.setBrush(QBrush(Qt.GlobalColor.white))
        self.setZValue(100)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)

    @property
    def annotation(self) -> Any:
        return self.parent_box.annotation

    @property
    def track_id(self) -> Optional[int]:
        return self.parent_box.track_id

    @property
    def frame_index(self) -> int:
        return self.parent_box.frame_index

    def get_cursor(self) -> Qt.CursorShape:
        mapping = {
            ResizeHandle.TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
            ResizeHandle.BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
            ResizeHandle.TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
            ResizeHandle.BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
            ResizeHandle.TOP: Qt.CursorShape.SizeVerCursor,
            ResizeHandle.BOTTOM: Qt.CursorShape.SizeVerCursor,
            ResizeHandle.LEFT: Qt.CursorShape.SizeHorCursor,
            ResizeHandle.RIGHT: Qt.CursorShape.SizeHorCursor,
        }
        return mapping.get(self.handle_type, Qt.CursorShape.ArrowCursor)

    def hoverEnterEvent(self, event):
        self.setCursor(QCursor(self.get_cursor()))
        self.setBrush(QBrush(self.parent_box.color))
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.setBrush(QBrush(Qt.GlobalColor.white))
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        # Ignore right clicks so view can handle box drawing
        if event.button() == Qt.MouseButton.RightButton:
            event.ignore()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.parent_box.start_resize(self.handle_type, event.scenePos())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if self.parent_box._is_resizing:
            self.parent_box.update_resize(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            event.ignore()
            return
        if self.parent_box._is_resizing:
            self.parent_box.finish_resize()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent):
        event.ignore()


class AnnotationProxy:
    """
    Fallback proxy ensuring `.annotation` never evaluates to None.
    Allows code like `it.annotation.track_id == track_id` to run safely.
    """

    def __init__(self, item: "AnnotationBBoxItem"):
        super().__setattr__("_item", item)

    @property
    def track_id(self) -> Optional[int]:
        return self._item.track_id

    @track_id.setter
    def track_id(self, val: Optional[int]):
        self._item.set_track_id(val)

    @property
    def frame_index(self) -> int:
        return self._item.frame_index

    @frame_index.setter
    def frame_index(self, val: int):
        self._item.frame_index = val

    @property
    def frame_idx(self) -> int:
        return self._item.frame_index

    @frame_idx.setter
    def frame_idx(self, val: int):
        self._item.frame_index = val

    @property
    def class_id(self) -> int:
        return self._item.class_id

    @class_id.setter
    def class_id(self, val: int):
        self._item.class_id = val

    @property
    def category_id(self) -> int:
        return self._item.class_id

    @category_id.setter
    def category_id(self, val: int):
        self._item.class_id = val

    @property
    def label(self) -> str:
        return self._item.label

    @label.setter
    def label(self, val: str):
        self._item.set_label(val)

    @property
    def category_name(self) -> str:
        return self._item.label

    @category_name.setter
    def category_name(self, val: str):
        self._item.set_label(val)

    @property
    def confidence(self) -> float:
        return self._item.confidence

    @confidence.setter
    def confidence(self, val: float):
        self._item.confidence = val

    @property
    def occluded(self) -> bool:
        return self._item.occluded

    @occluded.setter
    def occluded(self, val: bool):
        self._item.set_occluded(val)

    @property
    def keyframe(self) -> bool:
        return self._item.keyframe

    @keyframe.setter
    def keyframe(self, val: bool):
        self._item.set_keyframe(val)

    @property
    def rect(self) -> QRectF:
        return self._item.get_scene_rect()

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self._item.get_coordinates()

    @property
    def x(self) -> float:
        return self._item.pos().x()

    @property
    def y(self) -> float:
        return self._item.pos().y()

    @property
    def w(self) -> float:
        return self._item._rect.width()

    @property
    def h(self) -> float:
        return self._item._rect.height()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._item, name, None)

    def __setattr__(self, name: str, value: Any):
        if name == "_item":
            super().__setattr__(name, value)
        elif hasattr(self._item, name):
            setattr(self._item, name, value)
        else:
            super().__setattr__(name, value)


class AnnotationBBoxItem(QGraphicsItem):
    """
    Interactive bounding box item for video annotation.
    Supports 8-directional handle resizing, box dragging, boundary clamping,
    and label badge rendering (track ID, class label, confidence, occlusion).
    """

    HANDLE_SIZE = 8.0
    MIN_SIZE = 4.0

    def __init__(
        self,
        *args,
        rect: Optional[QRectF] = None,
        class_id: int = 0,
        label: str = "",
        track_id: Optional[int] = None,
        color: Optional[QColor] = None,
        confidence: float = 1.0,
        occluded: bool = False,
        keyframe: bool = True,
        frame_index: int = 0,
        annotation: Optional[Any] = None,
        parent: Optional[QGraphicsItem] = None,
        **kwargs,
    ):
        super().__init__(parent)

        # Boundary limit placeholder
        self._bounds_limit: Optional[QRectF] = None

        # Resolve positional arguments flexibly across all application caller signatures
        init_rect: Optional[QRectF] = None
        extracted_ann: Optional[Any] = annotation

        if len(args) >= 1:
            first = args[0]
            if isinstance(first, QRectF):
                init_rect = first
            elif len(args) >= 4 and all(isinstance(a, (int, float)) for a in args[:4]):
                init_rect = QRectF(float(args[0]), float(args[1]), float(args[2]), float(args[3]))
            elif hasattr(first, "bbox") and first.bbox is not None and len(first.bbox) == 4:
                init_rect = QRectF(float(first.bbox[0]), float(first.bbox[1]), float(first.bbox[2]), float(first.bbox[3]))
                extracted_ann = first
            elif hasattr(first, "rect") and first.rect is not None:
                r = first.rect
                init_rect = r if isinstance(r, QRectF) else QRectF(float(r[0]), float(r[1]), float(r[2]), float(r[3]))
                extracted_ann = first
            elif hasattr(first, "x") and hasattr(first, "width"):
                init_rect = QRectF(float(first.x), float(first.y), float(first.width), float(first.height))
                extracted_ann = first

            if len(args) >= 2 and isinstance(args[1], QRectF):
                self._bounds_limit = args[1]

        if init_rect is None and rect is not None:
            init_rect = rect

        if init_rect is None:
            init_rect = QRectF(0.0, 0.0, 50.0, 50.0)

        norm_rect = init_rect.normalized()
        self.setPos(norm_rect.topLeft())
        self._rect = QRectF(0.0, 0.0, max(self.MIN_SIZE, norm_rect.width()), max(self.MIN_SIZE, norm_rect.height()))

        # Metadata
        self.class_id: int = class_id
        self.label: str = label if label else f"Class {class_id}"
        self.track_id: Optional[int] = track_id
        self.confidence: float = confidence
        self.occluded: bool = occluded
        self.keyframe: bool = keyframe
        self._frame_index: int = kwargs.get("frame_idx", frame_index)
        self.is_locked: bool = kwargs.get("is_locked", False)

        # Model backing
        self._annotation: Optional[Any] = extracted_ann
        self._proxy: Optional[AnnotationProxy] = None

        if self._annotation is not None:
            if hasattr(self._annotation, "track_id") and self._annotation.track_id is not None:
                self.track_id = self._annotation.track_id
            if hasattr(self._annotation, "class_id") and self._annotation.class_id is not None:
                self.class_id = self._annotation.class_id
            if hasattr(self._annotation, "class_name") and self._annotation.class_name:
                self.label = self._annotation.class_name
            elif hasattr(self._annotation, "label") and self._annotation.label:
                self.label = self._annotation.label
            if hasattr(self._annotation, "frame_index") and self._annotation.frame_index is not None:
                self._frame_index = self._annotation.frame_index
        elif Annotation is not None:
            try:
                self._annotation = Annotation(
                    track_id=self.track_id,
                    class_id=self.class_id,
                    label=self.label,
                    confidence=self.confidence,
                    occluded=self.occluded,
                    keyframe=self.keyframe,
                    frame_index=self._frame_index,
                    bbox=self.get_coordinates(),
                )
            except Exception:
                self._annotation = None

        self.color: QColor = color if color is not None else get_track_color(self.track_id)

        # Resize interaction state
        self.active_handle: int = ResizeHandle.NONE
        self._is_resizing: bool = False
        self._drag_start_scene_pos: QPointF = QPointF()
        self._drag_start_pos: QPointF = QPointF()
        self._drag_start_rect: QRectF = QRectF()

        # Display settings
        self.show_label: bool = True
        self.show_track_id: bool = True
        self.show_confidence: bool = False
        self._font = QFont("Segoe UI", 9, QFont.Weight.Bold)

        # Graphic item flags
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            | QGraphicsItem.GraphicsItemFlag.ItemIsFocusable
        )
        if self.is_locked:
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

        self.setAcceptHoverEvents(True)

        # Child handles
        self.handles: Dict[int, ResizeHandle] = {}
        for ht in (
            ResizeHandle.TOP_LEFT,
            ResizeHandle.TOP,
            ResizeHandle.TOP_RIGHT,
            ResizeHandle.RIGHT,
            ResizeHandle.BOTTOM_RIGHT,
            ResizeHandle.BOTTOM,
            ResizeHandle.BOTTOM_LEFT,
            ResizeHandle.LEFT,
        ):
            h_item = ResizeHandle(ht, self, size=self.HANDLE_SIZE)
            h_item.setVisible(False)
            self.handles[ht] = h_item

        self.update_handles()

    # -------------------------------------------------------------------------
    # Frame Index Properties
    # -------------------------------------------------------------------------
    @property
    def frame_index(self) -> int:
        if self._annotation is not None and hasattr(self._annotation, "frame_index"):
            return self._annotation.frame_index
        return self._frame_index

    @frame_index.setter
    def frame_index(self, val: int):
        self._frame_index = val
        if self._annotation is not None and hasattr(self._annotation, "frame_index"):
            self._annotation.frame_index = val

    @property
    def frame_idx(self) -> int:
        return self.frame_index

    @frame_idx.setter
    def frame_idx(self, val: int):
        self.frame_index = val

    # -------------------------------------------------------------------------
    # Annotation Property
    # -------------------------------------------------------------------------
    @property
    def annotation(self) -> Any:
        if self._annotation is not None:
            return self._annotation
        if self._proxy is None:
            self._proxy = AnnotationProxy(self)
        return self._proxy

    @annotation.setter
    def annotation(self, ann: Any):
        self._annotation = ann
        if ann is not None:
            if hasattr(ann, "track_id") and ann.track_id is not None:
                self.track_id = ann.track_id
                self.color = get_track_color(self.track_id)
            if hasattr(ann, "class_id") and ann.class_id is not None:
                self.class_id = ann.class_id
            if hasattr(ann, "class_name") and ann.class_name:
                self.label = ann.class_name
            elif hasattr(ann, "label") and ann.label:
                self.label = ann.label
            if hasattr(ann, "confidence") and ann.confidence is not None:
                self.confidence = ann.confidence
            if hasattr(ann, "occluded"):
                self.occluded = ann.occluded
            if hasattr(ann, "keyframe"):
                self.keyframe = ann.keyframe
            if hasattr(ann, "frame_index"):
                self._frame_index = ann.frame_index
            if hasattr(ann, "rect"):
                r = ann.rect if isinstance(ann.rect, QRectF) else QRectF(ann.rect[0], ann.rect[1], ann.rect[2], ann.rect[3])
                self.set_rect(r)
            elif hasattr(ann, "bbox") and ann.bbox:
                b = ann.bbox
                self.set_rect(QRectF(float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            self.update()

    def _sync_annotation(self):
        if self._annotation is not None:
            if hasattr(self._annotation, "track_id"):
                self._annotation.track_id = self.track_id
            if hasattr(self._annotation, "frame_index"):
                self._annotation.frame_index = self.frame_index
            if hasattr(self._annotation, "class_id"):
                self._annotation.class_id = self.class_id
            if hasattr(self._annotation, "label"):
                self._annotation.label = self.label
            if hasattr(self._annotation, "class_name"):
                self._annotation.class_name = self.label
            if hasattr(self._annotation, "confidence"):
                self._annotation.confidence = self.confidence
            if hasattr(self._annotation, "occluded"):
                self._annotation.occluded = self.occluded
            if hasattr(self._annotation, "keyframe"):
                self._annotation.keyframe = self.keyframe
            r = self.get_scene_rect()
            if hasattr(self._annotation, "rect"):
                self._annotation.rect = r
            if hasattr(self._annotation, "bbox"):
                self._annotation.bbox = (float(r.x()), float(r.y()), float(r.width()), float(r.height()))
            if hasattr(self._annotation, "x"):
                self._annotation.x = float(r.x())
                self._annotation.y = float(r.y())
                self._annotation.width = float(r.width())
                self._annotation.height = float(r.height())

    # -------------------------------------------------------------------------
    # Bounds & Geometry
    # -------------------------------------------------------------------------
    def set_bounds_limit(self, bounds: Optional[QRectF]):
        self._bounds_limit = bounds

    def boundingRect(self) -> QRectF:
        margin = self.HANDLE_SIZE + 4.0
        base = self._rect.adjusted(-margin, -margin, margin, margin)
        if self.show_label:
            base = base.adjusted(0, -28.0, 0, 0)
        return base

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self._rect)
        if self.isSelected():
            for h in self.handles.values():
                path.addRect(h.mapRectToParent(h.rect()))
        return path

    def rect(self) -> QRectF:
        return self._rect

    def setRect(self, rect: QRectF):
        self.set_rect(rect)

    def set_rect(self, rect: QRectF):
        self.prepareGeometryChange()
        norm = rect.normalized()
        self.setPos(norm.topLeft())
        self._rect = QRectF(0, 0, max(self.MIN_SIZE, norm.width()), max(self.MIN_SIZE, norm.height()))
        self.update_handles()
        self._sync_annotation()
        self.update()

    def get_scene_rect(self) -> QRectF:
        return QRectF(self.pos().x(), self.pos().y(), self._rect.width(), self._rect.height())

    def to_rect(self) -> QRectF:
        return self.get_scene_rect()

    def get_coordinates(self) -> Tuple[float, float, float, float]:
        return (float(self.pos().x()), float(self.pos().y()), float(self._rect.width()), float(self._rect.height()))

    def set_coordinates(self, x: float, y: float, w: float, h: float):
        self.prepareGeometryChange()
        self.setPos(x, y)
        self._rect = QRectF(0, 0, max(self.MIN_SIZE, w), max(self.MIN_SIZE, h))
        self.update_handles()
        self._sync_annotation()
        self.update()

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self.get_coordinates()

    @property
    def coords(self) -> Tuple[float, float, float, float]:
        return self.get_coordinates()

    @property
    def category_id(self) -> int:
        return self.class_id

    @category_id.setter
    def category_id(self, val: int):
        self.class_id = val

    def set_class_id(self, class_id: int):
        self.class_id = class_id
        self._sync_annotation()
        self.update()

    def set_locked(self, locked: bool):
        self.is_locked = locked
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, not locked)
        for h in self.handles.values():
            h.setVisible(not locked and self.isSelected())
        self.update()

    # -------------------------------------------------------------------------
    # Handle Updates & Layout
    # -------------------------------------------------------------------------
    def update_handles(self):
        if not hasattr(self, "handles") or not self.handles:
            return

        r = self._rect
        positions = {
            ResizeHandle.TOP_LEFT: QPointF(r.left(), r.top()),
            ResizeHandle.TOP: QPointF(r.center().x(), r.top()),
            ResizeHandle.TOP_RIGHT: QPointF(r.right(), r.top()),
            ResizeHandle.RIGHT: QPointF(r.right(), r.center().y()),
            ResizeHandle.BOTTOM_RIGHT: QPointF(r.right(), r.bottom()),
            ResizeHandle.BOTTOM: QPointF(r.center().x(), r.bottom()),
            ResizeHandle.BOTTOM_LEFT: QPointF(r.left(), r.bottom()),
            ResizeHandle.LEFT: QPointF(r.left(), r.center().y()),
        }

        is_sel = self.isSelected() and not self.is_locked
        for ht, handle_item in self.handles.items():
            if ht in positions:
                handle_item.setPos(positions[ht])
                handle_item.setBrush(QBrush(self.color))
            handle_item.setVisible(is_sel)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            is_sel = bool(value)
            if hasattr(self, "handles"):
                for h in self.handles.values():
                    h.setVisible(is_sel and not getattr(self, "is_locked", False))
        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            if getattr(self, "_bounds_limit", None) is not None and not self._is_resizing:
                new_pos = value
                if self._bounds_limit.width() > 0 and self._bounds_limit.height() > 0:
                    clamped_x = min(
                        max(new_pos.x(), self._bounds_limit.left()),
                        max(self._bounds_limit.left(), self._bounds_limit.right() - self._rect.width()),
                    )
                    clamped_y = min(
                        max(new_pos.y(), self._bounds_limit.top()),
                        max(self._bounds_limit.top(), self._bounds_limit.bottom() - self._rect.height()),
                    )
                    return QPointF(clamped_x, clamped_y)
        return super().itemChange(change, value)

    # -------------------------------------------------------------------------
    # Resize Engine
    # -------------------------------------------------------------------------
    def start_resize(self, handle_type: int, scene_pos: QPointF):
        self.active_handle = handle_type
        self._is_resizing = True
        self._drag_start_scene_pos = scene_pos
        self._drag_start_pos = self.pos()
        self._drag_start_rect = QRectF(self._rect)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

    def update_resize(self, scene_pos: QPointF):
        if not self._is_resizing or self.active_handle == ResizeHandle.NONE:
            return

        delta = scene_pos - self._drag_start_scene_pos
        dx = delta.x()
        dy = delta.y()

        orig_x = self._drag_start_pos.x()
        orig_y = self._drag_start_pos.y()
        orig_w = self._drag_start_rect.width()
        orig_h = self._drag_start_rect.height()

        left = orig_x
        top = orig_y
        right = orig_x + orig_w
        bottom = orig_y + orig_h

        ht = self.active_handle
        if ht in (ResizeHandle.TOP_LEFT, ResizeHandle.LEFT, ResizeHandle.BOTTOM_LEFT):
            left = min(orig_x + dx, right - self.MIN_SIZE)
        if ht in (ResizeHandle.TOP_LEFT, ResizeHandle.TOP, ResizeHandle.TOP_RIGHT):
            top = min(orig_y + dy, bottom - self.MIN_SIZE)
        if ht in (ResizeHandle.TOP_RIGHT, ResizeHandle.RIGHT, ResizeHandle.BOTTOM_RIGHT):
            right = max(orig_x + orig_w + dx, left + self.MIN_SIZE)
        if ht in (ResizeHandle.BOTTOM_LEFT, ResizeHandle.BOTTOM, ResizeHandle.BOTTOM_RIGHT):
            bottom = max(orig_y + orig_h + dy, top + self.MIN_SIZE)

        if self._bounds_limit is not None and self._bounds_limit.width() > 0 and self._bounds_limit.height() > 0:
            left = max(self._bounds_limit.left(), left)
            top = max(self._bounds_limit.top(), top)
            right = min(self._bounds_limit.right(), right)
            bottom = min(self._bounds_limit.bottom(), bottom)

        new_width = max(self.MIN_SIZE, right - left)
        new_height = max(self.MIN_SIZE, bottom - top)

        self.prepareGeometryChange()
        self.setPos(left, top)
        self._rect = QRectF(0, 0, new_width, new_height)
        self.update_handles()
        self._sync_annotation()
        self.update()

        if self.scene() and hasattr(self.scene(), "on_box_modified"):
            self.scene().on_box_modified(self)

    def finish_resize(self):
        self._is_resizing = False
        self.active_handle = ResizeHandle.NONE
        if not self.is_locked:
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.update_handles()
        self._sync_annotation()
        self.update()

        if self.scene() and hasattr(self.scene(), "on_box_modified"):
            self.scene().on_box_modified(self)

    def get_handle_at(self, pos: QPointF) -> int:
        if not self.isSelected() or self.is_locked:
            return ResizeHandle.NONE
        for ht, handle_item in self.handles.items():
            if handle_item.mapRectToParent(handle_item.rect()).contains(pos):
                return ht
        return ResizeHandle.NONE

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        # Ignore right clicks so canvas can draw a box immediately
        if event.button() == Qt.MouseButton.RightButton:
            event.ignore()
            return

        if event.button() == Qt.MouseButton.LeftButton and not self.is_locked:
            handle = self.get_handle_at(event.pos())
            if handle != ResizeHandle.NONE:
                self.start_resize(handle, event.scenePos())
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if self._is_resizing:
            self.update_resize(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            event.ignore()
            return

        if self._is_resizing:
            self.finish_resize()
            event.accept()
            return

        super().mouseReleaseEvent(event)
        self._sync_annotation()

        if self.scene() and hasattr(self.scene(), "on_box_modified"):
            self.scene().on_box_modified(self)

    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent):
        event.ignore()

    # -------------------------------------------------------------------------
    # Visual Styling
    # -------------------------------------------------------------------------
    def set_color(self, color: QColor):
        self.color = color
        self.update()

    def set_label(self, label: str):
        self.label = label
        self.prepareGeometryChange()
        self._sync_annotation()
        self.update()

    def set_track_id(self, track_id: Optional[int]):
        self.track_id = track_id
        self.color = get_track_color(track_id)
        self.prepareGeometryChange()
        self._sync_annotation()
        self.update()

    def set_occluded(self, occluded: bool):
        self.occluded = occluded
        self._sync_annotation()
        self.update()

    def set_keyframe(self, keyframe: bool):
        self.keyframe = keyframe
        self._sync_annotation()
        self.update()

    def set_confidence(self, conf: float):
        self.confidence = conf
        self._sync_annotation()
        self.update()

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None):
        is_sel = self.isSelected()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Border stroke
        pen_width = 2.5 if is_sel else 1.8
        pen = QPen(self.color, pen_width)
        if self.occluded:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)

        # Fill
        fill_color = QColor(self.color)
        fill_color.setAlpha(65 if is_sel else 25)
        painter.setBrush(QBrush(fill_color))
        painter.drawRect(self._rect)

        # Label Banner
        if self.show_label:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            badge_text = self.label
            if self.show_track_id and self.track_id is not None:
                badge_text = f"#{self.track_id} {badge_text}"
            if self.show_confidence and self.confidence < 1.0:
                badge_text += f" {self.confidence:.2f}"

            painter.setFont(self._font)
            metrics = QFontMetrics(self._font)
            text_rect = metrics.boundingRect(badge_text)

            badge_width = text_rect.width() + 10.0
            badge_height = max(16.0, text_rect.height() + 4.0)

            badge_y = -badge_height if self.pos().y() >= badge_height else 0.0
            badge_rect = QRectF(0, badge_y, badge_width, badge_height)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self.color))
            painter.drawRoundedRect(badge_rect, 3, 3)

            luminance = (self.color.red() * 0.299 + self.color.green() * 0.587 + self.color.blue() * 0.114)
            text_color = Qt.GlobalColor.black if luminance > 150 else Qt.GlobalColor.white
            painter.setPen(QPen(text_color))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)


# Aliases for backwards compatibility
BoundingBoxItem = AnnotationBBoxItem
BBoxItem = AnnotationBBoxItem
HandleType = ResizeHandle
HandleItem = ResizeHandle

__all__ = [
    "AnnotationBBoxItem",
    "BoundingBoxItem",
    "BBoxItem",
    "ResizeHandle",
    "HandleType",
    "HandleItem",
    "get_track_color",
    "AnnotationProxy",
]