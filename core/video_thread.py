import glob
import inspect
import os
import re
from typing import Any

import cv2
import numpy as np

# Universal Qt compatibility layer (PyQt5, PySide6, PyQt6)
try:
    from PyQt5.QtCore import QMutex, QMutexLocker, QObject, QThread, QTimer, pyqtSignal as Signal
    from PyQt5.QtGui import QImage
except ImportError:
    try:
        from PySide6.QtCore import QMutex, QMutexLocker, QObject, QThread, QTimer, Signal
        from PySide6.QtGui import QImage
    except ImportError:
        from PyQt6.QtCore import QMutex, QMutexLocker, QObject, QThread, QTimer, pyqtSignal as Signal
        from PyQt6.QtGui import QImage


def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


class CallableBool(int):
    """Behaves as both a boolean value and a callable function returning bool."""
    def __call__(self):
        return bool(self)
    def __bool__(self):
        return int(self) != 0


class SignalConnector(QObject):
    """Dynamic signal connector matching slot parameters by name and position."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._slots = []

    def connect(self, slot):
        if slot not in self._slots:
            self._slots.append(slot)
        return True

    def disconnect(self, slot=None):
        if slot is None:
            self._slots.clear()
        elif slot in self._slots:
            self._slots.remove(slot)

    def emit(self, *args, **kwargs):
        context = kwargs.get("context", {})

        for slot in list(self._slots):
            try:
                sig = inspect.signature(slot)
                pos_params = [
                    p for p in sig.parameters.values()
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())

                if has_varargs:
                    slot(*args)
                    continue

                if len(pos_params) == 0:
                    slot()
                    continue

                call_args = []
                for i, p in enumerate(pos_params):
                    p_name = p.name.lower()
                    if p_name in context:
                        call_args.append(context[p_name])
                    elif i < len(args):
                        call_args.append(args[i])
                    elif p.default is not inspect.Parameter.empty:
                        call_args.append(p.default)
                    else:
                        call_args.append(None)

                slot(*call_args)
            except Exception as err:
                print(f"[SignalConnector] Error dispatching to {slot}: {err}")


class VideoProvider(QObject):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.frame_ready = SignalConnector(self)
        self.video_loaded = SignalConnector(self)
        self.video_opened = SignalConnector(self)
        self.source_opened = SignalConnector(self)
        self.source_closed = SignalConnector(self)
        self.source_changed = SignalConnector(self)
        self.video_closed = SignalConnector(self)
        self.finished = SignalConnector(self)
        self.video_finished = SignalConnector(self)
        self.playback_finished = SignalConnector(self)
        self.error = SignalConnector(self)
        self.error_occurred = SignalConnector(self)
        self.position_changed = SignalConnector(self)
        self.frame_changed = SignalConnector(self)
        self.state_changed = SignalConnector(self)
        self.playback_state_changed = SignalConnector(self)
        self.fps_changed = SignalConnector(self)
        self.progress = SignalConnector(self)
        self.buffer_progress = SignalConnector(self)
        self.cache_updated = SignalConnector(self)

        video_path = ""
        cache_size_mb = 512
        if args:
            if isinstance(args[0], str):
                video_path = args[0]
                if len(args) > 1 and isinstance(args[1], (int, float)):
                    cache_size_mb = int(args[1])
            elif isinstance(args[0], (int, float)):
                cache_size_mb = int(args[0])
                if len(args) > 1 and isinstance(args[1], str):
                    video_path = args[1]

        self.source_path = kwargs.get("video_path", kwargs.get("path", kwargs.get("source", video_path)))
        self.video_path = self.source_path
        self.cache_size_mb = int(kwargs.get("cache_size_mb", kwargs.get("cache_size", cache_size_mb)))

        self.cap = None
        self._image_files = []
        self._source_type = "none"
        self._total_frames = 0
        self._fps = 30.0
        self._width = 0
        self._height = 0
        self._current_frame_idx = 0
        self._mutex = QMutex()
        self._cache = {}
        self._max_cache_entries = max(16, int((self.cache_size_mb * 1024 * 1024) / (1920 * 1080 * 3)))

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_playback_tick)
        self._is_playing = False

        if self.source_path and os.path.exists(str(self.source_path)):
            self.open_source(self.source_path)

    def __getattr__(self, name: str):
        if any(kw in name for kw in ("signal", "ready", "changed", "loaded", "opened", "closed", "finished", "error", "progress")):
            connector = SignalConnector(self)
            setattr(self, name, connector)
            return connector
        def _dummy(*args, **kwargs):
            return None
        return _dummy

    def open_source(self, source, *args, **kwargs) -> bool:
        with QMutexLocker(self._mutex):
            self.pause()
            self._release_resources()
            self.source_path = str(source)
            self.video_path = self.source_path

            if os.path.isdir(self.source_path):
                exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff")
                files = []
                for ext in exts:
                    files.extend(glob.glob(os.path.join(self.source_path, ext)))
                    files.extend(glob.glob(os.path.join(self.source_path, ext.upper())))
                self._image_files = sorted(list(set(files)), key=natural_sort_key)
                self._total_frames = len(self._image_files)
                if self._total_frames == 0:
                    self.error.emit(f"No image files found in {self.source_path}")
                    return False
                first = cv2.imread(self._image_files[0])
                if first is not None:
                    self._height, self._width = first.shape[:2]
                self._fps = 30.0
                self._source_type = "images"
            elif str(source).isdigit():
                self.cap = cv2.VideoCapture(int(source))
                if not self.cap.isOpened():
                    self.cap = None
                    return False
                self._source_type = "camera"
                self._fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
                self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._total_frames = 1000000
            else:
                self.cap = cv2.VideoCapture(self.source_path)
                if not self.cap.isOpened():
                    self.cap = None
                    self.error.emit(f"Failed to open video source: {self.source_path}")
                    return False
                self._source_type = "video"
                self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(self.cap.get(cv2.CAP_PROP_FPS))
                self._fps = fps if (fps and fps > 0 and not np.isnan(fps)) else 30.0
                self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self._current_frame_idx = 0
            self._cache.clear()

        ext = os.path.splitext(self.source_path)[1].lstrip('.').lower()
        fmt = ext if ext else self._source_type

        # Multi-alias context ensures slots match parameters regardless of naming convention
        context = {
            'total_frames': int(self._total_frames),
            'frame_count': int(self._total_frames),
            'num_frames': int(self._total_frames),
            'n_frames': int(self._total_frames),
            'frames': int(self._total_frames),
            'count': int(self._total_frames),
            'total': int(self._total_frames),
            'fps': float(self._fps),
            'framerate': float(self._fps),
            'frame_rate': float(self._fps),
            'w': int(self._width),
            'width': int(self._width),
            'video_width': int(self._width),
            'h': int(self._height),
            'height': int(self._height),
            'video_height': int(self._height),
            'fmt': str(fmt),
            'format': str(fmt),
            'ext': str(ext),
            'path': str(self.source_path),
            'source': str(self.source_path),
            'source_path': str(self.source_path),
            'video_path': str(self.source_path),
            'file_path': str(self.source_path),
        }

        self.source_opened.emit(self.source_path, context=context)
        self.video_opened.emit(self.source_path, context=context)
        # Passes integer frame count first to populate timeline and timecode duration
        self.video_loaded.emit(
            int(self._total_frames),
            float(self._fps),
            int(self._width),
            int(self._height),
            str(fmt),
            context=context
        )
        self.fps_changed.emit(self._fps, context=context)

        self.seek(0)
        return True

    def open(self, path: str) -> bool:
        return self.open_source(path)

    def load(self, path: str) -> bool:
        return self.open_source(path)

    def load_source(self, path: str) -> bool:
        return self.open_source(path)

    def load_video(self, path: str) -> bool:
        return self.open_source(path)

    def open_video(self, path: str) -> bool:
        return self.open_source(path)

    @property
    def source(self) -> str:
        return self.source_path

    @property
    def path(self) -> str:
        return self.source_path

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def is_container(self):
        return CallableBool(self._source_type == "video" and bool(self.source_path))

    @property
    def is_video_container(self):
        return CallableBool(self._source_type == "video" and bool(self.source_path))

    @property
    def is_container_file(self):
        return CallableBool(self._source_type == "video" and bool(self.source_path))

    @property
    def has_container(self):
        return CallableBool(self._source_type == "video" and bool(self.source_path))

    @property
    def is_video(self):
        return CallableBool(self._source_type == "video")

    @property
    def is_video_file(self):
        return CallableBool(self._source_type == "video" and bool(self.source_path))

    @property
    def is_image_sequence(self):
        return CallableBool(self._source_type == "images")

    @property
    def is_sequence(self):
        return CallableBool(self._source_type == "images")

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def frame_count(self) -> int:
        return self._total_frames

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def resolution(self):
        return (self._width, self._height)

    @property
    def duration(self) -> float:
        return (self._total_frames / self._fps) if self._fps > 0 else 0.0

    @property
    def is_opened(self) -> bool:
        if self._source_type == "images":
            return len(self._image_files) > 0
        return self.cap is not None and self.cap.isOpened()

    def isOpened(self) -> bool:
        return self.is_opened

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def current_frame_idx(self) -> int:
        return self._current_frame_idx

    @property
    def current_frame(self) -> int:
        return self._current_frame_idx

    @property
    def position(self) -> int:
        return self._current_frame_idx

    def get_frame(self, frame_idx: int = None) -> np.ndarray:
        if frame_idx is None:
            frame_idx = self._current_frame_idx

        with QMutexLocker(self._mutex):
            if not self.is_opened or frame_idx < 0:
                return None

            if frame_idx in self._cache:
                return self._cache[frame_idx]

            frame = None
            if self._source_type == "images":
                if 0 <= frame_idx < len(self._image_files):
                    frame = cv2.imread(self._image_files[frame_idx])
            elif self.cap is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                if not ret:
                    frame = None

            if frame is not None:
                if len(self._cache) >= self._max_cache_entries:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[frame_idx] = frame

            return frame

    def read_frame(self, frame_idx: int = None) -> np.ndarray:
        return self.get_frame(frame_idx)

    def get_current_frame(self) -> np.ndarray:
        return self.get_frame(self._current_frame_idx)

    def seek(self, frame_idx: int) -> np.ndarray:
        if self._total_frames > 0:
            frame_idx = max(0, min(frame_idx, self._total_frames - 1))
        self._current_frame_idx = frame_idx

        frame = self.get_frame(frame_idx)
        if frame is not None:
            ctx = {'frame_idx': frame_idx, 'idx': frame_idx, 'index': frame_idx, 'frame': frame, 'image': frame}
            self.position_changed.emit(frame_idx, context=ctx)
            self.frame_changed.emit(frame_idx, context=ctx)
            self.frame_ready.emit(frame_idx, frame, context=ctx)
        return frame

    def seek_frame(self, frame_idx: int) -> np.ndarray:
        return self.seek(frame_idx)

    def set_frame(self, frame_idx: int) -> np.ndarray:
        return self.seek(frame_idx)

    def set_position(self, frame_idx: int) -> np.ndarray:
        return self.seek(frame_idx)

    def jump_to(self, frame_idx: int) -> np.ndarray:
        return self.seek(frame_idx)

    def request_frame(self, frame_idx: int):
        self.seek(frame_idx)

    def play(self):
        if not self.is_opened:
            return
        interval = max(10, int(1000.0 / self._fps))
        self._is_playing = True
        self.playback_state_changed.emit(True)
        self._timer.start(interval)

    def pause(self):
        self._is_playing = False
        self._timer.stop()
        self.playback_state_changed.emit(False)

    def stop(self):
        self.pause()
        self.seek(0)

    def toggle_play(self):
        if self._is_playing:
            self.pause()
        else:
            self.play()

    def _on_playback_tick(self):
        if not self.is_opened:
            self.pause()
            return

        if self._current_frame_idx < self._total_frames - 1:
            self.seek(self._current_frame_idx + 1)
        else:
            self.pause()
            self.finished.emit()
            self.video_finished.emit()
            self.playback_finished.emit()

    def step(self, step: int = 1):
        self.seek(self._current_frame_idx + step)

    def step_forward(self, step: int = 1):
        self.seek(self._current_frame_idx + step)

    def step_backward(self, step: int = 1):
        self.seek(self._current_frame_idx - step)

    def next_frame(self):
        self.step_forward(1)

    def prev_frame(self):
        self.step_backward(1)

    def get_qimage(self, frame_idx: int = None) -> QImage:
        frame = self.get_frame(frame_idx)
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

    def get_current_qimage(self) -> QImage:
        return self.get_qimage(self._current_frame_idx)

    def clear_cache(self):
        with QMutexLocker(self._mutex):
            self._cache.clear()

    def _release_resources(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._image_files = []
        self._source_type = "none"
        self._cache.clear()

    def release(self):
        with QMutexLocker(self._mutex):
            self._release_resources()

    def close(self):
        self.release()

    def close_source(self):
        self.release()
        self.source_closed.emit()

    def __len__(self) -> int:
        return self._total_frames


class VideoDecoderWorker(QThread):
    frame_decoded = Signal(int, object)
    progress = Signal(int, int)
    finished = Signal()

    def __init__(self, provider: VideoProvider = None, parent=None, *args, **kwargs):
        super().__init__(parent)
        self.provider = provider
        self.is_running = False
        self.start_frame = 0
        self.end_frame = 0

    def set_range(self, start_frame: int, end_frame: int):
        self.start_frame = start_frame
        self.end_frame = end_frame

    def run(self):
        self.is_running = True
        if self.provider and self.provider.is_opened:
            for idx in range(self.start_frame, self.end_frame + 1):
                if not self.is_running:
                    break
                frame = self.provider.get_frame(idx)
                if frame is not None:
                    self.frame_decoded.emit(idx, frame)
                total = max(1, self.end_frame - self.start_frame + 1)
                curr = idx - self.start_frame + 1
                self.progress.emit(curr, total)
        self.finished.emit()

    def stop(self):
        self.is_running = False
        self.wait()


VideoThread = VideoProvider
__all__ = ["VideoProvider", "VideoDecoderWorker", "VideoThread", "SignalConnector"]