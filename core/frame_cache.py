"""
core/frame_cache.py

LRU RAM frame cache for DarkLabel Modern:
- Tracks memory consumption in megabytes (MB) based on raw frame buffer nbytes.
- Thread-safe eviction using collections.OrderedDict and threading.RLock.
- Configurable capacity limits with runtime adjustment via set_max_size_mb.
"""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Optional

import numpy as np


class FrameCache:
    """
    Thread-safe LRU cache storing decoded video frames with memory-based eviction.
    """

    def __init__(
        self,
        max_size_mb: int = 512,
        max_frames: Optional[int] = None,
        *args,
        **kwargs
    ):
        # Support alternative argument names gracefully
        if "max_mb" in kwargs:
            max_size_mb = kwargs["max_mb"]
        if "capacity" in kwargs and max_frames is None:
            max_frames = kwargs["capacity"]

        self._lock = threading.RLock()
        self.max_bytes: int = max(16, max_size_mb) * 1024 * 1024
        self.max_frames: Optional[int] = max_frames
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._current_bytes: int = 0

    @property
    def max_size_mb(self) -> int:
        with self._lock:
            return self.max_bytes // (1024 * 1024)

    def set_max_size_mb(self, mb: int) -> None:
        """Adjusts maximum allocated cache size and triggers eviction if necessary."""
        with self._lock:
            self.max_bytes = max(16, mb) * 1024 * 1024
            self._evict()

    def get(self, frame_idx: int) -> Optional[np.ndarray]:
        """Retrieves a cached frame and marks it as most recently used."""
        with self._lock:
            if frame_idx in self._cache:
                self._cache.move_to_end(frame_idx)
                return self._cache[frame_idx]
            return None

    def put(self, frame_idx: int, frame: np.ndarray) -> None:
        """Inserts or updates a frame in the cache, evicting oldest items if needed."""
        if frame is None or not isinstance(frame, np.ndarray):
            return

        frame_bytes = frame.nbytes
        with self._lock:
            # If already present, deduct previous size first
            if frame_idx in self._cache:
                self._current_bytes -= self._cache[frame_idx].nbytes
                del self._cache[frame_idx]

            self._cache[frame_idx] = frame
            self._current_bytes += frame_bytes
            self._cache.move_to_end(frame_idx)
            self._evict()

    def clear(self) -> None:
        """Flushes the cache entirely."""
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0

    def _evict(self) -> None:
        """Evicts oldest frames until within memory and frame count limits."""
        if self.max_frames is not None:
            while len(self._cache) > self.max_frames:
                _, old_frame = self._cache.popitem(last=False)
                self._current_bytes -= old_frame.nbytes

        while self._current_bytes > self.max_bytes and len(self._cache) > 1:
            _, old_frame = self._cache.popitem(last=False)
            self._current_bytes -= old_frame.nbytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, frame_idx: int) -> bool:
        with self._lock:
            return frame_idx in self._cache