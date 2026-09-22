"""
core/project_manager.py

Project serialization, session persistence (.dlm format), and crash recovery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import glob
import json
import os
import shutil
import threading
import time
from typing import Any, Dict, List, Optional

from core.annotation_manager import AnnotationManager
from core.history_manager import HistoryManager


@dataclass
class VideoMetadata:
    source_path: str = ""
    total_frames: int = 0
    fps: float = 30.0
    width: int = 0
    height: int = 0
    codec: str = "RAW"


@dataclass
class ClassCatalogEntry:
    class_id: int
    name: str
    color: str
    attributes_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryEntry:
    filepath: str
    timestamp: float
    track_count: int
    frame_count: int

    @property
    def formatted_time(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")


# Backward-compatibility alias
RecoveryInfo = RecoveryEntry


class ProjectManager:
    """Manages project save, load, and automated crash recovery daemon."""

    RECOVERY_DIR = os.path.expanduser("~/.darklabel_modern/recoveries")

    def __init__(self, annotation_manager: AnnotationManager, history_manager: HistoryManager):
        self.annotation_manager = annotation_manager
        self.history_manager = history_manager

        self.project_path: Optional[str] = None
        self.video_metadata = VideoMetadata()
        self.classes: Dict[int, ClassCatalogEntry] = {}

        self._autosave_thread: Optional[threading.Thread] = None
        self._autosave_stop_event = threading.Event()

        os.makedirs(self.RECOVERY_DIR, exist_ok=True)

    def save_project(self, out_path: Optional[str] = None) -> str:
        save_file = out_path or self.project_path
        if not save_file:
            raise ValueError("No destination filepath specified for project save.")

        os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)

        payload = {
            "format": "DarkLabel Modern Project",
            "version": "1.0",
            "saved_at": datetime.now().isoformat(),
            "video_metadata": asdict(self.video_metadata),
            "classes": {str(cid): asdict(entry) for cid, entry in self.classes.items()},
            "database": self.annotation_manager.to_dict(),
        }

        temp_file = f"{save_file}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        shutil.move(temp_file, save_file)

        self.project_path = save_file
        return save_file

    def load_project(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Project file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        v_meta = payload.get("video_metadata", {})
        self.video_metadata = VideoMetadata(**v_meta)

        self.classes.clear()
        for cid_str, c_data in payload.get("classes", {}).items():
            entry = ClassCatalogEntry(
                class_id=int(c_data["class_id"]),
                name=str(c_data["name"]),
                color=str(c_data["color"]),
                attributes_schema=dict(c_data.get("attributes_schema", {}))
            )
            self.classes[entry.class_id] = entry

        db_payload = payload.get("database", {})
        self.annotation_manager.load_dict(db_payload, clear_existing=True)
        self.history_manager.clear()
        self.project_path = filepath

    def start_autosave_daemon(self, interval_seconds: int = 60) -> None:
        self.stop_autosave_daemon()
        self._autosave_stop_event.clear()

        def daemon_loop():
            while not self._autosave_stop_event.wait(interval_seconds):
                if self.annotation_manager.has_annotations(0) or len(self.annotation_manager.track_manager.tracks) > 0:
                    self._create_recovery_snapshot()

        self._autosave_thread = threading.Thread(target=daemon_loop, daemon=True)
        self._autosave_thread.start()

    def stop_autosave_daemon(self) -> None:
        self._autosave_stop_event.set()
        if self._autosave_thread and self._autosave_thread.is_alive():
            self._autosave_thread.join(timeout=1.0)
            self._autosave_thread = None

    def _create_recovery_snapshot(self) -> None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_path = os.path.join(self.RECOVERY_DIR, f"recovery_{timestamp_str}.dlm")
        try:
            self.save_project(rec_path)
            files = sorted(glob.glob(os.path.join(self.RECOVERY_DIR, "recovery_*.dlm")))
            while len(files) > 5:
                oldest = files.pop(0)
                try:
                    os.remove(oldest)
                except OSError:
                    pass
        except Exception:
            pass

    def detect_available_recoveries(self) -> List[RecoveryEntry]:
        files = sorted(glob.glob(os.path.join(self.RECOVERY_DIR, "recovery_*.dlm")), reverse=True)
        recoveries = []
        for path in files:
            try:
                mtime = os.path.getmtime(path)
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tm_data = data.get("database", {}).get("track_manager", {})
                t_count = len(tm_data.get("tracks", {}))
                f_count = sum(len(t.get("annotations", {})) for t in tm_data.get("tracks", {}).values())
                recoveries.append(RecoveryEntry(filepath=path, timestamp=mtime, track_count=t_count, frame_count=f_count))
            except Exception:
                continue
        return recoveries

    def restore_recovery(self, recovery: RecoveryEntry) -> None:
        self.load_project(recovery.filepath)
        self.project_path = None