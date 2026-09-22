"""
core/history_manager.py

Action-based Command Pattern Undo/Redo engine for DarkLabel Modern.
Stores lightweight delta state transitions rather than heavy full-video project snapshots,
keeping memory footprints minimal even on large datasets with tens of thousands of frames.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from typing import Any, Callable, Dict, List, Optional

from core.annotation_manager import AnnotationManager
from core.annotation_models import Annotation, Track
from core.track_manager import MergeStrategy


# ==============================================================================
# Base Command Definition
# ==============================================================================

class Command(ABC):
    """Abstract base class for all reversible annotation and track actions."""

    def __init__(self, description: str = ""):
        self.description = description

    @abstractmethod
    def execute(self, manager: AnnotationManager) -> bool:
        """Executes the action on the annotation manager. Returns True if successful."""
        pass

    @abstractmethod
    def undo(self, manager: AnnotationManager) -> bool:
        """Rolls back the action, returning manager to its prior state."""
        pass

    def redo(self, manager: AnnotationManager) -> bool:
        """Re-applies the action after an undo. Defaults to calling execute()."""
        return self.execute(manager)


# ==============================================================================
# Annotation Commands (Delta Transitions)
# ==============================================================================

class AddAnnotationCommand(Command):
    """Adds a single annotation to the timeline."""

    def __init__(self, annotation: Annotation, description: str = ""):
        super().__init__(description or f"Add #{annotation.track_id} on frame {annotation.frame_index}")
        self.annotation = annotation.copy()

    def execute(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            manager.add_or_update_annotation(self.annotation.copy(), auto_create_track=True)
            return True

    def undo(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            manager.remove_annotation(self.annotation.frame_index, self.annotation.track_id, force=True)
            return True


class RemoveAnnotationCommand(Command):
    """Deletes an annotation, preserving its exact state for rollback."""

    def __init__(self, frame_index: int, track_id: int, description: str = ""):
        super().__init__(description or f"Delete #{track_id} on frame {frame_index}")
        self.frame_index = frame_index
        self.track_id = track_id
        self.previous_annotation: Optional[Annotation] = None

    def execute(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            existing = manager.get_annotation(self.frame_index, self.track_id)
            if existing is None:
                return False
            self.previous_annotation = existing.copy()
            manager.remove_annotation(self.frame_index, self.track_id, force=True)
            return True

    def undo(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            if self.previous_annotation is None:
                return False
            manager.add_or_update_annotation(self.previous_annotation.copy(), auto_create_track=True)
            return True


class UpdateAnnotationCommand(Command):
    """Updates geometry, flags, or attributes of an existing annotation."""

    def __init__(self, new_annotation: Annotation, description: str = ""):
        super().__init__(description or f"Modify #{new_annotation.track_id} on frame {new_annotation.frame_index}")
        self.new_annotation = new_annotation.copy()
        self.previous_annotation: Optional[Annotation] = None

    def execute(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            existing = manager.get_annotation(self.new_annotation.frame_index, self.new_annotation.track_id)
            if existing is not None:
                self.previous_annotation = existing.copy()
            manager.add_or_update_annotation(self.new_annotation.copy(), auto_create_track=True)
            return True

    def undo(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            if self.previous_annotation is not None:
                manager.add_or_update_annotation(self.previous_annotation.copy(), auto_create_track=True)
            else:
                # If it didn't exist before, undo means removing it
                manager.remove_annotation(
                    self.new_annotation.frame_index,
                    self.new_annotation.track_id,
                    force=True
                )
            return True


class BatchAddAnnotationCommand(Command):
    """Adds multiple annotations simultaneously (e.g., auto-tagging, interpolation)."""

    def __init__(self, annotations: List[Annotation], description: str = ""):
        super().__init__(description or f"Batch add {len(annotations)} annotations")
        self.annotations = [a.copy() for a in annotations]
        self.previous_states: Dict[Tuple[int, int], Optional[Annotation]] = {}

    def execute(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            self.previous_states.clear()
            for ann in self.annotations:
                prev = manager.get_annotation(ann.frame_index, ann.track_id)
                self.previous_states[(ann.frame_index, ann.track_id)] = prev.copy() if prev else None
                manager.add_or_update_annotation(ann.copy(), auto_create_track=True)
            return True

    def undo(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            for (f_idx, tid), prev in self.previous_states.items():
                if prev is not None:
                    manager.add_or_update_annotation(prev.copy(), auto_create_track=True)
                else:
                    manager.remove_annotation(f_idx, tid, force=True)
            return True


class DeleteTailCommand(Command):
    """DarkLabel track termination: deletes annotations from frame N+1 to end."""

    def __init__(self, track_id: int, from_frame: int, description: str = ""):
        super().__init__(description or f"Terminate #{track_id} after frame {from_frame}")
        self.track_id = track_id
        self.from_frame = from_frame
        self.purged_annotations: List[Annotation] = []

    def execute(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            track = manager.track_manager.get_track(self.track_id)
            if not track:
                return False
            # Save deleted items before removing
            self.purged_annotations = [
                ann.copy() for f, ann in track.annotations.items() if f > self.from_frame
            ]
            manager.terminate_track(self.track_id, self.from_frame, force=True)
            return True

    def undo(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            for ann in self.purged_annotations:
                manager.add_or_update_annotation(ann.copy(), auto_create_track=True)
            return True


class DeleteIntervalCommand(Command):
    """Deletes all annotations within a frame range."""

    def __init__(
        self,
        start_frame: int,
        end_frame: int,
        track_id: Optional[int] = None,
        description: str = ""
    ):
        super().__init__(
            description or f"Purge interval [{start_frame}, {end_frame}]"
            + (f" for #{track_id}" if track_id else "")
        )
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.track_id = track_id
        self.purged_annotations: List[Annotation] = []

    def execute(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            self.purged_annotations.clear()
            for f in range(self.start_frame, self.end_frame + 1):
                boxes = manager.get_annotations(f, visible_only=False, include_outside=True)
                for b in boxes:
                    if self.track_id is None or b.track_id == self.track_id:
                        self.purged_annotations.append(b.copy())
            manager.delete_interval(self.start_frame, self.end_frame, track_id=self.track_id, force=True)
            return True

    def undo(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            for ann in self.purged_annotations:
                manager.add_or_update_annotation(ann.copy(), auto_create_track=True)
            return True


# ==============================================================================
# Track Identity Commands (Split, Merge, Delete)
# ==============================================================================

class SplitTrackCommand(Command):
    """Splits a track into two distinct identities at split_frame."""

    def __init__(
        self,
        track_id: int,
        split_frame: int,
        new_track_id: Optional[int] = None,
        description: str = ""
    ):
        super().__init__(description or f"Split Track #{track_id} at frame {split_frame}")
        self.track_id = track_id
        self.split_frame = split_frame
        self.designated_new_id = new_track_id
        self.actual_new_id: Optional[int] = None

    def execute(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            res = manager.split_track(
                self.track_id,
                self.split_frame,
                new_track_id=self.designated_new_id,
                force=True
            )
            self.actual_new_id = res.new_track_id
            return True

    def undo(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            if self.actual_new_id is None:
                return False
            # Re-merge the newly created tail track back into the original track
            manager.merge_tracks(
                source_track_id=self.actual_new_id,
                target_track_id=self.track_id,
                strategy=MergeStrategy.PREFER_SOURCE,
                force=True
            )
            return True


class MergeTrackCommand(Command):
    """Merges source track into target track, resolving any temporal overlaps."""

    def __init__(
        self,
        source_track_id: int,
        target_track_id: int,
        strategy: MergeStrategy = MergeStrategy.FAIL_ON_CONFLICT,
        description: str = ""
    ):
        super().__init__(description or f"Merge Track #{source_track_id} into #{target_track_id}")
        self.source_track_id = source_track_id
        self.target_track_id = target_track_id
        self.strategy = strategy

        # Snapshot source track and target's overwritten boxes for exact rollback
        self.cached_source_track: Optional[Track] = None
        self.cached_overwritten_target_boxes: Dict[int, Annotation] = {}

    def execute(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            s_track = manager.track_manager.get_track(self.source_track_id)
            t_track = manager.track_manager.get_track(self.target_track_id)
            if not s_track or not t_track:
                return False

            self.cached_source_track = Track.from_dict(s_track.to_dict())

            # Detect overwrites if PREFER_SOURCE is chosen
            if self.strategy == MergeStrategy.PREFER_SOURCE:
                overlap = manager.track_manager.detect_temporal_overlap(
                    self.source_track_id, self.target_track_id
                )
                self.cached_overwritten_target_boxes = {
                    f: t_track.annotations[f].copy() for f in overlap if f in t_track.annotations
                }

            manager.merge_tracks(
                self.source_track_id,
                self.target_track_id,
                strategy=self.strategy,
                force=True
            )
            return True

    def undo(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            if not self.cached_source_track:
                return False

            target_track = manager.track_manager.get_track(self.target_track_id)
            if not target_track:
                return False

            # Remove migrated annotations from target track
            for f in self.cached_source_track.annotations.keys():
                target_track.remove_annotation(f)

            # Restore overwritten target boxes
            for f, ann in self.cached_overwritten_target_boxes.items():
                target_track.set_annotation(ann.copy())

            # Re-register source track
            restored_source = Track.from_dict(self.cached_source_track.to_dict())
            manager.track_manager.register_track(restored_source)

            manager.rebuild_frame_index()
            return True


class DeleteTrackCommand(Command):
    """Deletes an entire track trajectory, preserving state for undo."""

    def __init__(self, track_id: int, description: str = ""):
        super().__init__(description or f"Delete entire Track #{track_id}")
        self.track_id = track_id
        self.cached_track: Optional[Track] = None

    def execute(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            track = manager.track_manager.get_track(self.track_id)
            if not track:
                return False
            self.cached_track = Track.from_dict(track.to_dict())
            manager.delete_track(self.track_id, force=True)
            return True

    def undo(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            if not self.cached_track:
                return False
            restored = Track.from_dict(self.cached_track.to_dict())
            manager.track_manager.register_track(restored)
            manager.rebuild_frame_index()
            return True


# ==============================================================================
# Macro Command (Atomic Grouping)
# ==============================================================================

class MacroCommand(Command):
    """Groups multiple sub-commands into a single atomic action on the undo stack."""

    def __init__(self, description: str = "Batch Operation"):
        super().__init__(description)
        self.commands: List[Command] = []

    def add_command(self, cmd: Command) -> None:
        self.commands.append(cmd)

    def execute(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            for cmd in self.commands:
                if not cmd.execute(manager):
                    return False
            return True

    def undo(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            # Undo in reverse execution order
            for cmd in reversed(self.commands):
                if not cmd.undo(manager):
                    return False
            return True

    def redo(self, manager: AnnotationManager) -> bool:
        with manager.lock:
            for cmd in self.commands:
                if not cmd.redo(manager):
                    return False
            return True


# ==============================================================================
# History Manager & Transaction Engine
# ==============================================================================

class HistoryManager:
    """
    Manages undo/redo stacks, transactions, and project modification flags.
    """

    def __init__(self, manager: AnnotationManager, max_depth: int = 150):
        self.manager = manager
        self.max_depth = max_depth

        self.undo_stack: List[Command] = []
        self.redo_stack: List[Command] = []

        # Macro transaction nesting
        self._active_macro: Optional[MacroCommand] = None
        self._transaction_depth = 0

        # Change listeners
        self._on_change_callbacks: List[Callable[[bool, bool], None]] = []
        self._saved_command_index = 0

    def register_change_callback(self, callback: Callable[[bool, bool], None]) -> None:
        """Callback signature: fn(can_undo: bool, can_redo: bool)"""
        self._on_change_callbacks.append(callback)

    def _notify(self) -> None:
        can_u = self.can_undo
        can_r = self.can_redo
        for cb in self._on_change_callbacks:
            try:
                cb(can_u, can_r)
            except Exception:
                pass

    @property
    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    @property
    def is_dirty(self) -> bool:
        """Returns True if current history index differs from last saved state."""
        return len(self.undo_stack) != self._saved_command_index

    def mark_saved(self) -> None:
        """Designates current state as clean / synchronized with disk."""
        self._saved_command_index = len(self.undo_stack)

    def execute(self, command: Command) -> bool:
        """Executes a command and pushes it onto the undo stack."""
        if self._active_macro is not None:
            # Inside a transaction: record for batch commit
            success = command.execute(self.manager)
            if success:
                self._active_macro.add_command(command)
            return success

        success = command.execute(self.manager)
        if success:
            if len(self.undo_stack) >= self.max_depth:
                self.undo_stack.pop(0)
                self._saved_command_index = max(-1, self._saved_command_index - 1)

            self.undo_stack.append(command)
            self.redo_stack.clear()
            self._notify()
        return success

    def undo(self) -> Optional[Command]:
        """Rolls back the most recent command."""
        if not self.undo_stack:
            return None

        cmd = self.undo_stack.pop()
        success = cmd.undo(self.manager)
        if success:
            self.redo_stack.append(cmd)
            self._notify()
            return cmd
        else:
            # Rollback failed, restore stack state
            self.undo_stack.append(cmd)
            return None

    def redo(self) -> Optional[Command]:
        """Re-applies the most recently undone command."""
        if not self.redo_stack:
            return None

        cmd = self.redo_stack.pop()
        success = cmd.redo(self.manager)
        if success:
            self.undo_stack.append(cmd)
            self._notify()
            return cmd
        else:
            # Redo failed, restore redo stack
            self.redo_stack.append(cmd)
            return None

    def begin_transaction(self, description: str = "Batch Operation") -> None:
        """Begins grouping operations into an atomic MacroCommand."""
        if self._transaction_depth == 0:
            self._active_macro = MacroCommand(description)
        self._transaction_depth += 1

    def commit_transaction(self) -> None:
        """Commits the active transaction as a single item on the undo stack."""
        if self._transaction_depth <= 0:
            return

        self._transaction_depth -= 1
        if self._transaction_depth == 0 and self._active_macro is not None:
            macro = self._active_macro
            self._active_macro = None
            if macro.commands:
                if len(self.undo_stack) >= self.max_depth:
                    self.undo_stack.pop(0)
                    self._saved_command_index = max(-1, self._saved_command_index - 1)
                self.undo_stack.append(macro)
                self.redo_stack.clear()
                self._notify()

    def rollback_transaction(self) -> None:
        """Aborts the transaction and undoes any sub-commands executed so far."""
        if self._active_macro is not None:
            self._active_macro.undo(self.manager)
            self._active_macro = None
        self._transaction_depth = 0
        self._notify()

    def clear(self) -> None:
        """Wipes undo and redo history."""
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._active_macro = None
        self._transaction_depth = 0
        self._saved_command_index = 0
        self._notify()