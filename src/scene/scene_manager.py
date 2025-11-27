"""Scene manager for tracking scene state during pipeline execution."""

from typing import Dict, Optional
from src.scene.scene_metadata import SceneMetadata, SceneSegment


class SceneManager:
    """Manages scene state during video processing pipeline.

    This class tracks the current scene and detects scene boundaries
    as the pipeline processes frames sequentially.
    """

    def __init__(self, metadata: SceneMetadata):
        """Initialize scene manager.

        Args:
            metadata: Scene metadata containing all segments
        """
        self.metadata = metadata
        self.current_scene: Optional[SceneSegment] = None
        self.last_frame = -1
        self.scene_change_count = 0

        print(f"[SceneManager] Initialized with {len(metadata.segments)} scenes")
        print(f"[SceneManager] Source: {metadata.source}, FPS: {metadata.fps}")

        # Print scene summary
        scene_types = {}
        for seg in metadata.segments:
            scene_types[seg.label] = scene_types.get(seg.label, 0) + 1

        print(f"[SceneManager] Scene distribution: {scene_types}")

    def update(self, frame_num: int) -> Dict[str, any]:
        """Update scene state for the given frame.

        This method should be called for each frame in sequential order.
        It detects scene boundaries and tracks the current scene type.

        Args:
            frame_num: Current frame number

        Returns:
            Dictionary with scene information:
            - is_boundary: True if this is the start of a new scene
            - scene_type: Current scene type label
            - scene_changed: True if scene just changed
            - frame_num: Current frame number
            - scene_index: Index of current scene in metadata
        """
        # Get scene for current frame
        scene = self.metadata.get_scene_at_frame(frame_num)

        # Check if scene changed
        scene_changed = False
        is_boundary = False

        if scene != self.current_scene:
            scene_changed = True
            is_boundary = True
            self.scene_change_count += 1

            if scene:
                print(f"[SceneManager] Frame {frame_num}: New scene '{scene.label}' "
                      f"({scene.start_seconds:.1f}s - {scene.end_seconds:.1f}s)")
            else:
                print(f"[SceneManager] Frame {frame_num}: No scene data (gap in metadata)")

            self.current_scene = scene

        self.last_frame = frame_num

        # Find scene index
        scene_index = -1
        if scene:
            try:
                scene_index = self.metadata.segments.index(scene)
            except ValueError:
                pass

        return {
            'is_boundary': is_boundary,
            'scene_type': scene.label if scene else 'unknown',
            'scene_changed': scene_changed,
            'frame_num': frame_num,
            'scene_index': scene_index,
            'scene': scene
        }

    def get_current_scene_info(self) -> Dict[str, any]:
        """Get information about the current scene.

        Returns:
            Dictionary with current scene details or empty dict if no scene
        """
        if not self.current_scene:
            return {}

        return {
            'label': self.current_scene.label,
            'start_frame': self.current_scene.start_frame,
            'end_frame': self.current_scene.end_frame,
            'start_seconds': self.current_scene.start_seconds,
            'end_seconds': self.current_scene.end_seconds,
            'duration': self.current_scene.duration
        }

    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about scene processing.

        Returns:
            Dictionary with processing statistics
        """
        return {
            'total_scenes': len(self.metadata.segments),
            'scenes_encountered': self.scene_change_count,
            'last_frame_processed': self.last_frame,
            'current_scene_type': self.current_scene.label if self.current_scene else None
        }

    def reset(self):
        """Reset manager state (useful for reprocessing)."""
        self.current_scene = None
        self.last_frame = -1
        self.scene_change_count = 0
        print("[SceneManager] State reset")
