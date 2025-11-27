"""Scene awareness module for dynamic reframing.

This module provides scene detection and classification capabilities
for optimizing video reframing based on scene types (wide/close/audience/replay).
"""

from src.scene.scene_metadata import (
    SceneSegment,
    SceneMetadata,
    load_from_auto_tagger_json,
    validate_scene_metadata
)
from src.scene.scene_manager import SceneManager

__all__ = [
    'SceneSegment',
    'SceneMetadata',
    'SceneManager',
    'load_from_auto_tagger_json',
    'validate_scene_metadata'
]
