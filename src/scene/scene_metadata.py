"""Scene metadata models and JSON parsing utilities."""

from dataclasses import dataclass
from typing import List, Optional
import json
from pathlib import Path


@dataclass
class SceneSegment:
    """Single scene segment with type classification.

    Attributes:
        label: Scene type ("wide", "close", "audience", "replay")
        start_seconds: Start time in seconds
        end_seconds: End time in seconds
        start_frame: Start frame number (calculated from fps)
        end_frame: End frame number (calculated from fps)
        duration: Scene duration in seconds
    """
    label: str
    start_seconds: float
    end_seconds: float
    start_frame: int
    end_frame: int
    duration: float

    def contains_frame(self, frame_num: int) -> bool:
        """Check if this segment contains the given frame."""
        return self.start_frame <= frame_num < self.end_frame


@dataclass
class SceneMetadata:
    """Complete scene metadata for a video.

    Attributes:
        segments: List of scene segments
        fps: Video frames per second
        source: Source of scene data ("auto_tagger", "pyscenedetect")
    """
    segments: List[SceneSegment]
    fps: float
    source: str = "auto_tagger"

    def get_scene_at_frame(self, frame_num: int) -> Optional[SceneSegment]:
        """Get the scene segment containing the given frame.

        Args:
            frame_num: Frame number to query

        Returns:
            SceneSegment if found, None otherwise
        """
        for segment in self.segments:
            if segment.contains_frame(frame_num):
                return segment
        return None

    def is_scene_boundary(self, frame_num: int, tolerance: int = 0) -> bool:
        """Check if the given frame is at a scene boundary.

        Args:
            frame_num: Frame number to check
            tolerance: Number of frames tolerance for boundary detection

        Returns:
            True if frame is at the start of a new scene
        """
        for segment in self.segments:
            if segment.start_frame - tolerance <= frame_num <= segment.start_frame + tolerance:
                return True
        return False

    def get_scene_type(self, frame_num: int) -> str:
        """Get the scene type for the given frame.

        Args:
            frame_num: Frame number to query

        Returns:
            Scene type label, or "unknown" if not found
        """
        scene = self.get_scene_at_frame(frame_num)
        return scene.label if scene else "unknown"


def load_from_auto_tagger_json(json_path: str, fps: float) -> SceneMetadata:
    """Load scene metadata from auto_tagger JSON output.

    Expected JSON format (from auto_tagger/test.py):
    [
        {
            "label": "wide",
            "start_time": "00분 15초",
            "end_time": "00분 20초",
            "duration": 5.2,
            "start_seconds": 15.0,
            "end_seconds": 20.2
        },
        ...
    ]

    Args:
        json_path: Path to auto_tagger JSON file
        fps: Video frame rate for frame number calculation

    Returns:
        SceneMetadata object with parsed segments

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON format is invalid or missing required fields
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Scene JSON not found: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_path}: {e}")

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    segments = []
    for i, item in enumerate(data):
        # Validate required fields
        required_fields = ['label', 'start_seconds', 'end_seconds']
        missing_fields = [f for f in required_fields if f not in item]
        if missing_fields:
            raise ValueError(f"Segment {i} missing fields: {missing_fields}")

        # Extract data
        label = item['label']
        start_seconds = float(item['start_seconds'])
        end_seconds = float(item['end_seconds'])
        duration = item.get('duration', end_seconds - start_seconds)

        # Calculate frame numbers
        start_frame = int(start_seconds * fps)
        end_frame = int(end_seconds * fps)

        # Create segment
        segment = SceneSegment(
            label=label,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            start_frame=start_frame,
            end_frame=end_frame,
            duration=duration
        )
        segments.append(segment)

    # Sort segments by start time
    segments.sort(key=lambda s: s.start_seconds)

    return SceneMetadata(
        segments=segments,
        fps=fps,
        source="auto_tagger"
    )


def validate_scene_metadata(metadata: SceneMetadata) -> List[str]:
    """Validate scene metadata for consistency.

    Args:
        metadata: SceneMetadata to validate

    Returns:
        List of warning messages (empty if all valid)
    """
    warnings = []

    # Check for overlapping segments
    for i, seg1 in enumerate(metadata.segments):
        for seg2 in metadata.segments[i+1:]:
            if seg1.end_frame > seg2.start_frame:
                warnings.append(
                    f"Overlapping scenes: {seg1.label} ({seg1.start_frame}-{seg1.end_frame}) "
                    f"and {seg2.label} ({seg2.start_frame}-{seg2.end_frame})"
                )

    # Check for very short scenes (< 0.5 seconds)
    for seg in metadata.segments:
        if seg.duration < 0.5:
            warnings.append(
                f"Very short scene ({seg.duration:.2f}s): {seg.label} "
                f"at {seg.start_seconds:.1f}s"
            )

    # Check for unknown scene types
    valid_types = {'wide', 'close', 'audience', 'replay', 'unknown'}
    for seg in metadata.segments:
        if seg.label not in valid_types:
            warnings.append(
                f"Unknown scene type '{seg.label}' at {seg.start_seconds:.1f}s"
            )

    return warnings
