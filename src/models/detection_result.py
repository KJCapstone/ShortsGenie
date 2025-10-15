"""Data models for detection and tracking results."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate
    width: int  # Box width
    height: int  # Box height

    @property
    def center_x(self) -> int:
        """Get center x coordinate."""
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        """Get center y coordinate."""
        return self.y + self.height // 2

    @property
    def area(self) -> int:
        """Get bounding box area."""
        return self.width * self.height


@dataclass
class Detection:
    """Single object detection result."""
    bbox: BoundingBox
    confidence: float  # Detection confidence [0.0-1.0]
    class_id: int  # YOLO class ID (0=person, 32=sports ball)
    track_id: Optional[int] = None  # Tracking ID from ByteTrack
    importance_score: float = 0.0  # Importance score for key player detection

    @property
    def class_name(self) -> str:
        """Get human-readable class name."""
        class_names = {
            0: "person",
            32: "sports ball"
        }
        return class_names.get(self.class_id, f"class_{self.class_id}")

    @property
    def is_ball(self) -> bool:
        """Check if detection is a sports ball."""
        return self.class_id == 32

    @property
    def is_person(self) -> bool:
        """Check if detection is a person."""
        return self.class_id == 0


@dataclass
class FrameDetections:
    """All detections for a single frame."""
    frame_number: int
    timestamp: float  # Timestamp in seconds
    detections: List[Detection] = field(default_factory=list)

    @property
    def ball_detections(self) -> List[Detection]:
        """Get all ball detections in this frame."""
        return [d for d in self.detections if d.is_ball]

    @property
    def person_detections(self) -> List[Detection]:
        """Get all person detections in this frame."""
        return [d for d in self.detections if d.is_person]

    @property
    def key_player_detections(self) -> List[Detection]:
        """Get detections marked as key players (importance_score >= 4.0)."""
        return [d for d in self.person_detections if d.importance_score >= 4.0]

    def add_detection(self, detection: Detection):
        """Add a detection to this frame."""
        self.detections.append(detection)


@dataclass
class ROI:
    """Region of Interest for dynamic reframing."""
    center_x: int
    center_y: int
    width: int
    height: int
    confidence: float = 1.0  # ROI confidence based on detections

    def to_bbox(self) -> BoundingBox:
        """Convert ROI to BoundingBox format."""
        return BoundingBox(
            x=self.center_x - self.width // 2,
            y=self.center_y - self.height // 2,
            width=self.width,
            height=self.height
        )
