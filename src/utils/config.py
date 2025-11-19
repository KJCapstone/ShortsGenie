"""Configuration management for the application."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class VideoConfig:
    """Video processing configuration."""
    output_width: int = 1080
    output_height: int = 1920
    fps: int = 30
    codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 23  # Constant Rate Factor (lower = better quality)


@dataclass
class DetectionConfig:
    """Object detection configuration."""
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.45
    target_classes: List[int] = field(default_factory=lambda: [0, 32])  # person, sports ball
    max_detections: int = 300

    # Tracking parameters
    track_buffer: int = 30  # ByteTrack buffer size
    track_thresh: float = 0.5  # High confidence threshold
    match_thresh: float = 0.8  # IOU threshold for matching


@dataclass
class ROIConfig:
    """ROI calculation configuration."""
    ball_weight: float = 3.0  # Weight for ball in ROI calculation
    person_weight: float = 1.0  # Weight for person in ROI calculation
    min_roi_size: int = 200  # Minimum ROI size in pixels
    use_smoothing: bool = True


@dataclass
class KalmanConfig:
    """Kalman filter configuration."""
    process_variance: float = 1e-3  # Process noise variance
    measurement_variance: float = 1e-1  # Measurement noise variance
    initial_state_covariance: float = 1.0


@dataclass
class KeyPlayerConfig:
    """Key player detection configuration."""
    distance_to_ball_threshold: int = 200  # Max distance to ball (pixels)
    speed_threshold: float = 50.0  # Min movement speed (pixels/frame)
    center_distance_threshold: int = 300  # Max distance to screen center
    confidence_threshold: float = 0.7  # Min detection confidence
    importance_threshold: float = 4.0  # Min importance score to be "key player"

    # Scoring weights
    distance_score_weight: float = 3.0
    speed_score_weight: float = 2.0
    center_score_weight: float = 1.0
    confidence_score_weight: float = 1.0


@dataclass
class AudioConfig:
    """Audio filtering and transcription configuration."""
    # Whisper STT settings
    whisper_model: str = "base"  # tiny, base, small, medium, large
    whisper_device: str = "auto"  # auto, mps, cuda, cpu
    whisper_language: str = "ko"  # Language code or None for auto-detect

    # Highlight filtering settings
    threshold_percentile: float = 60.0  # Top N% segments to keep (40% reduction)
    merge_gap: float = 2.0  # Merge segments within N seconds
    segment_duration: float = 5.0  # Each segment duration in seconds

    # Audio analysis weights
    rms_weight: float = 0.7  # RMS energy weight (volume)
    spectral_weight: float = 0.3  # Spectral centroid weight (frequency)

    # Processing options
    verbose: bool = True  # Show progress information


@dataclass
class CropperConfig:
    """Video cropping and letterboxing configuration."""
    letterbox_top: int = 288  # Top black bar size in pixels (15% of 1920)
    letterbox_bottom: int = 288  # Bottom black bar size in pixels (15% of 1920)
    # Note: No horizontal letterbox - width always fills completely (1080px)


@dataclass
class AppConfig:
    """Main application configuration."""
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    key_player: KeyPlayerConfig = field(default_factory=KeyPlayerConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    cropper: CropperConfig = field(default_factory=CropperConfig)
