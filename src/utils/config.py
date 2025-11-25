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
    model_path: str = "yolov8m.pt"  # Upgraded from yolov8n.pt for better accuracy
    confidence_threshold: float = 0.05  # Lowered from 0.3 for higher recall
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

    # Hysteresis parameters (dead zone to prevent jittery camera movement)
    use_hysteresis: bool = True  # Enable hysteresis-based ROI update
    hysteresis_threshold: int = 100  # Dead zone radius in pixels
    min_movement_frames: int = 3  # Frames before confirming movement
    adaptive_threshold: bool = True  # Scale threshold by confidence

    # Scene locking parameters (lock ROI when ball stays in area)
    use_scene_locking: bool = True  # Enable scene-coherent ROI locking
    lock_threshold: int = 150  # Distance to trigger unlock (pixels)
    lock_min_frames: int = 15  # Minimum frames before establishing lock
    lock_decay_rate: float = 0.95  # Lock confidence decay per frame


@dataclass
class KalmanConfig:
    """Kalman filter configuration."""
    process_variance: float = 1e-3  # Process noise variance
    measurement_variance: float = 1e-1  # Measurement noise variance
    initial_state_covariance: float = 1.0


@dataclass
class SmootherConfig:
    """Smoothing configuration."""
    method: str = "adaptive_ema"  # Options: "kalman", "ema", "adaptive_ema"

    # Kalman parameters (existing)
    process_variance: float = 1e-3
    measurement_variance: float = 1e-1
    initial_state_covariance: float = 1.0

    # EMA parameters
    ema_alpha: float = 0.1  # Fixed EMA smoothing factor (0-1)
    ema_alpha_min: float = 0.05  # Adaptive EMA minimum (very smooth)
    ema_alpha_max: float = 0.25  # Adaptive EMA maximum (fast response)


@dataclass
class SceneDetectionConfig:
    """Scene detection configuration."""
    enabled: bool = True
    method: str = "pyscenedetect"  # Options: "pyscenedetect", "transnetv2", "optical"

    # PySceneDetect parameters
    adaptive: bool = True  # Use AdaptiveDetector vs ContentDetector
    threshold: float = 3.0  # Detection sensitivity (lower = more sensitive)
    min_scene_len: int = 15  # Minimum frames per scene (0.5s at 30fps)

    # TransNetV2 parameters
    transnet_model_dir: str = "resources/models/transnetv2/"
    transnet_threshold: float = 0.5  # Probability threshold

    # Optical flow parameters
    hist_threshold: float = 0.7  # Histogram correlation threshold
    flow_threshold: float = 15.0  # Optical flow magnitude threshold


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
    smoother: SmootherConfig = field(default_factory=SmootherConfig)
    scene_detection: SceneDetectionConfig = field(default_factory=SceneDetectionConfig)
    key_player: KeyPlayerConfig = field(default_factory=KeyPlayerConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    cropper: CropperConfig = field(default_factory=CropperConfig)
