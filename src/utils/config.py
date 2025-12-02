"""Configuration management for the application."""

from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path
import os


def get_project_root() -> Path:
    """Get the project root directory (app/ folder)."""
    # This file is at app/src/utils/config.py, so go up 2 levels to get app/
    return Path(__file__).parent.parent.parent


def get_model_path(model_name: str) -> str:
    """
    Get the full path to a model file.

    Args:
        model_name: Name of the model file (e.g., "best.pt", "yolov8n.pt")

    Returns:
        Absolute path to the model file as a string
    """
    project_root = get_project_root()

    # Check in resources/models/ first (preferred location)
    models_dir = project_root / "resources" / "models"
    model_path = models_dir / model_name

    if model_path.exists():
        return str(model_path)

    # Fallback to app root for backward compatibility
    fallback_path = project_root / model_name
    if fallback_path.exists():
        return str(fallback_path)

    # If not found, return the preferred path (will be created when downloaded)
    return str(model_path)


@dataclass
class VideoConfig:
    """Video processing configuration."""
    output_width: int = 1080   # Full HD width for vertical video
    output_height: int = 1920  # Full HD height (9:16 aspect ratio)
    fps: int = 30
    codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 28  # Constant Rate Factor (18-28 recommended, higher=smaller file)


@dataclass
class DetectionConfig:
    """Object detection configuration."""
    model_path: str = field(default_factory=lambda: get_model_path("best.pt"))  # Fine-tuned model (soccer-specific)
    confidence_threshold: float = 0.05  # Lowered from 0.3 for higher recall
    iou_threshold: float = 0.45
    target_classes: List[int] = field(default_factory=lambda: [0, 32])  # person, sports ball
    max_detections: int = 300

    # Tracking parameters
    track_buffer: int = 30  # ByteTrack buffer size
    track_thresh: float = 0.5  # High confidence threshold
    match_thresh: float = 0.8  # IOU threshold for matching

    # Detector backend selection
    detector_backend: str = "yolo"  # Options: "yolo", "soccernet"


@dataclass
class ROIConfig:
    """ROI calculation configuration."""
    ball_weight: float = 3.0  # Weight for ball in ROI calculation
    person_weight: float = 1.0  # Weight for person in ROI calculation
    min_roi_size: int = 200  # Minimum ROI size in pixels
    use_smoothing: bool = True

    # ROI scaling parameter (zoom level) - NOT USED, use CropperConfig.letterbox instead
    roi_scale_factor: float = 1.0  # Scale factor for ROI size (>1.0 = wider view, <1.0 = tighter crop)

    # Hysteresis parameters (dead zone to prevent jittery camera movement)
    use_hysteresis: bool = True  # Enable hysteresis-based ROI update
    hysteresis_threshold: int = 150  # Dead zone radius in pixels (increased from 100)
    min_movement_frames: int = 5  # Frames before confirming movement (increased from 3)
    adaptive_threshold: bool = True  # Scale threshold by confidence

    # Scene locking parameters (lock ROI when ball stays in area)
    use_scene_locking: bool = True  # Enable scene-coherent ROI locking
    lock_threshold: int = 150  # Distance to trigger unlock (pixels)
    lock_min_frames: int = 15  # Minimum frames before establishing lock
    lock_decay_rate: float = 0.95  # Lock confidence decay per frame

    # Scene-aware ROI parameters
    use_scene_type_weights: bool = False  # Apply different weights per scene type
    scene_type_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'wide': {
            'ball_weight': 4.0,      # Focus on ball in wide shots
            'person_weight': 0.8,
        },
        'close': {
            'ball_weight': 3.0,      # Balanced framing
            'person_weight': 1.2,
        },
        'audience': {
            'ball_weight': 0.0,      # No ball expected in audience shots
            'person_weight': 2.0,
        },
        'replay': {
            'ball_weight': 5.0,      # Maximum focus on ball in replays
            'person_weight': 0.5,
        }
    })


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
    ema_alpha_min: float = 0.03  # Adaptive EMA minimum (very smooth, reduced from 0.05)
    ema_alpha_max: float = 0.20  # Adaptive EMA maximum (fast response, reduced from 0.25)


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
    transnet_model_dir: str = field(default_factory=lambda: str(get_project_root() / "resources" / "models" / "transnetv2"))
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
    letterbox_top: int = 480  # Top black bar size in pixels (25% of 1920, increased for wider view)
    letterbox_bottom: int = 480  # Bottom black bar size in pixels (25% of 1920, increased for wider view)
    # Note: No horizontal letterbox - width always fills completely (1080px)


@dataclass
class SceneConfig:
    """Scene awareness configuration for dynamic reframing."""
    enabled: bool = True  # Scene-aware ROI for stability (enabled by default)

    # Scene metadata source
    use_auto_tagger: bool = True  # Use auto_tagger ResNet-18 model
    auto_tagger_model_path: str = field(default_factory=lambda: str(get_project_root() / "auto_tagger" / "model" / "soccer_model.pth"))

    # Fallback to PySceneDetect if auto_tagger unavailable
    fallback_to_pyscenedetect: bool = True

    # Scene validation
    min_scene_duration: float = 0.5  # Minimum scene length in seconds
    validate_on_load: bool = True  # Validate metadata when loading


@dataclass
class AppConfig:
    """Main application configuration."""
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    smoother: SmootherConfig = field(default_factory=SmootherConfig)
    scene_detection: SceneDetectionConfig = field(default_factory=SceneDetectionConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    key_player: KeyPlayerConfig = field(default_factory=KeyPlayerConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    cropper: CropperConfig = field(default_factory=CropperConfig)
