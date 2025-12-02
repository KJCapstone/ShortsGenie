"""Pipeline configuration for hybrid highlight generation system.

This module provides flexible configuration for enabling/disabling different
processing modules and defining mode-specific strategies.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ModuleConfig:
    """Base configuration for a processing module."""
    enabled: bool = True
    priority: int = 10  # Lower = higher priority


@dataclass
class OCRDetectionConfig(ModuleConfig):
    """Scoreboard OCR-based goal detection configuration."""
    enabled: bool = False  # Disabled by default (too slow)
    priority: int = 1  # Highest priority for goal detection

    # Detection parameters
    confidence_threshold: float = 0.7
    min_score_change: int = 1  # Minimum score change to consider as goal
    search_region: Optional[tuple] = None  # (x, y, w, h) or None for auto-detect

    # Performance
    frame_skip: int = 5  # Process every Nth frame for speed


@dataclass
class AudioAnalysisConfig(ModuleConfig):
    """Audio-based highlight detection configuration."""
    enabled: bool = True
    priority: int = 2

    # Energy analysis
    window_duration: float = 2.0  # seconds
    energy_threshold: float = 0.7  # Normalized threshold

    # Spectral analysis
    use_spectral_features: bool = True

    # Excitement detection
    excitement_threshold: float = 0.75


@dataclass
class TranscriptAnalysisConfig(ModuleConfig):
    """Whisper + AI transcript analysis configuration."""
    enabled: bool = True
    priority: int = 3

    # Backend selection
    backend: str = "whisper"  # "whisper" (local) or "groq" (cloud API)

    # Whisper settings (for local backend)
    model_size: str = "base"  # tiny, base, small, medium, large
    language: str = "auto"  # Auto-detect language (supports 99 languages)

    # Groq settings (for cloud backend)
    groq_api_key: Optional[str] = None  # API key (from GUI or .env)
    groq_model: str = "whisper-large-v3-turbo"  # whisper-large-v3-turbo or whisper-large-v3

    # AI analysis
    use_gemini: bool = True
    extract_context: bool = True  # Add descriptions to highlights


@dataclass
class SceneDetectionConfig(ModuleConfig):
    """Scene boundary detection configuration."""
    enabled: bool = False  # Disabled by default (performance)
    priority: int = 4

    method: str = "pyscenedetect"  # pyscenedetect, transnetv2
    threshold: float = 3.0
    min_scene_len: int = 15


@dataclass
class SceneClassificationConfig(ModuleConfig):
    """Auto_tagger ResNet18 scene classification configuration."""
    enabled: bool = True
    priority: int = 5  # After highlight extraction, before reframing

    # Model settings
    model_path: str = "resources/models/scene_classifier/soccer_model_ver2.pth"
    device: str = "auto"  # auto, mps, cuda, cpu

    # PySceneDetect parameters (for boundary detection)
    threshold: float = 27.0  # ContentDetector threshold
    min_scene_len: int = 15  # Minimum frames per scene

    # Classification parameters
    min_scene_duration: float = 0.5  # Skip very short scenes

    # Output settings
    save_json: bool = True
    json_output_dir: str = "output/scene_metadata"


@dataclass
class ReframingConfig(ModuleConfig):
    """Dynamic 16:9 → 9:16 reframing configuration."""
    enabled: bool = True
    priority: int = 5

    # Detection
    use_soccernet_model: bool = True  # Use fine-tuned model if available
    confidence_threshold: float = 0.05  # Low for high recall

    # Detector backend selection
    detector_backend: str = "yolo"  # Options: "yolo", "soccernet"

    # Filtering
    use_temporal_filter: bool = True
    use_kalman_smoothing: bool = True

    # ROI
    ball_weight: float = 3.0
    person_weight: float = 1.0


@dataclass
class VideoEditingConfig(ModuleConfig):
    """Video editing and export configuration."""
    enabled: bool = True
    priority: int = 6

    # Output format
    output_width: int = 720   # Reduced from 1080 for smaller file size
    output_height: int = 1280  # Reduced from 1920 (still HD quality for shorts)
    fps: int = 30
    codec: str = "libx264"
    crf: int = 28  # Quality (18-28 recommended, higher=smaller file, lower=better quality)

    # Clip parameters
    clip_padding_before: float = 5.0  # seconds before event
    clip_padding_after: float = 5.0   # seconds after event
    max_clip_duration: float = 30.0   # Maximum clip length


@dataclass
class PipelineConfig:
    """Complete pipeline configuration with mode-specific presets."""

    # Module configurations
    ocr_detection: OCRDetectionConfig = field(default_factory=OCRDetectionConfig)
    audio_analysis: AudioAnalysisConfig = field(default_factory=AudioAnalysisConfig)
    transcript_analysis: TranscriptAnalysisConfig = field(default_factory=TranscriptAnalysisConfig)
    scene_detection: SceneDetectionConfig = field(default_factory=SceneDetectionConfig)
    scene_classification: SceneClassificationConfig = field(default_factory=SceneClassificationConfig)
    reframing: ReframingConfig = field(default_factory=ReframingConfig)
    video_editing: VideoEditingConfig = field(default_factory=VideoEditingConfig)

    # Pipeline settings
    mode: str = "골"  # 골, 경기, 밈
    max_highlights: int = 6  # Maximum number of highlights to generate (~3min total)
    min_highlight_duration: float = 10.0  # Minimum seconds
    max_highlight_duration: float = 60.0  # Maximum seconds

    # Performance
    use_gpu: bool = True
    num_workers: int = 4
    cache_dir: Path = field(default_factory=lambda: Path("cache/pipeline"))

    # Parallel processing
    enable_parallel_clip_generation: bool = True  # Enable parallel clip processing
    max_parallel_workers: int = 2  # Max workers for parallel clip generation (GPU constrained)

    # Output
    output_dir: Path = field(default_factory=lambda: Path("output"))
    temp_dir: Path = field(default_factory=lambda: Path("output/temp"))
    save_debug_info: bool = False  # Save intermediate results

    @classmethod
    def for_goal_mode(cls) -> "PipelineConfig":
        """Create optimized configuration for goal highlight mode.

        Goal mode prioritizes:
        - Audio analysis (main detection method)
        - Transcript for context
        - Reframing for vertical format
        - OCR detection (disabled by default - too slow)

        Returns:
            PipelineConfig optimized for goal detection
        """
        config = cls()
        config.mode = "골"

        # Enable core modules
        config.ocr_detection.enabled = False  # Disabled (too slow)
        config.audio_analysis.enabled = False  # Disabled
        config.transcript_analysis.enabled = True  # Whisper + Gemini
        config.reframing.enabled = True

        # Enable scene classification for scene-aware ROI
        config.scene_classification.enabled = True

        # Disable old scene detection module
        config.scene_detection.enabled = False

        # Goal-specific parameters
        config.max_highlights = 10
        config.min_highlight_duration = 15.0  # Goals need ~15-30s
        config.max_highlight_duration = 50.0  # Increased to 50s to accommodate longer highlights

        config.video_editing.clip_padding_before = 10.0  # Show build-up
        config.video_editing.clip_padding_after = 15.0   # Show celebration

        return config

    @classmethod
    def for_match_mode(cls) -> "PipelineConfig":
        """Create optimized configuration for match highlight mode.

        Match mode prioritizes:
        - Audio analysis (excitement peaks)
        - Scene detection (key moments)
        - Transcript for context
        - Optional reframing

        Returns:
            PipelineConfig optimized for match highlights
        """
        config = cls()
        config.mode = "경기"

        # Enable audio-based detection
        config.ocr_detection.enabled = False  # Not needed for general highlights
        config.audio_analysis.enabled = True
        config.audio_analysis.priority = 1  # Highest priority
        config.transcript_analysis.enabled = True
        config.scene_detection.enabled = True
        config.reframing.enabled = True

        # Match-specific parameters
        config.max_highlights = 15
        config.min_highlight_duration = 5.0   # Shorter clips OK
        config.max_highlight_duration = 20.0

        config.audio_analysis.energy_threshold = 0.65  # More sensitive

        return config

    @classmethod
    def for_meme_mode(cls) -> "PipelineConfig":
        """Create optimized configuration for meme/funny moment mode.

        Meme mode prioritizes:
        - Transcript analysis (funny commentary)
        - Audio analysis (crowd reactions)
        - Short clips
        - Optional reframing

        Returns:
            PipelineConfig optimized for meme generation
        """
        config = cls()
        config.mode = "밈"

        # Enable transcript-focused modules
        config.ocr_detection.enabled = False
        config.audio_analysis.enabled = True
        config.transcript_analysis.enabled = True
        config.transcript_analysis.priority = 1  # Highest priority
        config.scene_detection.enabled = False
        config.reframing.enabled = True

        # Meme-specific parameters
        config.max_highlights = 20  # More clips
        config.min_highlight_duration = 3.0   # Very short clips
        config.max_highlight_duration = 10.0

        config.video_editing.clip_padding_before = 1.0
        config.video_editing.clip_padding_after = 2.0

        return config

    def get_enabled_modules(self) -> list[tuple[str, ModuleConfig]]:
        """Get list of enabled modules sorted by priority.

        Returns:
            List of (module_name, config) tuples, sorted by priority
        """
        modules = [
            ("ocr_detection", self.ocr_detection),
            ("audio_analysis", self.audio_analysis),
            ("transcript_analysis", self.transcript_analysis),
            ("scene_detection", self.scene_detection),
            ("reframing", self.reframing),
            ("video_editing", self.video_editing),
        ]

        # Filter enabled modules and sort by priority
        enabled = [(name, cfg) for name, cfg in modules if cfg.enabled]
        enabled.sort(key=lambda x: x[1].priority)

        return enabled

    def disable_module(self, module_name: str) -> None:
        """Disable a specific module at runtime.

        Args:
            module_name: Name of module to disable
        """
        if hasattr(self, module_name):
            getattr(self, module_name).enabled = False

    def enable_module(self, module_name: str) -> None:
        """Enable a specific module at runtime.

        Args:
            module_name: Name of module to enable
        """
        if hasattr(self, module_name):
            getattr(self, module_name).enabled = True


def create_config_from_mode(mode: str) -> PipelineConfig:
    """Factory function to create config based on mode string.

    Args:
        mode: One of "골", "경기", "밈"

    Returns:
        PipelineConfig optimized for the specified mode

    Raises:
        ValueError: If mode is not recognized
    """
    mode_map = {
        "골": PipelineConfig.for_goal_mode,
        "경기": PipelineConfig.for_match_mode,
        "밈": PipelineConfig.for_meme_mode,
    }

    if mode not in mode_map:
        raise ValueError(f"Unknown mode: {mode}. Must be one of {list(mode_map.keys())}")

    return mode_map[mode]()
