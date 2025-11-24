"""Scene transition detection for sports videos."""

from typing import List, Tuple, Optional
from pathlib import Path

try:
    from scenedetect import detect, AdaptiveDetector, ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    print("[SceneDetector] WARNING: scenedetect not installed. Scene detection will be disabled.")
    print("[SceneDetector] Install with: pip install scenedetect[opencv]")

from src.utils.config import SceneDetectionConfig


class SceneDetector:
    """
    Detect scene transitions in sports videos using PySceneDetect.

    Features:
    - AdaptiveDetector for handling fast camera movement (ideal for sports)
    - ContentDetector for standard scene change detection
    - Configurable sensitivity and minimum scene length

    Use cases:
    - Reset ROI locking on scene changes
    - Detect transitions between gameplay, crowd shots, replays
    - Enable scene-specific processing strategies
    """

    def __init__(self, config: Optional[SceneDetectionConfig] = None):
        """Initialize scene detector.

        Args:
            config: Scene detection configuration. Uses defaults if None.
        """
        self.config = config or SceneDetectionConfig()

        if not SCENEDETECT_AVAILABLE:
            self.config.enabled = False
            print("[SceneDetector] Scene detection disabled (library not available)")
            return

        print(f"[SceneDetector] Method: {self.config.method}")
        print(f"[SceneDetector] Adaptive: {self.config.adaptive}")
        print(f"[SceneDetector] Threshold: {self.config.threshold}")
        print(f"[SceneDetector] Min scene length: {self.config.min_scene_len} frames")

    def detect_scenes(self, video_path: str) -> List[Tuple[int, int]]:
        """Detect scene boundaries in video.

        Args:
            video_path: Path to input video file

        Returns:
            List of (start_frame, end_frame) tuples for each scene.
            Returns empty list if detection disabled or fails.
        """
        if not self.config.enabled or not SCENEDETECT_AVAILABLE:
            print("[SceneDetector] Scene detection is disabled")
            return []

        # Check if video file exists
        if not Path(video_path).exists():
            print(f"[SceneDetector] ERROR: Video file not found: {video_path}")
            return []

        try:
            if self.config.method == "pyscenedetect":
                return self._detect_with_pyscenedetect(video_path)
            else:
                print(f"[SceneDetector] WARNING: Unsupported method '{self.config.method}', falling back to pyscenedetect")
                return self._detect_with_pyscenedetect(video_path)

        except Exception as e:
            print(f"[SceneDetector] ERROR during detection: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _detect_with_pyscenedetect(self, video_path: str) -> List[Tuple[int, int]]:
        """Detect scenes using PySceneDetect library.

        Args:
            video_path: Path to video file

        Returns:
            List of (start_frame, end_frame) scene boundaries
        """
        # Create detector
        if self.config.adaptive:
            detector = AdaptiveDetector(
                adaptive_threshold=self.config.threshold,
                min_scene_len=self.config.min_scene_len
            )
            print("[SceneDetector] Using AdaptiveDetector (optimized for sports)")
        else:
            detector = ContentDetector(
                threshold=self.config.threshold,
                min_scene_len=self.config.min_scene_len
            )
            print("[SceneDetector] Using ContentDetector (standard detection)")

        # Detect scenes
        print(f"[SceneDetector] Analyzing video: {video_path}")
        scene_list = detect(video_path, detector, show_progress=True)

        # Convert to frame numbers
        scenes = []
        for scene in scene_list:
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            scenes.append((start_frame, end_frame))

        print(f"[SceneDetector] Found {len(scenes)} scenes")

        # Log scene boundaries
        for i, (start, end) in enumerate(scenes):
            duration_frames = end - start
            print(f"  Scene {i+1}: frames {start}-{end} ({duration_frames} frames)")

        return scenes

    def get_scene_for_frame(self, frame_num: int, scenes: List[Tuple[int, int]]) -> int:
        """Get scene ID for a specific frame number.

        Args:
            frame_num: Frame number to query
            scenes: List of (start_frame, end_frame) tuples from detect_scenes()

        Returns:
            Scene ID (0-indexed), or -1 if not in any scene
        """
        for scene_id, (start, end) in enumerate(scenes):
            if start <= frame_num < end:
                return scene_id
        return -1

    def is_scene_boundary(self, frame_num: int, scenes: List[Tuple[int, int]]) -> bool:
        """Check if frame is at a scene boundary.

        Useful for detecting scene transitions to reset ROI locks.

        Args:
            frame_num: Frame number to check
            scenes: List of scene boundaries

        Returns:
            True if frame is the first frame of a new scene
        """
        for start, _ in scenes:
            if frame_num == start:
                return True
        return False

    def get_scene_info(self, frame_num: int, scenes: List[Tuple[int, int]]) -> dict:
        """Get detailed information about the scene containing a frame.

        Args:
            frame_num: Frame number to query
            scenes: List of scene boundaries

        Returns:
            Dictionary with scene information:
            - scene_id: Scene index (0-based)
            - start_frame: Scene start frame
            - end_frame: Scene end frame
            - duration: Scene duration in frames
            - position: Relative position in scene (0.0-1.0)
        """
        scene_id = self.get_scene_for_frame(frame_num, scenes)

        if scene_id == -1:
            return {
                "scene_id": -1,
                "start_frame": -1,
                "end_frame": -1,
                "duration": 0,
                "position": 0.0
            }

        start, end = scenes[scene_id]
        duration = end - start
        position = (frame_num - start) / duration if duration > 0 else 0.0

        return {
            "scene_id": scene_id,
            "start_frame": start,
            "end_frame": end,
            "duration": duration,
            "position": position
        }


class DummySceneDetector(SceneDetector):
    """Dummy scene detector for when PySceneDetect is not available.

    Returns a single scene spanning the entire video.
    """

    def __init__(self, config: Optional[SceneDetectionConfig] = None):
        """Initialize dummy detector."""
        self.config = config or SceneDetectionConfig()
        self.config.enabled = False
        print("[DummySceneDetector] Using dummy detector (no scene detection)")

    def detect_scenes(self, video_path: str) -> List[Tuple[int, int]]:
        """Return empty scene list (disables scene detection)."""
        return []
