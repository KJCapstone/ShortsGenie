"""Complete dynamic reframing pipeline."""

from typing import Dict, List, Optional
import numpy as np
from pathlib import Path

from src.core.soccernet_detector import SoccerNetDetector
from src.core.detector import ObjectDetector
from src.core.temporal_filter import TemporalBallFilter
from src.core.roi_calculator import ROICalculator
from src.core.smoother import Smoother
from src.core.cropper import Cropper
from src.utils.video_utils import VideoReader
from src.utils.config import AppConfig, DetectionConfig, ROIConfig, KalmanConfig
from src.models.detection_result import Detection


class ReframingPipeline:
    """
    Complete pipeline for dynamic video reframing.

    PHASE 2 Implementation:
    - SoccerNet YOLO detection for ball and players
    - Temporal filtering of ball trajectory
    - Weighted ROI calculation
    - Kalman filter smoothing
    - Video cropping and encoding
    """

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize reframing pipeline.

        Args:
            config: Application configuration. Uses defaults if None.
        """
        self.config = config or AppConfig()

        print("="*60)
        print("REFRAMING PIPELINE - PHASE 2")
        print("="*60)

    def process_goal_clip(
        self,
        clip_path: str,
        output_path: str,
        use_soccernet_model: bool = True,
        use_temporal_filter: bool = True,
        use_kalman_smoothing: bool = True,
        scene_metadata_path: Optional[str] = None
    ) -> Dict[str, any]:
        """Process a pre-extracted goal clip with SoccerNet detection.

        This method implements PHASE 2 of the pipeline:
        0. Load scene metadata (optional, scene-aware feature)
        1. Detect objects with SoccerNet YOLO
        2. Apply temporal filtering to ball trajectory
        3. Calculate weighted ROI centers
        4. Apply Kalman smoothing
        5. Crop and encode output video

        Args:
            clip_path: Path to input goal clip (e.g., "goal_1.mp4")
            output_path: Path to output vertical video
            use_soccernet_model: Use SoccerNet fine-tuned model (vs generic YOLO)
            use_temporal_filter: Apply temporal filtering to ball trajectory
            use_kalman_smoothing: Apply Kalman smoothing to ROI trajectory
            scene_metadata_path: Path to auto_tagger JSON (optional, for scene-aware ROI)

        Returns:
            Dict with processing statistics:
            - frames_processed: Total frames processed
            - ball_detection_rate: Ratio of frames with ball detected
            - average_players_per_frame: Average number of players detected
            - output_path: Path to output video
            - processing_time: Time taken (seconds)
        """
        import time
        start_time = time.time()

        print(f"\n[Pipeline] Processing goal clip: {clip_path}")
        print(f"[Pipeline] Output: {output_path}")
        print(f"[Pipeline] SoccerNet model: {use_soccernet_model}")
        print(f"[Pipeline] Temporal filter: {use_temporal_filter}")
        print(f"[Pipeline] Kalman smoothing: {use_kalman_smoothing}")
        print(f"[Pipeline] Scene awareness: {scene_metadata_path is not None}")
        print()

        # Step 0: Load scene metadata (optional, scene-aware feature)
        scene_manager = None
        if scene_metadata_path and self.config.scene.enabled:
            try:
                from src.scene.scene_metadata import load_from_auto_tagger_json
                from src.scene.scene_manager import SceneManager

                # We need fps, so we'll load it after getting video info
                # For now, mark that we need to load it
                scene_metadata_to_load = scene_metadata_path
            except ImportError:
                print("[Warning] Scene module not found, disabling scene awareness")
                scene_metadata_to_load = None
        else:
            scene_metadata_to_load = None

        # Step 1: Initialize components
        print("[Step 1/6] Initializing components...")

        # Initialize detector based on config backend
        backend = self.config.detection.detector_backend

        if backend == "footandball":
            from src.core.footandball_detector import FootAndBallDetector
            detector = FootAndBallDetector(
                model_path=self.config.detection.footandball_model_path,
                ball_threshold=self.config.detection.footandball_ball_threshold,
                player_threshold=self.config.detection.footandball_player_threshold,
                device=None  # Auto-detect (MPS/CUDA/CPU)
            )
            print(f"[Pipeline] Using FootAndBall detector")
        elif backend == "soccernet" or use_soccernet_model:
            detector = SoccerNetDetector(self.config.detection)
            print(f"[Pipeline] Using SoccerNet YOLO detector")
        else:  # backend == "yolo"
            detector = ObjectDetector(self.config.detection)
            print(f"[Pipeline] Using generic YOLO detector")

        if use_temporal_filter:
            temporal_filter = TemporalBallFilter()
        else:
            temporal_filter = None

        roi_calculator = ROICalculator(self.config.roi)

        if use_kalman_smoothing:
            smoother = Smoother(self.config.kalman)
        else:
            smoother = None

        cropper = Cropper(
            output_width=self.config.video.output_width,
            output_height=self.config.video.output_height,
            config=self.config.cropper
        )

        # Step 2: Run detection on all frames
        print("\n[Step 2/6] Running object detection...")
        ball_detections_per_frame = {}
        player_detections_per_frame = {}

        with VideoReader(clip_path) as reader:
            total_frames = reader.frame_count
            frame_width = reader.width
            frame_height = reader.height
            fps = reader.fps

            # Now load scene metadata if requested (we have fps now)
            if scene_metadata_to_load:
                try:
                    from src.scene.scene_metadata import load_from_auto_tagger_json
                    from src.scene.scene_manager import SceneManager

                    print(f"\n[Step 0/6] Loading scene metadata from {scene_metadata_to_load}...")
                    metadata = load_from_auto_tagger_json(scene_metadata_to_load, fps)
                    scene_manager = SceneManager(metadata)
                    print(f"[Scene] ✓ Loaded {len(metadata.segments)} scenes")
                except Exception as e:
                    print(f"[Warning] Failed to load scene metadata: {e}")
                    scene_manager = None

            for frame_num, frame in enumerate(reader):
                # All detectors now have the same interface
                if hasattr(detector, 'detect_frame_multi_ball'):
                    # SoccerNet-specific multi-ball detection
                    balls, players = detector.detect_frame_multi_ball(
                        frame, frame_num, frame_num / fps
                    )
                else:
                    # Generic interface: works for YOLO, SoccerNet (single-ball mode), and FootAndBall
                    frame_detections = detector.detect_frame(frame, frame_num, frame_num / fps)
                    balls = frame_detections.ball_detections
                    players = frame_detections.person_detections

                ball_detections_per_frame[frame_num] = balls
                player_detections_per_frame[frame_num] = players

                if (frame_num + 1) % 100 == 0:
                    print(f"[Detection] Processed {frame_num + 1}/{total_frames} frames")

        # Calculate detection statistics
        frames_with_ball = sum(1 for balls in ball_detections_per_frame.values() if balls)
        ball_detection_rate = frames_with_ball / total_frames
        avg_players = np.mean([len(players) for players in player_detections_per_frame.values()])

        print(f"[Detection] Ball detected in {frames_with_ball}/{total_frames} frames ({ball_detection_rate:.1%})")
        print(f"[Detection] Average players per frame: {avg_players:.1f}")

        # Step 3: Apply temporal filtering to ball trajectory
        if use_temporal_filter and temporal_filter is not None:
            print("\n[Step 3/6] Applying temporal filtering to ball trajectory...")
            ball_trajectory = temporal_filter.filter_trajectory(
                ball_detections_per_frame,
                total_frames
            )
            print(f"[TemporalFilter] ✓ Ball trajectory smoothed")
        else:
            print("\n[Step 3/6] Skipping temporal filtering (disabled)")
            # Create simple trajectory from best ball per frame
            from src.core.temporal_filter import BallTrajectory
            ball_trajectory = BallTrajectory()
            for frame_num in range(total_frames):
                balls = ball_detections_per_frame.get(frame_num, [])
                if balls:
                    best_ball = max(balls, key=lambda d: d.confidence)
                    ball_trajectory.frame_numbers.append(frame_num)
                    ball_trajectory.positions.append((best_ball.bbox.center_x, best_ball.bbox.center_y))
                    ball_trajectory.confidences.append(best_ball.confidence)
                    ball_trajectory.is_interpolated.append(False)
                else:
                    ball_trajectory.frame_numbers.append(frame_num)
                    ball_trajectory.positions.append((frame_width // 2, frame_height // 2))
                    ball_trajectory.confidences.append(0.0)
                    ball_trajectory.is_interpolated.append(True)

        # Step 4: Calculate ROI trajectory (with scene awareness if available)
        print("\n[Step 4/6] Calculating ROI trajectory...")
        roi_trajectory = roi_calculator.calculate_roi_trajectory(
            ball_trajectory,
            player_detections_per_frame,
            frame_width,
            frame_height,
            total_frames,
            scene_manager=scene_manager  # Pass scene manager for scene-aware ROI
        )
        print(f"[ROICalculator] ✓ Calculated {len(roi_trajectory)} ROIs")

        # Step 5: Apply Kalman smoothing
        if use_kalman_smoothing and smoother is not None:
            print("\n[Step 5/6] Applying Kalman filter smoothing...")
            roi_trajectory = smoother.smooth_trajectory(roi_trajectory)
            print(f"[Smoother] ✓ Trajectory smoothed")
        else:
            print("\n[Step 5/6] Skipping Kalman smoothing (disabled)")

        # Step 6: Crop and encode
        print("\n[Step 6/6] Cropping and encoding output...")
        cropper.process(clip_path, output_path, roi_trajectory)

        # Calculate final statistics
        processing_time = time.time() - start_time

        stats = {
            "frames_processed": total_frames,
            "ball_detection_rate": ball_detection_rate,
            "average_players_per_frame": float(avg_players),
            "output_path": output_path,
            "processing_time": processing_time,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "fps": fps
        }

        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Input: {clip_path}")
        print(f"Output: {output_path}")
        print(f"Frames: {total_frames}")
        print(f"Resolution: {frame_width}x{frame_height} → {self.config.video.output_width}x{self.config.video.output_height}")
        print(f"Ball detection rate: {ball_detection_rate:.1%}")
        print(f"Processing time: {processing_time:.1f}s ({total_frames/processing_time:.1f} fps)")
        print("="*60)

        return stats


def process_video_simple(
    input_path: str,
    output_path: str,
    output_width: int = 1080,
    output_height: int = 1920,
    use_soccernet: bool = True,
    letterbox_top: int = 288,
    letterbox_bottom: int = 288
) -> Dict[str, any]:
    """Simple convenience function for processing a video.

    Args:
        input_path: Input video path
        output_path: Output video path
        output_width: Output video width
        output_height: Output video height
        use_soccernet: Use SoccerNet fine-tuned model
        letterbox_top: Top letterbox size in pixels (default 288)
        letterbox_bottom: Bottom letterbox size in pixels (default 288)

    Returns:
        Processing statistics dict
    """
    config = AppConfig()
    config.video.output_width = output_width
    config.video.output_height = output_height
    config.cropper.letterbox_top = letterbox_top
    config.cropper.letterbox_bottom = letterbox_bottom

    pipeline = ReframingPipeline(config)
    return pipeline.process_goal_clip(input_path, output_path, use_soccernet_model=use_soccernet)
