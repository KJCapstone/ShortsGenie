"""Test script for object detection and key player visualization.

This script tests the detection system on a soccer match video and visualizes:
1. Ball detection (red bounding box)
2. Player detection (green bounding box)
3. Key player identification (yellow bounding box)

Key player criteria:
- Distance to ball < threshold
- Movement speed > threshold
- Position near screen center
- High confidence score
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

from src.core.detector import ObjectDetector
from src.models.detection_result import Detection, FrameDetections
from src.utils.config import AppConfig


class KeyPlayerAnalyzer:
    """Analyzes detections to identify key players."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.prev_positions: Dict[int, Tuple[int, int]] = {}  # track_id -> (x, y)

    def calculate_importance_score(
        self,
        detection: Detection,
        ball_position: Optional[Tuple[int, int]],
        frame_center: Tuple[int, int],
        frame_shape: Tuple[int, int]
    ) -> float:
        """Calculate importance score for a player detection.

        Args:
            detection: Player detection
            ball_position: (x, y) position of ball, or None if no ball detected
            frame_center: (x, y) center of the frame
            frame_shape: (height, width) of the frame

        Returns:
            Importance score (higher = more important)
        """
        score = 0.0
        cfg = self.config.key_player

        player_center = (detection.bbox.center_x, detection.bbox.center_y)

        # 1. Distance to ball (if ball detected)
        if ball_position is not None:
            distance = np.sqrt(
                (player_center[0] - ball_position[0]) ** 2 +
                (player_center[1] - ball_position[1]) ** 2
            )
            if distance < cfg.distance_to_ball_threshold:
                # Closer to ball = higher score
                score += cfg.distance_score_weight * (1 - distance / cfg.distance_to_ball_threshold)

        # 2. Movement speed (if tracked)
        if detection.track_id is not None and detection.track_id in self.prev_positions:
            prev_pos = self.prev_positions[detection.track_id]
            speed = np.sqrt(
                (player_center[0] - prev_pos[0]) ** 2 +
                (player_center[1] - prev_pos[1]) ** 2
            )
            if speed > cfg.speed_threshold:
                score += cfg.speed_score_weight

        # 3. Distance to screen center
        center_distance = np.sqrt(
            (player_center[0] - frame_center[0]) ** 2 +
            (player_center[1] - frame_center[1]) ** 2
        )
        if center_distance < cfg.center_distance_threshold:
            # Closer to center = higher score
            score += cfg.center_score_weight * (1 - center_distance / cfg.center_distance_threshold)

        # 4. Detection confidence
        if detection.confidence > cfg.confidence_threshold:
            score += cfg.confidence_score_weight

        # Update position history
        if detection.track_id is not None:
            self.prev_positions[detection.track_id] = player_center

        return score

    def analyze_frame(self, frame_detections: FrameDetections, frame_shape: Tuple[int, int]):
        """Analyze frame and assign importance scores to all player detections.

        Args:
            frame_detections: Detections for this frame
            frame_shape: (height, width) of the frame
        """
        frame_center = (frame_shape[1] // 2, frame_shape[0] // 2)

        # Find ball position
        ball_position = None
        ball_detections = frame_detections.ball_detections
        if ball_detections:
            # Use highest confidence ball detection
            best_ball = max(ball_detections, key=lambda d: d.confidence)
            ball_position = (best_ball.bbox.center_x, best_ball.bbox.center_y)

        # Calculate importance scores for all players
        for detection in frame_detections.person_detections:
            score = self.calculate_importance_score(
                detection, ball_position, frame_center, frame_shape
            )
            detection.importance_score = score


class DetectionVisualizer:
    """Visualizes detection results with bounding boxes and labels."""

    def __init__(self, config: AppConfig):
        self.config = config

    def draw_bounding_box(
        self,
        frame: np.ndarray,
        detection: Detection,
        color: Tuple[int, int, int],
        label: str,
        thickness: int = 2
    ):
        """Draw a bounding box with label on the frame.

        Args:
            frame: Frame to draw on
            detection: Detection to visualize
            color: BGR color tuple
            label: Label text
            thickness: Box line thickness
        """
        bbox = detection.bbox
        x1, y1 = bbox.x, bbox.y
        x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw label background
        label_text = f"{label} {detection.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, font_thickness
        )

        # Background rectangle for text
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1  # Filled
        )

        # Draw text
        cv2.putText(
            frame,
            label_text,
            (x1, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),  # White text
            font_thickness
        )

    def visualize_frame(self, frame: np.ndarray, frame_detections: FrameDetections) -> np.ndarray:
        """Visualize all detections on a frame.

        Args:
            frame: Input frame
            frame_detections: Detections for this frame

        Returns:
            Frame with visualizations drawn
        """
        vis_frame = frame.copy()

        # Draw balls (red)
        for ball in frame_detections.ball_detections:
            self.draw_bounding_box(vis_frame, ball, (0, 0, 255), "BALL", thickness=3)

        # Draw key players (yellow)
        key_players = frame_detections.key_player_detections
        for player in key_players:
            track_label = f"KEY #{player.track_id}" if player.track_id else "KEY PLAYER"
            self.draw_bounding_box(vis_frame, player, (0, 255, 255), track_label, thickness=3)

        # Draw regular players (green)
        regular_players = [p for p in frame_detections.person_detections
                          if p.importance_score < self.config.key_player.importance_threshold]
        for player in regular_players:
            track_label = f"#{player.track_id}" if player.track_id else "PLAYER"
            self.draw_bounding_box(vis_frame, player, (0, 255, 0), track_label, thickness=2)

        # Draw statistics overlay
        self._draw_statistics(vis_frame, frame_detections)

        return vis_frame

    def _draw_statistics(self, frame: np.ndarray, frame_detections: FrameDetections):
        """Draw statistics overlay on the frame.

        Args:
            frame: Frame to draw on
            frame_detections: Detections for this frame
        """
        stats_text = [
            f"Frame: {frame_detections.frame_number}",
            f"Time: {frame_detections.timestamp:.2f}s",
            f"Balls: {len(frame_detections.ball_detections)}",
            f"Players: {len(frame_detections.person_detections)}",
            f"Key Players: {len(frame_detections.key_player_detections)}",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        line_height = 25

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 10 + len(stats_text) * line_height + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw text
        for i, text in enumerate(stats_text):
            y_pos = 30 + i * line_height
            cv2.putText(frame, text, (20, y_pos), font, font_scale, (255, 255, 255), font_thickness)


def process_video(
    input_path: str,
    output_path: str,
    max_frames: Optional[int] = None,
    skip_frames: int = 0
):
    """Process video with detection and visualization.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        max_frames: Maximum number of frames to process (None = all)
        skip_frames: Number of frames to skip at the beginning
    """
    print("\n" + "=" * 60)
    print("Object Detection & Key Player Visualization Test")
    print("=" * 60)

    # Initialize components
    config = AppConfig()
    detector = ObjectDetector(config.detection)
    analyzer = KeyPlayerAnalyzer(config)
    visualizer = DetectionVisualizer(config)

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nInput video: {input_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.2f}s")

    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Skip initial frames if requested
    if skip_frames > 0:
        print(f"\nSkipping first {skip_frames} frames...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)

    # Process frames
    print(f"\nProcessing frames...")
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            timestamp = current_frame_num / fps

            # Detect and track
            frame_detections = detector.detect_and_track(frame, current_frame_num, timestamp)

            # Analyze key players
            analyzer.analyze_frame(frame_detections, (height, width))

            # Visualize
            vis_frame = visualizer.visualize_frame(frame, frame_detections)

            # Write output
            out.write(vis_frame)

            frame_count += 1

            # Print progress
            if frame_count % 30 == 0 or frame_count == 1:
                elapsed = time.time() - start_time
                processing_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frame {current_frame_num}: "
                      f"{len(frame_detections.ball_detections)} balls, "
                      f"{len(frame_detections.person_detections)} players, "
                      f"{len(frame_detections.key_player_detections)} key players "
                      f"({processing_fps:.1f} FPS)")

            # Check max frames limit
            if max_frames is not None and frame_count >= max_frames:
                print(f"\nReached max frames limit ({max_frames})")
                break

    finally:
        # Cleanup
        cap.release()
        out.release()

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Processed {frame_count} frames in {elapsed:.2f}s")
        print(f"Average FPS: {frame_count/elapsed:.1f}")
        print(f"Output saved to: {output_path}")
        print(f"{'='*60}\n")


def main():
    """Main entry point."""
    # Configuration
    INPUT_VIDEO = "input/korea_vs_brazil.mp4"
    OUTPUT_VIDEO = "output/test_detection.mp4"
    MAX_FRAMES = 300  # Process first 10 seconds at 30fps (adjust as needed)
    SKIP_FRAMES = 0  # Skip first N frames (useful for testing specific sections)

    # Resolve paths
    script_dir = Path(__file__).parent.parent
    input_path = script_dir / INPUT_VIDEO
    output_path = script_dir / OUTPUT_VIDEO

    if not input_path.exists():
        print(f"ERROR: Input video not found: {input_path}")
        print(f"Please place your video at: {input_path}")
        return

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process video
    process_video(
        str(input_path),
        str(output_path),
        max_frames=MAX_FRAMES,
        skip_frames=SKIP_FRAMES
    )


if __name__ == "__main__":
    main()
