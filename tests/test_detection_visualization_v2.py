"""Enhanced test script with better ball detection and configurable settings.

Improvements:
1. Lower ball confidence threshold for better detection
2. Support for multiple video segments testing
3. Better key player scoring without ball dependency
4. Frame sampling option for faster testing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import argparse

from src.core.detector import ObjectDetector
from src.models.detection_result import Detection, FrameDetections
from src.utils.config import AppConfig


class EnhancedKeyPlayerAnalyzer:
    """Enhanced analyzer that works better without ball detection."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.prev_positions: Dict[int, Tuple[int, int]] = {}
        self.position_history: Dict[int, List[Tuple[int, int]]] = {}  # For speed calculation

    def calculate_importance_score(
        self,
        detection: Detection,
        ball_position: Optional[Tuple[int, int]],
        frame_center: Tuple[int, int],
        all_players: List[Detection]
    ) -> float:
        """Calculate importance score with improved logic."""
        score = 0.0
        cfg = self.config.key_player
        player_center = (detection.bbox.center_x, detection.bbox.center_y)

        # 1. Distance to ball (bonus if available, but not required)
        if ball_position is not None:
            distance = np.sqrt(
                (player_center[0] - ball_position[0]) ** 2 +
                (player_center[1] - ball_position[1]) ** 2
            )
            if distance < cfg.distance_to_ball_threshold:
                bonus = cfg.distance_score_weight * (1 - distance / cfg.distance_to_ball_threshold)
                score += bonus * 2.0  # Double weight when ball is detected

        # 2. Movement speed (using history for better accuracy)
        if detection.track_id is not None:
            if detection.track_id not in self.position_history:
                self.position_history[detection.track_id] = []

            history = self.position_history[detection.track_id]
            history.append(player_center)

            # Keep last 10 positions
            if len(history) > 10:
                history.pop(0)

            # Calculate speed from last 3 frames
            if len(history) >= 3:
                prev_pos = history[-3]
                speed = np.sqrt(
                    (player_center[0] - prev_pos[0]) ** 2 +
                    (player_center[1] - prev_pos[1]) ** 2
                ) / 3.0  # Average over 3 frames

                if speed > cfg.speed_threshold:
                    score += cfg.speed_score_weight
                elif speed > cfg.speed_threshold * 0.5:  # Partial credit
                    score += cfg.speed_score_weight * 0.5

        # 3. Screen center proximity (important for main action)
        center_distance = np.sqrt(
            (player_center[0] - frame_center[0]) ** 2 +
            (player_center[1] - frame_center[1]) ** 2
        )
        if center_distance < cfg.center_distance_threshold:
            center_score = cfg.center_score_weight * (1 - center_distance / cfg.center_distance_threshold)
            score += center_score * 1.5  # Boost center importance

        # 4. Confidence score
        if detection.confidence > cfg.confidence_threshold:
            score += cfg.confidence_score_weight

        # 5. Size-based importance (larger = closer = more important)
        bbox_area = detection.bbox.area
        avg_area = np.mean([p.bbox.area for p in all_players]) if all_players else 1
        if bbox_area > avg_area:
            score += 1.0  # Bonus for larger detections

        # 6. Lower part of screen (where action happens)
        frame_height = frame_center[1] * 2
        if player_center[1] > frame_height * 0.4:  # Lower 60% of screen
            score += 0.5

        return score

    def analyze_frame(self, frame_detections: FrameDetections, frame_shape: Tuple[int, int]):
        """Analyze frame with enhanced key player detection."""
        frame_center = (frame_shape[1] // 2, frame_shape[0] // 2)

        # Find ball position
        ball_position = None
        ball_detections = frame_detections.ball_detections
        if ball_detections:
            best_ball = max(ball_detections, key=lambda d: d.confidence)
            ball_position = (best_ball.bbox.center_x, best_ball.bbox.center_y)

        # Get all players for context
        all_players = frame_detections.person_detections

        # Calculate importance scores
        for detection in all_players:
            score = self.calculate_importance_score(
                detection, ball_position, frame_center, all_players
            )
            detection.importance_score = score


class EnhancedVisualizer:
    """Enhanced visualizer with better colors and info."""

    def __init__(self, config: AppConfig):
        self.config = config

    def draw_bounding_box(self, frame: np.ndarray, detection: Detection,
                         color: Tuple[int, int, int], label: str, thickness: int = 2):
        """Draw bounding box with enhanced styling."""
        bbox = detection.bbox
        x1, y1 = bbox.x, bbox.y
        x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Prepare label
        if detection.importance_score > 0:
            label_text = f"{label} {detection.confidence:.2f} [S:{detection.importance_score:.1f}]"
        else:
            label_text = f"{label} {detection.confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, font_thickness
        )

        # Background rectangle
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width + 5, y1),
            color,
            -1
        )

        # Text
        cv2.putText(
            frame, label_text, (x1 + 2, y1 - baseline - 2),
            font, font_scale, (255, 255, 255), font_thickness
        )

    def visualize_frame(self, frame: np.ndarray, frame_detections: FrameDetections) -> np.ndarray:
        """Visualize with enhanced display."""
        vis_frame = frame.copy()

        # Sort players by importance
        all_players = sorted(
            frame_detections.person_detections,
            key=lambda p: p.importance_score,
            reverse=True
        )

        # Draw regular players first (green)
        regular_players = [p for p in all_players
                          if p.importance_score < self.config.key_player.importance_threshold]
        for player in regular_players:
            label = f"#{player.track_id}" if player.track_id else "P"
            self.draw_bounding_box(vis_frame, player, (0, 200, 0), label, thickness=2)

        # Draw key players on top (yellow/orange based on score)
        key_players = [p for p in all_players
                      if p.importance_score >= self.config.key_player.importance_threshold]
        for player in key_players:
            # Color intensity based on score (yellow -> orange)
            intensity = min(player.importance_score / 10.0, 1.0)
            color = (0, int(255 * (1 - intensity * 0.5)), 255)  # BGR
            label = f"KEY #{player.track_id}" if player.track_id else "KEY"
            self.draw_bounding_box(vis_frame, player, color, label, thickness=3)

        # Draw balls last (red, thick)
        for ball in frame_detections.ball_detections:
            self.draw_bounding_box(vis_frame, ball, (0, 0, 255), "BALL", thickness=4)

        # Draw statistics
        self._draw_statistics(vis_frame, frame_detections)

        return vis_frame

    def _draw_statistics(self, frame: np.ndarray, frame_detections: FrameDetections):
        """Draw enhanced statistics."""
        key_players = frame_detections.key_player_detections

        stats_text = [
            f"Frame: {frame_detections.frame_number}",
            f"Time: {frame_detections.timestamp:.2f}s",
            f"Balls: {len(frame_detections.ball_detections)}",
            f"Players: {len(frame_detections.person_detections)}",
            f"Key Players: {len(key_players)}",
        ]

        if key_players:
            top_score = max(p.importance_score for p in key_players)
            stats_text.append(f"Top Score: {top_score:.1f}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        line_height = 25

        # Semi-transparent background
        overlay = frame.copy()
        bg_height = len(stats_text) * line_height + 20
        cv2.rectangle(overlay, (10, 10), (320, 10 + bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text
        for i, text in enumerate(stats_text):
            y_pos = 30 + i * line_height
            cv2.putText(frame, text, (20, y_pos), font, font_scale,
                       (255, 255, 255), font_thickness)


def process_video(
    input_path: str,
    output_path: str,
    start_second: float = 0,
    duration_seconds: float = 10,
    frame_skip: int = 1
):
    """Process video segment with enhanced detection.

    Args:
        input_path: Input video path
        output_path: Output video path
        start_second: Start time in seconds
        duration_seconds: Duration to process
        frame_skip: Process every Nth frame (1=all frames)
    """
    print("\n" + "=" * 70)
    print("Enhanced Object Detection & Key Player Visualization")
    print("=" * 70)

    # Initialize with optimized config
    config = AppConfig()
    config.detection.confidence_threshold = 0.25  # Lower for better ball detection
    config.key_player.importance_threshold = 3.0  # Lower threshold
    config.key_player.distance_to_ball_threshold = 300  # Larger radius

    detector = ObjectDetector(config.detection)
    analyzer = EnhancedKeyPlayerAnalyzer(config)
    visualizer = EnhancedVisualizer(config)

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nInput: {input_path}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    print(f"Processing: {start_second}s - {start_second + duration_seconds}s")
    print(f"Frame skip: {frame_skip} (processing every {frame_skip} frame)")

    # Calculate frame range
    start_frame = int(start_second * fps)
    end_frame = int((start_second + duration_seconds) * fps)
    end_frame = min(end_frame, total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Setup output
    output_fps = fps // frame_skip
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    print(f"\nProcessing {end_frame - start_frame} frames...")

    frame_count = 0
    processed_count = 0
    start_time = time.time()

    try:
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            # Frame skipping
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            timestamp = current_frame / fps

            # Detect and analyze
            frame_detections = detector.detect_and_track(frame, current_frame, timestamp)
            analyzer.analyze_frame(frame_detections, (height, width))

            # Visualize
            vis_frame = visualizer.visualize_frame(frame, frame_detections)
            out.write(vis_frame)

            processed_count += 1
            frame_count += 1

            # Progress
            if processed_count % 30 == 0 or processed_count == 1:
                elapsed = time.time() - start_time
                proc_fps = processed_count / elapsed if elapsed > 0 else 0
                print(f"Frame {current_frame}: "
                      f"{len(frame_detections.ball_detections)}B "
                      f"{len(frame_detections.person_detections)}P "
                      f"{len(frame_detections.key_player_detections)}K "
                      f"({proc_fps:.1f} FPS)")

    finally:
        cap.release()
        out.release()

        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Complete! Processed {processed_count} frames in {elapsed:.2f}s")
        print(f"Average: {processed_count/elapsed:.1f} FPS")
        print(f"Output: {output_path}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Enhanced detection visualization test")
    parser.add_argument("--start", type=float, default=0, help="Start time (seconds)")
    parser.add_argument("--duration", type=float, default=10, help="Duration (seconds)")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--output", type=str, default="output/test_detection_v2.mp4")

    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent.parent
    input_path = script_dir / "input" / "korea_vs_brazil.mp4"
    output_path = script_dir / args.output

    if not input_path.exists():
        print(f"ERROR: Video not found: {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    process_video(
        str(input_path),
        str(output_path),
        start_second=args.start,
        duration_seconds=args.duration,
        frame_skip=args.skip
    )


if __name__ == "__main__":
    main()
