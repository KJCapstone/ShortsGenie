"""Improved ball detection test with multiple strategies.

Improvements:
1. Larger YOLO model (yolov8s or yolov8m)
2. Separate confidence thresholds for ball vs person
3. Higher input resolution for YOLO
4. Two-pass detection (full frame + zoom)
5. Temporal tracking to fill gaps

Usage:
    python tests/test_ball_detection_improved.py --strategy [basic|large_model|two_pass|temporal]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import argparse
from collections import deque

from ultralytics import YOLO
import torch

from src.models.detection_result import Detection, BoundingBox, FrameDetections
from src.utils.config import AppConfig
from test_detection_visualization_v2 import EnhancedKeyPlayerAnalyzer, EnhancedVisualizer


class ImprovedBallDetector:
    """Enhanced detector with strategies for better ball detection."""

    def __init__(self, strategy: str = "basic"):
        """Initialize detector with specific strategy.

        Args:
            strategy: "basic", "large_model", "two_pass", or "temporal"
        """
        self.strategy = strategy

        # Auto-detect device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"[ImprovedDetector] Strategy: {strategy}")
        print(f"[ImprovedDetector] Device: {self.device}")

        # Load appropriate model based on strategy
        if strategy == "large_model":
            model_path = "yolov8m.pt"  # Medium model
            print(f"[ImprovedDetector] Using LARGER model: {model_path}")
        else:
            model_path = "yolov8s.pt"  # Small model (better than nano)
            print(f"[ImprovedDetector] Using model: {model_path}")

        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Confidence thresholds
        self.person_conf = 0.3
        self.ball_conf = 0.15  # Much lower for ball!
        print(f"[ImprovedDetector] Thresholds - Person: {self.person_conf}, Ball: {self.ball_conf}")

        # Input resolution
        if strategy in ["large_model", "two_pass"]:
            self.imgsz = 1280  # Higher resolution
            print(f"[ImprovedDetector] Input resolution: {self.imgsz}x{self.imgsz}")
        else:
            self.imgsz = 640

        # Temporal tracking buffer
        self.ball_history = deque(maxlen=10)  # Last 10 ball positions

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0
    ) -> FrameDetections:
        """Detect with improved strategy."""

        if self.strategy == "two_pass":
            return self._detect_two_pass(frame, frame_number, timestamp)
        elif self.strategy == "temporal":
            return self._detect_with_temporal(frame, frame_number, timestamp)
        else:
            return self._detect_basic(frame, frame_number, timestamp)

    def _detect_basic(self, frame: np.ndarray, frame_number: int, timestamp: float) -> FrameDetections:
        """Basic detection with separate thresholds."""
        frame_detections = FrameDetections(frame_number=frame_number, timestamp=timestamp)

        # Detect persons
        results_person = self.model.track(
            frame,
            conf=self.person_conf,
            classes=[0],  # person only
            imgsz=self.imgsz,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )[0]

        self._parse_results(results_person, frame_detections)

        # Detect balls with lower threshold
        results_ball = self.model(
            frame,
            conf=self.ball_conf,
            classes=[32],  # sports ball only
            imgsz=self.imgsz,
            verbose=False
        )[0]

        self._parse_results(results_ball, frame_detections, is_ball=True)

        return frame_detections

    def _detect_two_pass(self, frame: np.ndarray, frame_number: int, timestamp: float) -> FrameDetections:
        """Two-pass detection: full frame + zoomed regions."""
        # First pass: full frame detection
        frame_detections = self._detect_basic(frame, frame_number, timestamp)

        # If no ball found, try zoomed detection in likely areas
        if len(frame_detections.ball_detections) == 0:
            height, width = frame.shape[:2]

            # Define regions to zoom (center 60% of frame where ball is likely)
            regions = [
                (int(width * 0.2), int(height * 0.2), int(width * 0.8), int(height * 0.8)),  # Center
            ]

            for x1, y1, x2, y2 in regions:
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Detect in ROI with very low threshold
                results_ball = self.model(
                    roi,
                    conf=0.1,  # Very low threshold
                    classes=[32],
                    imgsz=self.imgsz,
                    verbose=False
                )[0]

                # Adjust coordinates back to full frame
                if results_ball.boxes is not None and len(results_ball.boxes) > 0:
                    boxes = results_ball.boxes.xyxy.cpu().numpy()
                    confidences = results_ball.boxes.conf.cpu().numpy()

                    for box, conf in zip(boxes, confidences):
                        bx1, by1, bx2, by2 = box
                        # Translate back to full frame coordinates
                        bbox = BoundingBox(
                            x=int(bx1 + x1),
                            y=int(by1 + y1),
                            width=int(bx2 - bx1),
                            height=int(by2 - by1)
                        )

                        detection = Detection(
                            bbox=bbox,
                            confidence=float(conf),
                            class_id=32
                        )
                        frame_detections.add_detection(detection)
                        break  # Use first ball found

        return frame_detections

    def _detect_with_temporal(self, frame: np.ndarray, frame_number: int, timestamp: float) -> FrameDetections:
        """Detection with temporal smoothing."""
        frame_detections = self._detect_basic(frame, frame_number, timestamp)

        # Update ball history
        ball_detections = frame_detections.ball_detections
        if ball_detections:
            best_ball = max(ball_detections, key=lambda d: d.confidence)
            self.ball_history.append((best_ball.bbox.center_x, best_ball.bbox.center_y, frame_number))
        else:
            # No ball detected, try to predict from history
            if len(self.ball_history) >= 3:
                # Predict ball position based on recent trajectory
                recent = list(self.ball_history)[-3:]

                # Simple linear extrapolation
                if recent[-1][2] == frame_number - 1:  # Last detection was previous frame
                    x_positions = [pos[0] for pos in recent]
                    y_positions = [pos[1] for pos in recent]

                    # Calculate velocity
                    vx = (x_positions[-1] - x_positions[0]) / 2.0
                    vy = (y_positions[-1] - y_positions[0]) / 2.0

                    # Predict position
                    pred_x = int(x_positions[-1] + vx)
                    pred_y = int(y_positions[-1] + vy)

                    # Add predicted ball (with low confidence marker)
                    bbox = BoundingBox(x=pred_x - 15, y=pred_y - 15, width=30, height=30)
                    predicted_ball = Detection(
                        bbox=bbox,
                        confidence=0.3,  # Lower confidence for predicted
                        class_id=32
                    )
                    frame_detections.add_detection(predicted_ball)

        return frame_detections

    def _parse_results(self, results, frame_detections: FrameDetections, is_ball: bool = False):
        """Parse YOLO results into FrameDetections."""
        if results.boxes is None or len(results.boxes) == 0:
            return

        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        # Get track IDs if available (for persons)
        track_ids = None
        if not is_ball and hasattr(results.boxes, 'id') and results.boxes.id is not None:
            track_ids = results.boxes.id.cpu().numpy().astype(int)

        for idx, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box
            bbox = BoundingBox(
                x=int(x1),
                y=int(y1),
                width=int(x2 - x1),
                height=int(y2 - y1)
            )

            track_id = int(track_ids[idx]) if track_ids is not None else None

            detection = Detection(
                bbox=bbox,
                confidence=float(conf),
                class_id=int(cls_id),
                track_id=track_id
            )

            frame_detections.add_detection(detection)


def process_video_improved(
    input_path: str,
    output_path: str,
    strategy: str,
    start_second: float = 0,
    duration_seconds: float = 10,
    frame_skip: int = 1
):
    """Process video with improved ball detection."""
    print("\n" + "=" * 70)
    print(f"Improved Ball Detection Test - Strategy: {strategy.upper()}")
    print("=" * 70)

    # Initialize
    config = AppConfig()
    detector = ImprovedBallDetector(strategy=strategy)
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

    # Calculate frame range
    start_frame = int(start_second * fps)
    end_frame = int((start_second + duration_seconds) * fps)
    end_frame = min(end_frame, total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Setup output
    output_fps = fps // frame_skip
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    print(f"\nProcessing frames...")

    frame_count = 0
    processed_count = 0
    ball_detected_count = 0
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

            # Detect
            frame_detections = detector.detect_frame(frame, current_frame, timestamp)
            analyzer.analyze_frame(frame_detections, (height, width))

            # Count ball detections
            if len(frame_detections.ball_detections) > 0:
                ball_detected_count += 1

            # Visualize
            vis_frame = visualizer.visualize_frame(frame, frame_detections)

            # Add strategy label
            cv2.putText(
                vis_frame, f"Strategy: {strategy.upper()}",
                (width - 350, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

            out.write(vis_frame)

            processed_count += 1
            frame_count += 1

            # Progress
            if processed_count % 30 == 0 or processed_count == 1:
                elapsed = time.time() - start_time
                proc_fps = processed_count / elapsed if elapsed > 0 else 0
                ball_rate = ball_detected_count / processed_count * 100
                print(f"Frame {current_frame}: "
                      f"{len(frame_detections.ball_detections)}B "
                      f"{len(frame_detections.person_detections)}P "
                      f"{len(frame_detections.key_player_detections)}K "
                      f"(Ball: {ball_rate:.1f}% | {proc_fps:.1f} FPS)")

    finally:
        cap.release()
        out.release()

        elapsed = time.time() - start_time
        ball_rate = ball_detected_count / processed_count * 100 if processed_count > 0 else 0

        print(f"\n{'='*70}")
        print(f"Complete! Strategy: {strategy.upper()}")
        print(f"Processed: {processed_count} frames in {elapsed:.2f}s")
        print(f"Average FPS: {processed_count/elapsed:.1f}")
        print(f"Ball Detection Rate: {ball_rate:.1f}% ({ball_detected_count}/{processed_count} frames)")
        print(f"Output: {output_path}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Improved ball detection test")
    parser.add_argument(
        "--strategy",
        type=str,
        default="basic",
        choices=["basic", "large_model", "two_pass", "temporal"],
        help="Detection strategy"
    )
    parser.add_argument("--start", type=float, default=548, help="Start time (seconds)")
    parser.add_argument("--duration", type=float, default=10, help="Duration (seconds)")
    parser.add_argument("--skip", type=int, default=2, help="Process every Nth frame")

    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent.parent
    input_path = script_dir / "input" / "korea_vs_brazil.mp4"
    output_path = script_dir / "output" / f"ball_detection_{args.strategy}.mp4"

    if not input_path.exists():
        print(f"ERROR: Video not found: {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    process_video_improved(
        str(input_path),
        str(output_path),
        strategy=args.strategy,
        start_second=args.start,
        duration_seconds=args.duration,
        frame_skip=args.skip
    )


if __name__ == "__main__":
    main()
