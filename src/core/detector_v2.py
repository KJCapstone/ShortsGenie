"""Improved object detection with separate thresholds for ball and person.

This is an enhanced version of detector.py with better ball detection.
Key improvements:
- Larger YOLO model (yolov8s or yolov8m)
- Separate confidence thresholds for ball vs person
- Higher input resolution
- Two detection passes (person tracking + ball detection)
"""

import numpy as np
from ultralytics import YOLO
from typing import List, Optional
import torch

from src.models.detection_result import Detection, BoundingBox, FrameDetections
from src.utils.config import DetectionConfig


class ObjectDetectorV2:
    """Enhanced YOLOv8-based detector optimized for soccer ball detection."""

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        person_conf: float = 0.3,
        ball_conf: float = 0.15,
        imgsz: int = 1280
    ):
        """Initialize improved detector.

        Args:
            model_path: YOLO model path (yolov8s.pt or yolov8m.pt recommended)
            person_conf: Confidence threshold for person detection
            ball_conf: Confidence threshold for ball detection (lower is better)
            imgsz: Input image size for YOLO (higher = better small object detection)
        """
        # Auto-detect device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"[DetectorV2] Device: {self.device}")
        print(f"[DetectorV2] Model: {model_path}")
        print(f"[DetectorV2] Thresholds - Person: {person_conf}, Ball: {ball_conf}")
        print(f"[DetectorV2] Input resolution: {imgsz}x{imgsz}")

        # Load model
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Configuration
        self.person_conf = person_conf
        self.ball_conf = ball_conf
        self.imgsz = imgsz
        self.iou_threshold = 0.45

    def detect_and_track(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0
    ) -> FrameDetections:
        """Detect and track objects with optimized ball detection.

        Uses two-pass approach:
        1. Track persons with ByteTrack (higher confidence)
        2. Detect balls with lower confidence threshold

        Args:
            frame: Input frame
            frame_number: Frame number
            timestamp: Timestamp in seconds

        Returns:
            FrameDetections with both persons and balls
        """
        frame_detections = FrameDetections(
            frame_number=frame_number,
            timestamp=timestamp
        )

        # Pass 1: Detect and track persons
        results_person = self.model.track(
            frame,
            conf=self.person_conf,
            iou=self.iou_threshold,
            classes=[0],  # person only
            imgsz=self.imgsz,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )[0]

        self._parse_results(results_person, frame_detections, track=True)

        # Pass 2: Detect balls with lower threshold
        results_ball = self.model(
            frame,
            conf=self.ball_conf,
            iou=self.iou_threshold,
            classes=[32],  # sports ball only
            imgsz=self.imgsz,
            verbose=False
        )[0]

        self._parse_results(results_ball, frame_detections, track=False)

        return frame_detections

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0
    ) -> FrameDetections:
        """Detect objects without tracking.

        Args:
            frame: Input frame
            frame_number: Frame number
            timestamp: Timestamp in seconds

        Returns:
            FrameDetections
        """
        frame_detections = FrameDetections(
            frame_number=frame_number,
            timestamp=timestamp
        )

        # Detect persons
        results_person = self.model(
            frame,
            conf=self.person_conf,
            iou=self.iou_threshold,
            classes=[0],
            imgsz=self.imgsz,
            verbose=False
        )[0]

        self._parse_results(results_person, frame_detections, track=False)

        # Detect balls
        results_ball = self.model(
            frame,
            conf=self.ball_conf,
            iou=self.iou_threshold,
            classes=[32],
            imgsz=self.imgsz,
            verbose=False
        )[0]

        self._parse_results(results_ball, frame_detections, track=False)

        return frame_detections

    def _parse_results(
        self,
        results,
        frame_detections: FrameDetections,
        track: bool = False
    ):
        """Parse YOLO results into FrameDetections.

        Args:
            results: YOLO results object
            frame_detections: FrameDetections to add to
            track: Whether to include track IDs
        """
        if results.boxes is None or len(results.boxes) == 0:
            return

        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        # Get track IDs if available
        track_ids = None
        if track and hasattr(results.boxes, 'id') and results.boxes.id is not None:
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

    def reset_tracking(self):
        """Reset tracking state."""
        try:
            if hasattr(self.model, 'predictor') and hasattr(self.model.predictor, 'trackers'):
                self.model.predictor.trackers = []
            print("[DetectorV2] Tracking state reset")
        except Exception as e:
            print(f"[DetectorV2] Warning: Could not reset tracking: {e}")


# Convenience function for quick initialization
def create_detector(quality: str = "balanced") -> ObjectDetectorV2:
    """Create detector with preset quality settings.

    Args:
        quality: "fast", "balanced", or "high"

    Returns:
        Configured ObjectDetectorV2 instance
    """
    presets = {
        "fast": {
            "model_path": "yolov8s.pt",
            "person_conf": 0.35,
            "ball_conf": 0.2,
            "imgsz": 640
        },
        "balanced": {
            "model_path": "yolov8s.pt",
            "person_conf": 0.3,
            "ball_conf": 0.15,
            "imgsz": 1280
        },
        "high": {
            "model_path": "yolov8m.pt",
            "person_conf": 0.3,
            "ball_conf": 0.15,
            "imgsz": 1280
        }
    }

    if quality not in presets:
        raise ValueError(f"Quality must be one of {list(presets.keys())}")

    config = presets[quality]
    print(f"[DetectorV2] Creating detector with '{quality}' preset")

    return ObjectDetectorV2(**config)
