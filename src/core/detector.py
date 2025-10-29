"""Object detection using YOLOv8."""

import numpy as np
from ultralytics import YOLO
from typing import List, Optional
import torch

from src.models.detection_result import Detection, BoundingBox, FrameDetections
from src.utils.config import DetectionConfig


class ObjectDetector:
    """YOLOv8-based object detector with ByteTrack tracking."""

    def __init__(self, config: Optional[DetectionConfig] = None):
        """Initialize detector.

        Args:
            config: Detection configuration. Uses defaults if None.
        """
        self.config = config or DetectionConfig()

        # Auto-detect device (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"[Detector] Using device: {self.device}")

        # Load YOLO model
        print(f"[Detector] Loading model: {self.config.model_path}")
        self.model = YOLO(self.config.model_path)
        self.model.to(self.device)

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0
    ) -> FrameDetections:
        """Detect objects in a single frame without tracking.

        Args:
            frame: Input frame (numpy array)
            frame_number: Frame number
            timestamp: Timestamp in seconds

        Returns:
            FrameDetections object containing all detections
        """
        # Run YOLO inference
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.target_classes,
            max_det=self.config.max_detections,
            verbose=False
        )[0]

        # Parse results
        frame_detections = FrameDetections(
            frame_number=frame_number,
            timestamp=timestamp
        )

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                bbox = BoundingBox(
                    x=int(x1),
                    y=int(y1),
                    width=int(x2 - x1),
                    height=int(y2 - y1)
                )

                detection = Detection(
                    bbox=bbox,
                    confidence=float(conf),
                    class_id=int(cls_id)
                )

                frame_detections.add_detection(detection)

        return frame_detections

    def detect_and_track(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0
    ) -> FrameDetections:
        """Detect and track objects in a single frame using ByteTrack.

        Args:
            frame: Input frame (numpy array)
            frame_number: Frame number
            timestamp: Timestamp in seconds

        Returns:
            FrameDetections object with track_id assigned
        """
        # Run YOLO inference with tracking
        results = self.model.track(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.target_classes,
            max_det=self.config.max_detections,
            tracker="bytetrack.yaml",
            persist=True,  # Persist tracks across frames
            verbose=False
        )[0]

        # Parse results
        frame_detections = FrameDetections(
            frame_number=frame_number,
            timestamp=timestamp
        )

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            # Get track IDs if available
            track_ids = None
            if hasattr(results.boxes, 'id') and results.boxes.id is not None:
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

        return frame_detections

    def reset_tracking(self):
        """Reset tracking state (useful when processing new videos)."""
        # Reset the tracker by reinitializing the model's tracker
        try:
            if hasattr(self.model, 'predictor') and hasattr(self.model.predictor, 'trackers'):
                self.model.predictor.trackers = []
            print("[Detector] Tracking state reset")
        except Exception as e:
            print(f"[Detector] Warning: Could not reset tracking: {e}")
