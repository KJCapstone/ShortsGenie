"""SoccerNet-specific YOLO detector for soccer ball and player detection."""

import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict
import torch
from pathlib import Path

from src.models.detection_result import Detection, BoundingBox, FrameDetections
from src.utils.config import DetectionConfig


class SoccerNetDetector:
    """
    Fine-tuned YOLOv8 detector optimized for SoccerNet dataset.

    Key Features:
    - Uses model trained on SoccerNet: yolov8n_soccernet.pt
    - Class mapping: SoccerNet (0=ball, 1=player) → Internal (32=ball, 0=person)
    - Ultra-low confidence threshold (0.05) for high recall
    - Returns multiple ball candidates per frame for temporal filtering
    """

    # SoccerNet class mapping
    SOCCERNET_BALL_ID = 0
    SOCCERNET_PLAYER_ID = 1

    # Internal class IDs (COCO-compatible)
    INTERNAL_PERSON_ID = 0
    INTERNAL_BALL_ID = 32

    def __init__(self, config: Optional[DetectionConfig] = None):
        """Initialize SoccerNet detector.

        Args:
            config: Detection configuration. Uses SoccerNet defaults if None.
        """
        # Use custom config for SoccerNet
        if config is None:
            config = DetectionConfig()

        # Override defaults for SoccerNet
        self.config = config
        self.config.model_path = "resources/models/yolov8n_soccernet.pt"
        self.config.confidence_threshold = 0.05  # Ultra-low for high recall
        self.config.target_classes = None  # Detect all classes (ball and player)
        self.config.max_detections = 100  # Allow many detections for filtering

        # Auto-detect device (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"[SoccerNetDetector] Using device: {self.device}")

        # Load YOLO model
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            print(f"[SoccerNetDetector] WARNING: SoccerNet model not found at {model_path}")
            print("[SoccerNetDetector] Falling back to generic YOLOv8n model")
            self.config.model_path = "yolov8n.pt"
            self.use_generic_model = True
        else:
            self.use_generic_model = False

        print(f"[SoccerNetDetector] Loading model: {self.config.model_path}")
        self.model = YOLO(self.config.model_path)
        self.model.to(self.device)

        # Verify model classes
        self._verify_model_classes()

    def _verify_model_classes(self):
        """Verify that the model has expected SoccerNet classes."""
        try:
            model_names = self.model.names
            print(f"[SoccerNetDetector] Model classes: {model_names}")

            if not self.use_generic_model:
                # Check for SoccerNet-specific classes
                if 0 in model_names and 1 in model_names:
                    if 'ball' in model_names[0].lower() and 'player' in model_names[1].lower():
                        print("[SoccerNetDetector] ✓ SoccerNet classes verified")
                    else:
                        print(f"[SoccerNetDetector] WARNING: Unexpected class names: {model_names}")
        except Exception as e:
            print(f"[SoccerNetDetector] Could not verify model classes: {e}")

    def _map_class_id(self, soccernet_class_id: int) -> int:
        """Map SoccerNet class ID to internal COCO-compatible class ID.

        Args:
            soccernet_class_id: Class ID from SoccerNet model (0=ball, 1=player)

        Returns:
            Internal class ID (0=person, 32=sports ball)
        """
        if self.use_generic_model:
            # Generic model already uses COCO classes
            return soccernet_class_id

        # Map SoccerNet classes to internal classes
        mapping = {
            self.SOCCERNET_BALL_ID: self.INTERNAL_BALL_ID,
            self.SOCCERNET_PLAYER_ID: self.INTERNAL_PERSON_ID
        }
        return mapping.get(soccernet_class_id, soccernet_class_id)

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0
    ) -> FrameDetections:
        """Detect objects in a single frame (standard interface).

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

                # Map SoccerNet class ID to internal class ID
                internal_class_id = self._map_class_id(int(cls_id))

                detection = Detection(
                    bbox=bbox,
                    confidence=float(conf),
                    class_id=internal_class_id
                )

                frame_detections.add_detection(detection)

        return frame_detections

    def detect_frame_multi_ball(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0
    ) -> Tuple[List[Detection], List[Detection]]:
        """Detect with multiple ball candidates for temporal filtering.

        This method returns ALL ball detections (even low confidence) to allow
        temporal filtering to select the best trajectory. Player detections are
        filtered to keep only high-confidence ones.

        Args:
            frame: Input frame (numpy array)
            frame_number: Frame number
            timestamp: Timestamp in seconds

        Returns:
            Tuple of (ball_detections, player_detections):
            - ball_detections: ALL ball candidates (sorted by confidence desc)
            - player_detections: Top players by confidence
        """
        # Get all detections with ultra-low threshold
        frame_detections = self.detect_frame(frame, frame_number, timestamp)

        # Separate ball and player detections
        ball_detections = []
        player_detections = []

        for detection in frame_detections.detections:
            if detection.class_id == self.INTERNAL_BALL_ID:
                ball_detections.append(detection)
            elif detection.class_id == self.INTERNAL_PERSON_ID:
                player_detections.append(detection)

        # Sort balls by confidence (highest first)
        ball_detections.sort(key=lambda d: d.confidence, reverse=True)

        # Sort players by confidence and keep top N
        player_detections.sort(key=lambda d: d.confidence, reverse=True)
        player_detections = player_detections[:10]  # Keep top 10 players

        return ball_detections, player_detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        start_frame_number: int = 0,
        fps: float = 25.0
    ) -> List[Tuple[List[Detection], List[Detection]]]:
        """Detect objects in a batch of frames for improved performance.

        Args:
            frames: List of frames to process
            start_frame_number: Starting frame number
            fps: Frames per second (for timestamp calculation)

        Returns:
            List of (ball_detections, player_detections) tuples for each frame
        """
        results_list = []

        # Process frames in batches for efficiency
        batch_size = 8  # Adjust based on GPU memory
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]

            for j, frame in enumerate(batch):
                frame_number = start_frame_number + i + j
                timestamp = frame_number / fps

                ball_dets, player_dets = self.detect_frame_multi_ball(
                    frame, frame_number, timestamp
                )
                results_list.append((ball_dets, player_dets))

        return results_list

    def reset_tracking(self):
        """Reset tracking state (for compatibility with base detector)."""
        try:
            if hasattr(self.model, 'predictor') and hasattr(self.model.predictor, 'trackers'):
                self.model.predictor.trackers = []
            print("[SoccerNetDetector] Tracking state reset")
        except Exception as e:
            print(f"[SoccerNetDetector] Warning: Could not reset tracking: {e}")
