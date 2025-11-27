# FootAndBall Detector Integration
# Wrapper for the FootAndBall model to work with our reframing pipeline

import sys
from pathlib import Path
import torch
import numpy as np

# Add FootAndBall directory to path
FOOTANDBALL_PATH = Path(__file__).parent.parent.parent.parent / "FootAndBall"
sys.path.insert(0, str(FOOTANDBALL_PATH))

from network import footandball
from data.augmentation import PLAYER_LABEL, BALL_LABEL

from src.models.detection_result import Detection, BoundingBox, FrameDetections


class FootAndBallDetector:
    """
    FootAndBall detector wrapper for soccer player and ball detection.

    This detector is specialized for soccer and trained on the ISSIA-CNR dataset.
    It uses a custom FPN-based architecture that's optimized for soccer scenarios.

    Advantages over YOLO:
    - Trained specifically on soccer footage
    - Specialized architecture for ball and player detection
    - May provide better accuracy in soccer-specific scenarios

    Usage:
        detector = FootAndBallDetector(
            model_path="path/to/model.pth",
            ball_threshold=0.7,
            player_threshold=0.7
        )

        detections = detector.detect(frame)
    """

    def __init__(
        self,
        model_path: str = None,
        ball_threshold: float = 0.5,
        player_threshold: float = 0.5,
        max_ball_detections: int = 100,
        max_player_detections: int = 100,
        device: str = None
    ):
        """
        Initialize FootAndBall detector.

        Args:
            model_path: Path to .pth model weights. If None, uses default path.
            ball_threshold: Confidence threshold for ball detection (0.0-1.0)
            player_threshold: Confidence threshold for player detection (0.0-1.0)
            max_ball_detections: Maximum number of ball detections per frame
            max_player_detections: Maximum number of player detections per frame
            device: Device to run inference on ('cpu', 'cuda', 'mps', or None for auto)
        """
        # Set default model path
        if model_path is None:
            model_path = str(FOOTANDBALL_PATH / "models" / "model_20201019_1416_final.pth")

        self.model_path = model_path
        self.ball_threshold = ball_threshold
        self.player_threshold = player_threshold

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device

        # Create model
        self.model = footandball.model_factory(
            'fb1',
            'detect',
            ball_threshold=ball_threshold,
            player_threshold=player_threshold,
            max_ball_detections=max_ball_detections,
            max_player_detections=max_player_detections
        )

        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"FootAndBall detector loaded on {self.device}")

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for FootAndBall model.

        Args:
            frame: BGR image (H, W, 3) as numpy array

        Returns:
            Preprocessed tensor (1, 3, H, W) ready for inference
        """
        # Convert BGR to RGB
        frame_rgb = frame[:, :, ::-1].copy()

        # Convert to float32 and normalize to [0, 1]
        frame_float = frame_rgb.astype(np.float32) / 255.0

        # Convert to tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(frame_float).permute(2, 0, 1)

        # Add batch dimension (C, H, W) -> (1, C, H, W)
        tensor = tensor.unsqueeze(0)

        return tensor

    def detect(self, frame: np.ndarray) -> FrameDetections:
        """
        Detect players and ball in a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            FrameDetections object containing all detections
        """
        # Preprocess
        input_tensor = self._preprocess_frame(frame).to(self.device)

        # Inference
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # predictions is a list with one element (batch size = 1)
        pred = predictions[0]

        # Convert to our Detection format
        detections = []

        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box

            # Map FootAndBall labels to our format
            # PLAYER_LABEL = 1, BALL_LABEL = 2 in FootAndBall
            if label == PLAYER_LABEL:
                class_id = 0  # person in COCO/YOLO
            elif label == BALL_LABEL:
                class_id = 32  # sports ball in COCO/YOLO
            else:
                continue

            # Convert from (x1, y1, x2, y2) to (x, y, width, height)
            bbox = BoundingBox(
                x=int(x1),
                y=int(y1),
                width=int(x2 - x1),
                height=int(y2 - y1)
            )

            detection = Detection(
                bbox=bbox,
                confidence=float(score),
                class_id=class_id,
                track_id=None  # FootAndBall doesn't provide tracking
            )

            detections.append(detection)

        return FrameDetections(
            frame_number=0,
            timestamp=0.0,
            detections=detections
        )

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0
    ) -> FrameDetections:
        """
        Pipeline-compatible detection method.

        This method provides the same interface as ObjectDetector and SoccerNetDetector,
        making FootAndBallDetector a drop-in replacement for the reframing pipeline.

        Args:
            frame: Input frame (BGR image as numpy array)
            frame_number: Frame number for metadata
            timestamp: Timestamp in seconds for metadata

        Returns:
            FrameDetections with metadata populated
        """
        # Perform detection using the core detect() method
        detections = self.detect(frame)

        # Populate metadata for pipeline compatibility
        detections.frame_number = frame_number
        detections.timestamp = timestamp

        return detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[FrameDetections]:
        """
        Detect players and ball in a batch of frames.

        Args:
            frames: List of BGR images as numpy arrays

        Returns:
            List of FrameDetections objects
        """
        # FootAndBall supports batch processing
        batch_tensors = torch.cat([self._preprocess_frame(f) for f in frames], dim=0)
        batch_tensors = batch_tensors.to(self.device)

        with torch.no_grad():
            predictions = self.model(batch_tensors)

        # Convert each prediction to FrameDetections
        results = []
        for pred in predictions:
            detections = []

            boxes = pred['boxes'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box

                if label == PLAYER_LABEL:
                    class_id = 0  # person in COCO/YOLO
                elif label == BALL_LABEL:
                    class_id = 32  # sports ball in COCO/YOLO
                else:
                    continue

                # Convert from (x1, y1, x2, y2) to (x, y, width, height)
                bbox = BoundingBox(
                    x=int(x1),
                    y=int(y1),
                    width=int(x2 - x1),
                    height=int(y2 - y1)
                )
                detection = Detection(
                    bbox=bbox,
                    confidence=float(score),
                    class_id=class_id,
                    track_id=None
                )
                detections.append(detection)

            results.append(FrameDetections(detections=detections))

        return results

    @property
    def model_info(self) -> dict:
        """Get model information."""
        return {
            'name': 'FootAndBall',
            'version': 'fb1',
            'model_path': self.model_path,
            'device': self.device,
            'ball_threshold': self.ball_threshold,
            'player_threshold': self.player_threshold,
        }


if __name__ == "__main__":
    # Simple test
    import cv2

    # Create detector
    detector = FootAndBallDetector(
        ball_threshold=0.5,
        player_threshold=0.5
    )

    print("Model info:", detector.model_info)

    # Test with dummy frame
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)

    print(f"Detections: {len(detections.detections)} objects")
    print("FootAndBall detector test completed!")
