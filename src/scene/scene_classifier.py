"""Scene classification using ResNet18 for soccer video clips.

This module provides scene classification functionality for soccer videos,
identifying scene types: wide, close, audience, and replay.
"""

import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import json

from src.scene.scene_metadata import SceneMetadata, SceneSegment

logger = logging.getLogger(__name__)


class SceneClassifier:
    """ResNet18-based scene classifier for soccer video clips.

    This classifier analyzes video clips to identify scene types:
    - "wide": Wide camera angle showing full field
    - "close": Close-up shots of players/ball
    - "audience": Crowd/audience shots
    - "replay": Replay sequences

    Uses PySceneDetect ContentDetector for scene boundary detection,
    then classifies each scene using a fine-tuned ResNet18 model.
    """

    LABELS = ["wide", "close", "audience", "replay"]

    def __init__(self, config):
        """Initialize scene classifier.

        Args:
            config: SceneClassificationConfig object with model settings
        """
        self.config = config
        self.device = self._setup_device()
        self.model = self._load_model()
        self.transform = self._create_transform()

        logger.info(f"SceneClassifier initialized on device: {self.device}")

    def classify_clip(
        self,
        video_path: str,
        output_json_path: Optional[str] = None
    ) -> SceneMetadata:
        """Classify scenes in a video clip.

        Args:
            video_path: Path to input video clip
            output_json_path: Optional path to save JSON output

        Returns:
            SceneMetadata object with classified segments

        Process:
            1. Detect scene boundaries using PySceneDetect ContentDetector
            2. For each scene, extract middle frame
            3. Classify frame using ResNet18
            4. Build SceneMetadata with frame-level timestamps
            5. Optionally save to JSON
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Classifying scenes in: {video_path}")

        # Step 1: Detect scene boundaries
        scene_list = self._detect_scene_boundaries(video_path)
        logger.info(f"Detected {len(scene_list)} scenes")

        if len(scene_list) == 0:
            # No scenes detected - treat entire video as single scene
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            scene_list = [(0.0, duration)]
            logger.warning(f"No scene boundaries detected, treating as single scene ({duration:.1f}s)")

        # Step 2: Open video for frame extraction
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Step 3: Classify each scene
        segments = []
        timeline_data = []  # For JSON output

        for i, (start_time, end_time) in enumerate(scene_list):
            duration = end_time - start_time

            # Skip very short scenes
            if duration < self.config.min_scene_duration:
                logger.debug(f"Skipping short scene {i}: {duration:.2f}s")
                continue

            # Extract middle frame
            mid_time = start_time + (duration / 2)
            cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Failed to extract frame at {mid_time:.2f}s")
                continue

            # Classify frame
            label = self._classify_frame(frame)

            # Create SceneSegment
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            segment = SceneSegment(
                label=label,
                start_seconds=start_time,
                end_seconds=end_time,
                start_frame=start_frame,
                end_frame=end_frame,
                duration=duration
            )
            segments.append(segment)

            # For JSON output (auto_tagger format)
            timeline_data.append({
                "label": label,
                "start_time": self._format_time(start_time),
                "end_time": self._format_time(end_time),
                "duration": round(duration, 2),
                "start_seconds": round(start_time, 2),
                "end_seconds": round(end_time, 2)
            })

            logger.debug(f"Scene {i+1}: [{label}] {start_time:.1f}s - {end_time:.1f}s")

        cap.release()

        # Create SceneMetadata
        scene_metadata = SceneMetadata(
            segments=segments,
            fps=fps
        )

        # Save JSON if requested
        if output_json_path:
            self._save_json(timeline_data, output_json_path)
            logger.info(f"Scene JSON saved to: {output_json_path}")

        logger.info(f"Classification complete: {len(segments)} scenes")
        return scene_metadata

    def _setup_device(self) -> torch.device:
        """Auto-detect and setup device (MPS > CUDA > CPU).

        Returns:
            torch.device object
        """
        if self.config.device != "auto":
            return torch.device(self.config.device)

        # Auto-detect: MPS > CUDA > CPU
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _load_model(self) -> nn.Module:
        """Load ResNet18 model from .pth file.

        Returns:
            Loaded PyTorch model

        Raises:
            FileNotFoundError: If model file not found
        """
        model_path = Path(self.config.model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Scene classification model not found: {model_path}\n"
                f"Expected location: resources/models/scene_classifier/soccer_model_ver2.pth\n"
                f"Please ensure model file is installed."
            )

        # Create ResNet18 with custom output layer
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(self.LABELS))

        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()

        logger.info(f"Model loaded from: {model_path}")
        return model

    def _create_transform(self) -> transforms.Compose:
        """Create ImageNet-normalized transforms for ResNet18.

        Returns:
            torchvision.transforms.Compose object
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _detect_scene_boundaries(self, video_path: str) -> List[Tuple[float, float]]:
        """Use PySceneDetect ContentDetector to find scene cuts.

        Args:
            video_path: Path to video file

        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()

        # Add ContentDetector with config parameters
        scene_manager.add_detector(
            ContentDetector(
                threshold=self.config.threshold,
                min_scene_len=self.config.min_scene_len
            )
        )

        # Detect scenes
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list(video_manager.get_base_timecode())
        video_manager.release()

        # Convert to (start_seconds, end_seconds) tuples
        scenes = []
        for scene in scene_list:
            start = scene[0].get_seconds()
            end = scene[1].get_seconds()
            scenes.append((start, end))

        return scenes

    def _classify_frame(self, frame) -> str:
        """Classify a single frame.

        Args:
            frame: OpenCV BGR frame (numpy array)

        Returns:
            Scene label (one of LABELS)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        # Transform and add batch dimension
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, preds = torch.max(outputs, 1)

        label_idx = preds.item()
        return self.LABELS[label_idx]

    def _save_json(self, timeline_data: List[dict], output_path: str):
        """Save timeline data to JSON file (auto_tagger format).

        Args:
            timeline_data: List of scene dictionaries
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(timeline_data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as '00분 00초' format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string
        """
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}분 {s:02d}초"
