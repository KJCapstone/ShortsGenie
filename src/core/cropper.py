"""Video cropping utilities for dynamic reframing."""

import cv2
import numpy as np
from typing import List
from pathlib import Path

from src.models.detection_result import ROI
from src.utils.video_utils import VideoReader, VideoWriter


class Cropper:
    """Crop video frames according to ROI trajectory."""

    def __init__(self, output_width: int = 1080, output_height: int = 1920):
        """Initialize cropper.

        Args:
            output_width: Output video width
            output_height: Output video height
        """
        self.output_width = output_width
        self.output_height = output_height

        print(f"[Cropper] Output size: {output_width}x{output_height}")

    def process(
        self,
        input_path: str,
        output_path: str,
        roi_trajectory: List[ROI]
    ):
        """Crop video according to ROI trajectory.

        Args:
            input_path: Input video path
            output_path: Output video path
            roi_trajectory: List of ROI objects (one per frame)
        """
        print(f"[Cropper] Processing: {input_path} -> {output_path}")

        with VideoReader(input_path) as reader:
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Create video writer
            with VideoWriter(
                output_path,
                self.output_width,
                self.output_height,
                reader.fps,
                codec="mp4v"
            ) as writer:
                # Process each frame
                for frame_idx, frame in enumerate(reader):
                    if frame_idx >= len(roi_trajectory):
                        print(f"[Cropper] Warning: Not enough ROIs for frame {frame_idx}")
                        break

                    roi = roi_trajectory[frame_idx]

                    # Crop frame according to ROI
                    cropped = self._crop_frame(frame, roi)

                    # Write cropped frame
                    writer.write(cropped)

                    if (frame_idx + 1) % 100 == 0:
                        print(f"[Cropper] Processed {frame_idx + 1}/{reader.frame_count} frames")

        print(f"[Cropper] ✓ Complete: {output_path}")

    def _crop_frame(self, frame: np.ndarray, roi: ROI) -> np.ndarray:
        """Crop a single frame according to ROI with aspect ratio preservation.

        Args:
            frame: Input frame
            roi: ROI specification

        Returns:
            Cropped frame with letterboxing if needed (1080x1920)
        """
        frame_height, frame_width = frame.shape[:2]

        # Calculate crop boundaries
        half_width = roi.width // 2
        half_height = roi.height // 2

        x1 = max(0, roi.center_x - half_width)
        y1 = max(0, roi.center_y - half_height)
        x2 = min(frame_width, roi.center_x + half_width)
        y2 = min(frame_height, roi.center_y + half_height)

        # Crop
        cropped = frame[y1:y2, x1:x2]

        # Get actual cropped dimensions
        cropped_height, cropped_width = cropped.shape[:2]

        # Create black canvas for output (1080x1920)
        canvas = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)

        # Calculate scaling to fit within 1080x1920 while maintaining aspect ratio
        scale_width = self.output_width / cropped_width
        scale_height = self.output_height / cropped_height
        scale = min(scale_width, scale_height)  # Use smaller scale to fit within bounds

        # Calculate new dimensions maintaining aspect ratio
        new_width = int(cropped_width * scale)
        new_height = int(cropped_height * scale)

        # Only resize if necessary (if scale != 1.0)
        if scale != 1.0:
            resized = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            resized = cropped

        # Calculate position to center the video on canvas
        x_offset = (self.output_width - new_width) // 2
        y_offset = (self.output_height - new_height) // 2

        # Place resized video on canvas (centered with letterboxing)
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        return canvas
