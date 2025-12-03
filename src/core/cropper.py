"""Video cropping utilities for dynamic reframing."""

import cv2
import numpy as np
import subprocess
import logging
from typing import List, Optional
from pathlib import Path

from src.models.detection_result import ROI
from src.utils.video_utils import VideoReader, VideoWriter
from src.utils.config import CropperConfig

logger = logging.getLogger(__name__)


class Cropper:
    """Crop video frames according to ROI trajectory."""

    def __init__(
        self,
        output_width: int = 1080,
        output_height: int = 1920,
        config: Optional[CropperConfig] = None
    ):
        """Initialize cropper.

        Args:
            output_width: Output video width
            output_height: Output video height
            config: Cropper configuration (uses defaults if None)
        """
        self.output_width = output_width
        self.output_height = output_height
        self.config = config or CropperConfig()

        print(f"[Cropper] Output size: {output_width}x{output_height}")
        print(f"[Cropper] Letterbox - Top: {self.config.letterbox_top}px, "
              f"Bottom: {self.config.letterbox_bottom}px")

    def process(
        self,
        input_path: str,
        output_path: str,
        roi_trajectory: List[ROI],
        output_fps: Optional[float] = None
    ):
        """Crop video according to ROI trajectory with audio preservation.

        Args:
            input_path: Input video path (with audio)
            output_path: Output video path (will have audio)
            roi_trajectory: List of ROI objects (one per frame)
            output_fps: Output video fps. If None, uses source video fps.
        """
        print(f"[Cropper] Processing: {input_path} -> {output_path}")

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Crop video with OpenCV (no audio)
        temp_video_path = str(Path(output_path).with_suffix('.temp.mp4'))

        with VideoReader(input_path) as reader:
            # Use specified output_fps or fall back to source fps
            fps_to_use = output_fps if output_fps is not None else reader.fps

            # Create video writer for temp file
            with VideoWriter(
                temp_video_path,
                self.output_width,
                self.output_height,
                fps_to_use,
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

        # Step 2: Add audio from source using FFmpeg
        print(f"[Cropper] Adding audio from source...")
        success = self._add_audio_from_source(temp_video_path, output_path, input_path)

        # Cleanup temp file
        try:
            Path(temp_video_path).unlink()
        except Exception as e:
            logger.warning(f"Failed to delete temp file {temp_video_path}: {e}")

        if success:
            print(f"[Cropper] ✓ Complete: {output_path}")
        else:
            print(f"[Cropper] ⚠ Complete (no audio): {output_path}")

    def _add_audio_from_source(
        self,
        video_path: str,
        output_path: str,
        source_path: str
    ) -> bool:
        """Add audio from source video to cropped video using FFmpeg.

        Args:
            video_path: Cropped video path (no audio)
            output_path: Final output path (with audio)
            source_path: Source video path (with audio)

        Returns:
            True if audio was successfully added, False otherwise
        """
        try:
            # Method 1: Try to copy audio codec (fastest, works for most formats)
            cmd_copy = [
                'ffmpeg', '-y',
                '-i', video_path,      # Cropped video (no audio)
                '-i', source_path,     # Original video (with audio)
                '-map', '0:v:0',       # Use video from first input
                '-map', '1:a:0',       # Use audio from second input
                '-c:v', 'copy',        # Copy video (no re-encoding)
                '-c:a', 'copy',        # Copy audio (no re-encoding)
                '-shortest',           # Match shortest stream
                output_path
            ]

            logger.info(f"[Cropper] Attempting audio merge (codec copy)")
            result = subprocess.run(cmd_copy, check=True, capture_output=True, text=True)
            logger.info(f"[Cropper] ✓ Audio added successfully (codec copy)")
            return True

        except subprocess.CalledProcessError as e:
            logger.warning(f"[Cropper] Audio merge (codec copy) failed: {e.returncode}")

            # Method 2: Re-encode audio to AAC (more compatible, works with MKV/AVI)
            try:
                cmd_reencode = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-i', source_path,
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    '-c:v', 'copy',
                    '-c:a', 'aac',         # Re-encode to AAC
                    '-b:a', '128k',
                    '-shortest',
                    output_path
                ]

                logger.info(f"[Cropper] Attempting audio merge (AAC re-encode)")
                subprocess.run(cmd_reencode, check=True, capture_output=True, text=True)
                logger.info(f"[Cropper] ✓ Audio added successfully (AAC re-encode)")
                return True

            except subprocess.CalledProcessError as e2:
                logger.error(f"[Cropper] Audio merge (re-encode) also failed: {e2.returncode}")
                logger.error(f"[Cropper] FFmpeg stderr: {e2.stderr}")

                # Method 3: Fallback - copy video without audio
                logger.warning(f"[Cropper] Creating output without audio (fallback)")
                try:
                    import shutil
                    shutil.copy2(video_path, output_path)
                    return False
                except Exception as e3:
                    logger.error(f"[Cropper] Even fallback copy failed: {e3}")
                    raise

    def _crop_frame(self, frame: np.ndarray, roi: ROI) -> np.ndarray:
        """Crop a single frame with full vertical height and horizontal tracking.

        Args:
            frame: Input frame
            roi: ROI specification (only center_x is used for horizontal tracking)

        Returns:
            Cropped frame with vertical letterboxing (1080x1920)
            - Uses full frame height
            - Tracks horizontally based on ROI center
            - No horizontal letterbox (fills 1080px width)
        """
        frame_height, frame_width = frame.shape[:2]

        # Calculate available height for content (excluding letterbox)
        content_height = self.output_height - self.config.letterbox_top - self.config.letterbox_bottom

        # Calculate scale to fit full frame height into content_height
        scale = content_height / frame_height

        # Calculate required width from original frame to get output_width after scaling
        required_width = int(self.output_width / scale)

        # Calculate horizontal crop boundaries centered on ROI
        half_width = required_width // 2
        x1 = max(0, roi.center_x - half_width)
        x2 = min(frame_width, roi.center_x + half_width)

        # Adjust if we hit frame boundaries
        if x1 == 0:
            x2 = min(required_width, frame_width)
        elif x2 == frame_width:
            x1 = max(0, frame_width - required_width)

        # Use full frame height (no vertical cropping of original content)
        y1 = 0
        y2 = frame_height

        # Crop frame: full height, ROI-centered width
        cropped = frame[y1:y2, x1:x2]

        # Create black canvas for output (1080x1920)
        canvas = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)

        # Resize cropped frame to fit content area (should result in exactly 1080x1344)
        resized = cv2.resize(cropped, (self.output_width, content_height), interpolation=cv2.INTER_LINEAR)

        # Place resized video on canvas with vertical letterbox
        y_offset = self.config.letterbox_top

        canvas[y_offset:y_offset + content_height, 0:self.output_width] = resized

        return canvas
