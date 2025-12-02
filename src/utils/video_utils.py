"""Video processing utilities."""

import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


def get_video_resolution(video_path: str) -> Tuple[int, int]:
    """
    Get video resolution (width, height) without loading frames.

    This is a fast operation that only reads video metadata.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height) in pixels

    Raises:
        ValueError: If video cannot be opened or resolution cannot be determined

    Example:
        >>> width, height = get_video_resolution("video.mp4")
        >>> print(f"{width}x{height}")
        1920x1080
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video resolution: {width}x{height}")

    return (width, height)


class VideoReader:
    """Simple video reader using OpenCV with optional frame stepping for fps downsampling."""

    def __init__(self, video_path: str, target_fps: Optional[float] = None):
        """Initialize video reader.

        Args:
            video_path: Path to video file
            target_fps: Target fps for processing. If None or >= source fps, uses source fps.
                       If < source fps, skips frames to achieve target fps.
                       Example: 60fps source with target_fps=30 will read every 2nd frame.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame stepping for fps downsampling
        if target_fps is None or target_fps >= self.fps:
            # No downsampling needed
            self.target_fps = self.fps
            self.frame_step = 1
            self.effective_fps = self.fps
            self.effective_frame_count = self.frame_count
        else:
            # Downsample by skipping frames
            self.target_fps = target_fps
            self.frame_step = max(1, round(self.fps / target_fps))
            self.effective_fps = self.fps / self.frame_step
            self.effective_frame_count = self.frame_count // self.frame_step

        self._current_frame_idx = 0  # Track current frame index for stepping

        print(f"[VideoReader] Opened: {video_path}")
        print(f"[VideoReader] Size: {self.width}x{self.height}, FPS: {self.fps}, Frames: {self.frame_count}")
        if self.frame_step > 1:
            print(f"[VideoReader] Frame stepping: {self.frame_step} "
                  f"({self.fps:.2f}fps → {self.effective_fps:.2f}fps, "
                  f"{self.frame_count} → {self.effective_frame_count} frames)")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __iter__(self):
        """Iterator interface."""
        return self

    def __next__(self) -> np.ndarray:
        """Read next frame with frame stepping applied.

        Returns:
            Next frame after skipping frame_step-1 frames
        """
        # Skip frame_step-1 frames
        for _ in range(self.frame_step - 1):
            ret = self.cap.grab()  # Fast skip without decoding
            if not ret:
                raise StopIteration

        # Read the actual frame
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration

        self._current_frame_idx += 1
        return frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame.

        Returns:
            Tuple of (success, frame)
        """
        return self.cap.read()

    def close(self):
        """Release video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoWriter:
    """Simple video writer using OpenCV."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "mp4v"
    ):
        """Initialize video writer.

        Args:
            output_path: Output video path
            width: Frame width
            height: Frame height
            fps: Frames per second
            codec: FourCC codec (e.g., "mp4v", "avc1")
        """
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not self.writer.isOpened():
            raise ValueError(f"Cannot create video writer: {output_path}")

        print(f"[VideoWriter] Created: {output_path}")
        print(f"[VideoWriter] Size: {width}x{height}, FPS: {fps}, Codec: {codec}")

    def write(self, frame: np.ndarray):
        """Write a frame.

        Args:
            frame: Frame to write
        """
        self.writer.write(frame)

    def close(self):
        """Release video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
