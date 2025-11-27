"""Video processing utilities."""

import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class VideoReader:
    """Simple video reader using OpenCV."""

    def __init__(self, video_path: str):
        """Initialize video reader.

        Args:
            video_path: Path to video file
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

        print(f"[VideoReader] Opened: {video_path}")
        print(f"[VideoReader] Size: {self.width}x{self.height}, FPS: {self.fps}, Frames: {self.frame_count}")

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
        """Read next frame."""
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
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
