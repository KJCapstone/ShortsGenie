"""Temporal filtering for ball trajectory smoothing using Savitzky-Golay filter."""

from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.signal import savgol_filter
from dataclasses import dataclass, field

from src.models.detection_result import Detection


@dataclass
class BallTrajectory:
    """Smoothed ball trajectory across frames."""
    frame_numbers: List[int] = field(default_factory=list)
    positions: List[Tuple[int, int]] = field(default_factory=list)  # (x, y) center points
    confidences: List[float] = field(default_factory=list)
    is_interpolated: List[bool] = field(default_factory=list)  # True if position was interpolated

    def get_position(self, frame_number: int) -> Optional[Tuple[int, int]]:
        """Get ball position for specific frame.

        Args:
            frame_number: Frame number to query

        Returns:
            (x, y) position if available, None otherwise
        """
        try:
            idx = self.frame_numbers.index(frame_number)
            return self.positions[idx]
        except ValueError:
            return None

    def get_confidence(self, frame_number: int) -> float:
        """Get confidence score for specific frame.

        Args:
            frame_number: Frame number to query

        Returns:
            Confidence score (0.0-1.0), or 0.0 if not available
        """
        try:
            idx = self.frame_numbers.index(frame_number)
            return self.confidences[idx]
        except ValueError:
            return 0.0

    def is_detected(self, frame_number: int) -> bool:
        """Check if ball was actually detected (not interpolated) at frame.

        Args:
            frame_number: Frame number to query

        Returns:
            True if ball was detected (not interpolated)
        """
        try:
            idx = self.frame_numbers.index(frame_number)
            return not self.is_interpolated[idx]
        except ValueError:
            return False


class TemporalBallFilter:
    """
    Temporal filtering for ball detection using 3rd-order polynomial fitting.

    Algorithm:
    1. Collect all ball detections across frames
    2. Select best detection per frame (highest confidence)
    3. Fit Savitzky-Golay filter (3rd order polynomial)
    4. Reject outliers (>threshold px from fitted curve)
    5. Re-fit without outliers
    6. Interpolate missing frames
    """

    def __init__(
        self,
        window_size: int = 51,
        polyorder: int = 3,
        outlier_threshold: float = 100.0,
        min_detections_ratio: float = 0.3  # Need 30% frames with detections
    ):
        """Initialize temporal filter.

        Args:
            window_size: Savitzky-Golay window size (must be odd)
            polyorder: Polynomial order for fitting
            outlier_threshold: Distance threshold in pixels for outlier rejection
            min_detections_ratio: Minimum ratio of frames with ball detections
        """
        self.window_size = window_size
        self.polyorder = polyorder
        self.outlier_threshold = outlier_threshold
        self.min_detections_ratio = min_detections_ratio

        # Ensure window size is odd
        if window_size % 2 == 0:
            self.window_size += 1
            print(f"[TemporalFilter] Adjusted window size to {self.window_size} (must be odd)")

    def filter_trajectory(
        self,
        ball_detections_per_frame: Dict[int, List[Detection]],
        total_frames: int
    ) -> BallTrajectory:
        """Apply temporal filtering to ball detections.

        Args:
            ball_detections_per_frame: Dict mapping frame_number -> [Detection, ...]
            total_frames: Total number of frames in video

        Returns:
            BallTrajectory with smoothed positions for every frame
        """
        # Step 1: Select best detection per frame
        best_per_frame = self._select_best_detections(ball_detections_per_frame)

        # Check if we have enough detections
        detection_ratio = len(best_per_frame) / total_frames
        print(f"[TemporalFilter] Ball detected in {len(best_per_frame)}/{total_frames} frames ({detection_ratio:.1%})")

        if detection_ratio < self.min_detections_ratio:
            print(f"[TemporalFilter] WARNING: Low detection rate ({detection_ratio:.1%} < {self.min_detections_ratio:.1%})")
            print("[TemporalFilter] Returning sparse trajectory without filtering")
            return self._build_sparse_trajectory(best_per_frame, total_frames)

        # Step 2: Apply Savitzky-Golay filter
        smoothed_x, smoothed_y = self._apply_savgol_filter(best_per_frame, total_frames)

        # Step 3: Detect and remove outliers
        inlier_frames = self._detect_outliers(best_per_frame, smoothed_x, smoothed_y)

        # Step 4: Re-fit after outlier removal
        if len(inlier_frames) < len(best_per_frame):
            print(f"[TemporalFilter] Removed {len(best_per_frame) - len(inlier_frames)} outliers")
            inlier_detections = {f: best_per_frame[f] for f in inlier_frames}
            final_x, final_y = self._apply_savgol_filter(inlier_detections, total_frames)
        else:
            final_x, final_y = smoothed_x, smoothed_y

        # Step 5: Build trajectory with interpolation flags
        trajectory = self._build_trajectory(best_per_frame, final_x, final_y, total_frames)

        return trajectory

    def _select_best_detections(
        self,
        detections_per_frame: Dict[int, List[Detection]]
    ) -> Dict[int, Detection]:
        """Select highest confidence detection per frame.

        Args:
            detections_per_frame: Dict mapping frame -> list of detections

        Returns:
            Dict mapping frame -> single best detection
        """
        best_per_frame = {}

        for frame_num, detections in detections_per_frame.items():
            if not detections:
                continue

            # Select detection with highest confidence
            best_detection = max(detections, key=lambda d: d.confidence)
            best_per_frame[frame_num] = best_detection

        return best_per_frame

    def _apply_savgol_filter(
        self,
        detections: Dict[int, Detection],
        total_frames: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Savitzky-Golay filter to x and y coordinates.

        Args:
            detections: Dict mapping frame -> detection
            total_frames: Total number of frames

        Returns:
            Tuple of (smoothed_x, smoothed_y) arrays for all frames
        """
        if not detections:
            # No detections - return empty arrays
            return np.zeros(total_frames), np.zeros(total_frames)

        # Extract frame numbers and positions
        frames = sorted(detections.keys())
        x_coords = [detections[f].bbox.center_x for f in frames]
        y_coords = [detections[f].bbox.center_y for f in frames]

        # Create dense arrays with interpolation for missing frames
        dense_x = np.zeros(total_frames)
        dense_y = np.zeros(total_frames)

        if len(frames) >= 2:
            # Linear interpolation for all frames
            dense_x = np.interp(
                np.arange(total_frames),
                frames,
                x_coords
            )
            dense_y = np.interp(
                np.arange(total_frames),
                frames,
                y_coords
            )
        elif len(frames) == 1:
            # Only one detection - use constant value
            dense_x[:] = x_coords[0]
            dense_y[:] = y_coords[0]

        # Apply Savitzky-Golay filter if we have enough points
        if len(frames) >= self.window_size:
            try:
                smoothed_x = savgol_filter(dense_x, self.window_size, self.polyorder)
                smoothed_y = savgol_filter(dense_y, self.window_size, self.polyorder)
            except Exception as e:
                print(f"[TemporalFilter] Warning: Savitzky-Golay filter failed: {e}")
                smoothed_x, smoothed_y = dense_x, dense_y
        else:
            # Not enough points - use linear interpolation only
            print(f"[TemporalFilter] Not enough detections for Savitzky-Golay ({len(frames)} < {self.window_size})")
            smoothed_x, smoothed_y = dense_x, dense_y

        return smoothed_x, smoothed_y

    def _detect_outliers(
        self,
        detections: Dict[int, Detection],
        smoothed_x: np.ndarray,
        smoothed_y: np.ndarray
    ) -> List[int]:
        """Detect outliers based on distance from smoothed trajectory.

        Args:
            detections: Dict mapping frame -> detection
            smoothed_x: Smoothed x coordinates
            smoothed_y: Smoothed y coordinates

        Returns:
            List of frame numbers that are inliers (not outliers)
        """
        inlier_frames = []

        for frame_num, detection in detections.items():
            # Calculate distance from detection to smoothed position
            det_x = detection.bbox.center_x
            det_y = detection.bbox.center_y
            smooth_x = smoothed_x[frame_num]
            smooth_y = smoothed_y[frame_num]

            distance = np.sqrt((det_x - smooth_x)**2 + (det_y - smooth_y)**2)

            if distance <= self.outlier_threshold:
                inlier_frames.append(frame_num)

        return inlier_frames

    def _build_trajectory(
        self,
        original_detections: Dict[int, Detection],
        final_x: np.ndarray,
        final_y: np.ndarray,
        total_frames: int
    ) -> BallTrajectory:
        """Build final trajectory with interpolation flags.

        Args:
            original_detections: Original detections (to mark which were detected)
            final_x: Final smoothed x coordinates
            final_y: Final smoothed y coordinates
            total_frames: Total number of frames

        Returns:
            BallTrajectory object
        """
        trajectory = BallTrajectory()

        for frame_num in range(total_frames):
            x = int(final_x[frame_num])
            y = int(final_y[frame_num])

            # Check if this frame had an original detection
            was_detected = frame_num in original_detections

            # Get confidence (0.0 for interpolated frames)
            if was_detected:
                confidence = original_detections[frame_num].confidence
            else:
                confidence = 0.0

            trajectory.frame_numbers.append(frame_num)
            trajectory.positions.append((x, y))
            trajectory.confidences.append(confidence)
            trajectory.is_interpolated.append(not was_detected)

        return trajectory

    def _build_sparse_trajectory(
        self,
        detections: Dict[int, Detection],
        total_frames: int
    ) -> BallTrajectory:
        """Build sparse trajectory when not enough detections for filtering.

        Uses simple linear interpolation between detected positions.

        Args:
            detections: Dict mapping frame -> detection
            total_frames: Total number of frames

        Returns:
            BallTrajectory object with sparse detections
        """
        trajectory = BallTrajectory()

        if not detections:
            # No detections at all - return empty trajectory
            for frame_num in range(total_frames):
                trajectory.frame_numbers.append(frame_num)
                trajectory.positions.append((0, 0))
                trajectory.confidences.append(0.0)
                trajectory.is_interpolated.append(True)
            return trajectory

        # Extract detected positions
        frames = sorted(detections.keys())
        x_coords = [detections[f].bbox.center_x for f in frames]
        y_coords = [detections[f].bbox.center_y for f in frames]

        # Interpolate for all frames
        if len(frames) >= 2:
            all_x = np.interp(np.arange(total_frames), frames, x_coords)
            all_y = np.interp(np.arange(total_frames), frames, y_coords)
        elif len(frames) == 1:
            all_x = np.full(total_frames, x_coords[0])
            all_y = np.full(total_frames, y_coords[0])
        else:
            all_x = np.zeros(total_frames)
            all_y = np.zeros(total_frames)

        # Build trajectory
        for frame_num in range(total_frames):
            was_detected = frame_num in detections

            trajectory.frame_numbers.append(frame_num)
            trajectory.positions.append((int(all_x[frame_num]), int(all_y[frame_num])))
            trajectory.confidences.append(detections[frame_num].confidence if was_detected else 0.0)
            trajectory.is_interpolated.append(not was_detected)

        return trajectory
