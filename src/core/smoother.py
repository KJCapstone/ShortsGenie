"""Smoothing algorithms for ROI trajectory (Kalman Filter and EMA)."""

import numpy as np
from typing import List
from src.models.detection_result import ROI
from src.utils.config import KalmanConfig, SmootherConfig


class Smoother:
    """Kalman filter for smoothing ROI center point trajectory."""

    def __init__(self, config: KalmanConfig):
        """Initialize Kalman filter smoother.

        Args:
            config: Kalman filter configuration
        """
        self.config = config

        # Kalman filter state: [x, y, vx, vy] (position and velocity)
        self.state = None
        self.covariance = None

        print(f"[Smoother] Process variance: {config.process_variance}")
        print(f"[Smoother] Measurement variance: {config.measurement_variance}")

    def smooth_trajectory(self, roi_list: List[ROI]) -> List[ROI]:
        """Apply Kalman filter smoothing to ROI trajectory.

        Args:
            roi_list: List of ROI objects

        Returns:
            Smoothed list of ROI objects
        """
        if not roi_list:
            return roi_list

        # Initialize filter with first ROI
        self._initialize_filter(roi_list[0])

        smoothed_rois = []

        for roi in roi_list:
            # Predict step
            self._predict()

            # Update step with measurement
            measurement = np.array([roi.center_x, roi.center_y])
            self._update(measurement)

            # Create smoothed ROI
            smoothed_roi = ROI(
                center_x=int(self.state[0]),
                center_y=int(self.state[1]),
                width=roi.width,
                height=roi.height,
                confidence=roi.confidence
            )
            smoothed_rois.append(smoothed_roi)

        return smoothed_rois

    def _initialize_filter(self, first_roi: ROI):
        """Initialize Kalman filter state.

        Args:
            first_roi: First ROI in trajectory
        """
        # Initial state: [x, y, 0, 0] (position with zero velocity)
        self.state = np.array([
            first_roi.center_x,
            first_roi.center_y,
            0.0,
            0.0
        ], dtype=float)

        # Initial covariance matrix
        self.covariance = np.eye(4) * self.config.initial_state_covariance

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ])

        # Measurement matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise covariance
        self.Q = np.eye(4) * self.config.process_variance

        # Measurement noise covariance
        self.R = np.eye(2) * self.config.measurement_variance

    def _predict(self):
        """Kalman filter prediction step."""
        # Predict state
        self.state = self.F @ self.state

        # Predict covariance
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q

    def _update(self, measurement: np.ndarray):
        """Kalman filter update step.

        Args:
            measurement: Observed position [x, y]
        """
        # Innovation (measurement residual)
        y = measurement - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.covariance @ self.H.T + self.R

        # Kalman gain
        K = self.covariance @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(4)
        self.covariance = (I - K @ self.H) @ self.covariance


class EMASmoother:
    """Exponential Moving Average smoother - simpler alternative to Kalman filter.

    EMA formula: S_t = alpha * X_t + (1 - alpha) * S_{t-1}
    where:
    - S_t is the smoothed value at time t
    - X_t is the observation at time t
    - alpha is the smoothing factor (0-1)

    Characteristics:
    - Low alpha (0.05-0.15): Very smooth, slow response
    - High alpha (0.3-0.5): Less smooth, fast response
    """

    def __init__(self, alpha: float = 0.1):
        """Initialize EMA smoother.

        Args:
            alpha: Smoothing factor (0-1). Lower = smoother.
        """
        self.alpha = alpha
        self.smoothed_x = None
        self.smoothed_y = None

        print(f"[EMASmoother] Alpha: {alpha} (lower = smoother)")

    def smooth_trajectory(self, roi_list: List[ROI]) -> List[ROI]:
        """Apply EMA smoothing to ROI trajectory.

        Args:
            roi_list: List of ROI objects

        Returns:
            Smoothed list of ROI objects
        """
        if not roi_list:
            return roi_list

        smoothed_rois = []

        for roi in roi_list:
            if self.smoothed_x is None:
                # Initialize with first ROI
                self.smoothed_x = roi.center_x
                self.smoothed_y = roi.center_y
            else:
                # EMA formula
                self.smoothed_x = self.alpha * roi.center_x + (1 - self.alpha) * self.smoothed_x
                self.smoothed_y = self.alpha * roi.center_y + (1 - self.alpha) * self.smoothed_y

            smoothed_roi = ROI(
                center_x=int(self.smoothed_x),
                center_y=int(self.smoothed_y),
                width=roi.width,
                height=roi.height,
                confidence=roi.confidence
            )
            smoothed_rois.append(smoothed_roi)

        return smoothed_rois


class AdaptiveEMASmoother(EMASmoother):
    """EMA smoother with adaptive alpha based on scene activity.

    Adapts smoothing strength based on movement in recent frames:
    - High movement → High alpha (fast response, less smooth)
    - Low movement → Low alpha (slow response, very smooth)

    This provides:
    - Smooth, locked feel during steady scenes
    - Responsive tracking during fast action
    """

    def __init__(self, alpha_min: float = 0.05, alpha_max: float = 0.25, window: int = 10):
        """Initialize adaptive EMA smoother.

        Args:
            alpha_min: Minimum alpha (very smooth, for low movement)
            alpha_max: Maximum alpha (fast response, for high movement)
            window: Number of frames to analyze for movement detection
        """
        super().__init__(alpha=alpha_min)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.window = window

        print(f"[AdaptiveEMASmoother] Alpha range: {alpha_min}-{alpha_max}")
        print(f"[AdaptiveEMASmoother] Window: {window} frames")

    def smooth_trajectory(self, roi_list: List[ROI]) -> List[ROI]:
        """Apply adaptive EMA smoothing.

        Args:
            roi_list: List of ROI objects

        Returns:
            Smoothed list with adaptive alpha
        """
        if not roi_list:
            return roi_list

        smoothed_rois = []

        for frame_idx, roi in enumerate(roi_list):
            # Calculate adaptive alpha based on recent movement
            adaptive_alpha = self._calculate_adaptive_alpha(roi_list, frame_idx)

            if self.smoothed_x is None:
                # Initialize with first ROI
                self.smoothed_x = roi.center_x
                self.smoothed_y = roi.center_y
            else:
                # Apply EMA with adaptive alpha
                self.smoothed_x = adaptive_alpha * roi.center_x + (1 - adaptive_alpha) * self.smoothed_x
                self.smoothed_y = adaptive_alpha * roi.center_y + (1 - adaptive_alpha) * self.smoothed_y

            smoothed_roi = ROI(
                center_x=int(self.smoothed_x),
                center_y=int(self.smoothed_y),
                width=roi.width,
                height=roi.height,
                confidence=roi.confidence
            )
            smoothed_rois.append(smoothed_roi)

        return smoothed_rois

    def _calculate_adaptive_alpha(self, roi_list: List[ROI], frame_idx: int) -> float:
        """Calculate alpha based on movement in recent frames.

        Args:
            roi_list: Full ROI list
            frame_idx: Current frame index

        Returns:
            Adaptive alpha value (alpha_min to alpha_max)
        """
        if frame_idx < self.window:
            # Not enough history - use minimum alpha (smooth)
            return self.alpha_min

        # Calculate movement in recent window
        recent_rois = roi_list[max(0, frame_idx - self.window):frame_idx + 1]
        movements = []

        for i in range(1, len(recent_rois)):
            dx = recent_rois[i].center_x - recent_rois[i-1].center_x
            dy = recent_rois[i].center_y - recent_rois[i-1].center_y
            movement = np.sqrt(dx**2 + dy**2)
            movements.append(movement)

        if not movements:
            return self.alpha_min

        # Average movement per frame
        avg_movement = np.mean(movements)

        # Map movement to alpha
        # High movement (>200px) → alpha_max (fast response)
        # Low movement (<50px) → alpha_min (very smooth)
        normalized_movement = np.clip(avg_movement / 200.0, 0.0, 1.0)

        # Interpolate between alpha_min and alpha_max
        adaptive_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * normalized_movement

        return adaptive_alpha


def create_smoother(config: SmootherConfig):
    """Factory function to create smoother based on config.

    Args:
        config: Smoother configuration

    Returns:
        Appropriate smoother instance based on config.method
    """
    if config.method == "kalman":
        kalman_config = KalmanConfig(
            process_variance=config.process_variance,
            measurement_variance=config.measurement_variance,
            initial_state_covariance=config.initial_state_covariance
        )
        return Smoother(kalman_config)

    elif config.method == "ema":
        return EMASmoother(alpha=config.ema_alpha)

    elif config.method == "adaptive_ema":
        return AdaptiveEMASmoother(
            alpha_min=config.ema_alpha_min,
            alpha_max=config.ema_alpha_max
        )

    else:
        print(f"[SmootherFactory] WARNING: Unknown method '{config.method}', using Kalman")
        kalman_config = KalmanConfig(
            process_variance=config.process_variance,
            measurement_variance=config.measurement_variance,
            initial_state_covariance=config.initial_state_covariance
        )
        return Smoother(kalman_config)
