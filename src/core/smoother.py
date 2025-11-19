"""Kalman Filter for smoothing ROI trajectory."""

import numpy as np
from typing import List
from src.models.detection_result import ROI
from src.utils.config import KalmanConfig


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
