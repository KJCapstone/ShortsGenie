"""ROI (Region of Interest) calculator for dynamic reframing."""

from typing import List, Optional, Tuple, Dict
import numpy as np

from src.models.detection_result import Detection, FrameDetections, ROI
from src.core.temporal_filter import BallTrajectory
from src.utils.config import ROIConfig


class ROICalculator:
    """
    Calculate Region of Interest using weighted object positions.

    Weighting scheme:
    - Ball (filtered): weight = ball_weight (default 3.0)
    - Players (top N): weight = person_weight (default 1.0 each)
    - Total weight normalized to sum to 1.0
    """

    def __init__(self, config: Optional[ROIConfig] = None):
        """Initialize ROI calculator.

        Args:
            config: ROI configuration. Uses defaults if None.
        """
        self.config = config or ROIConfig()
        self.target_width = 1080   # Target width (may be adjusted based on source)
        self.target_height = 1920  # Target height (may be adjusted based on source)
        self.target_aspect_ratio = 9.0 / 16.0  # 9:16 aspect ratio

        print(f"[ROICalculator] Ball weight: {self.config.ball_weight}")
        print(f"[ROICalculator] Person weight: {self.config.person_weight}")
        print(f"[ROICalculator] Target size: {self.target_width}x{self.target_height}")

    def calculate_roi_trajectory(
        self,
        ball_trajectory: BallTrajectory,
        player_detections_per_frame: Dict[int, List[Detection]],
        frame_width: int,
        frame_height: int,
        total_frames: int
    ) -> List[ROI]:
        """Calculate ROI for each frame using ball and player positions.

        Args:
            ball_trajectory: Filtered ball trajectory from TemporalBallFilter
            player_detections_per_frame: Player detections by frame
            frame_width: Original video width
            frame_height: Original video height
            total_frames: Total number of frames

        Returns:
            List of ROI objects, one per frame
        """
        roi_list = []

        for frame_num in range(total_frames):
            # Get ball position (may be interpolated)
            ball_pos = ball_trajectory.get_position(frame_num)

            # Get top player positions
            players = self._get_top_players(
                player_detections_per_frame.get(frame_num, []),
                max_players=5
            )

            # Calculate weighted center
            roi_center, confidence = self._calculate_weighted_center(
                ball_pos,
                players,
                frame_width,
                frame_height
            )

            # Create ROI with bounds checking
            roi = self._create_roi(
                roi_center,
                frame_width,
                frame_height,
                confidence
            )

            roi_list.append(roi)

        return roi_list

    def calculate_roi_from_detections(
        self,
        frame_detections: FrameDetections,
        frame_width: int,
        frame_height: int
    ) -> ROI:
        """Calculate ROI from frame detections (legacy method).

        This method provides backward compatibility with the original pipeline
        that doesn't use temporal filtering.

        Args:
            frame_detections: Detections for a single frame
            frame_width: Original video width
            frame_height: Original video height

        Returns:
            ROI object for the frame
        """
        # Get ball and player detections
        ball_detections = frame_detections.ball_detections
        player_detections = frame_detections.person_detections

        # Get ball position (use highest confidence if multiple)
        ball_pos = None
        if ball_detections:
            best_ball = max(ball_detections, key=lambda d: d.confidence)
            ball_pos = (best_ball.bbox.center_x, best_ball.bbox.center_y)

        # Get top players
        players = self._get_top_players(player_detections, max_players=5)

        # Calculate weighted center
        roi_center, confidence = self._calculate_weighted_center(
            ball_pos,
            players,
            frame_width,
            frame_height
        )

        # Create ROI
        roi = self._create_roi(
            roi_center,
            frame_width,
            frame_height,
            confidence
        )

        return roi

    def _get_top_players(
        self,
        detections: List[Detection],
        max_players: int = 5
    ) -> List[Detection]:
        """Select top N players by confidence.

        Args:
            detections: List of player detections
            max_players: Maximum number of players to return

        Returns:
            Top players sorted by confidence
        """
        if not detections:
            return []

        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        # Return top N
        return sorted_detections[:max_players]

    def _calculate_weighted_center(
        self,
        ball_pos: Optional[Tuple[int, int]],
        players: List[Detection],
        frame_width: int,
        frame_height: int
    ) -> Tuple[Tuple[int, int], float]:
        """Calculate weighted average position.

        Handling:
        - If ball detected: weight = ball_weight
        - Each player: weight = person_weight
        - If no ball: center on players only
        - If no detections: use frame center

        Args:
            ball_pos: Ball position (x, y) or None
            players: List of player detections
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            Tuple of ((center_x, center_y), confidence)
        """
        positions = []
        weights = []

        # Add ball if detected
        if ball_pos is not None:
            positions.append(ball_pos)
            weights.append(self.config.ball_weight)

        # Add players
        for player in players:
            positions.append((
                player.bbox.center_x,
                player.bbox.center_y
            ))
            weights.append(self.config.person_weight)

        # Calculate weighted average
        if positions:
            weighted_x, weighted_y = self._weighted_average(positions, weights)
            # Confidence is higher when we have more detections
            confidence = min(1.0, len(positions) / 6.0)  # Max confidence at 6 objects
        else:
            # No detections - use frame center
            weighted_x = frame_width // 2
            weighted_y = frame_height // 2
            confidence = 0.0

        return (int(weighted_x), int(weighted_y)), confidence

    def _weighted_average(
        self,
        positions: List[Tuple[int, int]],
        weights: List[float]
    ) -> Tuple[float, float]:
        """Calculate weighted average of positions.

        Args:
            positions: List of (x, y) positions
            weights: Corresponding weights

        Returns:
            Weighted average (x, y) position
        """
        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1.0

        normalized_weights = [w / total_weight for w in weights]

        # Calculate weighted mean
        weighted_x = sum(pos[0] * w for pos, w in zip(positions, normalized_weights))
        weighted_y = sum(pos[1] * w for pos, w in zip(positions, normalized_weights))

        return weighted_x, weighted_y

    def _create_roi(
        self,
        center: Tuple[int, int],
        frame_width: int,
        frame_height: int,
        confidence: float
    ) -> ROI:
        """Create ROI with bounds checking and aspect ratio preservation.

        Ensures:
        - ROI stays within frame boundaries
        - Maintains 9:16 aspect ratio
        - ROI size adapts to source frame size
        - Handles edge cases gracefully

        Args:
            center: (x, y) center point
            frame_width: Original frame width
            frame_height: Original frame height
            confidence: ROI confidence score

        Returns:
            ROI object
        """
        center_x, center_y = center

        # Calculate optimal ROI size based on source frame dimensions
        # while maintaining 9:16 aspect ratio
        source_aspect = frame_width / frame_height
        target_aspect = self.target_aspect_ratio  # 9/16 = 0.5625

        if source_aspect > target_aspect:
            # Source is wider (e.g., 16:9) - use full height
            roi_height = frame_height
            roi_width = int(roi_height * target_aspect)
        else:
            # Source is narrower or already 9:16 - use full width
            roi_width = frame_width
            roi_height = int(roi_width / target_aspect)

        # Ensure ROI doesn't exceed frame dimensions
        roi_width = min(roi_width, frame_width)
        roi_height = min(roi_height, frame_height)

        # Calculate half dimensions for centering
        half_width = roi_width // 2
        half_height = roi_height // 2

        # Apply bounds checking - ensure ROI doesn't go outside frame
        # Adjust center_x if ROI would extend beyond frame
        if center_x - half_width < 0:
            center_x = half_width
        elif center_x + half_width > frame_width:
            center_x = frame_width - half_width

        # Adjust center_y if ROI would extend beyond frame
        if center_y - half_height < 0:
            center_y = half_height
        elif center_y + half_height > frame_height:
            center_y = frame_height - half_height

        # Create ROI
        roi = ROI(
            center_x=center_x,
            center_y=center_y,
            width=roi_width,
            height=roi_height,
            confidence=confidence
        )

        return roi

    def calculate_batch_rois(
        self,
        frame_detections_list: List[FrameDetections],
        frame_width: int,
        frame_height: int
    ) -> List[ROI]:
        """Calculate ROIs for a batch of frames (legacy method).

        Args:
            frame_detections_list: List of FrameDetections
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            List of ROI objects
        """
        return [
            self.calculate_roi_from_detections(fd, frame_width, frame_height)
            for fd in frame_detections_list
        ]
