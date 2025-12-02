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

        # Hysteresis state
        self.last_roi_center = None
        self.movement_counter = 0

        # Scene locking state
        self.locked_roi_center = None
        self.lock_confidence = 0.0
        self.frames_in_lock = 0

        print(f"[ROICalculator] Ball weight: {self.config.ball_weight}")
        print(f"[ROICalculator] Person weight: {self.config.person_weight}")
        print(f"[ROICalculator] Hysteresis: {self.config.use_hysteresis} (threshold={self.config.hysteresis_threshold}px)")
        print(f"[ROICalculator] Scene locking: {self.config.use_scene_locking} (threshold={self.config.lock_threshold}px)")
        print(f"[ROICalculator] Target size: {self.target_width}x{self.target_height}")

    def calculate_roi_trajectory(
        self,
        ball_trajectory: BallTrajectory,
        player_detections_per_frame: Dict[int, List[Detection]],
        frame_width: int,
        frame_height: int,
        total_frames: int,
        scene_manager: Optional['SceneManager'] = None
    ) -> List[ROI]:
        """Calculate ROI for each frame using ball and player positions.

        Args:
            ball_trajectory: Filtered ball trajectory from TemporalBallFilter
            player_detections_per_frame: Player detections by frame
            frame_width: Original video width
            frame_height: Original video height
            total_frames: Total number of frames
            scene_manager: Optional scene manager for scene-aware ROI calculation

        Returns:
            List of ROI objects, one per frame
        """
        roi_list = []

        # Scene-aware feature: fixed ROI per scene
        current_scene_roi = None  # ROI for current scene (locked)
        scene_roi_frames = 30      # Number of frames to analyze at scene start

        for frame_num in range(total_frames):
            # Check for scene boundary (scene-aware feature)
            if scene_manager:
                scene_info = scene_manager.update(frame_num)

                if scene_info['is_boundary']:
                    # New scene started - reset ROI state
                    self.reset_for_scene_boundary(scene_info)
                    current_scene_roi = None  # Force recalculation

                    # Apply scene-specific weights if enabled
                    if self.config.use_scene_type_weights:
                        self._apply_scene_weights(scene_info['scene_type'])

                # If we have a scene manager and no fixed ROI yet, calculate it
                if current_scene_roi is None:
                    # Calculate ROI from first N frames of this scene
                    scene = scene_info.get('scene')
                    if scene:
                        current_scene_roi = self._calculate_scene_fixed_roi(
                            ball_trajectory,
                            player_detections_per_frame,
                            frame_width,
                            frame_height,
                            scene.start_frame,
                            min(scene.end_frame, scene.start_frame + scene_roi_frames)
                        )
                        print(f"[ROI] Fixed ROI calculated for scene at frame {frame_num}: "
                              f"center=({current_scene_roi.center_x}, {current_scene_roi.center_y})")

                # Use fixed ROI for this scene
                if current_scene_roi is not None:
                    roi_list.append(current_scene_roi)
                    continue

            # No scene manager OR no scene data - use dynamic ROI (original behavior)
            ball_pos = ball_trajectory.get_position(frame_num)
            players = self._get_top_players(
                player_detections_per_frame.get(frame_num, []),
                max_players=5
            )
            roi_center, confidence = self._calculate_weighted_center(
                ball_pos, players, frame_width, frame_height
            )
            roi = self._create_roi(roi_center, frame_width, frame_height, confidence)
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

        # Apply scale factor to show wider view (less zoom)
        # roi_scale_factor > 1.0 = wider view with letterboxing
        # roi_scale_factor < 1.0 = tighter crop
        roi_width = int(roi_width * self.config.roi_scale_factor)
        roi_height = int(roi_height * self.config.roi_scale_factor)

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

    def apply_hysteresis(
        self,
        new_center: Tuple[int, int],
        confidence: float
    ) -> Tuple[int, int]:
        """Apply hysteresis (dead zone) to prevent jittery movement.

        Only updates ROI center when:
        1. Movement exceeds threshold distance
        2. Movement persists for min_movement_frames

        Args:
            new_center: Proposed new ROI center (x, y)
            confidence: Detection confidence (0-1)

        Returns:
            Final ROI center after hysteresis (x, y)
        """
        if not self.config.use_hysteresis:
            return new_center

        if self.last_roi_center is None:
            # First frame - initialize
            self.last_roi_center = new_center
            self.movement_counter = 0
            return new_center

        # Calculate distance from last ROI center
        dx = new_center[0] - self.last_roi_center[0]
        dy = new_center[1] - self.last_roi_center[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Adaptive threshold based on confidence
        # Lower confidence = larger dead zone (less responsive)
        if self.config.adaptive_threshold:
            adaptive_threshold = self.config.hysteresis_threshold * (1.0 / max(confidence, 0.3))
        else:
            adaptive_threshold = self.config.hysteresis_threshold

        if distance < adaptive_threshold:
            # Within dead zone - keep current ROI center
            self.movement_counter = 0
            return self.last_roi_center
        else:
            # Beyond dead zone - check if movement is sustained
            self.movement_counter += 1

            if self.movement_counter >= self.config.min_movement_frames:
                # Confirmed movement - update with smooth transition
                blend_factor = min(self.movement_counter / 10.0, 1.0)
                blended_x = int(self.last_roi_center[0] * (1 - blend_factor) + new_center[0] * blend_factor)
                blended_y = int(self.last_roi_center[1] * (1 - blend_factor) + new_center[1] * blend_factor)

                self.last_roi_center = (blended_x, blended_y)
                return self.last_roi_center
            else:
                # Movement detected but not yet confirmed
                return self.last_roi_center

    def apply_scene_locking(
        self,
        new_center: Tuple[int, int],
        confidence: float,
        frame_num: int
    ) -> Tuple[int, int]:
        """Apply scene-coherent ROI locking.

        Locks ROI to a region when ball stays in same area.
        Only unlocks on significant movement.

        Args:
            new_center: Proposed new ROI center (x, y)
            confidence: Detection confidence (0-1)
            frame_num: Current frame number (for logging)

        Returns:
            Final ROI center after locking (x, y)
        """
        if not self.config.use_scene_locking:
            return new_center

        if self.locked_roi_center is None:
            # Not locked - initialize
            self.locked_roi_center = new_center
            self.lock_confidence = confidence
            self.frames_in_lock = 1
            return new_center

        # Check distance from locked center
        dx = new_center[0] - self.locked_roi_center[0]
        dy = new_center[1] - self.locked_roi_center[1]
        distance = np.sqrt(dx**2 + dy**2)

        if distance < self.config.lock_threshold and confidence > 0.3:
            # Stay in lock - increment counter
            self.frames_in_lock += 1
            self.lock_confidence = min(1.0, self.lock_confidence * 1.02)  # Build confidence

            if self.frames_in_lock % 30 == 0:  # Log every second at 30fps
                print(f"[ROI] Locked to scene (frame {frame_num}, lock_frames={self.frames_in_lock}, distance={distance:.1f})")

            return self.locked_roi_center

        else:
            # Break lock - significant movement detected
            if self.frames_in_lock >= self.config.lock_min_frames:
                print(f"[ROI] Unlocking scene at frame {frame_num} (distance={distance:.1f} > {self.config.lock_threshold})")

            # Gradual transition to new center (10 frames)
            transition_frames = 10
            blend = min(1.0, 1.0 / transition_frames)
            new_center_x = int(self.locked_roi_center[0] * (1 - blend) + new_center[0] * blend)
            new_center_y = int(self.locked_roi_center[1] * (1 - blend) + new_center[1] * blend)

            self.locked_roi_center = (new_center_x, new_center_y)
            self.lock_confidence = confidence
            self.frames_in_lock = 1

            return (new_center_x, new_center_y)

    def reset_lock(self):
        """Reset scene locking state (call on scene changes)."""
        if self.locked_roi_center is not None:
            print("[ROI] Resetting scene lock")
        self.locked_roi_center = None
        self.lock_confidence = 0.0
        self.frames_in_lock = 0
        self.last_roi_center = None
        self.movement_counter = 0

    def reset_for_scene_boundary(self, scene_info: dict):
        """Reset ROI state when entering a new scene.

        This method is called when a scene boundary is detected to prevent
        ROI state from the previous scene affecting the new scene.

        Args:
            scene_info: Scene information dictionary containing:
                - scene_type: Type of the new scene
                - frame_num: Frame number of the boundary
                - is_boundary: True (always)
        """
        # Reset all ROI state
        self.last_roi_center = None
        self.movement_counter = 0
        self.locked_roi_center = None
        self.lock_confidence = 0.0
        self.frames_in_lock = 0

        scene_type = scene_info.get('scene_type', 'unknown')
        frame_num = scene_info.get('frame_num', '?')
        print(f"[ROI] Scene boundary at frame {frame_num}: '{scene_type}' (ROI state reset)")

    def _apply_scene_weights(self, scene_type: str):
        """Apply scene-specific ROI weights dynamically.

        This method temporarily updates the ROI weights based on the scene type.
        Weights are restored to defaults when the scene changes.

        Args:
            scene_type: Scene type label ("wide", "close", "audience", "replay")
        """
        if scene_type not in self.config.scene_type_weights:
            # Unknown scene type, keep default weights
            return

        weights = self.config.scene_type_weights[scene_type]

        # Update ball and person weights
        self.config.ball_weight = weights.get('ball_weight', 3.0)
        self.config.person_weight = weights.get('person_weight', 1.0)

        # Note: We could also adjust other parameters like lock_threshold here
        # but for now we only modify the core ROI weights

    def _calculate_scene_fixed_roi(
        self,
        ball_trajectory: BallTrajectory,
        player_detections_per_frame: Dict[int, List[Detection]],
        frame_width: int,
        frame_height: int,
        start_frame: int,
        end_frame: int
    ) -> ROI:
        """Calculate a fixed ROI for an entire scene based on initial frames.

        This analyzes the first N frames of a scene to determine the optimal
        fixed ROI that will be used for the entire scene.

        Args:
            ball_trajectory: Ball trajectory
            player_detections_per_frame: Player detections by frame
            frame_width: Frame width
            frame_height: Frame height
            start_frame: Scene start frame
            end_frame: Frame to analyze until (usually start + 30)

        Returns:
            Fixed ROI to use for entire scene
        """
        # Collect all ball and player positions in the analysis window
        all_positions = []
        all_weights = []

        for frame_num in range(start_frame, end_frame):
            # Get ball position
            ball_pos = ball_trajectory.get_position(frame_num)
            if ball_pos is not None:
                all_positions.append(ball_pos)
                all_weights.append(self.config.ball_weight)

            # Get player positions
            players = self._get_top_players(
                player_detections_per_frame.get(frame_num, []),
                max_players=5
            )
            for player in players:
                all_positions.append((player.bbox.center_x, player.bbox.center_y))
                all_weights.append(self.config.person_weight)

        # If no detections, use frame center
        if not all_positions:
            center = (frame_width // 2, frame_height // 2)
            confidence = 0.0
        else:
            # Calculate weighted average of all positions
            total_weight = sum(all_weights)
            weighted_x = sum(pos[0] * w for pos, w in zip(all_positions, all_weights)) / total_weight
            weighted_y = sum(pos[1] * w for pos, w in zip(all_positions, all_weights)) / total_weight
            center = (int(weighted_x), int(weighted_y))
            confidence = min(1.0, len(all_positions) / (end_frame - start_frame))

        # Create fixed ROI
        roi = self._create_roi(center, frame_width, frame_height, confidence)

        return roi

    def calculate_roi_with_stabilization(
        self,
        ball_pos: Optional[Tuple[int, int]],
        players: List[Detection],
        frame_width: int,
        frame_height: int,
        frame_num: int = 0
    ) -> ROI:
        """Calculate ROI with hysteresis and scene locking applied.

        This is the enhanced version that combines:
        1. Weighted center calculation
        2. Hysteresis (dead zone)
        3. Scene locking

        Args:
            ball_pos: Ball position (x, y) or None
            players: List of player detections
            frame_width: Frame width
            frame_height: Frame height
            frame_num: Current frame number

        Returns:
            ROI object with stabilized center
        """
        # Step 1: Calculate raw weighted center
        raw_center, confidence = self._calculate_weighted_center(
            ball_pos, players, frame_width, frame_height
        )

        # Step 2: Apply hysteresis (dead zone)
        center_after_hysteresis = self.apply_hysteresis(raw_center, confidence)

        # Step 3: Apply scene locking
        final_center = self.apply_scene_locking(center_after_hysteresis, confidence, frame_num)

        # Step 4: Create ROI with final center
        roi = self._create_roi(final_center, frame_width, frame_height, confidence)

        return roi
