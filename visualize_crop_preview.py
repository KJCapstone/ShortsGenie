"""Visualize crop region preview on horizontal video.

This script shows where the vertical crop will be applied on the original horizontal video.
Red box indicates the crop region, detected objects are shown with colored boxes.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.soccernet_detector import SoccerNetDetector
from src.core.temporal_filter import TemporalBallFilter
from src.core.roi_calculator import ROICalculator
from src.core.smoother import Smoother
from src.utils.video_utils import VideoReader
from src.utils.config import AppConfig
from src.models.detection_result import Detection, ROI, FrameDetections


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw detection boxes on frame.

    Args:
        frame: Input frame
        detections: List of Detection objects

    Returns:
        Frame with detection boxes drawn
    """
    result = frame.copy()

    for det in detections:
        # Convert bbox (x, y, width, height) to (x1, y1, x2, y2)
        x1 = det.bbox.x
        y1 = det.bbox.y
        x2 = det.bbox.x + det.bbox.width
        y2 = det.bbox.y + det.bbox.height

        # Color based on class: Ball = Blue, Person = Green
        if det.is_ball:
            color = (255, 0, 0)  # Blue for ball
            thickness = 3
            label = f"Ball {det.confidence:.2f}"
        else:
            color = (0, 255, 0)  # Green for person/player
            thickness = 2
            label = f"Player {det.track_id}" if det.track_id else f"Player {det.confidence:.2f}"

        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)

        # Draw label text
        cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result


def draw_crop_region(frame: np.ndarray, roi: ROI, crop_width: int, crop_height: int) -> np.ndarray:
    """Draw the crop region that will be used for vertical video.

    Args:
        frame: Input frame
        roi: ROI object with center coordinates
        crop_width: Width of crop region (9:16 aspect)
        crop_height: Height of crop region

    Returns:
        Frame with crop region drawn
    """
    result = frame.copy()
    frame_h, frame_w = frame.shape[:2]

    # Calculate crop region bounds
    half_w = crop_width // 2
    half_h = crop_height // 2

    center_x = int(roi.center_x)
    center_y = int(roi.center_y)

    # Ensure crop region stays within frame
    x1 = max(0, center_x - half_w)
    x2 = min(frame_w, center_x + half_w)
    y1 = max(0, center_y - half_h)
    y2 = min(frame_h, center_y + half_h)

    # Draw red rectangle for crop region
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 4)

    # Draw crosshair at ROI center
    crosshair_size = 20
    cv2.line(result,
             (center_x - crosshair_size, center_y),
             (center_x + crosshair_size, center_y),
             (0, 0, 255), 2)
    cv2.line(result,
             (center_x, center_y - crosshair_size),
             (center_x, center_y + crosshair_size),
             (0, 0, 255), 2)

    # Draw ROI center point
    cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1)

    # Add text label for crop region
    label = f"Crop Region ({crop_width}x{crop_height})"
    cv2.putText(result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return result


def add_info_overlay(frame: np.ndarray, frame_num: int, total_frames: int,
                     num_balls: int, num_players: int, fps: float) -> np.ndarray:
    """Add information overlay to frame.

    Args:
        frame: Input frame
        frame_num: Current frame number
        total_frames: Total number of frames
        num_balls: Number of balls detected
        num_players: Number of players detected
        fps: Processing FPS

    Returns:
        Frame with info overlay
    """
    result = frame.copy()

    # Create semi-transparent overlay
    overlay = result.copy()
    h, w = result.shape[:2]
    cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)

    # Add text information
    y_offset = 35
    cv2.putText(result, f"Frame: {frame_num}/{total_frames}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 25
    cv2.putText(result, f"Balls: {num_balls} | Players: {num_players}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 25
    cv2.putText(result, f"Processing FPS: {fps:.1f}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return result


def visualize_crop_preview(
    input_path: str,
    output_path: str,
    crop_width: int = 1080,
    crop_height: int = 1920,
    use_temporal_filter: bool = True,
    use_kalman_smoothing: bool = True
):
    """Create preview video showing crop region on horizontal video.

    Args:
        input_path: Path to input horizontal video
        output_path: Path to output preview video
        crop_width: Width of vertical crop (default: 1080)
        crop_height: Height of vertical crop (default: 1920)
        use_temporal_filter: Apply temporal filtering to ball trajectory
        use_kalman_smoothing: Apply Kalman smoothing to ROI trajectory
    """
    print("\n" + "=" * 70)
    print("CROP REGION VISUALIZATION")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Crop size: {crop_width}x{crop_height} (9:16 aspect ratio)")
    print(f"Temporal filter: {use_temporal_filter}")
    print(f"Kalman smoothing: {use_kalman_smoothing}")
    print("=" * 70)

    # Initialize components
    print("\n[1/5] Initializing components...")
    config = AppConfig()
    config.detection.confidence_threshold = 0.05

    detector = SoccerNetDetector(config.detection)
    temporal_filter = TemporalBallFilter() if use_temporal_filter else None
    roi_calculator = ROICalculator(config.roi)
    smoother = Smoother(config.kalman) if use_kalman_smoothing else None

    # Open video
    print("[2/5] Opening video...")
    reader = VideoReader(input_path)

    frame_width = reader.width
    frame_height = reader.height
    fps = reader.fps
    total_frames = reader.frame_count

    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")

    # Create output video writer
    print("[3/5] Creating output video...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to create output video: {output_path}")

    # Process frames
    print("[4/5] Processing frames...")
    frame_num = 0
    start_time = time.time()

    # Storage for temporal filtering
    all_detections = []

    # First pass: detect all objects
    print("  First pass: Detecting objects...")
    for frame in reader:
        frame_detections = detector.detect_frame(frame, frame_num, frame_num / fps)
        detections = frame_detections.detections  # Extract list of Detection objects
        all_detections.append(detections)
        frame_num += 1
        if frame_num % 30 == 0:
            print(f"    Processed {frame_num}/{total_frames} frames")

    # Apply temporal filter if enabled
    if temporal_filter:
        print("  Applying temporal filter to ball trajectory...")
        # Convert to dict format: frame_number -> [ball_detections]
        ball_detections_dict = {}
        for frame_idx, detections in enumerate(all_detections):
            balls = [d for d in detections if d.is_ball]
            if balls:
                ball_detections_dict[frame_idx] = balls

        # Apply temporal filtering
        ball_trajectory = temporal_filter.filter_trajectory(ball_detections_dict, total_frames)

        # Merge filtered ball back with player detections
        for frame_idx in range(len(all_detections)):
            # Remove old ball detections
            all_detections[frame_idx] = [d for d in all_detections[frame_idx] if not d.is_ball]

            # Add filtered ball if available
            ball_pos = ball_trajectory.get_position(frame_idx)
            if ball_pos and ball_trajectory.is_detected(frame_idx):
                # Find original ball detection for this frame
                if frame_idx in ball_detections_dict and ball_detections_dict[frame_idx]:
                    # Use the filtered position but keep original detection
                    ball_det = ball_detections_dict[frame_idx][0]
                    all_detections[frame_idx].append(ball_det)

    # Second pass: calculate ROI
    print("  Second pass: Calculating ROI...")
    all_rois = []
    for frame_idx in range(total_frames):
        detections = all_detections[frame_idx]

        # Create FrameDetections object
        frame_detections = FrameDetections(
            frame_number=frame_idx,
            timestamp=frame_idx / fps,
            detections=detections
        )

        # Calculate ROI
        roi = roi_calculator.calculate_roi_from_detections(frame_detections, frame_width, frame_height)
        all_rois.append(roi)

        if (frame_idx + 1) % 100 == 0:
            print(f"    Calculated ROI for {frame_idx + 1}/{total_frames} frames")

    # Apply smoothing if enabled
    if smoother:
        print("  Applying Kalman smoothing to ROI trajectory...")
        all_rois = smoother.smooth_trajectory(all_rois)

    # Third pass: visualize
    print("  Third pass: Visualizing crop regions...")
    reader = VideoReader(input_path)  # Reset reader
    frame_num = 0

    for frame in reader:
        detections = all_detections[frame_num]
        roi = all_rois[frame_num]

        # Draw visualizations
        # 1. Draw detections
        vis_frame = draw_detections(frame, detections)

        # 2. Draw crop region
        vis_frame = draw_crop_region(vis_frame, roi, crop_width, crop_height)

        # 3. Add info overlay
        num_balls = sum(1 for d in detections if d.is_ball)
        num_players = sum(1 for d in detections if not d.is_ball)
        elapsed = time.time() - start_time
        processing_fps = (frame_num + 1) / elapsed if elapsed > 0 else 0

        # vis_frame = add_info_overlay(vis_frame, frame_num + 1, total_frames,
        #                              num_balls, num_players, processing_fps)

        # Write frame
        out.write(vis_frame)

        frame_num += 1
        if frame_num % 30 == 0:
            print(f"    Visualized {frame_num}/{total_frames} frames")

    # Cleanup
    print("[5/5] Finalizing...")
    out.release()
    reader.close()

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("✓ VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"Output saved to: {output_path}")
    print(f"Processing time: {elapsed:.2f}s")
    print(f"Average FPS: {total_frames / elapsed:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n" + "=" * 70)
        print("USAGE")
        print("=" * 70)
        print(f"  python {Path(__file__).name} <input_video> [output_video]")
        print("\nExample:")
        print(f"  python {Path(__file__).name} input/goal_clip.mp4")
        print(f"  python {Path(__file__).name} input/goal_clip.mp4 output/preview.mp4")
        print("\nThis will create a preview showing:")
        print("  - Red box: Crop region for vertical video (9:16)")
        print("  - Blue box: Detected ball")
        print("  - Green box: Detected players")
        print("  - Red crosshair: ROI center point")
        print("=" * 70)
        sys.exit(1)

    input_video = sys.argv[1]

    if not Path(input_video).exists():
        print(f"\n✗ Error: Input video not found: {input_video}")
        sys.exit(1)

    # Generate output path if not provided
    if len(sys.argv) >= 3:
        output_video = sys.argv[2]
    else:
        input_stem = Path(input_video).stem
        output_video = f"output/{input_stem}_crop_preview.mp4"

    try:
        visualize_crop_preview(
            input_path=input_video,
            output_path=output_video,
            crop_width=1080,
            crop_height=1920,
            use_temporal_filter=True,
            use_kalman_smoothing=True
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
