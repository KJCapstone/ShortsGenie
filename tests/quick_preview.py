"""Quick preview tool to test detection on multiple video segments.

Usage:
    python tests/quick_preview.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from test_detection_visualization_v2 import (
    EnhancedKeyPlayerAnalyzer, EnhancedVisualizer, process_video
)
from src.core.detector import ObjectDetector
from src.utils.config import AppConfig


def sample_video_segments(video_path: str, num_samples: int = 5, sample_duration: float = 2.0):
    """Sample and analyze multiple segments from a video to find interesting parts.

    Args:
        video_path: Path to video
        num_samples: Number of segments to sample
        sample_duration: Duration of each sample in seconds

    Returns:
        List of (timestamp, detection_count, ball_count) tuples
    """
    print("\n" + "=" * 70)
    print("Video Segment Sampler - Finding Best Segments")
    print("=" * 70)

    config = AppConfig()
    config.detection.confidence_threshold = 0.25
    detector = ObjectDetector(config.detection)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"\nVideo: {video_path}")
    print(f"Duration: {duration:.1f}s ({total_frames} frames @ {fps} FPS)")
    print(f"\nSampling {num_samples} segments of {sample_duration}s each...")

    # Sample evenly spaced segments
    sample_interval = duration / (num_samples + 1)
    results = []

    for i in range(num_samples):
        timestamp = (i + 1) * sample_interval
        frame_num = int(timestamp * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        # Detect objects
        detections = detector.detect_frame(frame, frame_num, timestamp)

        num_players = len(detections.person_detections)
        num_balls = len(detections.ball_detections)
        total_detections = len(detections.detections)

        results.append((timestamp, num_players, num_balls))

        status = "✓" if num_players > 3 else "·"
        ball_status = "⚽" if num_balls > 0 else " "
        print(f"{status} Segment {i+1}: {timestamp:6.1f}s - "
              f"{num_players:2d} players, {num_balls} balls {ball_status}")

    cap.release()

    # Find best segments (most players)
    results.sort(key=lambda x: (x[2], x[1]), reverse=True)  # Sort by balls, then players

    print(f"\n{'='*70}")
    print("Top 3 Segments (Best for Testing):")
    for i, (ts, players, balls) in enumerate(results[:3], 1):
        print(f"{i}. Time: {ts:.1f}s - {players} players, {balls} balls")
    print(f"{'='*70}\n")

    return results


def create_comparison_video(input_path: str, timestamps: list, output_path: str):
    """Create a comparison video showing detection on multiple segments.

    Args:
        input_path: Input video path
        timestamps: List of (start_time, duration) tuples
        output_path: Output path
    """
    print(f"\nCreating comparison video with {len(timestamps)} segments...")

    config = AppConfig()
    config.detection.confidence_threshold = 0.25
    config.key_player.importance_threshold = 3.0

    detector = ObjectDetector(config.detection)
    analyzer = EnhancedKeyPlayerAnalyzer(config)
    visualizer = EnhancedVisualizer(config)

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    processed = 0

    for seg_idx, (start_time, duration) in enumerate(timestamps, 1):
        print(f"Processing segment {seg_idx}/{len(timestamps)} at {start_time:.1f}s...")

        start_frame = int(start_time * fps)
        end_frame = int((start_time + duration) * fps)

        detector.reset_tracking()  # Reset tracking for each segment
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_num / fps
            detections = detector.detect_and_track(frame, frame_num, timestamp)
            analyzer.analyze_frame(detections, (height, width))
            vis_frame = visualizer.visualize_frame(frame, detections)

            # Add segment label
            cv2.putText(
                vis_frame, f"Segment {seg_idx}/{len(timestamps)}",
                (width - 300, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            out.write(vis_frame)
            processed += 1

    cap.release()
    out.release()

    print(f"✓ Saved {processed} frames to: {output_path}\n")


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent.parent
    input_path = script_dir / "input" / "korea_vs_brazil.mp4"

    if not input_path.exists():
        print(f"ERROR: Video not found: {input_path}")
        return

    # Step 1: Sample video to find interesting segments
    results = sample_video_segments(str(input_path), num_samples=10, sample_duration=2.0)

    # Step 2: Create comparison video from top 3 segments
    top_segments = [(ts, 5.0) for ts, _, _ in results[:3]]  # 5 seconds each

    output_path = script_dir / "output" / "detection_comparison.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    create_comparison_video(str(input_path), top_segments, str(output_path))

    print("=" * 70)
    print("Quick preview complete!")
    print(f"Output: {output_path}")
    print("\nTo test a specific segment, run:")
    print("  python tests/test_detection_visualization_v2.py --start <time> --duration 10")
    print("=" * 70)


if __name__ == "__main__":
    main()
