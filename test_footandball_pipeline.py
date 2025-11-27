"""
Test script for FootAndBall detector integration with the reframing pipeline.

This script validates that:
1. FootAndBall detector can be selected via config
2. The detector works with the full reframing pipeline
3. All three backends (yolo, soccernet, footandball) can process the same video

Usage:
    python test_footandball_pipeline.py [input_video_path]

If no input video is provided, it will look for test videos in the input/ directory.
"""

import sys
import os
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.reframing_pipeline import ReframingPipeline
from src.utils.config import AppConfig


def test_footandball_detector():
    """Test FootAndBall detector with a sample video."""

    # Find a test video
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
    else:
        # Look for test videos in input directory
        input_dir = Path("input")
        if input_dir.exists():
            videos = list(input_dir.glob("*.mp4"))
            if videos:
                input_video = str(videos[0])
            else:
                print("Error: No .mp4 files found in input/ directory")
                print("Usage: python test_footandball_pipeline.py [input_video_path]")
                return
        else:
            print("Error: input/ directory not found")
            print("Usage: python test_footandball_pipeline.py [input_video_path]")
            return

    if not os.path.exists(input_video):
        print(f"Error: Video file not found: {input_video}")
        return

    print("="*70)
    print("FootAndBall Detector Integration Test")
    print("="*70)
    print(f"Input video: {input_video}")
    print()

    # Test 1: FootAndBall detector
    print("\n" + "="*70)
    print("TEST 1: FootAndBall Detector")
    print("="*70)

    config = AppConfig()
    config.detection.detector_backend = "footandball"
    config.detection.footandball_ball_threshold = 0.5
    config.detection.footandball_player_threshold = 0.5

    pipeline = ReframingPipeline(config)

    output_path = "output/test_footandball.mp4"
    os.makedirs("output", exist_ok=True)

    try:
        stats = pipeline.process_goal_clip(
            clip_path=input_video,
            output_path=output_path,
            use_soccernet_model=False,  # Ignored when detector_backend is set
            use_temporal_filter=True,
            use_kalman_smoothing=True
        )

        print("\n" + "-"*70)
        print("FootAndBall Results:")
        print(f"  Frames processed: {stats.get('frames_processed', 'N/A')}")
        print(f"  Ball detection rate: {stats.get('ball_detection_rate', 0)*100:.1f}%")
        print(f"  Avg players/frame: {stats.get('average_players_per_frame', 0):.1f}")
        print(f"  Processing time: {stats.get('processing_time', 0):.2f}s")
        print(f"  Output: {output_path}")
        print("-"*70)

    except Exception as e:
        print(f"\n❌ FootAndBall test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✅ FootAndBall detector integration test PASSED!")


def test_all_backends():
    """Test all three detection backends on the same video."""

    # Find a test video
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
    else:
        input_dir = Path("input")
        if input_dir.exists():
            videos = list(input_dir.glob("*.mp4"))
            if videos:
                input_video = str(videos[0])
            else:
                print("Error: No .mp4 files found in input/ directory")
                return
        else:
            print("Error: input/ directory not found")
            return

    if not os.path.exists(input_video):
        print(f"Error: Video file not found: {input_video}")
        return

    print("="*70)
    print("Testing All Detection Backends")
    print("="*70)
    print(f"Input video: {input_video}")
    print()

    backends = [
        ("footandball", "FootAndBall"),
        ("soccernet", "SoccerNet YOLO"),
        ("yolo", "Generic YOLO")
    ]

    results = {}

    for backend_id, backend_name in backends:
        print(f"\n{'='*70}")
        print(f"Testing {backend_name} ({backend_id})")
        print('='*70)

        config = AppConfig()
        config.detection.detector_backend = backend_id

        if backend_id == "footandball":
            config.detection.footandball_ball_threshold = 0.5
            config.detection.footandball_player_threshold = 0.5

        pipeline = ReframingPipeline(config)
        output_path = f"output/test_{backend_id}.mp4"

        try:
            stats = pipeline.process_goal_clip(
                clip_path=input_video,
                output_path=output_path,
                use_temporal_filter=True,
                use_kalman_smoothing=True
            )

            results[backend_name] = stats
            print(f"\n✅ {backend_name} completed successfully")

        except Exception as e:
            print(f"\n❌ {backend_name} FAILED: {e}")
            results[backend_name] = {"error": str(e)}

    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Backend':<20} {'Frames':<10} {'Ball%':<10} {'Players':<10} {'Time(s)':<10}")
    print("-"*70)

    for backend_name, stats in results.items():
        if "error" in stats:
            print(f"{backend_name:<20} ERROR: {stats['error']}")
        else:
            frames = stats.get('frames_processed', 'N/A')
            ball_rate = f"{stats.get('ball_detection_rate', 0)*100:.1f}%"
            avg_players = f"{stats.get('average_players_per_frame', 0):.1f}"
            proc_time = f"{stats.get('processing_time', 0):.2f}"
            print(f"{backend_name:<20} {frames:<10} {ball_rate:<10} {avg_players:<10} {proc_time:<10}")

    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test FootAndBall detector integration")
    parser.add_argument("--all", action="store_true", help="Test all backends (footandball, soccernet, yolo)")
    parser.add_argument("input_video", nargs="?", help="Input video path")

    args = parser.parse_args()

    # Override sys.argv if input_video was provided
    if args.input_video:
        sys.argv = [sys.argv[0], args.input_video]

    if args.all:
        test_all_backends()
    else:
        test_footandball_detector()
