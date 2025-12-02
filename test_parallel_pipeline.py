"""Test script for parallel clip generation pipeline.

This script demonstrates the performance difference between
sequential and parallel clip processing.

Usage:
    python test_parallel_pipeline.py [video_path]
"""

import sys
from pathlib import Path
import time

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.highlight_pipeline import HighlightPipeline
from src.pipeline.pipeline_config import PipelineConfig


def progress_callback(stage: str, progress: int, message: str):
    """Simple progress callback for testing."""
    print(f"[{progress}%] {stage}: {message}")


def test_parallel_pipeline(video_path: str):
    """Test parallel pipeline with a video file."""
    print("=" * 80)
    print("TESTING PARALLEL CLIP GENERATION PIPELINE")
    print("=" * 80)

    # Test 1: Parallel processing (enabled)
    print("\n" + "=" * 80)
    print("TEST 1: PARALLEL PROCESSING (ENABLED)")
    print("=" * 80)

    config_parallel = PipelineConfig.for_goal_mode()
    config_parallel.enable_parallel_clip_generation = True
    config_parallel.max_parallel_workers = 2

    pipeline_parallel = HighlightPipeline(config_parallel, progress_callback)

    start_time = time.time()
    highlights_parallel = pipeline_parallel.process(video_path)
    parallel_time = time.time() - start_time

    print(f"\n✅ Parallel processing complete!")
    print(f"   Time: {parallel_time:.1f}s")
    print(f"   Highlights: {len(highlights_parallel)}")

    # Test 2: Sequential processing (disabled)
    print("\n" + "=" * 80)
    print("TEST 2: SEQUENTIAL PROCESSING (DISABLED)")
    print("=" * 80)

    config_sequential = PipelineConfig.for_goal_mode()
    config_sequential.enable_parallel_clip_generation = False

    pipeline_sequential = HighlightPipeline(config_sequential, progress_callback)

    start_time = time.time()
    highlights_sequential = pipeline_sequential.process(video_path)
    sequential_time = time.time() - start_time

    print(f"\n✅ Sequential processing complete!")
    print(f"   Time: {sequential_time:.1f}s")
    print(f"   Highlights: {len(highlights_sequential)}")

    # Compare results
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"Parallel time:   {parallel_time:.1f}s")
    print(f"Sequential time: {sequential_time:.1f}s")
    print(f"Speedup:         {sequential_time / parallel_time:.2f}x")
    print(f"Time saved:      {sequential_time - parallel_time:.1f}s")
    print("=" * 80)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default test video
        video_path = "input/test_video.mp4"

    video_file = Path(video_path)

    if not video_file.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        print("\nUsage:")
        print("  python test_parallel_pipeline.py [video_path]")
        print("\nExample:")
        print("  python test_parallel_pipeline.py input/soccer_match.mp4")
        sys.exit(1)

    print(f"📹 Input video: {video_path}")
    test_parallel_pipeline(str(video_file))


if __name__ == "__main__":
    main()
