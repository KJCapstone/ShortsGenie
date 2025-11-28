"""Test script for scene-aware reframing pipeline.

This script demonstrates how to use the scene-aware features of the
reframing pipeline with auto_tagger scene metadata.
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.reframing_pipeline import ReframingPipeline
from src.utils.config import AppConfig


def test_baseline():
    """Test baseline pipeline without scene awareness."""
    print("\n" + "="*70)
    print("TEST 1: Baseline (No Scene Awareness)")
    print("="*70)

    config = AppConfig()
    config.scene.enabled = False

    pipeline = ReframingPipeline(config)

    stats = pipeline.process_goal_clip(
        clip_path="input/test_clip.mp4",
        output_path="output/baseline.mp4",
        use_soccernet_model=True,
        use_temporal_filter=True,
        use_kalman_smoothing=True
    )

    print(f"\n✓ Baseline processing complete:")
    print(f"  - Processing time: {stats['processing_time']:.1f}s")
    print(f"  - Ball detection rate: {stats['ball_detection_rate']:.1%}")
    print(f"  - Avg players/frame: {stats['average_players_per_frame']:.1f}")

    return stats


def test_scene_aware_with_json():
    """Test scene-aware pipeline with pre-generated JSON."""
    print("\n" + "="*70)
    print("TEST 2: Scene-Aware (With auto_tagger JSON)")
    print("="*70)

    config = AppConfig()
    config.scene.enabled = True  # Enable scene awareness

    pipeline = ReframingPipeline(config)

    # Path to pre-generated auto_tagger JSON
    # Generate this first using: cd auto_tagger && python test.py
    scene_json_path = "auto_tagger/full_match_log.json"

    if not Path(scene_json_path).exists():
        print(f"\n⚠️  Scene JSON not found: {scene_json_path}")
        print("   To generate, run:")
        print("     cd auto_tagger")
        print("     python test.py")
        return None

    stats = pipeline.process_goal_clip(
        clip_path="input/test_clip.mp4",
        output_path="output/scene_aware.mp4",
        use_soccernet_model=True,
        use_temporal_filter=True,
        use_kalman_smoothing=True,
        scene_metadata_path=scene_json_path  # NEW: Scene metadata
    )

    print(f"\n✓ Scene-aware processing complete:")
    print(f"  - Processing time: {stats['processing_time']:.1f}s")
    print(f"  - Ball detection rate: {stats['ball_detection_rate']:.1%}")
    print(f"  - Avg players/frame: {stats['average_players_per_frame']:.1f}")

    return stats


def test_scene_aware_with_weights():
    """Test scene-aware pipeline with dynamic weight adjustment."""
    print("\n" + "="*70)
    print("TEST 3: Scene-Aware + Dynamic Weights")
    print("="*70)

    config = AppConfig()
    config.scene.enabled = True
    config.roi.use_scene_type_weights = True  # Enable dynamic weight adjustment

    # Optionally customize scene type weights
    config.roi.scene_type_weights['replay']['ball_weight'] = 6.0  # Extra focus on ball in replays

    pipeline = ReframingPipeline(config)

    scene_json_path = "auto_tagger/full_match_log.json"

    if not Path(scene_json_path).exists():
        print(f"\n⚠️  Scene JSON not found: {scene_json_path}")
        return None

    stats = pipeline.process_goal_clip(
        clip_path="input/test_clip.mp4",
        output_path="output/scene_aware_weighted.mp4",
        use_soccernet_model=True,
        use_temporal_filter=True,
        use_kalman_smoothing=True,
        scene_metadata_path=scene_json_path
    )

    print(f"\n✓ Scene-aware (weighted) processing complete:")
    print(f"  - Processing time: {stats['processing_time']:.1f}s")
    print(f"  - Ball detection rate: {stats['ball_detection_rate']:.1%}")
    print(f"  - Avg players/frame: {stats['average_players_per_frame']:.1f}")

    return stats


def compare_results(baseline_stats, scene_aware_stats):
    """Compare baseline vs scene-aware results."""
    if not baseline_stats or not scene_aware_stats:
        print("\n⚠️  Cannot compare - one or both tests failed")
        return

    print("\n" + "="*70)
    print("COMPARISON: Baseline vs Scene-Aware")
    print("="*70)

    time_diff = scene_aware_stats['processing_time'] - baseline_stats['processing_time']
    overhead_pct = (time_diff / baseline_stats['processing_time']) * 100

    print(f"\nProcessing Time:")
    print(f"  Baseline:     {baseline_stats['processing_time']:.1f}s")
    print(f"  Scene-Aware:  {scene_aware_stats['processing_time']:.1f}s")
    print(f"  Difference:   {time_diff:+.1f}s ({overhead_pct:+.1f}%)")

    print(f"\nBall Detection Rate:")
    print(f"  Baseline:     {baseline_stats['ball_detection_rate']:.1%}")
    print(f"  Scene-Aware:  {scene_aware_stats['ball_detection_rate']:.1%}")

    print(f"\nNote: Scene awareness adds scene boundary detection and ROI resets.")
    print(f"      Processing overhead should be minimal (<5%).")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SCENE-AWARE REFRAMING PIPELINE - TEST SUITE")
    print("="*70)

    # Check if test input exists
    if not Path("input/test_clip.mp4").exists():
        print("\n⚠️  Test video not found: input/test_clip.mp4")
        print("   Please place a test video in the input directory.")
        return

    # Create output directory
    Path("output").mkdir(exist_ok=True)

    try:
        # Run tests
        baseline_stats = test_baseline()
        scene_aware_stats = test_scene_aware_with_json()

        # Optional: Test with dynamic weights
        # weighted_stats = test_scene_aware_with_weights()

        # Compare results
        if baseline_stats and scene_aware_stats:
            compare_results(baseline_stats, scene_aware_stats)

        print("\n" + "="*70)
        print("✓ ALL TESTS COMPLETE")
        print("="*70)
        print("\nOutput videos saved to output/ directory:")
        print("  - baseline.mp4          (no scene awareness)")
        print("  - scene_aware.mp4       (with scene awareness)")
        # print("  - scene_aware_weighted.mp4  (with dynamic weights)")

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
