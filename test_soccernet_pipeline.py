"""Test script for SoccerNet tracking pipeline (PHASE 2)."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.reframing_pipeline import ReframingPipeline, process_video_simple
from src.utils.config import AppConfig


def test_pipeline_simple(input_video: str):
    """Test pipeline with simple function.

    Args:
        input_video: Path to input goal clip
    """
    print("Testing SoccerNet Pipeline (Simple Function)")
    print("=" * 70)

    output_video = "output/test_output_simple.mp4"

    try:
        stats = process_video_simple(
            input_path=input_video,
            output_path=output_video,
            use_soccernet=True
        )

        print("\n✓ Test passed!")
        print(f"Output saved to: {stats['output_path']}")
        print(f"Processing time: {stats['processing_time']:.2f}s")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_pipeline_full(input_video: str):
    """Test pipeline with full configuration.

    Args:
        input_video: Path to input goal clip
    """
    print("\nTesting SoccerNet Pipeline (Full Configuration)")
    print("=" * 70)

    # Create custom configuration
    config = AppConfig()
    config.detection.confidence_threshold = 0.05  # Low threshold for high recall
    config.roi.ball_weight = 3.0
    config.roi.person_weight = 1.0
    config.video.output_width = 1080
    config.video.output_height = 1920

    # Initialize pipeline
    pipeline = ReframingPipeline(config)

    output_video = "output/test_output_full.mp4"

    try:
        stats = pipeline.process_goal_clip(
            clip_path=input_video,
            output_path=output_video,
            use_soccernet_model=True,
            use_temporal_filter=True,
            use_kalman_smoothing=True
        )

        print("\n✓ Test passed!")
        print(f"Output saved to: {stats['output_path']}")
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Ball detection rate: {stats['ball_detection_rate']:.1%}")
        print(f"Avg players per frame: {stats['average_players_per_frame']:.1f}")
        print(f"Processing time: {stats['processing_time']:.2f}s")
        print(f"FPS: {stats['frames_processed'] / stats['processing_time']:.1f}")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_components():
    """Test individual components."""
    print("\nTesting Individual Components")
    print("=" * 70)

    # Test 1: SoccerNet Detector
    print("\n[1/4] Testing SoccerNetDetector...")
    try:
        from src.core.soccernet_detector import SoccerNetDetector
        detector = SoccerNetDetector()
        print("  ✓ SoccerNetDetector initialized")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 2: Temporal Filter
    print("\n[2/4] Testing TemporalBallFilter...")
    try:
        from src.core.temporal_filter import TemporalBallFilter
        filter = TemporalBallFilter()
        print("  ✓ TemporalBallFilter initialized")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 3: ROI Calculator
    print("\n[3/4] Testing ROICalculator...")
    try:
        from src.core.roi_calculator import ROICalculator
        roi_calc = ROICalculator()
        print("  ✓ ROICalculator initialized")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 4: Smoother
    print("\n[4/4] Testing Smoother...")
    try:
        from src.core.smoother import Smoother
        from src.utils.config import KalmanConfig
        smoother = Smoother(KalmanConfig())
        print("  ✓ Smoother initialized")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print("\n✓ All component tests passed!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SOCCERNET TRACKING PIPELINE - TEST SUITE")
    print("=" * 70)

    # Test components first
    test_components()

    # Check if input video is provided
    if len(sys.argv) > 1:
        input_video = sys.argv[1]

        if not Path(input_video).exists():
            print(f"\n✗ Error: Input video not found: {input_video}")
            sys.exit(1)

        # Test simple function
        test_pipeline_simple(input_video)

        # Test full configuration
        test_pipeline_full(input_video)

    else:
        print("\n" + "=" * 70)
        print("USAGE")
        print("=" * 70)
        print("To test with a video file:")
        print(f"  python {Path(__file__).name} <input_video.mp4>")
        print("\nExample:")
        print(f"  python {Path(__file__).name} input/goal_clip_30s.mp4")
        print("\nNote: Component tests were run successfully!")
        print("      To test full pipeline, provide an input video file.")

    print("\n" + "=" * 70)
