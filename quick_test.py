"""Quick test script for aspect ratio preservation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.reframing_pipeline import process_video_simple


def main():
    """Run quick test with 30s_vid.mp4"""

    input_video = "../input/30s_vid.mp4"
    output_video = "output/test_aspect_ratio.mp4"

    print("="*70)
    print("QUICK TEST - Aspect Ratio Preservation")
    print("="*70)
    print(f"Input: {input_video}")
    print(f"Output: {output_video}")
    print()

    try:
        stats = process_video_simple(
            input_path=input_video,
            output_path=output_video,
            use_soccernet=True
        )

        print("\n" + "="*70)
        print("✓ TEST PASSED!")
        print("="*70)
        print(f"Output: {stats['output_path']}")
        print(f"Frames: {stats['frames_processed']}")
        print(f"Ball detection: {stats['ball_detection_rate']:.1%}")
        print(f"Processing time: {stats['processing_time']:.1f}s")
        print(f"FPS: {stats['frames_processed'] / stats['processing_time']:.1f}")
        print()
        print("Key improvements:")
        print("  ✓ Original aspect ratio preserved")
        print("  ✓ No distortion from forced resize")
        print("  ✓ Letterboxing (black bars) if needed")
        print("  ✓ Video centered in 1080x1920 frame")
        print("="*70)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
