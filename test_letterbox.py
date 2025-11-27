"""Test script for letterbox functionality."""

import sys
from pathlib import Path

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.reframing_pipeline import process_video_simple

def test_letterbox():
    """Test the letterbox feature with default settings."""

    print("="*60)
    print("LETTERBOX FEATURE TEST")
    print("="*60)
    print()
    print("Testing with default letterbox settings:")
    print("  - Top letterbox: 288px (15% of 1920)")
    print("  - Bottom letterbox: 288px (15% of 1920)")
    print("  - Content area: 1344px (70% of 1920)")
    print()

    input_path = "input/30s_vid.mp4"
    output_path = "output/test_letterbox_default.mp4"

    # Test with default letterbox (288px top and bottom)
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    print()

    stats = process_video_simple(
        input_path=input_path,
        output_path=output_path,
        output_width=1080,
        output_height=1920,
        use_soccernet=True,
        letterbox_top=288,  # Default
        letterbox_bottom=288  # Default
    )

    print()
    print("="*60)
    print("TEST COMPLETED!")
    print("="*60)
    print(f"Output video: {output_path}")
    print(f"Frames processed: {stats.get('frames_processed', 'N/A')}")
    print(f"Processing time: {stats.get('processing_time', 0):.2f}s")
    print()
    print("Expected result:")
    print("  ✓ Video should have 288px black bars on top and bottom")
    print("  ✓ Content should be centered in the middle 1344px height")
    print("  ✓ Width should always be full 1080px (no horizontal bars)")
    print()

def test_custom_letterbox():
    """Test with custom letterbox settings."""

    print("="*60)
    print("CUSTOM LETTERBOX TEST")
    print("="*60)
    print()
    print("Testing with larger letterbox settings:")
    print("  - Top letterbox: 400px")
    print("  - Bottom letterbox: 400px")
    print("  - Content area: 1120px")
    print()

    input_path = "input/30s_vid.mp4"
    output_path = "output/test_letterbox_large.mp4"

    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    print()

    stats = process_video_simple(
        input_path=input_path,
        output_path=output_path,
        output_width=1080,
        output_height=1920,
        use_soccernet=True,
        letterbox_top=400,
        letterbox_bottom=400
    )

    print()
    print("="*60)
    print("CUSTOM TEST COMPLETED!")
    print("="*60)
    print(f"Output video: {output_path}")
    print(f"Frames processed: {stats.get('frames_processed', 'N/A')}")
    print(f"Processing time: {stats.get('processing_time', 0):.2f}s")
    print()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test letterbox functionality")
    parser.add_argument("--custom", action="store_true", help="Test with custom (larger) letterbox")

    args = parser.parse_args()

    if args.custom:
        test_custom_letterbox()
    else:
        test_letterbox()
