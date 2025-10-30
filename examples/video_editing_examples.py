"""
Examples demonstrating various video editing scenarios using the VideoEditor.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.video_editor import (
    VideoEditor,
    VideoEditConfig,
    EditSegment,
    CropRegion,
    PanKeyframe,
    edit_video,
)

VIDEO_PATH = "input/korea_vs_brazil.mp4"


def example1_simple_cut():
    """Example 1: Cut a segment from 10s to 30s"""
    print("Example 1: Simple time-based cut")

    config = VideoEditConfig(
        input_path=VIDEO_PATH,
        output_path="output/example1_cut.mp4",
        segments=[EditSegment(start_time=10.0, end_time=30.0)],
    )

    edit_video(config)
    print("✅ Done: output/example1_cut.mp4")


def example2_cut_and_crop():
    """Example 2: Cut segment and crop to vertical format (9:16)"""
    print("\nExample 2: Cut and crop to vertical")

    # For a 1920x1080 source, we need to crop to 9:16 aspect ratio
    # 9:16 aspect ratio with max height of 1080 = 607.5 x 1080
    # Round to 608 x 1080 to keep even dimensions
    crop_width = 608
    crop_height = 1080
    crop_x = (1920 - crop_width) // 2  # Center horizontally

    config = VideoEditConfig(
        input_path=VIDEO_PATH,
        output_path="output/example2_crop.mp4",
        segments=[
            EditSegment(
                start_time=5.0,
                end_time=25.0,
                crop_region=CropRegion(
                    x=crop_x,  # Center crop
                    y=0,  # Start from top
                    width=crop_width,  # 608px wide for 9:16 ratio
                    height=crop_height,  # Full height (1080)
                ),
            )
        ],
        output_width=1080,  # Scale up to standard vertical resolution
        output_height=1920,
    )

    edit_video(config)
    print("✅ Done: output/example2_crop.mp4")


def example3_dynamic_pan():
    """Example 3: Dynamic panning crop that follows action"""
    print("\nExample 3: Dynamic panning crop")

    # Calculate crop dimensions for 9:16 aspect ratio
    crop_width = 608
    crop_height = 1080
    max_x = 1920 - crop_width  # Maximum x position for crop

    config = VideoEditConfig(
        input_path=VIDEO_PATH,
        output_path="output/example3_pan.mp4",
        segments=[
            EditSegment(
                start_time=0.0,
                end_time=20.0,
                crop_width=crop_width,  # Specify crop dimensions
                crop_height=crop_height,
                pan_keyframes=[
                    PanKeyframe(time=0.0, x=0, y=0),  # Start at left
                    PanKeyframe(time=5.0, x=max_x // 2, y=0),  # Move to center
                    PanKeyframe(time=10.0, x=max_x, y=0),  # Move to right
                    PanKeyframe(time=15.0, x=max_x // 2, y=0),  # Back to center
                    PanKeyframe(time=20.0, x=0, y=0),  # Back to left
                ],
            )
        ],
        output_width=1080,
        output_height=1920,
    )

    edit_video(config)
    print("✅ Done: output/example3_pan.mp4")


def example4_multiple_segments():
    """Example 4: Extract and combine multiple highlights"""
    print("\nExample 4: Multiple segments (highlight compilation)")

    # Calculate crop dimensions for 9:16 aspect ratio
    crop_width = 608
    crop_height = 1080
    crop_x = (1920 - crop_width) // 2  # Center horizontally

    config = VideoEditConfig(
        input_path=VIDEO_PATH,
        output_path="output/example4_highlights.mp4",
        segments=[
            # First highlight: 5s-10s
            EditSegment(
                start_time=5.0,
                end_time=10.0,
                crop_region=CropRegion(x=crop_x, y=0, width=crop_width, height=crop_height),
            ),
            # Second highlight: 25s-35s
            EditSegment(
                start_time=25.0,
                end_time=35.0,
                crop_region=CropRegion(x=crop_x, y=0, width=crop_width, height=crop_height),
            ),
            # Third highlight: 50s-60s
            EditSegment(
                start_time=50.0,
                end_time=60.0,
                crop_region=CropRegion(x=crop_x, y=0, width=crop_width, height=crop_height),
            ),
        ],
        output_width=1080,
        output_height=1920,
    )

    edit_video(config)
    print("✅ Done: output/example4_highlights.mp4")


def example5_advanced_tracking():
    """Example 5: Simulate ball tracking with smooth panning"""
    print("\nExample 5: Advanced tracking simulation")

    # Calculate crop dimensions for 9:16 aspect ratio
    crop_width = 608
    crop_height = 1080

    # Simulate tracking data (in real scenario, this comes from detector)
    tracking_data = [
        (0.0, 400, 500),   # time, x_center, y_center
        (2.0, 600, 600),
        (4.0, 900, 700),
        (6.0, 1200, 800),
        (8.0, 1400, 600),
        (10.0, 1000, 400),
    ]

    # Convert tracking data to keyframes (adjust for crop window)
    keyframes = []

    for time, x_center, y_center in tracking_data:
        # Calculate top-left corner of crop window to center on tracked point
        crop_x = max(0, min(x_center - crop_width // 2, 1920 - crop_width))
        crop_y = max(0, min(y_center - crop_height // 2, 1080 - crop_height))
        keyframes.append(PanKeyframe(time=time, x=int(crop_x), y=int(crop_y)))

    config = VideoEditConfig(
        input_path=VIDEO_PATH,
        output_path="output/example5_tracking.mp4",
        segments=[
            EditSegment(
                start_time=0.0,
                end_time=10.0,
                crop_width=crop_width,
                crop_height=crop_height,
                pan_keyframes=keyframes
            )
        ],
        output_width=1080,
        output_height=1920,
    )

    edit_video(config)
    print("✅ Done: output/example5_tracking.mp4")


def example6_manual_control():
    """Example 6: Manual control using VideoEditor instance"""
    print("\nExample 6: Manual control with VideoEditor")

    # Calculate crop dimensions for 9:16 aspect ratio
    crop_width = 608
    crop_height = 1080
    crop_x = (1920 - crop_width) // 2  # Center horizontally

    editor = VideoEditor()

    # Just cut a segment (no re-encoding, fast)
    editor.cut_segment(
        input_path=VIDEO_PATH,
        output_path="output/example6_segment.mp4",
        start_time=10.0,
        end_time=20.0,
        copy_streams=True,  # Fast mode (may be less precise)
    )

    # Apply crop and scale separately
    editor.crop_and_scale(
        input_path="output/example6_segment.mp4",
        output_path="output/example6_final.mp4",
        crop=CropRegion(x=crop_x, y=0, width=crop_width, height=crop_height),
        output_width=1080,
        output_height=1920,
    )

    print("✅ Done: output/example6_final.mp4")


def main():
    """Run all examples (comment out ones you don't want to run)"""
    # Create output directory
    Path("output").mkdir(exist_ok=True)

    print("=" * 60)
    print("Video Editing Examples")
    print("=" * 60)

    # Uncomment the examples you want to run:
    example1_simple_cut()
    example2_cut_and_crop()
    example3_dynamic_pan()
    example4_multiple_segments()
    example5_advanced_tracking()
    example6_manual_control()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Make sure to replace VIDEO_PATH with your actual video file
    # before running these examples!

    print(
        """
    ⚠️  Before running examples:
    1. Replace 'input.mp4' with your actual video path
    2. Make sure FFmpeg is installed
    3. Create 'output/' directory if it doesn't exist
    4. Uncomment the example you want to run in main()
    """
    )

    # Uncomment to run:
    main()
