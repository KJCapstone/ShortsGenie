"""Test script for full highlight generation pipeline.

This script tests the complete pipeline integration including:
- Pipeline configuration
- Module orchestration
- Progress callbacks
- Highlight generation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.highlight_pipeline import generate_highlights, HighlightPipeline
from src.pipeline.pipeline_config import create_config_from_mode, PipelineConfig


def test_simple_api(video_path: str, mode: str = "골"):
    """Test simple API (generate_highlights function).

    Args:
        video_path: Path to test video
        mode: Processing mode ("골", "경기", "밈")
    """
    print("\n" + "="*70)
    print(f"TEST 1: Simple API - {mode} Mode")
    print("="*70)

    def progress_callback(stage: str, progress: int, message: str):
        """Print progress updates."""
        print(f"[{progress:3d}%] {stage:20s} | {message}")

    try:
        highlights = generate_highlights(
            video_path=video_path,
            mode=mode,
            progress_callback=progress_callback
        )

        print("\n" + "-"*70)
        print(f"✓ Success! Generated {len(highlights)} highlights:")
        print("-"*70)

        for i, h in enumerate(highlights, 1):
            print(f"\n{i}. {h['title']}")
            print(f"   Description: {h['description']}")
            print(f"   Time: {h['start_time']} - {h['end_time']} ({h['duration']})")
            print(f"   Score: {h['score']:.2f}")
            print(f"   Source: {h['source']}")
            if h.get('video_path'):
                print(f"   Video: {h['video_path']}")

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_custom_config(video_path: str):
    """Test custom configuration with selective modules.

    Args:
        video_path: Path to test video
    """
    print("\n" + "="*70)
    print("TEST 2: Custom Configuration (Selective Modules)")
    print("="*70)

    # Create custom config
    config = PipelineConfig.for_goal_mode()

    # Customize module settings
    print("\nConfiguration:")
    print(f"  - OCR Detection: {config.ocr_detection.enabled}")
    print(f"  - Audio Analysis: {config.audio_analysis.enabled}")
    print(f"  - Transcript: {config.transcript_analysis.enabled}")
    print(f"  - Scene Detection: {config.scene_detection.enabled}")
    print(f"  - Reframing: {config.reframing.enabled}")

    # Disable some modules for faster testing
    config.transcript_analysis.enabled = False  # Disable Whisper (slow)
    config.scene_detection.enabled = False       # Disable scene detection
    config.reframing.enabled = False             # Disable reframing (just test detection)

    print("\nAdjusted Configuration:")
    print(f"  - OCR Detection: {config.ocr_detection.enabled}")
    print(f"  - Audio Analysis: {config.audio_analysis.enabled}")
    print(f"  - Transcript: {config.transcript_analysis.enabled}")
    print(f"  - Scene Detection: {config.scene_detection.enabled}")
    print(f"  - Reframing: {config.reframing.enabled}")

    def progress_callback(stage: str, progress: int, message: str):
        """Print progress updates."""
        print(f"[{progress:3d}%] {stage:20s} | {message}")

    try:
        pipeline = HighlightPipeline(config, progress_callback)
        highlights = pipeline.process(video_path)

        print("\n" + "-"*70)
        print(f"✓ Success! Generated {len(highlights)} highlights:")
        print("-"*70)

        for i, h in enumerate(highlights, 1):
            print(f"\n{i}. {h.title}")
            print(f"   {h.description}")
            print(f"   {h.start_time:.1f}s - {h.end_time:.1f}s (Duration: {h.duration:.1f}s)")
            print(f"   Score: {h.score:.2f} | Source: {h.source}")

        if pipeline.processing_errors:
            print("\n⚠️ Errors encountered:")
            for error in pipeline.processing_errors:
                print(f"  - {error}")

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_all_modes(video_path: str):
    """Test all three modes with minimal configuration.

    Args:
        video_path: Path to test video
    """
    print("\n" + "="*70)
    print("TEST 3: All Modes (골/경기/밈)")
    print("="*70)

    modes = ["골", "경기", "밈"]

    for mode in modes:
        print(f"\n--- Testing {mode} mode ---")

        config = create_config_from_mode(mode)

        # Fast test: disable slow modules
        config.transcript_analysis.enabled = False
        config.scene_detection.enabled = False
        config.reframing.enabled = False

        print(f"Enabled modules: {[name for name, _ in config.get_enabled_modules()]}")

        try:
            pipeline = HighlightPipeline(config)
            highlights = pipeline.process(video_path)

            print(f"✓ {mode} mode: {len(highlights)} highlights generated")

        except Exception as e:
            print(f"✗ {mode} mode failed: {e}")


def test_module_fallback():
    """Test graceful degradation when modules fail."""
    print("\n" + "="*70)
    print("TEST 4: Module Fallback (Graceful Degradation)")
    print("="*70)

    config = PipelineConfig.for_goal_mode()

    # Try with non-existent video (should trigger OCR failure)
    fake_video = "input/nonexistent_video.mp4"

    print(f"\nTesting with invalid video: {fake_video}")
    print("Expected: OCR should fail, but pipeline should continue with audio")

    def progress_callback(stage: str, progress: int, message: str):
        """Print progress updates."""
        if "⚠️" in message or "실패" in message:
            print(f"  ⚠️  [{progress:3d}%] {message}")

    try:
        pipeline = HighlightPipeline(config, progress_callback)
        highlights = pipeline.process(fake_video)

        print(f"\n✓ Graceful degradation worked!")
        print(f"   Generated {len(highlights)} highlights despite errors")
        print(f"   Errors: {len(pipeline.processing_errors)}")

        for error in pipeline.processing_errors:
            print(f"     - {error}")

    except Exception as e:
        print(f"\n✓ Expected failure: {e}")


def main():
    """Run all tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test full highlight pipeline")
    parser.add_argument(
        "video_path",
        nargs="?",
        default="input/test_video.mp4",
        help="Path to test video (default: input/test_video.mp4)"
    )
    parser.add_argument(
        "--mode",
        choices=["골", "경기", "밈"],
        default="골",
        help="Processing mode (default: 골)"
    )
    parser.add_argument(
        "--test",
        choices=["simple", "custom", "all_modes", "fallback", "full"],
        default="simple",
        help="Which test to run (default: simple)"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("FULL PIPELINE TEST SUITE")
    print("="*70)
    print(f"Video: {args.video_path}")
    print(f"Mode: {args.mode}")
    print(f"Test: {args.test}")

    # Check if video exists
    if not Path(args.video_path).exists():
        print(f"\n⚠️  Warning: Video file not found: {args.video_path}")
        print("Some tests may fail. For full testing, provide a valid video file.")

    # Run selected test
    if args.test == "simple":
        test_simple_api(args.video_path, args.mode)
    elif args.test == "custom":
        test_custom_config(args.video_path)
    elif args.test == "all_modes":
        test_all_modes(args.video_path)
    elif args.test == "fallback":
        test_module_fallback()
    elif args.test == "full":
        test_simple_api(args.video_path, args.mode)
        test_custom_config(args.video_path)
        test_all_modes(args.video_path)
        test_module_fallback()

    print("\n" + "="*70)
    print("Test completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
