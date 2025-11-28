"""Test scene classifier on video clips."""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from src.scene.scene_classifier import SceneClassifier
from src.scene.scene_metadata import validate_scene_metadata


# Temporary config class for testing
class SceneClassificationConfig:
    """Temporary config for testing before pipeline_config is updated."""
    def __init__(self):
        self.model_path = "resources/models/scene_classifier/soccer_model_ver2.pth"
        self.device = "auto"
        self.threshold = 27.0
        self.min_scene_len = 15
        self.min_scene_duration = 0.5
        self.save_json = True
        self.json_output_dir = "output/scene_metadata"


def test_scene_classifier():
    """Test scene classifier on a sample clip."""

    # Configuration
    config = SceneClassificationConfig()

    # Test video
    test_video = "input/30s_vid.mp4"  # Use existing test video

    if not Path(test_video).exists():
        print(f"❌ Test video not found: {test_video}")
        print("\nPlease ensure you have a test video at:")
        print(f"  {Path(test_video).absolute()}")
        print("\nYou can use any soccer video clip for testing.")
        return

    print("=" * 60)
    print("SceneClassifier Test")
    print("=" * 60)

    # Initialize classifier
    print("\n🔧 Initializing SceneClassifier...")
    try:
        classifier = SceneClassifier(config)
        print(f"✓ Initialized successfully")
        print(f"  Device: {classifier.device}")
        print(f"  Model: {config.model_path}")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Run classification
    print(f"\n🎬 Classifying: {test_video}")
    output_json = "output/test_scenes.json"

    try:
        scene_metadata = classifier.classify_clip(
            test_video,
            output_json_path=output_json
        )
    except Exception as e:
        print(f"\n❌ Classification failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Validate metadata
    print("\n🔍 Validating scene metadata...")
    warnings = validate_scene_metadata(scene_metadata)
    if warnings:
        print("⚠️  Validation warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("✓ Scene metadata validated successfully")

    # Print results
    print("\n📊 Results:")
    print(f"  Total scenes: {len(scene_metadata.segments)}")
    print(f"  JSON saved to: {output_json}")
    print(f"  Video FPS: {scene_metadata.fps:.2f}")

    # Scene breakdown by type
    print("\n📈 Scene breakdown:")
    scene_types = {}
    for seg in scene_metadata.segments:
        scene_types[seg.label] = scene_types.get(seg.label, 0) + 1

    for label, count in sorted(scene_types.items()):
        print(f"  {label:8s}: {count} scenes")

    # Scene timeline
    print("\n🎞️  Scene timeline:")
    for i, seg in enumerate(scene_metadata.segments):
        print(f"  {i+1:2d}. [{seg.label.upper():8s}] "
              f"{seg.start_seconds:6.1f}s - {seg.end_seconds:6.1f}s "
              f"({seg.duration:5.1f}s)")

    print("\n" + "=" * 60)
    print("✨ Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_scene_classifier()
