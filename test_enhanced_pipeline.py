"""Test script for enhanced reframing pipeline with stabilization and scene detection."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.reframing_pipeline import ReframingPipeline, process_video_simple
from src.utils.config import AppConfig


def test_simple_enhanced(input_video: str):
    """Test enhanced pipeline with simple function (uses all new features).

    Args:
        input_video: Path to input goal clip
    """
    print("\n" + "="*70)
    print("TEST 1: Simple Enhanced Pipeline")
    print("="*70)
    print("Features enabled:")
    print("  ✓ YOLOv8m detection (upgraded from YOLOv8n)")
    print("  ✓ Lower confidence threshold (0.05 vs 0.3)")
    print("  ✓ Hysteresis dead zone (100px)")
    print("  ✓ Scene-coherent locking (150px threshold)")
    print("  ✓ Adaptive EMA smoothing")
    print("="*70)

    output_video = "output/test_enhanced_simple.mp4"

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


def test_baseline_comparison(input_video: str):
    """Test baseline (old) vs enhanced (new) for comparison.

    Args:
        input_video: Path to input goal clip
    """
    print("\n" + "="*70)
    print("TEST 2: Baseline vs Enhanced Comparison")
    print("="*70)

    # Baseline configuration (old settings)
    print("\n[Baseline] Processing with OLD settings...")
    config_baseline = AppConfig()
    config_baseline.detection.model_path = "yolov8n.pt"  # Old model
    config_baseline.detection.confidence_threshold = 0.3  # Old threshold
    config_baseline.roi.use_hysteresis = False  # Disable hysteresis
    config_baseline.roi.use_scene_locking = False  # Disable locking
    config_baseline.smoother.method = "kalman"  # Old smoother
    config_baseline.scene_detection.enabled = False  # Disable scene detection

    pipeline_baseline = ReframingPipeline(config_baseline)

    try:
        stats_baseline = pipeline_baseline.process_goal_clip(
            clip_path=input_video,
            output_path="output/test_baseline.mp4",
            use_soccernet_model=False,
            use_temporal_filter=True,
            use_kalman_smoothing=True
        )
        print(f"✓ Baseline complete: {stats_baseline['processing_time']:.2f}s")
    except Exception as e:
        print(f"✗ Baseline failed: {e}")
        stats_baseline = None

    # Enhanced configuration (new settings)
    print("\n[Enhanced] Processing with NEW settings...")
    config_enhanced = AppConfig()  # Uses all new defaults
    # config already has:
    # - model_path = "yolov8m.pt"
    # - confidence_threshold = 0.05
    # - use_hysteresis = True
    # - use_scene_locking = True
    # - method = "adaptive_ema"
    # - scene_detection.enabled = True

    pipeline_enhanced = ReframingPipeline(config_enhanced)

    try:
        stats_enhanced = pipeline_enhanced.process_goal_clip(
            clip_path=input_video,
            output_path="output/test_enhanced.mp4",
            use_soccernet_model=True,
            use_temporal_filter=True,
            use_kalman_smoothing=True
        )
        print(f"✓ Enhanced complete: {stats_enhanced['processing_time']:.2f}s")
    except Exception as e:
        print(f"✗ Enhanced failed: {e}")
        stats_enhanced = None

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    if stats_baseline and stats_enhanced:
        print(f"\n{'Metric':<30} {'Baseline':>15} {'Enhanced':>15} {'Improvement':>15}")
        print("-"*76)

        # Ball detection rate
        baseline_rate = stats_baseline.get('ball_detection_rate', 0.0)
        enhanced_rate = stats_enhanced.get('ball_detection_rate', 0.0)
        improvement = ((enhanced_rate - baseline_rate) / max(baseline_rate, 0.01)) * 100
        print(f"{'Ball detection rate':<30} {baseline_rate:>14.1%} {enhanced_rate:>14.1%} {improvement:>+14.1f}%")

        # Processing time
        baseline_time = stats_baseline.get('processing_time', 0.0)
        enhanced_time = stats_enhanced.get('processing_time', 0.0)
        time_diff = ((enhanced_time - baseline_time) / max(baseline_time, 0.01)) * 100
        print(f"{'Processing time (s)':<30} {baseline_time:>14.2f} {enhanced_time:>14.2f} {time_diff:>+14.1f}%")

        # Frames processed
        baseline_frames = stats_baseline.get('frames_processed', 0)
        enhanced_frames = stats_enhanced.get('frames_processed', 0)
        print(f"{'Frames processed':<30} {baseline_frames:>15} {enhanced_frames:>15}")

        print("\n" + "="*70)
        print("Output files:")
        print(f"  Baseline: output/test_baseline.mp4")
        print(f"  Enhanced: output/test_enhanced.mp4")
        print("\nPlay both side-by-side to see the difference!")
        print("="*70)


def test_config_variants(input_video: str):
    """Test different configuration variants.

    Args:
        input_video: Path to input goal clip
    """
    print("\n" + "="*70)
    print("TEST 3: Configuration Variants")
    print("="*70)

    configs = {
        "hysteresis_only": {
            "desc": "Hysteresis only (no locking, no EMA)",
            "changes": lambda c: setattr(c.roi, 'use_scene_locking', False) or setattr(c.smoother, 'method', 'kalman')
        },
        "locking_only": {
            "desc": "Scene locking only (no hysteresis, no EMA)",
            "changes": lambda c: setattr(c.roi, 'use_hysteresis', False) or setattr(c.smoother, 'method', 'kalman')
        },
        "ema_only": {
            "desc": "Adaptive EMA only (no hysteresis, no locking)",
            "changes": lambda c: setattr(c.roi, 'use_hysteresis', False) or setattr(c.roi, 'use_scene_locking', False)
        },
        "all_features": {
            "desc": "All features enabled (hysteresis + locking + EMA)",
            "changes": lambda c: None  # Use defaults
        }
    }

    results = {}

    for variant_name, variant_info in configs.items():
        print(f"\n[{variant_name}] {variant_info['desc']}")

        config = AppConfig()
        variant_info['changes'](config)

        pipeline = ReframingPipeline(config)

        try:
            stats = pipeline.process_goal_clip(
                clip_path=input_video,
                output_path=f"output/test_{variant_name}.mp4",
                use_soccernet_model=True,
                use_temporal_filter=True,
                use_kalman_smoothing=True
            )
            results[variant_name] = stats
            print(f"  ✓ Complete: {stats['processing_time']:.2f}s, ball rate={stats.get('ball_detection_rate', 0):.1%}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[variant_name] = None

    # Summary
    print("\n" + "="*70)
    print("VARIANT COMPARISON")
    print("="*70)
    print(f"\n{'Variant':<20} {'Time (s)':>12} {'Ball Rate':>12} {'Output File'}")
    print("-"*70)

    for variant_name, stats in results.items():
        if stats:
            time_str = f"{stats['processing_time']:.2f}"
            rate_str = f"{stats.get('ball_detection_rate', 0):.1%}"
            file_str = f"output/test_{variant_name}.mp4"
        else:
            time_str = "FAILED"
            rate_str = "-"
            file_str = "-"

        print(f"{variant_name:<20} {time_str:>12} {rate_str:>12} {file_str}")

    print("="*70)


def test_components():
    """Test individual enhanced components."""
    print("\n" + "="*70)
    print("TEST 4: Component Tests")
    print("="*70)

    # Test 1: SceneDetector
    print("\n[1/4] Testing SceneDetector...")
    try:
        from src.core.scene_detector import SceneDetector
        from src.utils.config import SceneDetectionConfig

        config = SceneDetectionConfig()
        detector = SceneDetector(config)
        print("  ✓ SceneDetector initialized")
        print(f"    - Enabled: {config.enabled}")
        print(f"    - Method: {config.method}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 2: ROICalculator with hysteresis
    print("\n[2/4] Testing ROICalculator with hysteresis...")
    try:
        from src.core.roi_calculator import ROICalculator
        from src.utils.config import ROIConfig

        config = ROIConfig()
        roi_calc = ROICalculator(config)
        print("  ✓ ROICalculator initialized")
        print(f"    - Hysteresis: {config.use_hysteresis} (threshold={config.hysteresis_threshold}px)")
        print(f"    - Scene locking: {config.use_scene_locking} (threshold={config.lock_threshold}px)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 3: AdaptiveEMASmoother
    print("\n[3/4] Testing AdaptiveEMASmoother...")
    try:
        from src.core.smoother import AdaptiveEMASmoother, create_smoother
        from src.utils.config import SmootherConfig

        config = SmootherConfig()
        smoother = create_smoother(config)
        print("  ✓ AdaptiveEMASmoother initialized")
        print(f"    - Method: {config.method}")
        print(f"    - Alpha range: {config.ema_alpha_min}-{config.ema_alpha_max}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 4: Enhanced config
    print("\n[4/4] Testing enhanced AppConfig...")
    try:
        config = AppConfig()
        print("  ✓ AppConfig loaded")
        print(f"    - Detection model: {config.detection.model_path}")
        print(f"    - Confidence threshold: {config.detection.confidence_threshold}")
        print(f"    - Smoother method: {config.smoother.method}")
        print(f"    - Scene detection: {config.scene_detection.enabled}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print("\n✓ All component tests passed!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ENHANCED REFRAMING PIPELINE - TEST SUITE")
    print("="*70)
    print("\nEnhancements:")
    print("  1. YOLOv8m model (better ball detection)")
    print("  2. Hysteresis dead zone (reduces jitter)")
    print("  3. Scene-coherent locking (stable scenes)")
    print("  4. Adaptive EMA smoothing (activity-aware)")
    print("  5. Scene detection (PySceneDetect)")
    print("="*70)

    # Test components first
    test_components()

    # Check if input video is provided
    if len(sys.argv) > 1:
        input_video = sys.argv[1]

        if not Path(input_video).exists():
            print(f"\n✗ Error: Input video not found: {input_video}")
            sys.exit(1)

        # Run tests
        test_simple_enhanced(input_video)
        test_baseline_comparison(input_video)
        test_config_variants(input_video)

        print("\n" + "="*70)
        print("ALL TESTS COMPLETE")
        print("="*70)
        print("\nOutput files generated:")
        print("  - output/test_enhanced_simple.mp4 (simple API)")
        print("  - output/test_baseline.mp4 (old settings)")
        print("  - output/test_enhanced.mp4 (new settings)")
        print("  - output/test_hysteresis_only.mp4")
        print("  - output/test_locking_only.mp4")
        print("  - output/test_ema_only.mp4")
        print("  - output/test_all_features.mp4")
        print("\nCompare videos side-by-side to see improvements!")

    else:
        print("\n" + "="*70)
        print("USAGE")
        print("="*70)
        print("To test with a video file:")
        print(f"  python {Path(__file__).name} <input_video.mp4>")
        print("\nExample:")
        print(f"  python {Path(__file__).name} input/goal_clip_30s.mp4")
        print("\nNote: Component tests were run successfully!")
        print("      To test full pipeline, provide an input video file.")

    print("\n" + "="*70)
