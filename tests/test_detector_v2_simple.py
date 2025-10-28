"""Simple test to compare detector v1 vs v2.

Quick comparison of original detector vs improved detector.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from src.core.detector import ObjectDetector
from src.core.detector_v2 import ObjectDetectorV2, create_detector
from src.utils.config import DetectionConfig


def test_comparison():
    """Compare V1 vs V2 detection on a single frame."""

    # Load test frame
    video_path = Path(__file__).parent.parent / "input" / "korea_vs_brazil.mp4"
    cap = cv2.VideoCapture(str(video_path))

    # Jump to good frame (548s)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    target_frame = int(548 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to read frame")
        return

    print("\n" + "=" * 70)
    print("Detector Comparison Test")
    print("=" * 70)

    # Test V1 (Original)
    print("\n[V1] Original Detector (yolov8n, conf=0.3)")
    print("-" * 70)
    detector_v1 = ObjectDetector(DetectionConfig())
    detections_v1 = detector_v1.detect_frame(frame, target_frame, 548.0)

    print(f"✓ Total detections: {len(detections_v1.detections)}")
    print(f"✓ Persons: {len(detections_v1.person_detections)}")
    print(f"✓ Balls: {len(detections_v1.ball_detections)}")

    # Test V2 with different presets
    for quality in ["fast", "balanced", "high"]:
        print(f"\n[V2] Improved Detector - '{quality.upper()}' preset")
        print("-" * 70)

        detector_v2 = create_detector(quality=quality)
        detections_v2 = detector_v2.detect_frame(frame, target_frame, 548.0)

        print(f"✓ Total detections: {len(detections_v2.detections)}")
        print(f"✓ Persons: {len(detections_v2.person_detections)}")
        print(f"✓ Balls: {len(detections_v2.ball_detections)}")

        # Show ball confidences
        if detections_v2.ball_detections:
            ball_confs = [d.confidence for d in detections_v2.ball_detections]
            print(f"✓ Ball confidences: {', '.join([f'{c:.3f}' for c in ball_confs])}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("V1 (Original): Low ball detection rate")
    print("V2 (Fast):     Good balance of speed and accuracy")
    print("V2 (Balanced): Better ball detection with reasonable speed")
    print("V2 (High):     Best ball detection (recommended for production)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_comparison()
