"""장면별 고정 ROI 테스트

각 장면마다 ROI를 고정하여 흔들림 없는 리프레이밍을 테스트합니다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.reframing_pipeline import ReframingPipeline
from src.utils.config import AppConfig


def test_fixed_roi(input_video: str, scene_json: str = None, output_video: str = None):
    """
    장면별 고정 ROI 리프레이밍

    Args:
        input_video: 입력 영상 경로
        scene_json: auto_tagger JSON 경로 (없으면 기본값 사용)
        output_video: 출력 영상 경로
    """
    if output_video is None:
        output_video = "output/fixed_roi.mp4"

    if scene_json is None:
        scene_json = "full_match_log.json"

    # 출력 디렉토리 생성
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("장면별 고정 ROI 테스트")
    print("="*70)
    print(f"입력: {input_video}")
    print(f"Scene JSON: {scene_json}")
    print(f"출력: {output_video}")
    print()

    # Scene JSON 확인
    if not Path(scene_json).exists():
        print(f"⚠️  Scene JSON이 없습니다: {scene_json}")
        print("\nJSON 생성 방법:")
        print("  1. cd auto_tagger")
        print("  2. python test.py")
        print("\n또는 입력 영상을 auto_tagger에서 분석하세요.")
        return None

    # Scene-aware 설정 활성화
    config = AppConfig()
    config.scene.enabled = True  # ✅ Scene awareness 켜기 (중요!)

    pipeline = ReframingPipeline(config)

    # 리프레이밍 실행
    stats = pipeline.process_goal_clip(
        clip_path=input_video,
        output_path=output_video,
        use_soccernet_model=True,
        use_temporal_filter=True,
        use_kalman_smoothing=True,
        scene_metadata_path=scene_json  # ✅ Scene JSON 전달 (중요!)
    )

    # 결과 출력
    print("\n" + "="*70)
    print("✓ 장면별 고정 ROI 리프레이밍 완료!")
    print("="*70)
    print(f"처리 시간: {stats['processing_time']:.1f}초")
    print(f"총 프레임: {stats['frames_processed']}")
    print(f"공 감지율: {stats['ball_detection_rate']:.1%}")
    print(f"\n출력 파일: {output_video}")
    print("\n💡 각 장면마다 고정된 ROI를 사용했습니다.")
    print("   장면 전환 시에만 ROI가 변경됩니다.")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='장면별 고정 ROI 테스트')
    parser.add_argument('input', help='입력 영상 경로')
    parser.add_argument('-s', '--scene-json', help='Scene JSON 경로 (기본: auto_tagger/full_match_log.json)')
    parser.add_argument('-o', '--output', help='출력 영상 경로 (기본: output/fixed_roi.mp4)')

    args = parser.parse_args()

    # 입력 파일 확인
    if not Path(args.input).exists():
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    try:
        test_fixed_roi(args.input, args.scene_json, args.output)
    except Exception as e:
        print(f"\n❌ 에러 발생:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
