"""Simple reframing test script.

간단하게 리프레이밍만 테스트하는 스크립트입니다.
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.reframing_pipeline import ReframingPipeline
from src.utils.config import AppConfig


def test_simple_reframing(input_video: str, output_video: str = None):
    """
    간단한 리프레이밍 테스트

    Args:
        input_video: 입력 영상 경로 (예: "input/test.mp4")
        output_video: 출력 영상 경로 (기본값: "output/reframed.mp4")
    """
    # 출력 경로 설정
    if output_video is None:
        output_video = "output/reframed.mp4"

    # 출력 디렉토리 생성
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("간단 리프레이밍 테스트")
    print("="*70)
    print(f"입력: {input_video}")
    print(f"출력: {output_video}")
    print()

    # 기본 설정으로 파이프라인 생성
    config = AppConfig()
    pipeline = ReframingPipeline(config)

    # 리프레이밍 실행
    stats = pipeline.process_goal_clip(
        clip_path=input_video,
        output_path=output_video,
        use_soccernet_model=True,      # SoccerNet YOLO 사용
        use_temporal_filter=True,       # 볼 궤적 smoothing
        use_kalman_smoothing=True       # ROI smoothing
    )

    # 결과 출력
    print("\n" + "="*70)
    print("✓ 리프레이밍 완료!")
    print("="*70)
    print(f"처리 시간: {stats['processing_time']:.1f}초")
    print(f"총 프레임: {stats['frames_processed']}")
    print(f"공 감지율: {stats['ball_detection_rate']:.1%}")
    print(f"평균 선수 수: {stats['average_players_per_frame']:.1f}명")
    print(f"\n출력 파일: {output_video}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='간단한 리프레이밍 테스트')
    parser.add_argument('input', help='입력 영상 경로 (예: input/test.mp4)')
    parser.add_argument('-o', '--output', help='출력 영상 경로 (기본값: output/reframed.mp4)')

    args = parser.parse_args()

    # 입력 파일 확인
    if not Path(args.input).exists():
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    try:
        test_simple_reframing(args.input, args.output)
    except Exception as e:
        print(f"\n❌ 에러 발생:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
