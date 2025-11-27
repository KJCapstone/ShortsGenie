"""
FootAndBall 감지 결과 시각화 스크립트

공과 선수를 다른 색상의 바운딩 박스로 표시하여 감지 성능을 확인합니다.

사용법:
    # 단일 프레임 이미지 테스트
    python test_footandball_visualization.py image.jpg

    # 비디오 전체 처리
    python test_footandball_visualization.py input.mp4 output_visualized.mp4

    # 임계값 조정
    python test_footandball_visualization.py input.mp4 output.mp4 --ball-threshold 0.3 --player-threshold 0.5

    # 첫 N 프레임만 처리
    python test_footandball_visualization.py input.mp4 output.mp4 --max-frames 100
"""

import sys
import os
from pathlib import Path
import argparse

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from src.core.footandball_detector import FootAndBallDetector


def draw_detections(frame, detections, show_labels=True):
    """
    프레임에 감지 결과를 그립니다.

    Args:
        frame: 입력 프레임 (BGR)
        detections: FrameDetections 객체
        show_labels: 라벨과 신뢰도 표시 여부

    Returns:
        시각화된 프레임
    """
    output = frame.copy()

    # 색상 정의 (BGR 형식)
    BALL_COLOR = (0, 0, 255)      # 빨간색 - 공
    PLAYER_COLOR = (0, 255, 0)    # 초록색 - 선수
    TEXT_BG_COLOR = (0, 0, 0)     # 검은색 - 텍스트 배경
    TEXT_COLOR = (255, 255, 255)  # 흰색 - 텍스트

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_thickness = 1

    ball_count = 0
    player_count = 0

    for detection in detections.detections:
        # 바운딩 박스 좌표
        x1 = detection.bbox.x
        y1 = detection.bbox.y
        x2 = x1 + detection.bbox.width
        y2 = y1 + detection.bbox.height

        # 색상 선택
        if detection.is_ball:
            color = BALL_COLOR
            label = "Ball"
            ball_count += 1
        elif detection.is_person:
            color = PLAYER_COLOR
            label = "Player"
            player_count += 1
        else:
            continue

        # 바운딩 박스 그리기
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        if show_labels:
            # 라벨 텍스트 준비
            text = f"{label} {detection.confidence:.2f}"

            # 텍스트 크기 계산
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, text_thickness
            )

            # 텍스트 배경 그리기
            text_x = x1
            text_y = y1 - 10
            if text_y < text_height:
                text_y = y1 + text_height + 10

            cv2.rectangle(
                output,
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
                TEXT_BG_COLOR,
                -1
            )

            # 텍스트 그리기
            cv2.putText(
                output,
                text,
                (text_x, text_y),
                font,
                font_scale,
                TEXT_COLOR,
                text_thickness,
                cv2.LINE_AA
            )

    # 프레임 정보 표시 (좌측 상단)
    info_y = 30
    info_texts = [
        f"Balls: {ball_count}",
        f"Players: {player_count}",
        f"Total: {len(detections.detections)}"
    ]

    for text in info_texts:
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, text_thickness
        )

        # 배경
        cv2.rectangle(
            output,
            (10, info_y - text_height - baseline - 5),
            (20 + text_width, info_y + baseline + 5),
            TEXT_BG_COLOR,
            -1
        )

        # 텍스트
        cv2.putText(
            output,
            text,
            (15, info_y),
            font,
            font_scale,
            TEXT_COLOR,
            text_thickness,
            cv2.LINE_AA
        )

        info_y += 35

    return output


def process_image(image_path, detector, output_path=None):
    """단일 이미지 처리"""
    print(f"Processing image: {image_path}")

    # 이미지 로드
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return

    # 감지 실행
    detections = detector.detect_frame(frame)

    # 결과 출력
    print(f"  Detected {len(detections.ball_detections)} balls")
    print(f"  Detected {len(detections.person_detections)} players")

    # 시각화
    output = draw_detections(frame, detections)

    # 저장 또는 표시
    if output_path:
        cv2.imwrite(output_path, output)
        print(f"  Saved to: {output_path}")
    else:
        # 창 표시
        cv2.imshow('FootAndBall Detection', output)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(video_path, detector, output_path, max_frames=None):
    """비디오 처리"""
    print(f"Processing video: {video_path}")

    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # 비디오 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Frames to process: {total_frames}")

    # 출력 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 통계
    total_balls = 0
    total_players = 0
    frames_with_ball = 0
    frames_with_players = 0

    frame_num = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_num >= max_frames:
                break

            # 감지
            detections = detector.detect_frame(frame, frame_num, frame_num / fps)

            # 통계 업데이트
            ball_count = len(detections.ball_detections)
            player_count = len(detections.person_detections)

            total_balls += ball_count
            total_players += player_count

            if ball_count > 0:
                frames_with_ball += 1
            if player_count > 0:
                frames_with_players += 1

            # 시각화
            output_frame = draw_detections(frame, detections)

            # 프레임 번호 추가 (우측 하단)
            frame_text = f"Frame {frame_num + 1}/{total_frames}"
            cv2.putText(
                output_frame,
                frame_text,
                (width - 200, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # 저장
            out.write(output_frame)

            # 진행 상황 표시
            frame_num += 1
            if frame_num % 30 == 0 or frame_num == total_frames:
                print(f"  Progress: {frame_num}/{total_frames} frames", end='\r')

    finally:
        cap.release()
        out.release()

    print()  # 새 줄

    # 최종 통계
    print("\n" + "="*60)
    print("DETECTION STATISTICS")
    print("="*60)
    print(f"Total frames processed: {frame_num}")
    print(f"Frames with ball: {frames_with_ball} ({frames_with_ball/frame_num*100:.1f}%)")
    print(f"Frames with players: {frames_with_players} ({frames_with_players/frame_num*100:.1f}%)")
    print(f"Average balls per frame: {total_balls/frame_num:.2f}")
    print(f"Average players per frame: {total_players/frame_num:.2f}")
    print("="*60)
    print(f"\nOutput saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="FootAndBall 감지 결과 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 이미지 테스트
  python test_footandball_visualization.py image.jpg

  # 비디오 처리
  python test_footandball_visualization.py input.mp4 output_viz.mp4

  # 임계값 조정
  python test_footandball_visualization.py input.mp4 output.mp4 -b 0.3 -p 0.5

  # 첫 100프레임만
  python test_footandball_visualization.py input.mp4 output.mp4 --max-frames 100
        """
    )

    parser.add_argument('input', help='입력 파일 (이미지 또는 비디오)')
    parser.add_argument('output', nargs='?', help='출력 파일 (비디오인 경우 필수)')
    parser.add_argument('-b', '--ball-threshold', type=float, default=0.5,
                        help='공 감지 임계값 (기본값: 0.5)')
    parser.add_argument('-p', '--player-threshold', type=float, default=0.5,
                        help='선수 감지 임계값 (기본값: 0.5)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='처리할 최대 프레임 수 (비디오만)')
    parser.add_argument('--no-labels', action='store_true',
                        help='라벨 표시 안 함')

    args = parser.parse_args()

    # 입력 파일 확인
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # 파일 타입 확인
    ext = Path(args.input).suffix.lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg']
    is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp']

    if not is_video and not is_image:
        print(f"Error: Unsupported file format: {ext}")
        print("Supported: .mp4, .avi, .mov, .mkv (video) | .jpg, .png, .bmp (image)")
        return 1

    # 비디오인 경우 출력 경로 필수
    if is_video and not args.output:
        print("Error: Output path required for video files")
        print("Usage: python test_footandball_visualization.py input.mp4 output.mp4")
        return 1

    # 출력 디렉토리 생성
    if args.output:
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("FootAndBall Detection Visualization")
    print("="*60)
    print(f"Input: {args.input}")
    if args.output:
        print(f"Output: {args.output}")
    print(f"Ball threshold: {args.ball_threshold}")
    print(f"Player threshold: {args.player_threshold}")
    print()

    # Detector 초기화
    print("Loading FootAndBall detector...")
    detector = FootAndBallDetector(
        ball_threshold=args.ball_threshold,
        player_threshold=args.player_threshold
    )
    print(f"Detector loaded on {detector.device}")
    print()

    # 처리
    try:
        if is_image:
            process_image(args.input, detector, args.output)
        else:
            process_video(args.input, detector, args.output, args.max_frames)

        print("\n✅ Processing completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
