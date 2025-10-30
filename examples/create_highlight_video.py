"""
하이라이트 JSON 파일을 기반으로 영상 편집

step3_highlights.json의 시간 구간을 추출하여 하나의 영상으로 병합
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import subprocess
import tempfile
import shutil


def load_highlights(json_path: str) -> list:
    """
    하이라이트 JSON 파일 로드

    Args:
        json_path: highlights.json 파일 경로

    Returns:
        List[Dict]: 하이라이트 리스트
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        highlights = json.load(f)

    # start 시간 기준으로 정렬
    highlights.sort(key=lambda x: x['start'])

    return highlights


def extract_video_segments(
    input_video: str,
    highlights: list,
    temp_dir: str
) -> list:
    """
    각 하이라이트 구간을 개별 영상 파일로 추출

    Args:
        input_video: 원본 영상 경로
        highlights: 하이라이트 리스트
        temp_dir: 임시 파일 저장 디렉토리

    Returns:
        List[str]: 추출된 영상 파일 경로 리스트
    """
    segment_files = []

    print("\n🎬 하이라이트 구간 추출 중...")
    print("-" * 70)

    for i, highlight in enumerate(highlights, 1):
        start_time = highlight['start']
        end_time = highlight['end']
        duration = end_time - start_time

        segment_file = f"{temp_dir}/segment_{i:03d}.mp4"

        print(f"[{i}/{len(highlights)}] {start_time:.1f}초 ~ {end_time:.1f}초 "
              f"({duration:.1f}초) - {highlight['type']}")
        print(f"   설명: {highlight['description']}")

        # FFmpeg 명령어
        # -ss: 시작 시간
        # -to: 종료 시간
        # 재인코딩으로 호환성 확보
        cmd = [
            "ffmpeg",
            "-i", input_video,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c:v", "libx264",  # H.264 비디오 코덱
            "-preset", "fast",   # 빠른 인코딩
            "-crf", "23",        # 품질 (낮을수록 고품질)
            "-c:a", "aac",       # AAC 오디오 코덱
            "-b:a", "128k",      # 오디오 비트레이트
            "-y",  # 덮어쓰기
            segment_file
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            segment_files.append(segment_file)
            print(f"   ✅ 추출 완료: {segment_file}")

        except subprocess.CalledProcessError as e:
            print(f"   ❌ 추출 실패: {e.stderr}")
            continue

    print("-" * 70)
    print(f"✅ 총 {len(segment_files)}개 구간 추출 완료\n")

    return segment_files


def create_concat_file(segment_files: list, concat_file_path: str):
    """
    FFmpeg concat 파일 생성

    Args:
        segment_files: 영상 파일 경로 리스트
        concat_file_path: concat 파일 저장 경로
    """
    with open(concat_file_path, 'w', encoding='utf-8') as f:
        for segment_file in segment_files:
            # FFmpeg concat 파일 형식
            # file '/absolute/path/to/file.mp4'
            abs_path = Path(segment_file).resolve()
            f.write(f"file '{abs_path}'\n")


def merge_segments(segment_files: list, output_video: str):
    """
    추출된 영상 파일들을 하나로 병합

    Args:
        segment_files: 영상 파일 경로 리스트
        output_video: 최종 출력 영상 경로
    """
    print("🔗 영상 병합 중...")
    print("-" * 70)

    # 임시 concat 파일 생성
    concat_file = "output/concat_list.txt"
    create_concat_file(segment_files, concat_file)

    print(f"📄 Concat 파일 생성: {concat_file}")
    print(f"   총 {len(segment_files)}개 파일 병합 예정")

    # FFmpeg concat 명령어
    # -f concat: concat demuxer 사용
    # -safe 0: 절대 경로 허용
    # 재인코딩으로 호환성 확보
    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c:v", "libx264",  # H.264 비디오 코덱
        "-preset", "fast",   # 빠른 인코딩
        "-crf", "23",        # 품질
        "-c:a", "aac",       # AAC 오디오 코덱
        "-b:a", "128k",      # 오디오 비트레이트
        "-movflags", "+faststart",  # QuickTime 최적화
        "-y",  # 덮어쓰기
        output_video
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ 병합 완료: {output_video}")

    except subprocess.CalledProcessError as e:
        print(f"❌ 병합 실패: {e.stderr}")
        raise


def create_highlight_video(
    input_video: str,
    highlights_json: str,
    output_video: str = "output/highlights_video.mp4"
):
    """
    하이라이트 JSON을 기반으로 최종 영상 생성

    Args:
        input_video: 원본 영상 경로
        highlights_json: 하이라이트 JSON 경로
        output_video: 출력 영상 경로

    Returns:
        Dict: 통계 정보
    """
    print("=" * 70)
    print("🎥 하이라이트 영상 생성")
    print("=" * 70)
    print(f"📂 원본 영상: {input_video}")
    print(f"📋 하이라이트 JSON: {highlights_json}")
    print(f"💾 출력 영상: {output_video}")

    # 출력 디렉토리 생성
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    # 하이라이트 로드
    print("\n📖 하이라이트 JSON 읽기 중...")
    highlights = load_highlights(highlights_json)

    print(f"✅ {len(highlights)}개 하이라이트 로드 완료")
    print("\n📊 하이라이트 요약:")

    # 타입별 통계
    type_counts = {}
    total_duration = 0
    for h in highlights:
        htype = h['type']
        type_counts[htype] = type_counts.get(htype, 0) + 1
        total_duration += (h['end'] - h['start'])

    for htype, count in type_counts.items():
        print(f"   {htype}: {count}개")

    print(f"   총 하이라이트 길이: {total_duration:.1f}초 ({total_duration/60:.1f}분)")

    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp(prefix="highlight_segments_")
    print(f"\n📁 임시 디렉토리: {temp_dir}")

    try:
        # Step 1: 각 구간 추출
        segment_files = extract_video_segments(
            input_video,
            highlights,
            temp_dir
        )

        if not segment_files:
            print("❌ 추출된 영상이 없습니다.")
            return None

        # Step 2: 병합
        merge_segments(segment_files, output_video)

        # Step 3: 결과 확인
        print("\n" + "=" * 70)
        print("✨ 하이라이트 영상 생성 완료!")
        print("=" * 70)

        output_size = Path(output_video).stat().st_size / (1024 * 1024)  # MB
        print(f"\n📊 최종 결과:")
        print(f"   파일: {output_video}")
        print(f"   크기: {output_size:.1f} MB")
        print(f"   구간 수: {len(segment_files)}개")
        print(f"   총 길이: {total_duration:.1f}초 ({total_duration/60:.1f}분)")

        return {
            'output_video': output_video,
            'segment_count': len(segment_files),
            'total_duration': total_duration,
            'file_size_mb': output_size,
            'highlights': highlights
        }

    finally:
        # 임시 파일 정리
        print(f"\n🧹 임시 파일 정리 중...")
        shutil.rmtree(temp_dir)
        if Path("output/concat_list.txt").exists():
            Path("output/concat_list.txt").unlink()
        print(f"✅ 정리 완료")


def print_highlight_details(highlights: list):
    """하이라이트 상세 정보 출력"""
    print("\n" + "=" * 70)
    print("📋 하이라이트 상세 정보")
    print("=" * 70)

    for i, h in enumerate(highlights, 1):
        start_min = int(h['start'] // 60)
        start_sec = int(h['start'] % 60)
        end_min = int(h['end'] // 60)
        end_sec = int(h['end'] % 60)
        duration = h['end'] - h['start']

        print(f"\n{i}. [{start_min:02d}:{start_sec:02d} ~ {end_min:02d}:{end_sec:02d}] "
              f"({duration:.1f}초)")
        print(f"   타입: {h['type'].upper()}")
        print(f"   설명: {h['description']}")


if __name__ == "__main__":
    import sys

    print("🎥 하이라이트 영상 생성 도구")
    print("=" * 70)
    print()

    # 인자 확인
    if len(sys.argv) < 3:
        print("사용법: python create_highlight_video.py <원본영상> <highlights.json> [출력영상]")
        print()
        print("예시:")
        print("  python create_highlight_video.py input/korea_vs_brazil.mp4 output/step3_highlights.json")
        print("  python create_highlight_video.py input/korea_vs_brazil.mp4 output/step3_highlights.json output/my_highlights.mp4")
        print()
        sys.exit(1)

    input_video = sys.argv[1]
    highlights_json = sys.argv[2]
    output_video = sys.argv[3] if len(sys.argv) > 3 else "output/highlights_video.mp4"

    # 파일 존재 확인
    if not Path(input_video).exists():
        print(f"❌ 원본 영상을 찾을 수 없습니다: {input_video}")
        sys.exit(1)

    if not Path(highlights_json).exists():
        print(f"❌ 하이라이트 JSON을 찾을 수 없습니다: {highlights_json}")
        sys.exit(1)

    # 영상 생성
    result = create_highlight_video(input_video, highlights_json, output_video)

    if result:
        # 상세 정보 출력
        print_highlight_details(result['highlights'])

        print("\n" + "=" * 70)
        print("🎉 완료!")
        print("=" * 70)
        print(f"\n재생 명령어:")
        print(f"  open {output_video}  # macOS")
        print(f"  vlc {output_video}   # VLC Player")
