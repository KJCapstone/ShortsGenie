"""
재인코딩 없이 빠르게 하이라이트 영상 생성

Keyframe 기준으로 정확하게 자르기
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import subprocess
import tempfile
import shutil


def load_highlights(json_path: str) -> list:
    """하이라이트 JSON 파일 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        highlights = json.load(f)
    highlights.sort(key=lambda x: x['start'])
    return highlights


def extract_segments_with_reencoding_boundaries(
    input_video: str,
    highlights: list,
    temp_dir: str
) -> list:
    """
    재인코딩 없이 정확하게 자르기

    방법: -ss를 -i 앞에 배치 (입력 seeking) + 재인코딩
    """
    segment_files = []

    print("\n🎬 하이라이트 구간 추출 중 (빠른 모드)...")
    print("-" * 70)

    for i, highlight in enumerate(highlights, 1):
        start_time = highlight['start']
        end_time = highlight['end']
        duration = end_time - start_time

        segment_file = f"{temp_dir}/segment_{i:03d}.mp4"

        print(f"[{i}/{len(highlights)}] {start_time:.1f}초 ~ {end_time:.1f}초 "
              f"({duration:.1f}초) - {highlight['type']}")

        # 방법: -ss를 -i 앞에 (빠른 seek) + 짧게 재인코딩
        # -ss 위치가 중요!
        cmd = [
            "ffmpeg",
            "-ss", str(start_time),  # -i 앞에: 빠른 seek
            "-i", input_video,
            "-t", str(duration),     # 길이 지정
            "-c:v", "libx264",
            "-preset", "ultrafast",  # 가장 빠른 인코딩
            "-crf", "23",
            "-c:a", "copy",          # 오디오는 복사
            "-y",
            segment_file
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            segment_files.append(segment_file)
            print(f"   ✅ 완료")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ 실패: {e.stderr[:200]}")
            continue

    return segment_files


def extract_segments_copy_mode(
    input_video: str,
    highlights: list,
    temp_dir: str
) -> list:
    """
    복사 모드로 자르기 (가장 빠름, 재생 안될 수 있음)

    개선: -avoid_negative_ts 및 올바른 옵션 순서
    """
    segment_files = []

    print("\n🎬 하이라이트 구간 추출 중 (복사 모드)...")
    print("-" * 70)

    for i, highlight in enumerate(highlights, 1):
        start_time = highlight['start']
        end_time = highlight['end']
        duration = end_time - start_time

        segment_file = f"{temp_dir}/segment_{i:03d}.mp4"

        print(f"[{i}/{len(highlights)}] {start_time:.1f}초 ~ {end_time:.1f}초")

        # 정확한 복사 모드
        cmd = [
            "ffmpeg",
            "-ss", str(start_time),      # 입력 seeking
            "-i", input_video,
            "-t", str(duration),          # 길이
            "-c", "copy",                 # 복사
            "-avoid_negative_ts", "1",    # 타임스탬프 수정
            "-fflags", "+genpts",         # PTS 재생성
            "-y",
            segment_file
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            segment_files.append(segment_file)
            print(f"   ✅ 완료")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ 실패")
            continue

    return segment_files


def merge_segments_reencode(segment_files: list, output_video: str):
    """세그먼트 병합 (재인코딩)"""
    print("\n🔗 영상 병합 중 (재인코딩)...")

    concat_file = "output/concat_list.txt"
    with open(concat_file, 'w') as f:
        for seg in segment_files:
            f.write(f"file '{Path(seg).resolve()}'\n")

    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-y",
        output_video
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"✅ 병합 완료: {output_video}")


def merge_segments_copy(segment_files: list, output_video: str):
    """세그먼트 병합 (복사 모드)"""
    print("\n🔗 영상 병합 중 (복사 모드)...")

    concat_file = "output/concat_list.txt"
    with open(concat_file, 'w') as f:
        for seg in segment_files:
            f.write(f"file '{Path(seg).resolve()}'\n")

    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        "-movflags", "+faststart",
        "-y",
        output_video
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ 병합 완료: {output_video}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 복사 모드 실패, 재인코딩 시도...")
        merge_segments_reencode(segment_files, output_video)
        return False


def create_highlight_video(
    input_video: str,
    highlights_json: str,
    output_video: str,
    mode: str = "fast"
):
    """
    하이라이트 영상 생성

    Args:
        mode: 'fast' (빠른 재인코딩) 또는 'copy' (복사, 재생 안될 수 있음)
    """
    print("=" * 70)
    print(f"🎥 하이라이트 영상 생성 (모드: {mode})")
    print("=" * 70)

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    highlights = load_highlights(highlights_json)
    print(f"📋 {len(highlights)}개 하이라이트 로드")

    temp_dir = tempfile.mkdtemp(prefix="highlight_segments_")

    try:
        # 구간 추출
        if mode == "copy":
            segment_files = extract_segments_copy_mode(input_video, highlights, temp_dir)
            merge_segments_copy(segment_files, output_video)
        else:  # fast
            segment_files = extract_segments_with_reencoding_boundaries(
                input_video, highlights, temp_dir
            )
            merge_segments_reencode(segment_files, output_video)

        print("\n✨ 완료!")
        print(f"📁 {output_video}")

    finally:
        shutil.rmtree(temp_dir)
        if Path("output/concat_list.txt").exists():
            Path("output/concat_list.txt").unlink()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: python create_highlight_video_fast.py <원본영상> <highlights.json> [출력영상] [모드]")
        print()
        print("모드:")
        print("  fast - 빠른 재인코딩 (권장, 100% 재생 가능)")
        print("  copy - 복사 모드 (가장 빠름, 재생 안될 수 있음)")
        print()
        print("예시:")
        print("  python create_highlight_video_fast.py input/video.mp4 output/highlights.json")
        print("  python create_highlight_video_fast.py input/video.mp4 output/highlights.json output/result.mp4 fast")
        sys.exit(1)

    input_video = sys.argv[1]
    highlights_json = sys.argv[2]
    output_video = sys.argv[3] if len(sys.argv) > 3 else "output/highlights_video.mp4"
    mode = sys.argv[4] if len(sys.argv) > 4 else "fast"

    create_highlight_video(input_video, highlights_json, output_video, mode)
