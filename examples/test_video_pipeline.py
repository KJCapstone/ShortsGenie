"""
비디오 파일로 전체 파이프라인 테스트

비디오에서 오디오 추출 → 하이라이트 필터링 → STT → Gemini 분석
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
import time
from src.audio import AudioHighlightFilter, WhisperTranscriber
from src.ai.transcript_analyzer import TranscriptAnalyzer
import json


def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """
    비디오 파일에서 오디오 추출 (FFmpeg 사용)

    Args:
        video_path: 입력 비디오 파일 경로
        audio_path: 출력 오디오 파일 경로

    Returns:
        bool: 성공 여부
    """
    print("🎬 비디오에서 오디오 추출 중...")
    print(f"   입력: {video_path}")
    print(f"   출력: {audio_path}")

    # FFmpeg 명령어
    # -vn: 비디오 스트림 제거
    # -acodec libmp3lame: MP3 인코더
    # -q:a 2: 고품질 (0-9, 낮을수록 고품질)
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # 비디오 없음
        "-acodec", "libmp3lame",  # MP3 인코더
        "-q:a", "2",  # 고품질
        "-y",  # 덮어쓰기
        audio_path
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ 오디오 추출 완료: {audio_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ 오디오 추출 실패:")
        print(f"   {e.stderr}")
        return False


def get_video_info(video_path: str) -> dict:
    """
    비디오 정보 확인 (ffprobe 사용)

    Args:
        video_path: 비디오 파일 경로

    Returns:
        dict: 비디오 정보
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)

        # 비디오 정보 추출
        video_stream = None
        audio_stream = None

        for stream in info.get('streams', []):
            if stream['codec_type'] == 'video' and not video_stream:
                video_stream = stream
            elif stream['codec_type'] == 'audio' and not audio_stream:
                audio_stream = stream

        duration = float(info.get('format', {}).get('duration', 0))

        return {
            'duration': duration,
            'video_codec': video_stream.get('codec_name') if video_stream else None,
            'video_resolution': f"{video_stream.get('width')}x{video_stream.get('height')}" if video_stream else None,
            'audio_codec': audio_stream.get('codec_name') if audio_stream else None,
            'audio_sample_rate': audio_stream.get('sample_rate') if audio_stream else None,
        }

    except Exception as e:
        print(f"⚠️  비디오 정보 확인 실패: {e}")
        return {}


def test_video_pipeline(video_path: str, output_dir: str = "output"):
    """
    비디오 파일로 전체 파이프라인 테스트

    Args:
        video_path: 입력 비디오 파일
        output_dir: 출력 디렉토리

    Returns:
        Dict: 전체 결과
    """
    print("=" * 70)
    print("🎬 비디오 파일 전체 파이프라인 테스트")
    print("=" * 70)
    print(f"📂 입력 비디오: {video_path}\n")

    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 비디오 파일 존재 확인
    if not Path(video_path).exists():
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
        return None

    # 비디오 정보 확인
    print("📊 비디오 정보 확인 중...")
    video_info = get_video_info(video_path)

    if video_info:
        duration_minutes = int(video_info['duration'] // 60)
        duration_seconds = int(video_info['duration'] % 60)

        print(f"   ⏱️  길이: {duration_minutes}분 {duration_seconds}초")
        print(f"   🎥 비디오: {video_info.get('video_codec')} - {video_info.get('video_resolution')}")
        print(f"   🔊 오디오: {video_info.get('audio_codec')} - {video_info.get('audio_sample_rate')} Hz")
    print()

    total_start = time.time()

    # ========================================
    # Step 0: 오디오 추출
    # ========================================
    print("[Step 0] 비디오에서 오디오 추출")
    print("-" * 70)

    step0_start = time.time()

    video_name = Path(video_path).stem
    audio_path = f"{output_dir}/{video_name}_audio.mp3"

    if not extract_audio_from_video(video_path, audio_path):
        print("❌ 오디오 추출 실패. 종료합니다.")
        return None

    step0_time = time.time() - step0_start
    print(f"✅ Step 0 완료 ({step0_time:.1f}초)\n")

    # ========================================
    # Step 1: 오디오 하이라이트 필터링
    # ========================================
    print("[Step 1] 오디오 하이라이트 필터링")
    print("-" * 70)

    step1_start = time.time()

    audio_filter = AudioHighlightFilter(
        threshold_percentile=60,
        merge_gap=2.0,
        segment_duration=5.0,
        verbose=True
    )

    segments = audio_filter.filter_audio(audio_path)
    step1_time = time.time() - step1_start

    print(f"\n✅ Step 1 완료 ({step1_time:.1f}초)")
    print(f"   추출된 구간: {len(segments)}개")

    if segments:
        print(f"   구간 리스트 (처음 5개):")
        for i, (start, end) in enumerate(segments[:5], 1):
            minutes = int(start // 60)
            seconds = int(start % 60)
            print(f"      {i}. {minutes:02d}:{seconds:02d} ~ {int(end//60):02d}:{int(end%60):02d} (길이: {end-start:.1f}초)")
        if len(segments) > 5:
            print(f"      ... 외 {len(segments)-5}개")

        # Step 1 결과 저장
        segments_data = [
            {
                'start': s,
                'end': e,
                'duration': e-s,
                'timestamp': f"{int(s//60):02d}:{int(s%60):02d}"
            }
            for s, e in segments
        ]
        with open(f"{output_dir}/step1_segments.json", 'w', encoding='utf-8') as f:
            json.dump(segments_data, f, indent=2, ensure_ascii=False)
        print(f"   💾 저장: {output_dir}/step1_segments.json")

        # 총 하이라이트 시간
        total_highlight_time = sum(e - s for s, e in segments)
        print(f"   📊 총 하이라이트 시간: {total_highlight_time:.1f}초 ({total_highlight_time/60:.1f}분)")

        if video_info.get('duration'):
            reduction = (1 - total_highlight_time / video_info['duration']) * 100
            print(f"   📉 감소율: {reduction:.1f}%")
    else:
        print("❌ 하이라이트 구간을 찾지 못했습니다.")
        print("💡 해결책:")
        print("   1. threshold_percentile 값을 낮춰보세요 (예: 40)")
        print("   2. 오디오에 소리가 있는지 확인하세요")
        return None

    # ========================================
    # Step 2: Whisper STT
    # ========================================
    print("\n[Step 2] Whisper 음성-텍스트 변환")
    print("-" * 70)

    step2_start = time.time()

    transcriber = WhisperTranscriber(
        model_size="base",
        device="auto",
        language="ko",
        verbose=True
    )

    result = transcriber.transcribe(audio_path, segments=segments)
    step2_time = time.time() - step2_start

    print(f"\n✅ Step 2 완료 ({step2_time:.1f}초)")
    print(f"   전체 텍스트 길이: {len(result['text'])}자")
    print(f"   세그먼트 수: {len(result['segments'])}개")
    print(f"   감지된 언어: {result['language']}")

    if 'original_segments_count' in result:
        reduction = (1 - result['filtered_segments_count'] / result['original_segments_count']) * 100
        print(f"   세그먼트 감소율: {reduction:.1f}%")

    # Step 2 결과 저장
    transcriber.save_transcript(result, f"{output_dir}/step2_transcript.txt", format="txt")
    transcriber.save_transcript(result, f"{output_dir}/step2_transcript.json", format="json")
    print(f"   💾 저장: {output_dir}/step2_transcript.txt")
    print(f"   💾 저장: {output_dir}/step2_transcript.json")

    # 텍스트 미리보기
    if result['text']:
        print(f"\n   📄 텍스트 미리보기 (처음 300자):")
        preview = result['text'][:300]
        print(f"   {preview}{'...' if len(result['text']) > 300 else ''}")
    else:
        print(f"\n   ⚠️  텍스트가 비어있습니다. 오디오에 음성이 없을 수 있습니다.")

    # 타임스탬프 포함 텍스트 생성 (Gemini용)
    timestamp_text = ""
    for seg in result['segments']:
        timestamp_text += f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}\n"

    timestamp_path = f"{output_dir}/step2_transcript_with_timestamps.txt"
    with open(timestamp_path, 'w', encoding='utf-8') as f:
        f.write(timestamp_text)
    print(f"   💾 저장: {timestamp_path}")

    # ========================================
    # Step 3: Gemini LLM 분석
    # ========================================
    print("\n[Step 3] Gemini LLM 하이라이트 분석")
    print("-" * 70)

    step3_start = time.time()

    analyzer = TranscriptAnalyzer(verbose=True)

    highlights_path = f"{output_dir}/step3_highlights.json"
    highlights = analyzer.analyze_transcript(timestamp_path, highlights_path)
    step3_time = time.time() - step3_start

    print(f"\n✅ Step 3 완료 ({step3_time:.1f}초)")
    print(f"   추출된 하이라이트: {len(highlights)}개")
    print(f"   💾 저장: {highlights_path}")

    # 하이라이트 출력
    if highlights:
        print(f"\n   🎯 추출된 하이라이트:")
        for i, h in enumerate(highlights, 1):
            minutes = int(h['start'] // 60)
            seconds = int(h['start'] % 60)
            print(f"      {i}. [{minutes:02d}:{seconds:02d}] {h['type']}")
            print(f"         {h['description']}")

        # 타입별 통계
        type_counts = {}
        for h in highlights:
            type_counts[h['type']] = type_counts.get(h['type'], 0) + 1

        print(f"\n   📊 타입별 통계:")
        for htype, count in type_counts.items():
            print(f"      {htype}: {count}개")
    else:
        print("   ⚠️  하이라이트를 찾지 못했습니다.")

    # ========================================
    # 최종 요약
    # ========================================
    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("✨ 전체 파이프라인 완료!")
    print("=" * 70)

    print(f"\n⏱️  소요 시간:")
    print(f"   Step 0 (오디오 추출): {step0_time:.1f}초")
    print(f"   Step 1 (오디오 필터링): {step1_time:.1f}초")
    print(f"   Step 2 (Whisper STT): {step2_time:.1f}초")
    print(f"   Step 3 (Gemini 분석): {step3_time:.1f}초")
    print(f"   총 소요 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")

    print(f"\n📊 최종 통계:")
    if video_info.get('duration'):
        print(f"   원본 비디오: {video_info['duration']:.0f}초 ({video_info['duration']/60:.1f}분)")
    print(f"   추출된 오디오 구간: {len(segments)}개")
    print(f"   텍스트 세그먼트: {len(result['segments'])}개")
    print(f"   최종 하이라이트: {len(highlights)}개")

    print(f"\n📁 생성된 파일:")
    print(f"   ✓ {audio_path}")
    print(f"   ✓ {output_dir}/step1_segments.json")
    print(f"   ✓ {output_dir}/step2_transcript.txt")
    print(f"   ✓ {output_dir}/step2_transcript.json")
    print(f"   ✓ {output_dir}/step2_transcript_with_timestamps.txt")
    print(f"   ✓ {output_dir}/step3_highlights.json")

    return {
        'video_info': video_info,
        'audio_path': audio_path,
        'segments': segments,
        'transcript': result,
        'highlights': highlights,
        'timing': {
            'step0': step0_time,
            'step1': step1_time,
            'step2': step2_time,
            'step3': step3_time,
            'total': total_time
        }
    }


if __name__ == "__main__":
    import sys

    print("🎬 비디오 파일 전체 파이프라인 테스트")
    print("=" * 70)
    print()

    # 입력 파일 확인
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # 기본값: korea_vs_brazil.mp4
        video_path = "input/korea_vs_brazil.mp4"
        print(f"💡 사용법: python {Path(__file__).name} <video_file>")
        print(f"   기본 파일 사용: {video_path}\n")

    if not Path(video_path).exists():
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
        print(f"\n💡 해결 방법:")
        print(f"   1. 'input/' 디렉토리에 비디오 파일 넣기")
        print(f"   2. 파일 경로를 인자로 전달: python {Path(__file__).name} path/to/video.mp4")
        sys.exit(1)

    # 파이프라인 실행
    result = test_video_pipeline(video_path)

    if result and result['highlights']:
        print("\n" + "=" * 70)
        print("📋 최종 하이라이트 요약")
        print("=" * 70)

        for i, h in enumerate(result['highlights'], 1):
            minutes = int(h['start'] // 60)
            seconds = int(h['start'] % 60)
            print(f"{i}. [{minutes:02d}:{seconds:02d}] {h['type'].upper()}")
            print(f"   {h['description']}")

    print("\n" + "=" * 70)
    print("✅ 테스트 완료!")
    print("=" * 70)
