"""
오디오 하이라이트 필터링 및 STT 변환 예제

Phase 1 구현: librosa + Whisper
- RMS 에너지 및 Spectral 분석으로 하이라이트 구간 추출
- Whisper로 하이라이트 구간만 STT 변환
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio import AudioHighlightFilter, WhisperTranscriber


def example_1_audio_filtering():
    """
    예제 1: 오디오 하이라이트 구간 추출

    RMS 에너지와 Spectral 분석을 결합하여
    경기 하이라이트 구간을 자동으로 찾습니다.
    """
    print("=" * 70)
    print("예제 1: 오디오 하이라이트 구간 추출")
    print("=" * 70)

    # 입력 파일 (실제 경로로 변경하세요)
    audio_path = "input/sample_match.mp3"

    # 파일 존재 확인
    if not Path(audio_path).exists():
        print(f"❌ 오류: 파일을 찾을 수 없습니다: {audio_path}")
        print(f"💡 'input/' 디렉토리에 오디오 파일을 넣고 경로를 수정하세요.")
        return None

    # 하이라이트 필터 생성
    audio_filter = AudioHighlightFilter(
        threshold_percentile=60,  # 상위 40% 구간 유지
        merge_gap=2.0,            # 2초 이내 구간 병합
        segment_duration=5.0,     # 각 구간 5초
        rms_weight=0.7,           # RMS 70%
        spectral_weight=0.3,      # Spectral 30%
        verbose=True
    )

    # 하이라이트 구간 추출
    segments = audio_filter.filter_audio(audio_path)

    print(f"\n📋 추출된 하이라이트 구간:")
    for i, (start, end) in enumerate(segments, 1):
        print(f"   {i}. {start:.1f}초 ~ {end:.1f}초 (길이: {end-start:.1f}초)")

    return segments


def example_2_whisper_full():
    """
    예제 2: Whisper 전체 오디오 변환

    오디오 파일 전체를 텍스트로 변환합니다.
    (비교를 위한 예제)
    """
    print("\n" + "=" * 70)
    print("예제 2: Whisper 전체 오디오 변환 (비교용)")
    print("=" * 70)

    # 입력 파일
    audio_path = "input/sample_match.mp3"

    if not Path(audio_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {audio_path}")
        return

    # Whisper 변환기 생성
    transcriber = WhisperTranscriber(
        model_size="base",   # base 모델 사용
        device="auto",       # 자동 디바이스 선택
        language="ko",       # 한국어
        verbose=True
    )

    # 전체 오디오 변환
    result = transcriber.transcribe(audio_path)

    # 결과 저장
    output_path = "output/transcript_full.txt"
    transcriber.save_transcript(result, output_path, format="txt")
    transcriber.save_transcript(
        result,
        output_path.replace('.txt', '.json'),
        format="json"
    )

    print(f"\n📄 변환된 텍스트 (처음 500자):")
    print(result['text'][:500])
    if len(result['text']) > 500:
        print("...")


def example_3_combined_pipeline():
    """
    예제 3: 통합 파이프라인

    1. 오디오 하이라이트 필터링
    2. 하이라이트 구간만 STT 변환

    → 전체 시간 대비 60% 감소!
    """
    print("\n" + "=" * 70)
    print("예제 3: 통합 파이프라인 (필터링 + STT)")
    print("=" * 70)

    # 입력 파일
    audio_path = "input/sample_match.mp3"

    if not Path(audio_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {audio_path}")
        return

    # Step 1: 하이라이트 구간 추출
    print("\n🔍 Step 1: 하이라이트 구간 추출")
    print("-" * 70)

    audio_filter = AudioHighlightFilter(
        threshold_percentile=60,
        merge_gap=2.0,
        segment_duration=5.0,
        verbose=True
    )

    segments = audio_filter.filter_audio(audio_path)

    if not segments:
        print("❌ 하이라이트 구간을 찾지 못했습니다.")
        return

    # Step 2: 하이라이트 구간만 STT 변환
    print("\n🎙️ Step 2: 하이라이트 구간 STT 변환")
    print("-" * 70)

    transcriber = WhisperTranscriber(
        model_size="base",
        device="auto",
        language="ko",
        verbose=True
    )

    result = transcriber.transcribe(audio_path, segments=segments)

    # 결과 저장
    output_path = "output/transcript_filtered.txt"
    transcriber.save_transcript(result, output_path, format="txt")
    transcriber.save_transcript(
        result,
        output_path.replace('.txt', '.json'),
        format="json"
    )

    print(f"\n📄 변환된 텍스트 (처음 500자):")
    print(result['text'][:500])
    if len(result['text']) > 500:
        print("...")

    # 통계 출력
    print(f"\n📊 성능 비교:")
    print(f"   원본 세그먼트: {result.get('original_segments_count', 'N/A')}개")
    print(f"   필터링 후: {result.get('filtered_segments_count', 'N/A')}개")

    if result.get('original_segments_count'):
        reduction = (1 - result['filtered_segments_count'] / result['original_segments_count']) * 100
        print(f"   감소율: {reduction:.1f}%")


def example_4_custom_config():
    """
    예제 4: 사용자 설정 조정

    상황에 맞게 파라미터를 조정할 수 있습니다.
    """
    print("\n" + "=" * 70)
    print("예제 4: 사용자 설정 조정")
    print("=" * 70)

    audio_path = "input/sample_match.mp3"

    if not Path(audio_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {audio_path}")
        return

    # 설정 1: 공격적인 필터링 (80% 감소)
    print("\n🔥 설정 1: 공격적인 필터링 (상위 20%만 유지)")
    print("-" * 70)

    aggressive_filter = AudioHighlightFilter(
        threshold_percentile=80,  # 상위 20%만 유지
        merge_gap=1.0,            # 1초 이내만 병합
        segment_duration=3.0,     # 짧은 구간 (3초)
        rms_weight=0.8,           # 음량 우선
        spectral_weight=0.2,
        verbose=True
    )

    segments = aggressive_filter.filter_audio(audio_path)
    print(f"   추출된 구간: {len(segments)}개")

    # 설정 2: 보수적인 필터링 (40% 감소)
    print("\n🛡️ 설정 2: 보수적인 필터링 (상위 60% 유지)")
    print("-" * 70)

    conservative_filter = AudioHighlightFilter(
        threshold_percentile=40,  # 상위 60% 유지
        merge_gap=3.0,            # 3초 이내 병합
        segment_duration=7.0,     # 긴 구간 (7초)
        rms_weight=0.5,           # 균형
        spectral_weight=0.5,
        verbose=True
    )

    segments = conservative_filter.filter_audio(audio_path)
    print(f"   추출된 구간: {len(segments)}개")


if __name__ == "__main__":
    print("🎵 오디오 하이라이트 필터링 및 STT 예제")
    print("=" * 70)
    print()
    print("📌 준비사항:")
    print("   1. 'input/' 디렉토리에 오디오 파일 넣기 (sample_match.mp3)")
    print("   2. librosa, soundfile, openai-whisper 설치 확인")
    print("   3. Apple Silicon 사용 시 MPS 가속 자동 활성화")
    print()

    # 예제 선택
    print("실행할 예제를 선택하세요:")
    print("  1. 오디오 하이라이트 구간 추출만")
    print("  2. Whisper 전체 오디오 변환 (비교용)")
    print("  3. 통합 파이프라인 (필터링 + STT) ★ 추천")
    print("  4. 사용자 설정 조정 예제")
    print("  0. 모두 실행")

    choice = input("\n선택 (0-4): ").strip()

    if choice == "1":
        example_1_audio_filtering()
    elif choice == "2":
        example_2_whisper_full()
    elif choice == "3":
        example_3_combined_pipeline()
    elif choice == "4":
        example_4_custom_config()
    elif choice == "0":
        example_1_audio_filtering()
        example_2_whisper_full()
        example_3_combined_pipeline()
        example_4_custom_config()
    else:
        print("❌ 잘못된 선택입니다.")

    print("\n" + "=" * 70)
    print("✨ 예제 실행 완료!")
    print("=" * 70)
