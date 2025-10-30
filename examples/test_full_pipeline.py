"""
전체 파이프라인 통합 테스트

오디오 필터링 → Whisper STT → Gemini 분석 → JSON 출력
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio import AudioHighlightFilter, WhisperTranscriber
from src.ai.transcript_analyzer import TranscriptAnalyzer
import json
import time


def test_pipeline_step_by_step(audio_path: str):
    """
    단계별 전체 파이프라인 테스트

    Args:
        audio_path: 입력 오디오 파일 경로

    Returns:
        Dict: 전체 결과
    """

    print("=" * 70)
    print("🎬 전체 파이프라인 단계별 테스트")
    print("=" * 70)
    print(f"📂 입력 파일: {audio_path}\n")

    # 출력 디렉토리 생성
    Path("output").mkdir(exist_ok=True)

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
        print(f"   구간 리스트:")
        for i, (start, end) in enumerate(segments[:5], 1):  # 최대 5개만 표시
            print(f"      {i}. {start:.1f}초 ~ {end:.1f}초 (길이: {end-start:.1f}초)")
        if len(segments) > 5:
            print(f"      ... 외 {len(segments)-5}개")

        # Step 1 결과 저장
        segments_data = [{'start': s, 'end': e, 'duration': e-s} for s, e in segments]
        with open("output/step1_segments.json", 'w', encoding='utf-8') as f:
            json.dump(segments_data, f, indent=2, ensure_ascii=False)
        print(f"   💾 저장: output/step1_segments.json")
    else:
        print("❌ 하이라이트 구간을 찾지 못했습니다.")
        print("💡 해결책: threshold_percentile 값을 낮춰보세요 (예: 40)")
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
    transcriber.save_transcript(result, "output/step2_transcript.txt", format="txt")
    transcriber.save_transcript(result, "output/step2_transcript.json", format="json")
    print(f"   💾 저장: output/step2_transcript.txt")
    print(f"   💾 저장: output/step2_transcript.json")

    # 텍스트 미리보기
    print(f"\n   📄 텍스트 미리보기 (처음 200자):")
    preview = result['text'][:200]
    print(f"   {preview}{'...' if len(result['text']) > 200 else ''}")

    # 타임스탬프 포함 텍스트 생성 (Gemini용)
    timestamp_text = ""
    for seg in result['segments']:
        timestamp_text += f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}\n"

    timestamp_path = "output/step2_transcript_with_timestamps.txt"
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

    highlights_path = "output/step3_highlights.json"
    highlights = analyzer.analyze_transcript(timestamp_path, highlights_path)
    step3_time = time.time() - step3_start

    print(f"\n✅ Step 3 완료 ({step3_time:.1f}초)")
    print(f"   추출된 하이라이트: {len(highlights)}개")
    print(f"   💾 저장: {highlights_path}")

    # 하이라이트 출력
    if highlights:
        print(f"\n   🎯 추출된 하이라이트:")
        for i, h in enumerate(highlights, 1):
            print(f"      {i}. [{h['start']:.1f}s - {h['end']:.1f}s] {h['type']}")
            print(f"         {h['description']}")
    else:
        print("   ⚠️  하이라이트를 찾지 못했습니다.")

    # ========================================
    # 최종 요약
    # ========================================
    total_time = step1_time + step2_time + step3_time

    print("\n" + "=" * 70)
    print("✨ 전체 파이프라인 완료!")
    print("=" * 70)

    print(f"\n📊 최종 통계:")
    print(f"   Step 1: {len(segments)}개 오디오 구간 추출 ({step1_time:.1f}초)")
    print(f"   Step 2: {len(result['segments'])}개 텍스트 세그먼트 변환 ({step2_time:.1f}초)")
    print(f"   Step 3: {len(highlights)}개 하이라이트 추출 ({step3_time:.1f}초)")
    print(f"   총 소요 시간: {total_time:.1f}초")

    print(f"\n📁 생성된 파일:")
    print(f"   ✓ output/step1_segments.json")
    print(f"   ✓ output/step2_transcript.txt")
    print(f"   ✓ output/step2_transcript.json")
    print(f"   ✓ output/step2_transcript_with_timestamps.txt")
    print(f"   ✓ output/step3_highlights.json")

    return {
        'segments': segments,
        'transcript': result,
        'highlights': highlights,
        'timing': {
            'step1': step1_time,
            'step2': step2_time,
            'step3': step3_time,
            'total': total_time
        }
    }


def run_full_pipeline_fast(audio_path: str):
    """
    전체 파이프라인 빠른 실행 (진행 상황 최소화)

    Args:
        audio_path: 입력 오디오 파일 경로

    Returns:
        Dict: 전체 결과
    """
    from pathlib import Path
    import time

    Path("output").mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    print("🎬 전체 파이프라인 실행 (빠른 모드)")
    print("=" * 70)

    # Step 1
    print("⏳ [1/3] 오디오 분석 중...", end=" ", flush=True)
    step1_start = time.time()
    audio_filter = AudioHighlightFilter(verbose=False)
    segments = audio_filter.filter_audio(audio_path)
    step1_time = time.time() - step1_start
    print(f"✅ ({step1_time:.1f}초) - {len(segments)}개 구간")

    if not segments:
        print("❌ 하이라이트 구간을 찾지 못했습니다.")
        return None

    # Step 2
    print("⏳ [2/3] 음성-텍스트 변환 중...", end=" ", flush=True)
    step2_start = time.time()
    transcriber = WhisperTranscriber(model_size="base", device="auto", language="ko", verbose=False)
    result = transcriber.transcribe(audio_path, segments=segments)

    timestamp_text = "\n".join([
        f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}"
        for seg in result['segments']
    ])
    timestamp_path = "output/transcript_timestamps.txt"
    with open(timestamp_path, 'w', encoding='utf-8') as f:
        f.write(timestamp_text)

    step2_time = time.time() - step2_start
    print(f"✅ ({step2_time:.1f}초) - {len(result['segments'])}개 세그먼트")

    # Step 3
    print("⏳ [3/3] AI 분석 중...", end=" ", flush=True)
    step3_start = time.time()
    analyzer = TranscriptAnalyzer(verbose=False)
    highlights = analyzer.analyze_transcript(timestamp_path, "output/highlights.json")
    step3_time = time.time() - step3_start
    print(f"✅ ({step3_time:.1f}초) - {len(highlights)}개 하이라이트")

    total_time = time.time() - total_start

    print("=" * 70)
    print(f"✨ 완료! (총 {total_time:.1f}초)")
    print(f"   📁 출력: output/highlights.json")

    return {
        'segments': segments,
        'transcript': result,
        'highlights': highlights,
        'timing': {'total': total_time}
    }


if __name__ == "__main__":
    import sys

    print("🎵 전체 파이프라인 테스트")
    print("=" * 70)
    print()

    # 입력 파일 확인
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "input/sample_match.mp3"
        print(f"💡 사용법: python {Path(__file__).name} <audio_file>")
        print(f"   기본 파일 사용: {audio_path}\n")

    if not Path(audio_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {audio_path}")
        print(f"\n💡 해결 방법:")
        print(f"   1. 'input/' 디렉토리에 오디오 파일 넣기")
        print(f"   2. 파일 경로를 인자로 전달: python {Path(__file__).name} path/to/audio.mp3")
        sys.exit(1)

    # 모드 선택
    print("실행 모드를 선택하세요:")
    print("  1. 단계별 상세 모드 (권장) - 각 단계 결과 확인")
    print("  2. 빠른 실행 모드 - 진행 상황만 표시")

    choice = input("\n선택 (1-2, 기본값=1): ").strip() or "1"

    if choice == "1":
        result = test_pipeline_step_by_step(audio_path)
    elif choice == "2":
        result = run_full_pipeline_fast(audio_path)
    else:
        print("❌ 잘못된 선택입니다.")
        sys.exit(1)

    if result:
        print("\n📋 최종 하이라이트 요약:")
        for i, h in enumerate(result['highlights'], 1):
            minutes = int(h['start'] // 60)
            seconds = int(h['start'] % 60)
            print(f"  {i}. [{minutes:02d}:{seconds:02d}] {h['type']}: {h['description']}")

        print("\n" + "=" * 70)
        print("✅ 테스트 완료!")
        print("=" * 70)
