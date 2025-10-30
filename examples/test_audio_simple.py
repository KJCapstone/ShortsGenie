"""
간단한 오디오 필터링 테스트

실제 오디오 파일 없이도 기능을 테스트할 수 있는 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
from src.audio import AudioHighlightFilter, filter_highlight_segments


def create_test_audio():
    """
    테스트용 오디오 파일 생성

    - 30초 길이
    - 10초, 20초 지점에 큰 소리 (하이라이트 구간)
    - 나머지는 작은 소리 (배경 소음)
    """
    print("🔧 테스트 오디오 파일 생성 중...")

    sr = 22050  # 샘플링 레이트
    duration = 30  # 30초

    # 기본 배경 소음 (작은 소리)
    t = np.linspace(0, duration, sr * duration)
    y = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440Hz 사인파, 작은 음량

    # 10초 지점: 큰 소리 (하이라이트 1)
    start1 = int(10 * sr)
    end1 = int(15 * sr)
    y[start1:end1] += 0.8 * np.sin(2 * np.pi * 880 * t[start1:end1])  # 큰 음량, 높은 주파수

    # 20초 지점: 큰 소리 (하이라이트 2)
    start2 = int(20 * sr)
    end2 = int(25 * sr)
    y[start2:end2] += 0.9 * np.sin(2 * np.pi * 1760 * t[start2:end2])  # 매우 큰 음량, 더 높은 주파수

    # 정규화
    y = y / np.max(np.abs(y))

    # 저장
    output_path = Path("output/test_audio.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(output_path), y, sr)

    print(f"✅ 테스트 오디오 생성 완료: {output_path}")
    print(f"   길이: {duration}초")
    print(f"   하이라이트 구간: 10-15초, 20-25초")

    return str(output_path)


def test_audio_filtering():
    """오디오 하이라이트 필터링 테스트"""
    print("\n" + "=" * 70)
    print("🧪 오디오 하이라이트 필터링 테스트")
    print("=" * 70)

    # 테스트 오디오 생성
    audio_path = create_test_audio()

    # 하이라이트 구간 추출
    print("\n🔍 하이라이트 구간 추출 중...")
    segments = filter_highlight_segments(
        audio_path,
        threshold_percentile=50,  # 상위 50% 유지
        merge_gap=2.0,
        segment_duration=5.0,
        verbose=True
    )

    # 결과 검증
    print("\n✅ 추출된 하이라이트 구간:")
    for i, (start, end) in enumerate(segments, 1):
        print(f"   {i}. {start:.1f}초 ~ {end:.1f}초 (길이: {end-start:.1f}초)")

    # 예상 결과와 비교
    print("\n🎯 예상 하이라이트 구간:")
    print("   1. 10-15초 부근 (큰 소리)")
    print("   2. 20-25초 부근 (매우 큰 소리)")

    if len(segments) >= 2:
        print("\n✅ 테스트 성공: 2개 이상의 하이라이트 구간 발견")
    else:
        print(f"\n⚠️  경고: {len(segments)}개 구간만 발견됨")


def test_module_imports():
    """모듈 임포트 테스트"""
    print("\n" + "=" * 70)
    print("🧪 모듈 임포트 테스트")
    print("=" * 70)

    try:
        from src.audio import (
            AudioHighlightFilter,
            filter_highlight_segments,
            WhisperTranscriber,
            compute_rms_energy,
            compute_spectral_features,
            merge_segments,
            frames_to_time_segments
        )
        print("✅ src.audio 모듈 임포트 성공")

        from src.utils.config import AppConfig
        config = AppConfig()
        print("✅ AppConfig 임포트 성공")
        print(f"   오디오 설정: {config.audio}")

        print("\n✅ 모든 모듈 임포트 성공!")
        return True

    except ImportError as e:
        print(f"❌ 임포트 오류: {e}")
        return False


def test_config():
    """설정 테스트"""
    print("\n" + "=" * 70)
    print("🧪 설정 테스트")
    print("=" * 70)

    from src.utils.config import AppConfig

    config = AppConfig()

    print("🔧 AudioConfig 설정:")
    print(f"   Whisper 모델: {config.audio.whisper_model}")
    print(f"   Whisper 디바이스: {config.audio.whisper_device}")
    print(f"   Whisper 언어: {config.audio.whisper_language}")
    print(f"   임계값: {config.audio.threshold_percentile}%")
    print(f"   병합 간격: {config.audio.merge_gap}초")
    print(f"   RMS 가중치: {config.audio.rms_weight}")
    print(f"   Spectral 가중치: {config.audio.spectral_weight}")

    print("\n✅ 설정 테스트 성공!")


if __name__ == "__main__":
    print("🎵 오디오 처리 기능 테스트")
    print("=" * 70)

    # 1. 모듈 임포트 테스트
    if not test_module_imports():
        print("\n❌ 모듈 임포트 실패. 설치를 확인하세요:")
        print("   pip install librosa soundfile openai-whisper")
        sys.exit(1)

    # 2. 설정 테스트
    test_config()

    # 3. 오디오 필터링 테스트
    test_audio_filtering()

    print("\n" + "=" * 70)
    print("✨ 모든 테스트 완료!")
    print("=" * 70)
