"""
오디오 분석 유틸리티 함수들

librosa를 사용한 RMS 에너지, Spectral 특성 분석
"""

import librosa
import numpy as np
from typing import List, Tuple


def compute_rms_energy(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RMS 에너지 계산 (음량 분석)

    Args:
        y: 오디오 시계열 데이터
        sr: 샘플링 레이트
        frame_length: 프레임 길이
        hop_length: 홉 길이

    Returns:
        (rms_values, times): RMS 값 배열과 시간 배열
    """
    # RMS 에너지 계산
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # 프레임을 시간으로 변환
    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=hop_length
    )

    return rms, times


def compute_spectral_features(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spectral 특성 계산 (주파수 분석)

    고음 함성, 휘슬 등 특정 주파수 패턴 감지

    Args:
        y: 오디오 시계열 데이터
        sr: 샘플링 레이트
        hop_length: 홉 길이

    Returns:
        (centroid, rolloff, zcr): Spectral Centroid, Rolloff, Zero Crossing Rate
    """
    # Spectral Centroid (주파수 중심)
    spectral_centroids = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )[0]

    # Spectral Rolloff (에너지 집중 범위)
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, hop_length=hop_length
    )[0]

    # Zero Crossing Rate (고주파 소리 감지)
    zcr = librosa.feature.zero_crossing_rate(
        y, hop_length=hop_length
    )[0]

    return spectral_centroids, spectral_rolloff, zcr


def merge_segments(
    segments: List[Tuple[float, float]],
    gap: float = 2.0
) -> List[Tuple[float, float]]:
    """
    가까운 구간 병합

    Args:
        segments: [(start, end), ...] 형식의 구간 리스트
        gap: 병합할 최대 간격 (초)

    Returns:
        병합된 구간 리스트
    """
    if not segments:
        return []

    # 시작 시간 기준 정렬
    sorted_segments = sorted(segments, key=lambda x: x[0])

    merged = [sorted_segments[0]]

    for current_start, current_end in sorted_segments[1:]:
        last_start, last_end = merged[-1]

        # 현재 구간이 이전 구간과 가까우면 병합
        if current_start - last_end <= gap:
            # 끝 시간을 더 큰 값으로 업데이트
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # 새 구간 추가
            merged.append((current_start, current_end))

    return merged


def frames_to_time_segments(
    frame_indices: np.ndarray,
    sr: int,
    hop_length: int = 512,
    segment_duration: float = 5.0
) -> List[Tuple[float, float]]:
    """
    프레임 인덱스를 시간 구간으로 변환

    Args:
        frame_indices: 하이라이트 프레임 인덱스 배열
        sr: 샘플링 레이트
        hop_length: 홉 길이
        segment_duration: 각 구간 길이 (초)

    Returns:
        [(start, end), ...] 형식의 구간 리스트
    """
    if len(frame_indices) == 0:
        return []

    # 프레임을 시간으로 변환
    times = librosa.frames_to_time(
        frame_indices,
        sr=sr,
        hop_length=hop_length
    )

    # 각 시간 포인트 주변으로 구간 생성
    segments = []
    for time in times:
        start = max(0, time - segment_duration / 2)
        end = time + segment_duration / 2
        segments.append((start, end))

    return segments


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    신호 정규화 (평균 0, 표준편차 1)

    Args:
        signal: 입력 신호

    Returns:
        정규화된 신호
    """
    mean = np.mean(signal)
    std = np.std(signal)

    if std == 0:
        return signal - mean

    return (signal - mean) / std


def compute_combined_score(
    rms: np.ndarray,
    spectral_centroid: np.ndarray,
    rms_weight: float = 0.7,
    spectral_weight: float = 0.3
) -> np.ndarray:
    """
    RMS와 Spectral Centroid를 결합한 종합 점수 계산

    Args:
        rms: RMS 에너지 값
        spectral_centroid: Spectral Centroid 값
        rms_weight: RMS 가중치
        spectral_weight: Spectral 가중치

    Returns:
        정규화된 종합 점수
    """
    # 각 신호 정규화
    rms_norm = normalize_signal(rms)
    spectral_norm = normalize_signal(spectral_centroid)

    # 가중 평균
    combined = rms_weight * rms_norm + spectral_weight * spectral_norm

    return combined


if __name__ == "__main__":
    # 간단한 테스트
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("사용법: python audio_analyzer.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    print(f"오디오 파일 분석: {audio_path}")

    # 오디오 로드
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr

    print(f"샘플링 레이트: {sr} Hz")
    print(f"길이: {duration:.2f}초")

    # RMS 분석
    rms, times = compute_rms_energy(y, sr)
    print(f"\nRMS 에너지:")
    print(f"  평균: {np.mean(rms):.4f}")
    print(f"  최대: {np.max(rms):.4f}")
    print(f"  최소: {np.min(rms):.4f}")

    # Spectral 분석
    centroid, rolloff, zcr = compute_spectral_features(y, sr)
    print(f"\nSpectral 특성:")
    print(f"  Centroid 평균: {np.mean(centroid):.2f} Hz")
    print(f"  Rolloff 평균: {np.mean(rolloff):.2f} Hz")
    print(f"  ZCR 평균: {np.mean(zcr):.4f}")

    # 종합 점수
    combined = compute_combined_score(rms, centroid)
    print(f"\n종합 점수:")
    print(f"  평균: {np.mean(combined):.4f}")
    print(f"  표준편차: {np.std(combined):.4f}")
