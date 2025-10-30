"""
오디오 하이라이트 필터링

RMS 에너지와 Spectral 특성을 결합하여 하이라이트 구간 추출
"""

import librosa
import numpy as np
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import time

from .audio_analyzer import (
    compute_rms_energy,
    compute_spectral_features,
    compute_combined_score,
    merge_segments,
    frames_to_time_segments
)


class AudioHighlightFilter:
    """
    오디오 분석을 통한 하이라이트 구간 필터링

    RMS 에너지(음량)와 Spectral Centroid(주파수)를 결합하여
    경기 하이라이트 구간을 자동으로 추출합니다.

    Args:
        threshold_percentile: 상위 N% 구간을 하이라이트로 선택 (default: 60)
        merge_gap: 가까운 구간을 병합할 최대 간격(초) (default: 2.0)
        segment_duration: 각 하이라이트 구간 길이(초) (default: 5.0)
        rms_weight: RMS 가중치 (default: 0.7)
        spectral_weight: Spectral 가중치 (default: 0.3)
        verbose: 진행 상황 출력 여부 (default: True)
    """

    def __init__(
        self,
        threshold_percentile: float = 60,
        merge_gap: float = 2.0,
        segment_duration: float = 5.0,
        rms_weight: float = 0.7,
        spectral_weight: float = 0.3,
        verbose: bool = True
    ):
        self.threshold_percentile = threshold_percentile
        self.merge_gap = merge_gap
        self.segment_duration = segment_duration
        self.rms_weight = rms_weight
        self.spectral_weight = spectral_weight
        self.verbose = verbose

    def _log(self, message: str):
        """진행 상황 로그 출력"""
        if self.verbose:
            print(message)

    def filter_audio(
        self,
        audio_path: str,
        sr: Optional[int] = None
    ) -> List[Tuple[float, float]]:
        """
        오디오 파일에서 하이라이트 구간 추출

        Args:
            audio_path: 오디오 파일 경로
            sr: 샘플링 레이트 (None이면 원본 유지)

        Returns:
            [(start, end), ...] 형식의 하이라이트 구간 리스트
        """
        total_start = time.time()

        self._log("=" * 60)
        self._log("🎵 오디오 하이라이트 필터링 시작")
        self._log("=" * 60)
        self._log(f"📂 입력 파일: {audio_path}")

        # Step 1: 오디오 로드
        step_start = time.time()
        self._log("\n🔄 [1/4] 오디오 로드 중...")

        y, sr = librosa.load(audio_path, sr=sr)
        duration = len(y) / sr

        step_time = time.time() - step_start
        self._log(f"✅ [1/4] 오디오 로드 완료 ({step_time:.2f}초)")
        self._log(f"   📊 샘플링 레이트: {sr} Hz")
        self._log(f"   ⏱️  총 길이: {duration:.2f}초 ({duration/60:.1f}분)")

        # Step 2: RMS 에너지 분석
        step_start = time.time()
        self._log("\n🔄 [2/4] RMS 에너지 분석 중...")

        rms, times = compute_rms_energy(y, sr)

        step_time = time.time() - step_start
        self._log(f"✅ [2/4] RMS 분석 완료 ({step_time:.2f}초)")
        self._log(f"   📈 RMS 평균: {np.mean(rms):.4f}")
        self._log(f"   📈 RMS 최대: {np.max(rms):.4f}")

        # Step 3: Spectral 특성 분석
        step_start = time.time()
        self._log("\n🔄 [3/4] Spectral 특성 분석 중...")

        centroid, rolloff, zcr = compute_spectral_features(y, sr)

        step_time = time.time() - step_start
        self._log(f"✅ [3/4] Spectral 분석 완료 ({step_time:.2f}초)")
        self._log(f"   🎼 Centroid 평균: {np.mean(centroid):.2f} Hz")
        self._log(f"   🎼 Rolloff 평균: {np.mean(rolloff):.2f} Hz")

        # Step 4: 하이라이트 구간 추출
        step_start = time.time()
        self._log("\n🔄 [4/4] 하이라이트 구간 추출 중...")

        # 종합 점수 계산
        combined_score = compute_combined_score(
            rms, centroid,
            self.rms_weight, self.spectral_weight
        )

        # 임계값 계산 (상위 N% 구간)
        threshold = np.percentile(combined_score, self.threshold_percentile)

        # 임계값 이상인 프레임 찾기
        highlight_frames = np.where(combined_score >= threshold)[0]

        if len(highlight_frames) == 0:
            self._log("⚠️  경고: 하이라이트 구간을 찾지 못했습니다.")
            self._log("   💡 threshold_percentile 값을 낮춰보세요.")
            return []

        # 프레임을 시간 구간으로 변환
        segments = frames_to_time_segments(
            highlight_frames,
            sr=sr,
            hop_length=512,
            segment_duration=self.segment_duration
        )

        # 가까운 구간 병합
        merged_segments = merge_segments(segments, gap=self.merge_gap)

        # 전체 길이를 넘지 않도록 제한
        final_segments = []
        for start, end in merged_segments:
            end = min(end, duration)
            if start < duration:
                final_segments.append((start, end))

        step_time = time.time() - step_start
        self._log(f"✅ [4/4] 구간 추출 완료 ({step_time:.2f}초)")
        self._log(f"   🎯 추출된 구간 수: {len(final_segments)}개")

        # 통계 출력
        total_highlight_duration = sum(end - start for start, end in final_segments)
        reduction_rate = (1 - total_highlight_duration / duration) * 100

        self._log(f"\n📊 필터링 결과:")
        self._log(f"   ⏱️  원본 길이: {duration:.1f}초")
        self._log(f"   ⏱️  필터링 후: {total_highlight_duration:.1f}초")
        self._log(f"   📉 감소율: {reduction_rate:.1f}%")

        total_time = time.time() - total_start
        self._log("\n" + "=" * 60)
        self._log(f"✨ 전체 작업 완료! (총 {total_time:.2f}초)")
        self._log("=" * 60)

        return final_segments

    def extract_audio_segments(
        self,
        audio_path: str,
        segments: List[Tuple[float, float]],
        output_dir: str = "output/audio_segments"
    ) -> List[str]:
        """
        FFmpeg를 사용하여 하이라이트 구간만 추출

        Args:
            audio_path: 원본 오디오 파일 경로
            segments: [(start, end), ...] 형식의 구간 리스트
            output_dir: 출력 디렉토리

        Returns:
            추출된 오디오 파일 경로 리스트
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self._log(f"\n🎬 오디오 구간 추출 시작...")
        self._log(f"📂 출력 디렉토리: {output_dir}")

        output_files = []

        for i, (start, end) in enumerate(segments, 1):
            output_file = output_path / f"segment_{i:03d}.mp3"

            self._log(f"   [{i}/{len(segments)}] {start:.1f}초 ~ {end:.1f}초 추출 중...")

            # FFmpeg 명령어
            cmd = [
                "ffmpeg",
                "-i", str(audio_path),
                "-ss", str(start),
                "-to", str(end),
                "-c:a", "libmp3lame",  # MP3 인코더
                "-q:a", "2",  # 고품질
                "-y",  # 덮어쓰기
                str(output_file)
            ]

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                output_files.append(str(output_file))
            except subprocess.CalledProcessError as e:
                self._log(f"❌ 오류 발생: {e.stderr}")
                continue

        self._log(f"✅ 총 {len(output_files)}개 구간 추출 완료!")

        return output_files


def filter_highlight_segments(
    audio_path: str,
    threshold_percentile: float = 60,
    merge_gap: float = 2.0,
    segment_duration: float = 5.0,
    verbose: bool = True
) -> List[Tuple[float, float]]:
    """
    편의 함수: 오디오 하이라이트 구간 추출

    Args:
        audio_path: 오디오 파일 경로
        threshold_percentile: 상위 N% 구간 선택 (default: 60)
        merge_gap: 구간 병합 최대 간격(초) (default: 2.0)
        segment_duration: 각 구간 길이(초) (default: 5.0)
        verbose: 진행 상황 출력 여부 (default: True)

    Returns:
        [(start, end), ...] 형식의 하이라이트 구간 리스트
    """
    filter = AudioHighlightFilter(
        threshold_percentile=threshold_percentile,
        merge_gap=merge_gap,
        segment_duration=segment_duration,
        verbose=verbose
    )

    return filter.filter_audio(audio_path)


if __name__ == "__main__":
    # 간단한 테스트
    import sys

    if len(sys.argv) < 2:
        print("사용법: python highlight_filter.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    # 하이라이트 구간 추출
    segments = filter_highlight_segments(audio_path)

    print(f"\n추출된 하이라이트 구간:")
    for i, (start, end) in enumerate(segments, 1):
        print(f"  {i}. {start:.1f}초 ~ {end:.1f}초 (길이: {end-start:.1f}초)")
