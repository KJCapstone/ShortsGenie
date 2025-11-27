"""
OpenAI Whisper 기반 음성-텍스트 변환

CUDA, CPU 지원 (MPS는 현재 미지원)
"""

import whisper
import torch
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json


class WhisperTranscriber:
    """
    OpenAI Whisper를 사용한 음성-텍스트 변환

    CUDA, CPU 자동 감지 및 최적화
    주의: MPS(Apple Silicon)는 현재 Whisper와 호환성 문제로 CPU 사용

    Args:
        model_size: Whisper 모델 크기
            - tiny: 39M params (가장 빠름, 낮은 정확도)
            - base: 74M params (빠름, 적절한 정확도) ★ 추천
            - small: 244M params (중간 속도, 높은 정확도)
            - medium: 769M params (느림, 매우 높은 정확도)
            - large: 1550M params (매우 느림, 최고 정확도)
        device: 디바이스 ('auto', 'cuda', 'cpu')
        language: 언어 코드 (None=자동감지, 'ko'=한국어, 'en'=영어)
        verbose: 진행 상황 출력 여부
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        language: Optional[str] = None,
        verbose: bool = True
    ):
        self.model_size = model_size
        # "auto"를 None으로 변환 (Whisper는 None일 때 자동 감지)
        self.language = None if language == "auto" else language
        self.verbose = verbose

        # 디바이스 자동 감지
        if device == "auto":
            # Whisper는 현재 MPS를 완전히 지원하지 않으므로 CPU 사용
            if torch.cuda.is_available():
                self.device = "cuda"  # NVIDIA GPU
            else:
                self.device = "cpu"  # CPU (Apple Silicon 포함)
        else:
            self.device = device

        self._log(f"🔧 Whisper 초기화 중...")
        self._log(f"   모델: {model_size}")
        self._log(f"   디바이스: {self.device}")

        # Whisper 모델 로드
        load_start = time.time()
        self.model = whisper.load_model(model_size, device=self.device)
        load_time = time.time() - load_start

        self._log(f"✅ 모델 로드 완료 ({load_time:.2f}초)")

    def _log(self, message: str):
        """진행 상황 로그 출력"""
        if self.verbose:
            print(message)

    def transcribe(
        self,
        audio_path: str,
        segments: Optional[List[Tuple[float, float]]] = None
    ) -> Dict:
        """
        오디오 파일을 텍스트로 변환

        Args:
            audio_path: 오디오 파일 경로
            segments: 특정 구간만 변환 [(start, end), ...] (None이면 전체)

        Returns:
            Whisper 결과 딕셔너리
            {
                'text': 전체 텍스트,
                'segments': [
                    {
                        'start': 시작 시간(초),
                        'end': 종료 시간(초),
                        'text': 텍스트
                    },
                    ...
                ],
                'language': 감지된 언어
            }
        """
        total_start = time.time()

        self._log("=" * 60)
        self._log("🎙️  음성-텍스트 변환 시작")
        self._log("=" * 60)
        self._log(f"📂 입력 파일: {audio_path}")

        if segments:
            self._log(f"🎯 지정된 구간 수: {len(segments)}개")
            return self._transcribe_segments(audio_path, segments, total_start)
        else:
            self._log("🎯 전체 오디오 변환")
            return self._transcribe_full(audio_path, total_start)

    def _transcribe_full(
        self,
        audio_path: str,
        total_start: float
    ) -> Dict:
        """전체 오디오 변환"""
        self._log("\n🔄 Whisper 변환 중...")
        self._log("   ⏳ 시간이 다소 걸릴 수 있습니다...")

        transcribe_start = time.time()

        # Whisper 실행
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            verbose=False  # Whisper 자체 로그 비활성화
        )

        transcribe_time = time.time() - transcribe_start
        total_time = time.time() - total_start

        # 결과 출력
        self._log(f"\n✅ 변환 완료 ({transcribe_time:.2f}초)")
        self._log(f"   🌍 감지된 언어: {result.get('language', 'unknown')}")
        self._log(f"   📝 세그먼트 수: {len(result['segments'])}개")
        self._log(f"   📄 텍스트 길이: {len(result['text'])}자")

        self._log("\n" + "=" * 60)
        self._log(f"✨ 전체 작업 완료! (총 {total_time:.2f}초)")
        self._log("=" * 60)

        return result

    def _transcribe_segments(
        self,
        audio_path: str,
        segments: List[Tuple[float, float]],
        total_start: float
    ) -> Dict:
        """
        특정 구간만 변환 (하이라이트 필터링 후)

        FFmpeg로 임시 파일을 만들지 않고 Whisper의 타임스탬프 기능 활용
        """
        self._log("\n🔄 [1/2] 전체 오디오 변환 중...")
        self._log("   ⏳ 시간이 다소 걸릴 수 있습니다...")

        transcribe_start = time.time()

        # 전체 오디오 변환 (타임스탬프 포함)
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            verbose=False
        )

        transcribe_time = time.time() - transcribe_start
        self._log(f"✅ [1/2] 변환 완료 ({transcribe_time:.2f}초)")
        self._log(f"   📝 전체 세그먼트 수: {len(result['segments'])}개")

        # 지정된 구간에 해당하는 세그먼트만 필터링
        self._log("\n🔄 [2/2] 하이라이트 구간 필터링 중...")
        filter_start = time.time()

        filtered_segments = []
        for whisper_seg in result['segments']:
            seg_start = whisper_seg['start']
            seg_end = whisper_seg['end']

            # 이 세그먼트가 하이라이트 구간과 겹치는지 확인
            for highlight_start, highlight_end in segments:
                # 겹침 여부 확인
                if not (seg_end < highlight_start or seg_start > highlight_end):
                    filtered_segments.append(whisper_seg)
                    break

        # 필터링된 텍스트 재구성
        filtered_text = " ".join(seg['text'].strip() for seg in filtered_segments)

        filter_time = time.time() - filter_start
        total_time = time.time() - total_start

        self._log(f"✅ [2/2] 필터링 완료 ({filter_time:.2f}초)")
        self._log(f"   📝 필터링된 세그먼트: {len(filtered_segments)}개")
        self._log(f"   📄 텍스트 길이: {len(filtered_text)}자")

        reduction_rate = (1 - len(filtered_segments) / len(result['segments'])) * 100
        self._log(f"\n📊 필터링 결과:")
        self._log(f"   📉 세그먼트 감소율: {reduction_rate:.1f}%")

        self._log("\n" + "=" * 60)
        self._log(f"✨ 전체 작업 완료! (총 {total_time:.2f}초)")
        self._log("=" * 60)

        # 필터링된 결과 반환
        return {
            'text': filtered_text,
            'segments': filtered_segments,
            'language': result.get('language', 'unknown'),
            'original_segments_count': len(result['segments']),
            'filtered_segments_count': len(filtered_segments)
        }

    def save_transcript(
        self,
        result: Dict,
        output_path: str,
        format: str = "txt"
    ):
        """
        변환 결과 저장

        Args:
            result: transcribe() 결과
            output_path: 출력 파일 경로
            format: 저장 형식 ('txt', 'json', 'srt')
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "txt":
            # 텍스트 파일
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['text'])

        elif format == "json":
            # JSON 파일 (전체 정보 포함)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        elif format == "srt":
            # SRT 자막 파일
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(result['segments'], 1):
                    start_time = self._format_timestamp(seg['start'])
                    end_time = self._format_timestamp(seg['end'])
                    text = seg['text'].strip()

                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")

        else:
            raise ValueError(f"지원하지 않는 형식: {format}")

        self._log(f"💾 저장 완료: {output_file}")

    def _format_timestamp(self, seconds: float) -> str:
        """SRT 타임스탬프 형식으로 변환 (00:00:00,000)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


if __name__ == "__main__":
    # 간단한 테스트
    import sys

    if len(sys.argv) < 2:
        print("사용법: python whisper_transcriber.py <audio_file> [output.txt]")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/transcript.txt"

    # Whisper 변환
    transcriber = WhisperTranscriber(
        model_size="base",
        device="auto",
        language="ko"
    )

    result = transcriber.transcribe(audio_path)

    # 결과 저장
    transcriber.save_transcript(result, output_path, format="txt")
    transcriber.save_transcript(
        result,
        output_path.replace('.txt', '.json'),
        format="json"
    )

    print(f"\n📄 변환된 텍스트:")
    print(result['text'][:500])  # 처음 500자만 출력
    if len(result['text']) > 500:
        print("...")
