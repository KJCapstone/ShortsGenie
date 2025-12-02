"""
Groq Whisper API를 사용한 음성-텍스트 변환 (클라우드 기반)

- 216-299배 실시간 속도
- GPU 불필요
- 비용: $0.02-0.111/시간
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from groq import Groq
import time


class GroqTranscriber:
    """
    Groq Whisper API를 사용한 음성-텍스트 변환

    OpenAI Whisper와 동일한 인터페이스를 제공하여 drop-in replacement 가능

    Args:
        api_key: Groq API 키 (None이면 환경변수에서 로드)
        model: 모델 선택
            - "whisper-large-v3-turbo": 가장 빠름 (216x), $0.04/시간 ★ 추천
            - "whisper-large-v3": 더 정확 (299x), $0.111/시간
            - "distil-whisper-large-v3-en": 영어 전용, $0.02/시간
        language: 언어 코드 (None=자동감지, 'ko'=한국어, 'en'=영어)
        verbose: 진행 상황 출력 여부
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-large-v3-turbo",
        language: Optional[str] = None,
        verbose: bool = True
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please set it in .env file or pass as argument.\n"
                "Get your API key from: https://console.groq.com"
            )

        self.model = model
        # "auto"를 None으로 변환 (Groq는 None일 때 자동 감지)
        self.language = None if language == "auto" else language
        self.verbose = verbose

        try:
            self.client = Groq(api_key=self.api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize Groq client: {e}")

        self._log(f"🔧 Groq Transcriber 초기화")
        self._log(f"   모델: {model}")
        self._log(f"   언어: {language or '자동 감지'}")

    def _log(self, message: str):
        """진행 상황 로그 출력"""
        if self.verbose:
            print(message)

    def _compress_to_mp3(self, audio_path: str, target_bitrate: str = "64k") -> str:
        """
        오디오를 MP3로 압축 (25MB 제한 대응)

        Args:
            audio_path: 원본 오디오 파일 경로
            target_bitrate: 목표 비트레이트 (64k = 약 480KB/분, 128k = 약 960KB/분)

        Returns:
            압축된 MP3 파일 경로 (임시 파일)
        """
        # 임시 파일 생성
        temp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_mp3_path = temp_mp3.name
        temp_mp3.close()

        self._log(f"   📦 MP3 압축 중... (비트레이트: {target_bitrate})")

        try:
            # FFmpeg로 MP3 변환 (단일 채널, 16kHz - 음성 인식 최적)
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-vn",  # 비디오 제거
                "-ac", "1",  # 모노 (단일 채널)
                "-ar", "16000",  # 16kHz 샘플링 (Whisper 최적)
                "-b:a", target_bitrate,  # 비트레이트
                "-y",  # 덮어쓰기
                temp_mp3_path
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300  # 5분 타임아웃
            )

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg compression failed: {result.stderr.decode()}")

            # 압축된 파일 크기 확인
            compressed_size_mb = Path(temp_mp3_path).stat().st_size / (1024 * 1024)
            self._log(f"   ✅ 압축 완료: {compressed_size_mb:.1f}MB")

            return temp_mp3_path

        except Exception as e:
            # 실패 시 임시 파일 삭제
            if os.path.exists(temp_mp3_path):
                os.unlink(temp_mp3_path)
            raise RuntimeError(f"MP3 압축 실패: {e}")

    def transcribe(self, audio_path: str) -> Dict:
        """
        오디오 파일을 텍스트로 변환 (OpenAI Whisper 호환 인터페이스)

        Args:
            audio_path: 오디오 파일 경로 (mp3, mp4, wav, m4a 등)

        Returns:
            OpenAI Whisper 호환 딕셔너리:
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

        Raises:
            FileNotFoundError: 오디오 파일이 없을 때
            ValueError: 파일 크기가 25MB 초과일 때
            Exception: Groq API 호출 실패 시
        """
        audio_file = Path(audio_path)

        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 파일 크기 체크 (25MB 제한)
        file_size_mb = audio_file.stat().st_size / (1024 * 1024)

        total_start = time.time()

        self._log("=" * 60)
        self._log("🎙️  Groq Whisper API 변환 시작")
        self._log("=" * 60)
        self._log(f"📂 입력 파일: {audio_file.name}")
        self._log(f"📦 원본 크기: {file_size_mb:.1f}MB")

        # 25MB 초과 시 자동 압축
        compressed_file = None
        if file_size_mb > 25:
            self._log(f"⚠️  파일이 25MB를 초과합니다. MP3로 압축합니다...")

            # 다단계 압축 시도 (64k → 48k → 32k)
            bitrates = ["64k", "48k", "32k"]
            audio_to_send = None

            for bitrate in bitrates:
                try:
                    if compressed_file and os.path.exists(compressed_file):
                        os.unlink(compressed_file)

                    compressed_file = self._compress_to_mp3(audio_path, target_bitrate=bitrate)
                    final_size_mb = Path(compressed_file).stat().st_size / (1024 * 1024)

                    if final_size_mb <= 25:
                        audio_to_send = compressed_file
                        self._log(f"✅ 압축 성공: {final_size_mb:.1f}MB (비트레이트: {bitrate})")
                        break
                    else:
                        self._log(f"   {bitrate} 압축: {final_size_mb:.1f}MB (여전히 25MB 초과, 재시도...)")

                except Exception as e:
                    self._log(f"   {bitrate} 압축 실패: {e}")
                    continue

            if not audio_to_send:
                # 모든 압축 시도 실패
                if compressed_file and os.path.exists(compressed_file):
                    os.unlink(compressed_file)
                raise ValueError(
                    f"파일 크기({file_size_mb:.1f}MB)가 너무 큽니다.\n"
                    "압축 후에도 25MB를 초과합니다.\n\n"
                    "해결 방법:\n"
                    "1. 로컬 Whisper 사용 (무료, 제한 없음)\n"
                    "2. 더 짧은 영상 선택 (90분 이하 권장)"
                )
        else:
            audio_to_send = audio_path
            self._log(f"✅ 파일 크기 적합 (25MB 이하)")

        self._log(f"⚡ 예상 시간: 수 초 이내...")

        transcribe_start = time.time()

        try:
            # Groq API 호출
            with open(audio_to_send, "rb") as file:
                file_name = Path(audio_to_send).name
                transcription = self.client.audio.transcriptions.create(
                    file=(file_name, file.read()),
                    model=self.model,
                    language=self.language,
                    response_format="verbose_json",  # 세그먼트 정보 포함
                    timestamp_granularities=["segment"]  # 세그먼트 타임스탬프
                )
        except Exception as e:
            self._log(f"\n❌ Groq API 호출 실패: {e}")
            raise Exception(f"Groq API error: {e}")
        finally:
            # 압축 파일 정리
            if compressed_file and os.path.exists(compressed_file):
                os.unlink(compressed_file)
                self._log(f"🗑️  임시 파일 삭제: {Path(compressed_file).name}")

        transcribe_time = time.time() - transcribe_start
        total_time = time.time() - total_start

        # Groq 응답을 OpenAI Whisper 형식으로 변환
        result = self._convert_to_whisper_format(transcription)

        # 결과 출력
        self._log(f"\n✅ 변환 완료 ({transcribe_time:.2f}초)")
        self._log(f"   🌍 감지된 언어: {result.get('language', 'unknown')}")
        self._log(f"   📝 세그먼트 수: {len(result['segments'])}개")
        self._log(f"   📄 텍스트 길이: {len(result['text'])}자")

        # 속도 계산 (오디오 길이 대비)
        if result['segments']:
            audio_duration = result['segments'][-1]['end']
            speed_factor = audio_duration / transcribe_time if transcribe_time > 0 else 0
            self._log(f"   🚀 처리 속도: {speed_factor:.1f}x 실시간")

        self._log("\n" + "=" * 60)
        self._log(f"✨ 전체 작업 완료! (총 {total_time:.2f}초)")
        self._log("=" * 60)

        return result

    def _convert_to_whisper_format(self, groq_response) -> Dict:
        """Groq API 응답을 OpenAI Whisper 형식으로 변환"""

        # Groq의 segments를 OpenAI Whisper 형식으로 변환
        segments = []
        if hasattr(groq_response, 'segments') and groq_response.segments:
            for seg in groq_response.segments:
                segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text']
                })

        return {
            'text': groq_response.text,
            'segments': segments,
            'language': getattr(groq_response, 'language', 'unknown')
        }

    def save_transcript(
        self,
        result: Dict,
        output_path: str,
        format: str = "txt"
    ):
        """
        변환 결과 저장 (OpenAI Whisper와 동일한 인터페이스)

        Args:
            result: transcribe() 결과
            output_path: 출력 파일 경로
            format: 저장 형식 ('txt', 'json', 'srt')
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "txt":
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['text'])

        elif format == "json":
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        elif format == "srt":
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
        print("사용법: python groq_transcriber.py <audio_file> [output.txt]")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/transcript.txt"

    # Groq 변환
    transcriber = GroqTranscriber(
        model="whisper-large-v3-turbo",
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
    print(result['text'][:500])
    if len(result['text']) > 500:
        print("...")
