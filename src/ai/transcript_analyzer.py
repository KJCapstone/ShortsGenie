"""
경기 중계 텍스트를 분석하여 하이라이트를 추출하는 모듈
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai


class TranscriptAnalyzer:
    """중계 텍스트 분석기"""

    def __init__(self, api_key: str = None, verbose: bool = True):
        """
        초기화

        Args:
            api_key: Google API 키 (None이면 환경변수에서 로드)
            verbose: 진행 상황 출력 여부
        """
        self.verbose = verbose

        # .env 파일 로드
        load_dotenv()

        # API 키 설정
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY가 설정되지 않았습니다.\n"
                ".env 파일에 GOOGLE_API_KEY를 설정해주세요.\n"
                "발급: https://aistudio.google.com/apikey"
            )

        # Gemini 설정
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def analyze_transcript(
        self, transcript_path: str, output_json_path: str = None
    ) -> List[Dict]:
        """
        중계 텍스트를 분석하여 하이라이트 추출

        Args:
            transcript_path: 중계 텍스트 파일 경로
            output_json_path: 결과 JSON 저장 경로 (None이면 저장 안함)

        Returns:
            하이라이트 리스트
        """
        total_start_time = time.time()

        # 1단계: 중계 텍스트 읽기
        if self.verbose:
            print("\n" + "=" * 60)
            print("📄 중계 텍스트 분석 시작")
            print("=" * 60)
            print(f"📂 입력 파일: {transcript_path}")

        step_start = time.time()
        transcript_text = self._read_transcript(transcript_path)
        step_time = time.time() - step_start

        if self.verbose:
            text_length = len(transcript_text)
            lines = transcript_text.count('\n') + 1
            print(f"✅ [1/3] 파일 읽기 완료 ({step_time:.2f}초)")
            print(f"   📊 텍스트 길이: {text_length:,}자 ({lines}줄)")

        # 2단계: Gemini로 분석
        if self.verbose:
            print(f"\n🤖 [2/3] AI 분석 중...")
            print(f"   ⏳ Gemini API 호출 중 (약 10-30초 소요)")

        step_start = time.time()
        highlights = self._extract_highlights(transcript_text)
        step_time = time.time() - step_start

        if self.verbose:
            print(f"✅ [2/3] AI 분석 완료 ({step_time:.2f}초)")
            print(f"   📌 추출된 하이라이트: {len(highlights)}개")

        # 3단계: JSON으로 저장
        if output_json_path:
            if self.verbose:
                print(f"\n💾 [3/3] 결과 저장 중...")

            step_start = time.time()
            self._save_json(highlights, output_json_path)
            step_time = time.time() - step_start

            if self.verbose:
                print(f"✅ [3/3] 저장 완료 ({step_time:.2f}초)")
                print(f"   📁 출력 파일: {output_json_path}")

        total_time = time.time() - total_start_time

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"✨ 전체 작업 완료! (총 {total_time:.2f}초)")
            print("=" * 60)

        return highlights

    def _read_transcript(self, file_path: str) -> str:
        """중계 텍스트 파일 읽기"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _extract_highlights(self, transcript: str) -> List[Dict]:
        """Gemini를 사용하여 하이라이트 추출"""

        prompt = f"""
다음은 축구 경기 중계 텍스트입니다. 이 중계 텍스트를 분석하여 주요 하이라이트 장면들을 추출해주세요.

중계 텍스트:
```
{transcript}
```

요구사항:
1. 골, 결정적 기회, 중요한 수비 등 주요 하이라이트 장면만 추출
2. 각 하이라이트는 다음 정보를 포함해야 합니다:
   - start: 시작 시간(초 단위, 소수점 포함)
   - end: 종료 시간(초 단위, 소수점 포함)
   - type: 하이라이트 유형 ("goal", "chance", "save", "foul" 중 하나)
   - description: 한국어로 된 간단한 설명 (50자 이내)

3. 시간 형식:
   - 텍스트에서 시간이 "MM:SS.S" 형식이면 초로 변환 (예: "1:24:30.2" = 5070.2초)
   - start 시간은 하이라이트 시작 시점
   - end 시간은 start + 약 8-10초 (골 세레머니 포함)

4. 출력 형식은 **반드시 JSON 배열**이어야 합니다:
```json
[
  {{
    "start": 5070.2,
    "end": 5079.2,
    "type": "goal",
    "description": "황의조 득점 (1-0). 손흥민의 PK를 무슬레라가 막았으나, 황의조가 리바운드 볼을 밀어 넣어 선제골 기록."
  }}
]
```

**중요**:
- 응답은 JSON 배열만 출력하세요. 다른 텍스트나 설명은 포함하지 마세요.
- 코드 블록(```)을 사용하지 마세요.
- 순수 JSON만 출력하세요.
"""

        # Gemini API 호출
        response = self.model.generate_content(prompt)
        response_text = response.text.strip()

        # JSON 파싱
        highlights = self._parse_response(response_text)

        return highlights

    def _parse_response(self, response_text: str) -> List[Dict]:
        """Gemini 응답을 파싱하여 리스트로 변환"""
        try:
            # 코드 블록 제거 (있을 경우)
            response_text = response_text.strip()
            if response_text.startswith('```'):
                # ```json ... ``` 형식 처리
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1])

            # JSON 파싱
            highlights = json.loads(response_text)

            # 검증
            if not isinstance(highlights, list):
                raise ValueError("응답이 리스트 형식이 아닙니다")

            # 각 항목 검증
            for i, highlight in enumerate(highlights):
                required_fields = ['start', 'end', 'type', 'description']
                for field in required_fields:
                    if field not in highlight:
                        raise ValueError(
                            f"하이라이트 #{i+1}에 '{field}' 필드가 없습니다"
                        )

            return highlights

        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류: {e}")
            print(f"응답 텍스트:\n{response_text}")
            raise ValueError(f"Gemini 응답을 JSON으로 파싱할 수 없습니다: {e}")

    def _save_json(self, highlights: List[Dict], output_path: str) -> None:
        """하이라이트를 JSON 파일로 저장"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(highlights, f, ensure_ascii=False, indent=2)

        print(f"✅ 하이라이트 저장 완료: {output_path}")


def analyze_transcript_file(
    transcript_path: str,
    output_json_path: str = None,
    api_key: str = None
) -> List[Dict]:
    """
    편의 함수: 중계 텍스트 파일 분석

    Args:
        transcript_path: 중계 텍스트 파일 경로
        output_json_path: 결과 JSON 저장 경로
        api_key: Google API 키

    Returns:
        하이라이트 리스트
    """
    analyzer = TranscriptAnalyzer(api_key=api_key)
    return analyzer.analyze_transcript(transcript_path, output_json_path)


if __name__ == "__main__":
    # 테스트 실행
    import sys

    if len(sys.argv) < 2:
        print("사용법: python transcript_analyzer.py <transcript_file> [output_json]")
        print("\n예시:")
        print("  python transcript_analyzer.py input/match_transcript.txt")
        print("  python transcript_analyzer.py input/match_transcript.txt output/highlights.json")
        sys.exit(1)

    transcript_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        highlights = analyze_transcript_file(transcript_file, output_file)

        print(f"\n✅ 총 {len(highlights)}개의 하이라이트를 추출했습니다:\n")
        for i, h in enumerate(highlights, 1):
            print(f"{i}. [{h['type'].upper()}] {h['start']:.1f}s - {h['end']:.1f}s")
            print(f"   {h['description']}\n")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
