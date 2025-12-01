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
Analyze the following soccer match commentary and extract ONLY goals and key chances.

Commentary text:
```
{transcript}
```

Requirements:
1. Extract ONLY two types of highlights:
   - "goal": Actual goals scored
   - "chance": Clear goal-scoring opportunities (shots on target, near misses)

2. **Create SEPARATE highlights for EACH distinct event**:
   - Do NOT merge multiple events into one giant highlight
   - Each goal or chance should be its own highlight
   - Maximum duration per highlight: **40 seconds**
   - Minimum duration per highlight: 15 seconds
   - Recommended: 20-30 seconds per highlight

3. Timing is CRITICAL:
   - Include sufficient context BEFORE the moment (build-up play, 3-5 seconds)
   - Include sufficient context AFTER the moment (celebrations/replays, 3-5 seconds)
   - Analyze the commentary to determine appropriate padding
   - **Do NOT create highlights longer than 40 seconds**

4. **Sort highlights by TIME (chronological order)**:
   - ALWAYS return highlights in chronological order (earliest first)
   - Never put a later event before an earlier event

5. Merge adjacent highlights ONLY if necessary:
   - If two highlights are within 3 seconds of each other, merge them into ONE highlight
   - Example: If highlight A ends at 350s and highlight B starts at 352s → merge into single highlight from A.start to B.end
   - **After merging, if duration exceeds 40s, split into separate highlights**

6. Time format:
   - Convert "[MM:SS.S]" or "MM:SS" format to seconds (e.g., "[1:24.5]" = 84.5 seconds)
   - start: beginning of context (including build-up)
   - end: end of context (including celebration/replay)

7. Output format must be a JSON array:
```json
[
  {{
    "start": 24.0,
    "end": 52.0,
    "type": "goal",
    "description": "마트타 헤더골 (1-0). 라마스 골키퍼를 상대로 강력한 헤더로 선제골 기록."
  }},
  {{
    "start": 112.0,
    "end": 126.0,
    "type": "goal",
    "description": "지극해지 프리킥 원더골 (2-0). 환상적인 각도에서 골망을 흔들며 추가골 기록."
  }}
]
```

**Important**:
- Output ONLY the JSON array. No other text or explanations.
- Do NOT use code blocks (```).
- Output pure JSON only.
- Write descriptions in Korean.
- **CRITICAL**: Return highlights in CHRONOLOGICAL ORDER (sorted by start time)
- **CRITICAL**: Each highlight must be 15-40 seconds. Do NOT create highlights longer than 40 seconds.
- **CRITICAL**: Create SEPARATE highlights for each distinct event (goal/chance).
"""

        # Gemini API 호출
        response = self.model.generate_content(prompt)
        response_text = response.text.strip()

        # JSON 파싱
        highlights = self._parse_response(response_text)

        # 인접 하이라이트 병합 (Gemini가 못할 경우 대비)
        highlights = self._merge_adjacent_highlights(highlights)

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

    def _merge_adjacent_highlights(self, highlights: List[Dict], gap_threshold: float = 3.0) -> List[Dict]:
        """인접한 하이라이트를 병합

        Args:
            highlights: 하이라이트 리스트
            gap_threshold: 병합 기준 간격(초) - 3초 이하면 병합

        Returns:
            병합된 하이라이트 리스트
        """
        if not highlights:
            return highlights

        # 시작 시간 기준 정렬
        sorted_highlights = sorted(highlights, key=lambda x: x['start'])

        merged = []
        current = sorted_highlights[0].copy()

        for next_h in sorted_highlights[1:]:
            gap = next_h['start'] - current['end']

            if gap <= gap_threshold:
                # 병합: end 시간 연장, description 합침
                current['end'] = next_h['end']
                # 같은 타입이면 description 합침
                if current['type'] == next_h['type']:
                    current['description'] += f" / {next_h['description']}"
                else:
                    # 다른 타입이면 "goal + chance" 형식으로
                    current['type'] = f"{current['type']}+{next_h['type']}"
                    current['description'] += f" / {next_h['description']}"
            else:
                # 간격이 크면 별도 하이라이트로 유지
                merged.append(current)
                current = next_h.copy()

        # 마지막 하이라이트 추가
        merged.append(current)

        return merged

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
