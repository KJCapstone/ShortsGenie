"""
Google Gemini API 사용 예제

이 예제는 Google의 Generative AI (Gemini)를 사용하는 방법을 보여줍니다.
"""

import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
load_dotenv()

try:
    import google.generativeai as genai
except ImportError:
    print("❌ google-generativeai가 설치되지 않았습니다.")
    print("   다음 명령어로 설치하세요: pip install google-generativeai")
    sys.exit(1)


def setup_gemini():
    """Gemini API 설정"""
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or api_key == "your_google_api_key_here":
        print("❌ GOOGLE_API_KEY가 설정되지 않았습니다.")
        print("\n설정 방법:")
        print("1. .env.example 파일을 .env로 복사")
        print("   cp .env.example .env")
        print("\n2. https://aistudio.google.com/apikey 에서 API 키 발급")
        print("\n3. .env 파일에 API 키 입력")
        print("   GOOGLE_API_KEY=your_actual_api_key_here")
        sys.exit(1)

    genai.configure(api_key=api_key)
    return genai


def example1_simple_text():
    """예제 1: 간단한 텍스트 생성"""
    print("\n" + "=" * 60)
    print("예제 1: 간단한 텍스트 생성")
    print("=" * 60)

    genai = setup_gemini()
    model = genai.GenerativeModel('gemini-pro')

    prompt = "축구 경기 하이라이트 영상에 어울리는 10초 분량의 자막 3가지를 추천해주세요."

    print(f"\n프롬프트: {prompt}\n")

    response = model.generate_content(prompt)
    print("응답:")
    print(response.text)


def example2_video_analysis():
    """예제 2: 비디오 분석을 위한 프롬프트 생성"""
    print("\n" + "=" * 60)
    print("예제 2: 비디오 분석을 위한 프롬프트 생성")
    print("=" * 60)

    genai = setup_gemini()
    model = genai.GenerativeModel('gemini-pro')

    # 비디오 타임스탬프와 이벤트 정보 (실제로는 탐지 시스템에서 가져옴)
    video_info = {
        "duration": "2:15",
        "events": [
            {"time": "0:05", "type": "pass", "confidence": 0.92},
            {"time": "0:42", "type": "shot", "confidence": 0.88},
            {"time": "1:23", "type": "goal", "confidence": 0.95},
            {"time": "1:45", "type": "celebration", "confidence": 0.90},
        ]
    }

    prompt = f"""
다음은 {video_info['duration']} 길이의 축구 경기 영상에서 탐지된 이벤트입니다:

"""
    for event in video_info['events']:
        prompt += f"- {event['time']}: {event['type']} (신뢰도: {event['confidence']})\n"

    prompt += """
이 이벤트들을 기반으로:
1. 가장 흥미진진한 15초 하이라이트 구간을 추천해주세요
2. 각 이벤트에 어울리는 짧은 자막(5단어 이내)을 제안해주세요
3. 전체 하이라이트의 제목을 추천해주세요
"""

    print(f"\n프롬프트:\n{prompt}\n")

    response = model.generate_content(prompt)
    print("응답:")
    print(response.text)


def example3_caption_generation():
    """예제 3: 타임스탬프별 자막 생성"""
    print("\n" + "=" * 60)
    print("예제 3: 타임스탬프별 자막 생성")
    print("=" * 60)

    genai = setup_gemini()
    model = genai.GenerativeModel('gemini-pro')

    segments = [
        {"start": 0, "end": 5, "description": "패스 플레이"},
        {"start": 5, "end": 10, "description": "골 찬스"},
        {"start": 10, "end": 15, "description": "골 세리머니"},
    ]

    prompt = """
다음 축구 하이라이트 영상 구간에 어울리는 역동적이고 임팩트 있는 자막을 생성해주세요.
각 자막은 3-5단어로, 소셜미디어에 최적화된 형태로 작성해주세요.

"""

    for i, seg in enumerate(segments, 1):
        prompt += f"{i}. {seg['start']}-{seg['end']}초: {seg['description']}\n"

    prompt += "\n각 구간마다 2가지 옵션을 제시해주세요."

    print(f"\n프롬프트:\n{prompt}\n")

    response = model.generate_content(prompt)
    print("응답:")
    print(response.text)


def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("Google Gemini API 사용 예제")
    print("=" * 60)

    try:
        # 예제 1만 실행 (다른 예제는 필요시 주석 해제)
        example1_simple_text()

        # 다른 예제들을 실행하려면 아래 주석을 해제하세요
        # example2_video_analysis()
        # example3_caption_generation()

        print("\n" + "=" * 60)
        print("모든 예제 완료!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
