"""
중계 텍스트 분석 예제

이 예제는 축구 경기 중계 텍스트를 분석하여 하이라이트를 추출하는 방법을 보여줍니다.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.transcript_analyzer import analyze_transcript_file


def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("축구 경기 중계 텍스트 분석 예제")
    print("=" * 60)

    # 중계 텍스트 파일 경로
    transcript_file = "input/match_transcript.txt"
    output_file = "output/highlights.json"

    # 파일 존재 확인
    if not Path(transcript_file).exists():
        print(f"\n❌ 중계 텍스트 파일을 찾을 수 없습니다: {transcript_file}")
        print("\n다음과 같이 파일을 생성하세요:")
        print(f"  1. {transcript_file} 파일 생성")
        print("  2. 경기 중계 텍스트를 파일에 입력")
        print("\n예시:")
        print("```")
        print("1:24:30.2")
        print("손흥민의 PK. 무슬레라가 막았지만 황의조가 리바운드 골!")
        print("1-0 대한민국 선제골!")
        print("")
        print("1:31:17.2")
        print("우루과이의 역습! 토레이라가 패스, 베시노 골!")
        print("1-1 동점골!")
        print("```")
        return

    try:
        print(f"\n📄 중계 텍스트 분석 중: {transcript_file}")
        print("   (Gemini API 호출 중... 약 10-30초 소요)")

        # 분석 실행
        highlights = analyze_transcript_file(
            transcript_path=transcript_file,
            output_json_path=output_file
        )

        # 결과 출력
        print(f"\n✅ 총 {len(highlights)}개의 하이라이트를 추출했습니다:\n")

        for i, highlight in enumerate(highlights, 1):
            print(f"{i}. [{highlight['type'].upper()}] "
                  f"{highlight['start']:.1f}s - {highlight['end']:.1f}s")
            print(f"   📝 {highlight['description']}")
            print()

        print("=" * 60)
        print(f"✅ 결과 저장 완료: {output_file}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
