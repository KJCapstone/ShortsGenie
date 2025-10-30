"""
중계 텍스트 분석 성능 테스트

다양한 길이의 텍스트로 처리 속도를 측정합니다.
"""

import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.transcript_analyzer import TranscriptAnalyzer


def create_test_file(length: str, output_path: str) -> int:
    """테스트용 중계 텍스트 생성"""

    # 기본 템플릿 (약 300자)
    base_segment = """1:24:30.2
손흥민의 PK. 무슬레라가 막았지만 황의조가 리바운드 골!
1-0 대한민국 선제골!

1:27:45.8
우루과이의 빠른 역습입니다.
수아레스가 돌파를 시도하지만 김민재가 잘 막아냅니다.

"""

    if length == "short":
        # 짧은 텍스트: 약 300자 (1-2개 하이라이트)
        content = base_segment
    elif length == "medium":
        # 중간 텍스트: 약 1,500자 (5-8개 하이라이트)
        content = base_segment * 5
    elif length == "long":
        # 긴 텍스트: 약 3,000자 (10-15개 하이라이트)
        content = base_segment * 10
    elif length == "very_long":
        # 매우 긴 텍스트: 약 6,000자 (20-30개 하이라이트)
        content = base_segment * 20
    else:
        content = base_segment

    Path(output_path).write_text(content, encoding='utf-8')
    return len(content)


def test_performance():
    """성능 테스트 실행"""

    print("\n" + "=" * 70)
    print("중계 텍스트 분석 성능 테스트")
    print("=" * 70)

    test_cases = [
        ("short", "짧은 텍스트 (300자)"),
        ("medium", "중간 텍스트 (1,500자)"),
        ("long", "긴 텍스트 (3,000자)"),
        ("very_long", "매우 긴 텍스트 (6,000자)"),
    ]

    results = []

    for test_id, description in test_cases:
        print(f"\n{'─' * 70}")
        print(f"🧪 테스트: {description}")
        print(f"{'─' * 70}")

        # 테스트 파일 생성
        test_file = f"input/test_{test_id}.txt"
        text_length = create_test_file(test_id, test_file)

        print(f"📝 생성된 파일: {test_file}")
        print(f"📊 텍스트 길이: {text_length:,}자")

        # 분석 실행
        try:
            start_time = time.time()

            analyzer = TranscriptAnalyzer(verbose=False)
            highlights = analyzer.analyze_transcript(
                transcript_path=test_file,
                output_json_path=None  # 저장 안함
            )

            elapsed_time = time.time() - start_time

            # 결과 저장
            results.append({
                "description": description,
                "text_length": text_length,
                "highlights_count": len(highlights),
                "time": elapsed_time,
                "chars_per_sec": text_length / elapsed_time if elapsed_time > 0 else 0
            })

            print(f"✅ 완료!")
            print(f"   ⏱️  소요 시간: {elapsed_time:.2f}초")
            print(f"   📌 하이라이트: {len(highlights)}개")
            print(f"   ⚡ 처리 속도: {text_length / elapsed_time:.0f}자/초")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            results.append({
                "description": description,
                "text_length": text_length,
                "error": str(e)
            })

    # 결과 요약
    print("\n" + "=" * 70)
    print("📊 성능 테스트 결과 요약")
    print("=" * 70)
    print()
    print(f"{'텍스트 길이':<20} {'하이라이트':<12} {'소요 시간':<12} {'처리 속도':<15}")
    print("─" * 70)

    for result in results:
        if "error" not in result:
            print(f"{result['description']:<20} "
                  f"{result['highlights_count']:>3}개        "
                  f"{result['time']:>6.2f}초      "
                  f"{result['chars_per_sec']:>7.0f}자/초")
        else:
            print(f"{result['description']:<20} 오류 발생")

    print("=" * 70)

    # 평균 계산
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        avg_time = sum(r['time'] for r in valid_results) / len(valid_results)
        avg_speed = sum(r['chars_per_sec'] for r in valid_results) / len(valid_results)

        print(f"\n📈 평균 성능:")
        print(f"   • 평균 소요 시간: {avg_time:.2f}초")
        print(f"   • 평균 처리 속도: {avg_speed:.0f}자/초")

    print("\n💡 결론:")
    print("   • Gemini API 응답 시간은 주로 텍스트 길이에 비례합니다")
    print("   • 네트워크 상태에 따라 시간이 변동될 수 있습니다")
    print("   • 일반적으로 1,000-3,000자 텍스트가 최적입니다")
    print()


if __name__ == "__main__":
    try:
        test_performance()
    except KeyboardInterrupt:
        print("\n\n⚠️  테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
