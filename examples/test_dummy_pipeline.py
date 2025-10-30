"""
더미 데이터로 빠른 파이프라인 테스트

실제 오디오 파일 없이 Gemini LLM 분석 기능만 테스트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.transcript_analyzer import TranscriptAnalyzer


def test_gemini_only():
    """
    Gemini LLM만 테스트 (Step 3만)

    실제 오디오 파일 없이 더미 텍스트로 빠르게 테스트
    """

    print("=" * 70)
    print("🧪 더미 데이터로 Gemini LLM 테스트")
    print("=" * 70)

    # 더미 해설 텍스트 (실제 경기 해설 스타일)
    dummy_transcript = """
[10.5s - 15.5s] 경기 시작되었습니다. 홈팀이 먼저 공을 잡습니다.
[120.5s - 125.5s] 골! 골입니다! 환상적인 슈팅으로 선제골을 터뜨렸습니다!
[130.2s - 135.2s] 대단한 골이었습니다. 리플레이로 다시 보겠습니다.
[258.3s - 263.3s] 위험한 상황입니다. 프리킥 기회를 얻었습니다.
[268.5s - 273.5s] 프리킥입니다. 과연 이 기회를 살릴 수 있을까요?
[450.2s - 455.2s] 아! 아쉽습니다. 골대를 맞고 나갔습니다.
[678.8s - 683.8s] 코너킥입니다. 중요한 세트피스 상황입니다.
[789.1s - 794.1s] 헤딩 슈팅! 골키퍼가 막아냅니다.
[1020.5s - 1025.5s] 후반전이 시작되었습니다. 점수는 1대 0입니다.
[1230.5s - 1235.5s] 골! 골! 추가골입니다! 이제 2대 0으로 앞서갑니다!
[1245.2s - 1250.2s] 완벽한 팀플레이로 만들어낸 골이었습니다.
[1456.2s - 1461.2s] 경기 종료를 앞두고 마지막 찬스 상황입니다.
[1678.5s - 1683.5s] 경기 종료되었습니다. 최종 스코어 2대 0입니다.
"""

    print("📄 더미 해설 텍스트:")
    print("-" * 70)
    print(dummy_transcript)
    print("-" * 70)

    # 임시 파일 저장
    Path("output").mkdir(exist_ok=True)
    dummy_path = "output/dummy_transcript.txt"

    with open(dummy_path, 'w', encoding='utf-8') as f:
        f.write(dummy_transcript)

    print(f"\n💾 임시 파일 저장: {dummy_path}")

    # Gemini 분석
    print("\n🤖 Gemini LLM 분석 중...")
    print("-" * 70)

    analyzer = TranscriptAnalyzer(verbose=True)
    highlights = analyzer.analyze_transcript(dummy_path, "output/dummy_highlights.json")

    # 결과 출력
    print("\n" + "=" * 70)
    print("✅ 테스트 완료!")
    print("=" * 70)

    print(f"\n📊 추출된 하이라이트: {len(highlights)}개")
    print(f"💾 저장 위치: output/dummy_highlights.json")

    if highlights:
        print("\n🎯 하이라이트 상세:")
        print("-" * 70)

        # 타입별 분류
        type_counts = {}
        for h in highlights:
            type_counts[h['type']] = type_counts.get(h['type'], 0) + 1

        print("\n📈 타입별 통계:")
        for htype, count in type_counts.items():
            print(f"   {htype}: {count}개")

        print("\n📝 전체 리스트:")
        for i, h in enumerate(highlights, 1):
            minutes = int(h['start'] // 60)
            seconds = int(h['start'] % 60)
            print(f"\n{i}. [{minutes:02d}:{seconds:02d}] {h['type'].upper()}")
            print(f"   시간: {h['start']:.1f}초 ~ {h['end']:.1f}초")
            print(f"   설명: {h['description']}")

        # 예상 결과와 비교
        print("\n" + "=" * 70)
        print("🎯 예상 하이라이트 vs 실제 추출")
        print("=" * 70)

        expected_highlights = [
            "120.5초: 첫 골 (goal)",
            "258.3초 or 268.5초: 프리킥 (chance)",
            "678.8초: 코너킥 (chance)",
            "1230.5초: 추가골 (goal)",
            "1456.2초: 마지막 찬스 (chance)"
        ]

        print("\n예상 하이라이트:")
        for exp in expected_highlights:
            print(f"   ✓ {exp}")

        print(f"\n실제 추출: {len(highlights)}개")
        if len(highlights) >= 4:
            print("   ✅ 성공: 대부분의 하이라이트를 정확히 추출했습니다!")
        elif len(highlights) >= 2:
            print("   ⚠️  부분 성공: 일부 하이라이트를 놓쳤을 수 있습니다.")
        else:
            print("   ❌ 실패: 하이라이트 추출이 부족합니다.")

    else:
        print("\n❌ 하이라이트를 찾지 못했습니다.")
        print("💡 확인사항:")
        print("   1. .env 파일에 GOOGLE_API_KEY 설정 확인")
        print("   2. Gemini API 키가 유효한지 확인")

    return highlights


def test_custom_transcript():
    """
    사용자 정의 텍스트로 테스트
    """

    print("=" * 70)
    print("✏️  사용자 정의 텍스트 테스트")
    print("=" * 70)

    print("\n텍스트 입력 방법:")
    print("  1. 직접 입력")
    print("  2. 파일에서 읽기")

    choice = input("\n선택 (1-2): ").strip()

    if choice == "1":
        print("\n해설 텍스트를 입력하세요 (Ctrl+D 또는 Ctrl+Z로 종료):")
        print("형식: [시작초s - 종료초s] 텍스트")
        print("예: [120.5s - 125.5s] 골입니다!\n")

        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass

        transcript = "\n".join(lines)

    elif choice == "2":
        file_path = input("파일 경로를 입력하세요: ").strip()
        if not Path(file_path).exists():
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            transcript = f.read()

    else:
        print("❌ 잘못된 선택입니다.")
        return None

    # 임시 파일 저장
    Path("output").mkdir(exist_ok=True)
    custom_path = "output/custom_transcript.txt"

    with open(custom_path, 'w', encoding='utf-8') as f:
        f.write(transcript)

    print(f"\n💾 저장: {custom_path}")

    # Gemini 분석
    print("\n🤖 Gemini LLM 분석 중...")

    analyzer = TranscriptAnalyzer(verbose=True)
    highlights = analyzer.analyze_transcript(custom_path, "output/custom_highlights.json")

    print("\n✅ 분석 완료!")
    print(f"   추출된 하이라이트: {len(highlights)}개")
    print(f"   💾 저장: output/custom_highlights.json")

    for i, h in enumerate(highlights, 1):
        print(f"\n{i}. [{h['start']:.1f}s - {h['end']:.1f}s] {h['type']}")
        print(f"   {h['description']}")

    return highlights


def compare_with_without_timestamps():
    """
    타임스탬프 있는 것과 없는 것 비교
    """

    print("=" * 70)
    print("🔬 타임스탬프 유무 비교 테스트")
    print("=" * 70)

    # 타임스탬프 있는 버전
    with_timestamps = """
[120.5s - 125.5s] 골입니다! 환상적인 슈팅!
[450.2s - 455.2s] 프리킥 기회입니다.
[1230.5s - 1235.5s] 추가골입니다!
"""

    # 타임스탬프 없는 버전
    without_timestamps = """
골입니다! 환상적인 슈팅!
프리킥 기회입니다.
추가골입니다!
"""

    Path("output").mkdir(exist_ok=True)

    # 테스트 1: 타임스탬프 있음
    print("\n[테스트 1] 타임스탬프 포함")
    print("-" * 70)

    with open("output/with_timestamps.txt", 'w', encoding='utf-8') as f:
        f.write(with_timestamps)

    analyzer = TranscriptAnalyzer(verbose=False)
    highlights1 = analyzer.analyze_transcript(
        "output/with_timestamps.txt",
        "output/highlights_with_timestamps.json"
    )

    print(f"결과: {len(highlights1)}개 하이라이트")
    for h in highlights1:
        print(f"  - [{h['start']:.1f}s] {h['type']}: {h['description']}")

    # 테스트 2: 타임스탬프 없음
    print("\n[테스트 2] 타임스탬프 없음")
    print("-" * 70)

    with open("output/without_timestamps.txt", 'w', encoding='utf-8') as f:
        f.write(without_timestamps)

    highlights2 = analyzer.analyze_transcript(
        "output/without_timestamps.txt",
        "output/highlights_without_timestamps.json"
    )

    print(f"결과: {len(highlights2)}개 하이라이트")
    for h in highlights2:
        print(f"  - [{h['start']:.1f}s] {h['type']}: {h['description']}")

    # 비교
    print("\n" + "=" * 70)
    print("📊 비교 결과")
    print("=" * 70)
    print(f"타임스탬프 포함: {len(highlights1)}개")
    print(f"타임스탬프 없음: {len(highlights2)}개")

    if len(highlights1) > len(highlights2):
        print("\n✅ 타임스탬프가 있을 때 더 정확합니다!")
    elif len(highlights1) == len(highlights2):
        print("\n✓ 비슷한 결과입니다.")
    else:
        print("\n⚠️  예상과 다른 결과입니다.")


if __name__ == "__main__":
    print("🧪 더미 데이터 테스트 도구")
    print("=" * 70)
    print()
    print("테스트 종류:")
    print("  1. 기본 더미 데이터 테스트 (권장)")
    print("  2. 사용자 정의 텍스트 테스트")
    print("  3. 타임스탬프 유무 비교 테스트")

    choice = input("\n선택 (1-3, 기본값=1): ").strip() or "1"

    if choice == "1":
        test_gemini_only()
    elif choice == "2":
        test_custom_transcript()
    elif choice == "3":
        compare_with_without_timestamps()
    else:
        print("❌ 잘못된 선택입니다.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ 테스트 완료!")
    print("=" * 70)
