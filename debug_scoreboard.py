"""
스코어보드 위치 디버깅 스크립트
여러 시점의 프레임에서 OCR 실행하여 스코어 패턴 찾기
"""
import cv2
import re
from paddleocr import PaddleOCR

# 비디오 열기
video_path = "input/korea_vs_brazil.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"📹 비디오 정보:")
print(f"   FPS: {fps:.2f}")
print(f"   총 프레임: {total_frames:,}")
print(f"   길이: {duration:.1f}초 ({duration/60:.1f}분)\n")

# OCR 초기화
print("🔧 PaddleOCR 초기화 중...")
ocr = PaddleOCR(lang='en')

# 테스트할 시점들 (초 단위)
test_times = [30, 60, 120, 180, 300, 600]

print(f"\n🔍 여러 시점에서 스코어보드 탐색:\n")

for time_sec in test_times:
    if time_sec > duration:
        continue

    frame_num = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if not ret:
        continue

    print(f"⏱️  {time_sec}초 (프레임 {frame_num}):")

    height, width = frame.shape[:2]

    # 상단 20% 영역
    top_region = frame[0:int(height * 0.2), :]

    # OCR 실행
    result = ocr.predict(top_region)

    if not result or len(result) == 0:
        print("   ❌ OCR 결과 없음\n")
        continue

    ocr_result = result[0]
    rec_texts = ocr_result.get('rec_texts', [])
    rec_scores = ocr_result.get('rec_scores', [])

    if not rec_texts:
        print("   ❌ 텍스트 없음\n")
        continue

    # 모든 텍스트 출력
    print(f"   📝 감지된 텍스트 ({len(rec_texts)}개):")

    score_pattern = r'(\d+)\s*[-:]\s*(\d+)'

    for text, score in zip(rec_texts, rec_scores):
        is_score = re.search(score_pattern, text)
        marker = "✅" if is_score else "  "
        print(f"      {marker} '{text}' (신뢰도: {score:.2f})")

    # 전체 텍스트 결합해서 점수 찾기
    full_text = ' '.join(rec_texts)
    match = re.search(score_pattern, full_text)

    if match:
        print(f"   ⚽ 점수 발견: {match.group(0)}")

    print()

cap.release()
print("✅ 디버깅 완료")
