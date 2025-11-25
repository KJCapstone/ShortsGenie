"""
스코어보드 크롭 영역 시각화
실제로 어디를 자르고 있는지 확인
"""
import cv2
import numpy as np
from pathlib import Path

video_path = "input/korea_vs_brazil.mp4"
output_dir = Path("output/debug_crops")
output_dir.mkdir(exist_ok=True)

# 감지된 스코어보드 영역 (로그에서 가져옴)
scoreboard_region = (405, 48, 163, 111)  # x, y, w, h

print(f"📍 스코어보드 영역: x={scoreboard_region[0]}, y={scoreboard_region[1]}, w={scoreboard_region[2]}, h={scoreboard_region[3]}")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 테스트할 시점들 (초)
test_times = [120, 180, 300, 600]

for time_sec in test_times:
    if time_sec > total_frames / fps:
        continue

    frame_num = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if not ret:
        continue

    print(f"\n⏱️  {time_sec}초 (프레임 {frame_num}):")

    # 1. 전체 프레임 저장 (스코어보드 영역 표시)
    frame_with_box = frame.copy()
    x, y, w, h = scoreboard_region
    cv2.rectangle(frame_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(frame_with_box, f"Scoreboard ({x},{y},{w},{h})",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    full_path = output_dir / f"full_frame_{time_sec}s.jpg"
    cv2.imwrite(str(full_path), frame_with_box)
    print(f"   ✅ 전체 프레임: {full_path}")

    # 2. 크롭된 스코어보드 영역 저장
    scoreboard_crop = frame[y:y+h, x:x+w]
    crop_path = output_dir / f"crop_{time_sec}s.jpg"
    cv2.imwrite(str(crop_path), scoreboard_crop)
    print(f"   ✅ 크롭 영역: {crop_path} (크기: {scoreboard_crop.shape})")

    # 3. 상단 20% 영역도 확인 (초기화 때 사용한 영역)
    height, width = frame.shape[:2]
    top_region = frame[0:int(height * 0.2), :]
    top_path = output_dir / f"top20_frame_{time_sec}s.jpg"

    # 상단 영역에 박스 표시
    top_with_box = top_region.copy()
    cv2.rectangle(top_with_box, (x, y), (x+w, y+h), (255, 0, 0), 3)
    cv2.imwrite(str(top_path), top_with_box)
    print(f"   ✅ 상단 20%: {top_path}")

cap.release()

print(f"\n✅ 완료! 이미지들이 {output_dir}/ 에 저장되었습니다.")
print(f"\n📝 확인 사항:")
print(f"   1. full_frame_*.jpg - 전체 프레임에 초록색 박스가 스코어보드를 정확히 감싸는지")
print(f"   2. crop_*.jpg - 크롭된 영역에 점수가 보이는지")
print(f"   3. top20_*.jpg - 상단 20% 영역에 파란색 박스가 스코어보드 위치를 가리키는지")
