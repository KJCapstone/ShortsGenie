"""
새로운 스코어보드 영역 확인
"""
import cv2

video_path = "input/korea_vs_brazil.mp4"

# 새로 감지된 스코어보드 영역
scoreboard_region = (16, 0, 408, 216)  # x, y, w, h
x, y, w, h = scoreboard_region

print(f"📍 스코어보드 영역: x={x}, y={y}, w={w}, h={h}\n")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# 120초, 180초 프레임 확인
for time_sec in [120, 180]:
    frame_num = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if not ret:
        continue

    print(f"⏱️  {time_sec}초:")

    # 크롭된 영역
    scoreboard_crop = frame[y:y+h, x:x+w]

    # 저장
    crop_path = f"output/debug_crops/NEW_crop_{time_sec}s.jpg"
    cv2.imwrite(crop_path, scoreboard_crop)
    print(f"   ✅ {crop_path} (크기: {scoreboard_crop.shape})\n")

    # 전체 프레임에 박스 표시
    frame_with_box = frame.copy()
    cv2.rectangle(frame_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)
    full_path = f"output/debug_crops/NEW_full_{time_sec}s.jpg"
    cv2.imwrite(full_path, frame_with_box)
    print(f"   ✅ {full_path}\n")

cap.release()
print("✅ 완료!")
