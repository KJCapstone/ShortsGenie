# ============================================
# 0️⃣ 필수 라이브러리
# ============================================
import cv2
import json
import os
import time
from pathlib import Path

# ============================================
# 1️⃣ Heuristic Shot Detection
# ============================================
def analyze_shots(video_path, out_json="shots.json", threshold=30.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_frame = None
    shots = []
    shot_start = 0.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if prev_frame is not None:
                shots.append({"label":"unknown","start":shot_start,"end":frame_idx/fps})
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is None:
            prev_frame = gray
            frame_idx += 1
            continue

        diff = cv2.absdiff(gray, prev_frame)
        mean_diff = diff.mean()

        if mean_diff > threshold:
            end_time = frame_idx / fps
            avg_brightness = gray.mean()
            label = "wide" if avg_brightness>100 else "close" if avg_brightness>50 else "audience"
            shots.append({"label":label,"start":shot_start,"end":end_time})
            shot_start = end_time

        prev_frame = gray
        frame_idx += 1

    cap.release()
    with open(out_json,"w",encoding="utf-8") as f:
        json.dump(shots,f,indent=2,ensure_ascii=False)
    print(f"샷 분류 완료! → {out_json}")
    return shots

# ============================================
# 2️⃣ Segment Clip 생성
# ============================================
def make_segments(video_path, shots_json, out_dir="segments"):
    shots = json.load(open(shots_json))
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for idx, shot in enumerate(shots):
        start_frame = int(shot["start"]*fps)
        end_frame = int(shot["end"]*fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        width, height = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(f"{out_dir}/segment_{idx:04d}.mp4",
                              cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (width,height))
        for f in range(start_frame,end_frame):
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
        out.release()
    cap.release()
    print(f"{len(shots)} segments saved → {out_dir}")

# ============================================
# 3️⃣ JSON → HMS 변환 (선택)
# ============================================
def sec_to_hms(sec):
    m = int(sec//60)
    s = sec%60
    return f"{m:02d}:{s:05.2f}"

def convert_json_to_hms(in_json,out_json):
    shots = json.load(open(in_json))
    for shot in shots:
        shot["start_hms"] = sec_to_hms(shot["start"])
        shot["end_hms"] = sec_to_hms(shot["end"])
    with open(out_json,"w",encoding="utf-8") as f:
        json.dump(shots,f,indent=2,ensure_ascii=False)
    print(f"HMS 변환 완료! → {out_json}")

# ============================================
# 4️⃣ Main 실행
# ============================================
video_path = "test.mp4"
raw_json = "shots.json"
hms_json = "shots_hms.json"
segments_dir = "segments"

start_time = time.time()
analyze_shots(video_path, raw_json)
make_segments(video_path, raw_json, segments_dir)
convert_json_to_hms(raw_json, hms_json)
print(f"총 처리 시간: {time.time()-start_time:.2f}초")
