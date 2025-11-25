import cv2
import json
import os
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# ================= [설정값] =================
VIDEO_PATH = "test2.mp4"                 # 영상 파일명
TEMPLATE_PATH = "scoreboard_template.png" # 점수판 캡처 파일명
OUT_JSON = "shots2.json"

# 1. 장면 감지 민감도 (낮을수록 예민하게 자름)
SCENE_THRESHOLD = 15.0 

# 2. 점수판 매칭 기준 (0.6 이상이면 점수판 있다고 판단)
MATCH_THRESHOLD = 0.6
# ===========================================

def calculate_green_ratio(frame):
    """ 화면에 초록색(잔디)이 얼마나 있는지 %로 계산 """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 초록색 범위 (축구장 잔디색)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])

def step1_analyze():
    if not os.path.exists(VIDEO_PATH):
        print("❌ 영상 파일이 없습니다.")
        return

    print("🚀 [1단계] 지능형 분석 시작 (Brain Mode)...")

    # 1. 점수판 템플릿 로드
    template = None
    if os.path.exists(TEMPLATE_PATH):
        template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
        print("   ✅ 점수판 템플릿 로드됨 (리플레이 자동 감지 ON)")
    else:
        print("   ⚠️ 템플릿 없음 (리플레이 감지 불가)")

    # 2. 컷 감지 (PySceneDetect)
    print("   🔍 장면 전환 지점 찾는 중...")
    video_manager = VideoManager([VIDEO_PATH])
    scene_manager = SceneManager()
    # min_scene_len=15: 최소 0.5초 이상 되어야 컷으로 인정 (너무 잘게 쪼개짐 방지)
    scene_manager.add_detector(ContentDetector(threshold=SCENE_THRESHOLD, min_scene_len=15))
    
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(video_manager.get_base_timecode())
    video_manager.release()
    
    print(f"   ✅ 총 {len(scene_list)}개의 컷 발견.")

    # 3. 각 장면 상세 분석 (CV2 활용)
    cap = cv2.VideoCapture(VIDEO_PATH)
    final_shots = []

    print("   🧠 각 장면 내용 분석 중 (Replay & Shot Type)...")

    for i, scene in enumerate(scene_list):
        start = scene[0].get_seconds()
        end = scene[1].get_seconds()
        duration = end - start

        if duration < 0.5: continue # 너무 짧은 건 패스

        # 중간 프레임 추출
        mid_pos = start + (duration / 2)
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_pos * 1000)
        ret, frame = cap.read()
        if not ret: continue

        label = "unknown"
        is_replay = False

        # [A] 리플레이 검사 (점수판 찾기)
        if template is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 속도를 위해 좌측 상단(300x600)만 검사
            roi_h, roi_w = 300, 600
            if gray.shape[0] > roi_h and gray.shape[1] > roi_w:
                roi = gray[0:roi_h, 0:roi_w]
            else:
                roi = gray
            
            res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            if max_val < MATCH_THRESHOLD: # 점수판이 없으면?
                label = "replay"
                is_replay = True
        
        # [B] 샷 종류 검사 (잔디 비율) - 리플레이가 아닐 때만
        if not is_replay:
            green_ratio = calculate_green_ratio(frame)
            if green_ratio > 0.60:     # 60% 이상 잔디 -> Wide
                label = "wide"
            elif green_ratio > 0.20:   # 20~60% 잔디 -> Close
                label = "close"
            else:                      # 잔디 거의 없음 -> Audience
                label = "audience"

        final_shots.append({
            "label": label,
            "start": start,
            "end": end
        })

    cap.release()

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_shots, f, indent=2, ensure_ascii=False)

    print(f"\n✨ 분석 완료! '{OUT_JSON}' 생성됨.")
    print("👉 이제 step2_cut.py를 실행하세요.")

if __name__ == "__main__":
    step1_analyze()