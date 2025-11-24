import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from PIL import Image
from tqdm import tqdm
import pandas as pd
import json  # <--- JSON 라이브러리 추가

# ================= [설정값] =================
INPUT_VIDEO = "test.mp4"         # 분석할 영상
MODEL_PATH = "C:\\Users\\home\\Desktop\\shortsgenie\\ShortsGenie\\auto_tagger\\model\\soccer_model.pth" # 학습된 모델 경로

# 결과 파일 이름 2개
OUTPUT_CSV = "full_match_log.csv"      
OUTPUT_JSON = "full_match_log.json"    

# 4개 다 찾도록 설정
TARGET_LABELS = ['wide', 'close', 'audience', 'replay'] 
# ===========================================

LABELS = ['wide', 'close', 'audience', 'replay']

def sec_to_time(seconds):
    """초 단위를 '00분 00초' 형식으로 변환"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}분 {s:02d}초"

def analyze_full_video_json():
    # 0. 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"분석 중 (Device: {device})")

    # 1. 모델 로드
    if not os.path.exists(MODEL_PATH):
        print("모델 파일이 없습니다.")
        return

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # 2. 전처리 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. 장면 구간 나누기
    print("1. 영상의 모든 컷을 나누기")
    video_manager = VideoManager([INPUT_VIDEO])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=15))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(video_manager.get_base_timecode())
    video_manager.release()
    print(f" 총 {len(scene_list)}개의 컷을 분석합니다.")

    # 4. AI 분석 및 기록
    print("2. 각 컷의 장면의 종류 예측")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    timeline_data = [] 

    print("\n" + "="*60)
    print(f"   {'[장면 종류]':<10} |  {'시작 시간':<10} ~  {'종료 시간':<10} |  {'길이(초)':<5}")
    print("="*60)

    for scene in tqdm(scene_list):
        start = scene[0].get_seconds()
        end = scene[1].get_seconds()
        duration = end - start
        
        if duration < 0.5: continue

        mid_pos = start + (duration / 2)
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_pos * 1000)
        ret, frame = cap.read()
        if not ret: continue

        # AI 예측
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
            
        label_idx = preds.item()
        label_name = LABELS[label_idx]

        # 모든 라벨 기록
        if label_name in TARGET_LABELS:
            start_str = sec_to_time(start)
            end_str = sec_to_time(end)
            
            # 터미널 출력
            prefix = "★ " if label_name == 'replay' else "  "
            print(f" {prefix}[{label_name.upper():<8}] |  {start_str:<10} ~  {end_str:<10} |  {duration:.1f}s")
            
            timeline_data.append({
                "label": label_name,
                "start_time": start_str,
                "end_time": end_str,
                "duration": round(duration, 2),
                "start_seconds": round(start, 2),
                "end_seconds": round(end, 2)
            })

    cap.release()
    
    # 5. 파일 저장 (CSV + JSON)
    if timeline_data:
        # CSV 저장
        df = pd.DataFrame(timeline_data)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        
        # JSON 저장
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(timeline_data, f, indent=4, ensure_ascii=False)

        print("\n" + "="*60)
        print(f"분석 완료")
        print(f"엑셀 파일: {OUTPUT_CSV}")
        print(f"JSON 파일: {OUTPUT_JSON}")
        print(f"총 {len(timeline_data)}개의 컷이 기록되었습니다.")
    else:
        print("분석된 데이터가 없습니다.")

if __name__ == "__main__":
    analyze_full_video_json()