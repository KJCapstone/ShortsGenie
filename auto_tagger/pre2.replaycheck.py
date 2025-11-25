import cv2
import os

VIDEO_PATH = "test2.mp4"
TEMPLATE_PATH = "scoreboard_template.png"

def check_score():
    template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # 10초 지점으로 이동
    cap.set(cv2.CAP_PROP_POS_MSEC, 500000)
    ret, frame = cap.read()
    
    if not ret: return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 템플릿 매칭
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    print("="*30)
    print(f"현재 매칭 점수: {max_val:.4f}")
    print("="*30)
    
    if max_val > 0.6:
        print("점수판과 매칭이 잘됨")
    else:
        print("점수가 낮음. 점수판 다시 캡쳐하기")

check_score()