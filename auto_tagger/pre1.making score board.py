import cv2
import os

# ================= 설정값 =================
VIDEO_PATH = "test2.mp4"
SAVE_NAME = "scoreboard_template.png"

# ★ 여기를 바꾸세요! (몇 분부터 시작할지)
START_TIME_MIN = 5  # 5분 지점부터 시작
# =========================================

def interactive_template_maker():
    if not os.path.exists(VIDEO_PATH):
        print("영상 파일이 없습니다.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # -------------------------------------------------------
    # [추가된 기능] 지정된 시간으로 점프하기
    # -------------------------------------------------------
    if START_TIME_MIN > 0:
        target_ms = START_TIME_MIN * 60 * 1000 # 분 -> 밀리초 변환
        cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        print(f"⏩ {START_TIME_MIN}분 00초 지점으로 이동했습니다!")
    # -------------------------------------------------------

    print("영상이 재생됩니다.")
    print("점수판이 선명하게 나오면 [Space]를 눌러 멈추세요!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("영상이 끝났습니다")
            break

        cv2.imshow("Press SPACE to Pause", frame)
        
        # 33ms 대기 (약 30fps)
        key = cv2.waitKey(33) & 0xFF
        
        # 스페이스바 누르면 정지
        if key == ord(' '):
            print("점수판을 드래그하세요.")
            
            # ROI 선택 도구 실행
            roi = cv2.selectROI("Press SPACE to Pause", frame, showCrosshair=True, fromCenter=False)
            
            # ROI가 선택되었으면 (크기가 0이 아니면)
            if roi[2] > 0 and roi[3] > 0:
                x, y, w, h = roi
                crop_img = frame[y:y+h, x:x+w]
                
                # 저장
                cv2.imwrite(SAVE_NAME, crop_img)
                print(f"\n저장 완료! '{SAVE_NAME}' 파일이 생성되었습니다.")
                print("   이제 창을 닫고 다시 테스트해보세요.")
                break
            else:
                print("영역이 선택되지 않았습니다. 다시 스페이스바를 누르세요.")

        # q 누르면 종료
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    interactive_template_maker()