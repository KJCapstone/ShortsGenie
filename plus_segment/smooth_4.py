import os
from vidstab import VidStab
import subprocess
import imageio_ffmpeg as ffmpeg

# ================= [설정: 경로 확인] =================
INPUT_VIDEO = r"C:\Users\home\Desktop\shortsgenie\final_shorts.mp4"
OUTPUT_VIDEO = r"C:\Users\home\Desktop\shortsgenie\final_shorts_zoomed.mp4"

# 보정 강도 (20 정도가 적당)
SMOOTHING_WINDOW = 20 

# [핵심] 줌(Zoom) 비율 설정 (단위: 퍼센트)
# 5% 정도 확대하면 가장자리의 이상한 부분이 대부분 사라집니다.
ZOOM_PERCENT = 5 
# ====================================================

def convert_merge_flip_and_zoom(temp_video, original_video, final_output):
    """
    [최종 해결사 함수]
    1. 좌우 반전 교정 (hflip)
    2. 화면 살짝 확대해서 가장자리 잘라내기 (crop)
    3. 표준 코덱 변환 및 오디오 합치기
    """
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    
    print(f"🔄 [FFmpeg] 좌우 반전 교정 및 줌(Zoom) 적용 중...")

    # 1. 줌(Zoom) 필터 계산
    # 화면 중앙을 기준으로 지정된 퍼센트만큼 확대해서 잘라냅니다.
    # (예: 5% 줌 -> 화면의 95% 영역만 사용)
    crop_factor = 1 - (ZOOM_PERCENT / 100)
    crop_filter = f"crop=iw*{crop_factor}:ih*{crop_factor}"
    
    # 2. 필터 체인 연결 (좌우반전 -> 줌 -> 스케일링)
    # hflip: 좌우 반전
    # crop: 화면 잘라내기
    # scale: 잘라낸 화면을 다시 원래 해상도로 늘리기 (선택사항, 여기선 생략)
    filter_chain = f"hflip,{crop_filter}"
    
    cmd = [
        ffmpeg_exe, "-y",
        "-i", temp_video,       # 1. VidStab 영상 (반전됨, 테두리 이상함)
        "-i", original_video,   # 2. 원본 영상 (소리)
        
        # --- [필터 적용] ---
        "-vf", filter_chain,    # 좌우반전 + 줌 필터 동시 적용
        # -----------------

        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest",
        final_output
    ]
    
    try:
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.run(cmd, check=True, startupinfo=startupinfo)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg 변환 실패: {e}")
        return False

def run_stabilization():
    if not os.path.exists(INPUT_VIDEO):
        print("❌ 원본 파일이 없습니다.")
        return

    print(f"🌊 [보정 시작] 강도: {SMOOTHING_WINDOW}, 줌: {ZOOM_PERCENT}%")
    
    stabilizer = VidStab()
    temp_output = "temp_to_be_zoomed.mp4"
    
    # 1. 흔들림 잡기 (가장자리 늘리기 모드)
    stabilizer.stabilize(
        input_path=INPUT_VIDEO, 
        output_path=temp_output, 
        smoothing_window=SMOOTHING_WINDOW,
        border_type='replicate' # 일단 늘려놓고 나중에 잘라냄
    )

    print("✅ 1단계 완료. 이제 좌우를 바로잡고 줌을 적용합니다...")

    # 2. 최종 변환 (좌우반전 + 줌 + 오디오합치기)
    if convert_merge_flip_and_zoom(temp_output, INPUT_VIDEO, OUTPUT_VIDEO):
        if os.path.exists(temp_output): os.remove(temp_output)
        print("-" * 50)
        print(f"🎉 최종 완성본: {OUTPUT_VIDEO}")
        print("   👉 흔들림도 잡고, 가장자리도 깔끔하게 정리됐습니다!")
    else:
        print("❌ 변환 과정에서 문제가 생겼습니다.")

if __name__ == "__main__":
    run_stabilization()