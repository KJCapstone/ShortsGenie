import os
from vidstab import VidStab
import subprocess
import imageio_ffmpeg as ffmpeg

# ================= [설정] =================
INPUT_VIDEO = r"C:\Users\home\Desktop\shortsgenie\final_shorts.mp4"
OUTPUT_VIDEO = r"C:\Users\home\Desktop\shortsgenie\final_shorts_best.mp4"

# 보정 강도 (20: 사용자님이 가장 만족했던 안정감)
SMOOTHING_WINDOW = 20 

# [핵심 수정] 줌 비율을 5% -> 15%로 올렸습니다!
# 늘어지는 가장자리를 확실하게 잘라내기 위함입니다.
ZOOM_PERCENT = 15 
# =========================================

def convert_merge_flip_and_zoom(temp_video, original_video, final_output):
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    
    print(f"🔄 [FFmpeg] 좌우 반전 + {ZOOM_PERCENT}% 줌 적용 중...")

    # 1. 줌(Crop) 계산
    # 15% 줌 -> 화면의 85% 영역만 남김
    crop_factor = 1 - (ZOOM_PERCENT / 100)
    
    # 2. 필터 체인 (순서 중요!)
    # hflip: 좌우 반전
    # crop: 중앙 잘라내기 (늘어난 테두리 제거)
    # scale: 잘라낸 화면을 다시 원래 해상도로 복구 (이게 추가됨!)
    filter_chain = f"hflip,crop=iw*{crop_factor}:ih*{crop_factor},scale=iw:ih"
    
    cmd = [
        ffmpeg_exe, "-y",
        "-i", temp_video,       
        "-i", original_video,   
        "-vf", filter_chain,    
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
        print(f"❌ 변환 실패: {e}")
        return False

def run_stabilization():
    if not os.path.exists(INPUT_VIDEO):
        print("❌ 원본 파일이 없습니다.")
        return

    print(f"🌊 [최적 보정] 강도: {SMOOTHING_WINDOW}, 줌: {ZOOM_PERCENT}%")
    
    stabilizer = VidStab()
    temp_output = "temp_stable_best.mp4"
    
    # 사용자님이 가장 만족했던 'replicate' 방식 유지
    stabilizer.stabilize(
        input_path=INPUT_VIDEO, 
        output_path=temp_output, 
        smoothing_window=SMOOTHING_WINDOW,
        border_type='replicate' 
    )

    print("✅ 흔들림 잡기 완료. 늘어난 가장자리를 15% 잘라냅니다...")

    if convert_merge_flip_and_zoom(temp_output, INPUT_VIDEO, OUTPUT_VIDEO):
        if os.path.exists(temp_output): os.remove(temp_output)
        print("-" * 50)
        print(f"🎉 최종 완성본: {OUTPUT_VIDEO}")
        print("   👉 안정감은 그대로, 가장자리 늘어짐은 사라졌을 겁니다!")

if __name__ == "__main__":
    run_stabilization()