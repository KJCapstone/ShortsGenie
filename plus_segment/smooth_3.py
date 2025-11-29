import os
from vidstab import VidStab
import subprocess
import imageio_ffmpeg as ffmpeg

# ================= [설정: 파일 경로 확인!] =================
# 경로 앞에 r 붙이는 거 잊지 마세요!
INPUT_VIDEO = r"C:\Users\home\Desktop\shortsgenie\final_shorts.mp4"
OUTPUT_VIDEO = r"C:\Users\home\Desktop\shortsgenie\final_shorts_stable_fixed1.mp4" 
# 보정 강도를 30 -> 20으로 조금 낮췄습니다. (너무 세면 울렁거림)
SMOOTHING_WINDOW = 20 
# ========================================================

def convert_and_merge_with_flip(temp_video, original_video, final_output):
    """
    [해결사 함수 V2]
    1. mp4v -> H.264 표준 코덱 변환
    2. 오디오 합치기
    3. [NEW!] 좌우 반전된 영상을 다시 원래대로 뒤집기 (hflip 필터)
    """
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    
    print(f"🔄 [FFmpeg] 표준 변환 및 좌우 반전 교정 중...")
    
    cmd = [
        ffmpeg_exe, "-y",
        "-i", temp_video,       # 1. VidStab이 뱉은 영상 (반전됨, 소리 없음)
        "-i", original_video,   # 2. 원본 영상 (소리 있음)
        
        # --- [핵심: 강제 변환 및 필터 옵션] ---
        "-vf", "hflip",         # <--- [중요!] 수평 뒤집기(Horizontal Flip) 필터 적용
        "-c:v", "libx264",      # 비디오 코덱 H.264
        "-pix_fmt", "yuv420p",  # 호환성 픽셀 포맷
        "-preset", "fast",      # 변환 속도
        "-crf", "23",           # 화질
        # ---------------------------
        
        "-c:a", "aac",          # 오디오 AAC
        "-b:a", "192k",         # 오디오 고음질
        "-map", "0:v:0",        # 영상 스트림
        "-map", "1:a:0",        # 오디오 스트림
        "-shortest",            # 길이 맞춤
        final_output
    ]
    
    try:
        # 윈도우 팝업 숨김
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        subprocess.run(cmd, check=True, startupinfo=startupinfo)
        print("✅ 변환 및 교정 성공!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg 변환 실패! 에러 코드: {e}")
        return False

def run_stabilization():
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ 원본 파일이 없습니다: {INPUT_VIDEO}")
        return

    print(f"🌊 [VidStab] 흔들림 잡는 중... (강도: {SMOOTHING_WINDOW})")
    print("   (참고: 이 단계에서 영상이 잠시 반전될 수 있으나, 최종 결과물에서 해결됩니다.)")
    
    stabilizer = VidStab()
    temp_output = "temp_ignore_flipped.mp4"
    
    # 1. 흔들림 잡기
    # border_type을 'reflect'(거울)에서 'replicate'(늘리기)로 변경
    # -> 울렁거림이 덜하고 더 자연스럽습니다.
    stabilizer.stabilize(
        input_path=INPUT_VIDEO, 
        output_path=temp_output, 
        smoothing_window=SMOOTHING_WINDOW,
        border_type='replicate' 
    )

    print("✅ 1단계 완료. 이제 좌우를 바로잡고 소리를 합칩니다...")

    # 2. 좌우 반전 교정 + 포맷 변환 + 오디오 합치기
    if convert_and_merge_with_flip(temp_output, INPUT_VIDEO, OUTPUT_VIDEO):
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        print("-" * 50)
        print(f"🎉 완벽한 최종 결과물: {OUTPUT_VIDEO}")
        print("   👉 이제 좌우가 정상이고, 화면도 더 자연스러울 겁니다!")
    else:
        print("❌ 변환 과정에서 문제가 생겼습니다.")

if __name__ == "__main__":
    run_stabilization()