import os
from vidstab import VidStab
import subprocess
import imageio_ffmpeg as ffmpeg

# ================= [설정: 파일 경로 확인!] =================
# 경로 앞에 r 붙이는 거 잊지 마세요!
INPUT_VIDEO = r"C:\Users\home\Desktop\shortsgenie\final_shorts.mp4"
OUTPUT_VIDEO = r"C:\Users\home\Desktop\shortsgenie\final_shorts_stable_fixed.mp4" 
SMOOTHING_WINDOW = 30
# ========================================================

def convert_and_merge(temp_video, original_video, final_output):
    """
    [해결사 함수]
    VidStab이 만든 mp4v(재생 안됨) 영상을 -> H.264(재생 잘됨)로 바꾸고
    원본의 오디오를 합쳐서 저장합니다.
    """
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    
    print(f"🔄 [FFmpeg] 'mp4v'를 'H.264' 표준으로 변환 중... (재생 문제 해결)")
    
    cmd = [
        ffmpeg_exe, "-y",
        "-i", temp_video,       # 1. VidStab이 뱉은 영상 (mp4v, 소리 없음)
        "-i", original_video,   # 2. 원본 영상 (소리 있음)
        
        # --- [핵심: 강제 변환 옵션] ---
        "-c:v", "libx264",      # 비디오 코덱을 무조건 H.264로 설정
        "-pix_fmt", "yuv420p",  # 윈도우/맥/폰 어디서든 재생되게 픽셀 포맷 고정
        "-preset", "fast",      # 변환 속도 빠르게
        "-crf", "23",           # 화질 손상 없이
        # ---------------------------
        
        "-c:a", "aac",          # 오디오는 AAC (표준)
        "-b:a", "192k",         # 오디오 음질 좋게
        "-map", "0:v:0",        # 첫 번째 파일의 영상 사용
        "-map", "1:a:0",        # 두 번째 파일의 소리 사용
        "-shortest",            # 길이 맞추기
        final_output
    ]
    
    # 에러가 나면 이유를 보기 위해 try-except 사용
    try:
        # 윈도우에서 팝업창 안 뜨게 설정
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        subprocess.run(cmd, check=True, startupinfo=startupinfo)
        print("✅ 변환 및 저장 성공!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg 변환 실패! 에러 코드: {e}")
        return False

def run_stabilization():
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ 원본 파일이 없습니다: {INPUT_VIDEO}")
        return

    print(f"🌊 [VidStab] 흔들림 잡는 중... (잠시만 기다리세요)")
    
    stabilizer = VidStab()
    
    # 임시 파일 (이건 재생 안 돼도 상관없음. 재료로만 씀)
    temp_output = "temp_ignore_this.mp4"
    
    # 1. 흔들림 잡기 (OpenCV가 mp4v로 만듦 -> 신경 쓰지 마세요)
    stabilizer.stabilize(
        input_path=INPUT_VIDEO, 
        output_path=temp_output, 
        smoothing_window=SMOOTHING_WINDOW,
        border_type='reflect'
    )

    print("✅ 흔들림 잡기 완료. 이제 재생 가능한 파일로 만듭니다...")

    # 2. 재생 가능한 포맷으로 변환 + 오디오 합치기
    if convert_and_merge(temp_output, INPUT_VIDEO, OUTPUT_VIDEO):
        # 성공했으면 임시 파일 삭제
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        print("-" * 50)
        print(f"🎉 진짜_최종_완성본: {OUTPUT_VIDEO}")
        print("   👉 이제 더블 클릭하면 바로 재생됩니다!")
    else:
        print("❌ 변환 과정에서 문제가 생겼습니다.")

if __name__ == "__main__":
    run_stabilization()