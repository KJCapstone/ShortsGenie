import os
from vidstab import VidStab
import subprocess
import imageio_ffmpeg as ffmpeg

# ================= [설정] =================
INPUT_VIDEO = r"C:\Users\home\Desktop\shortsgenie\final_shorts.mp4"
OUTPUT_VIDEO = r"C:\Users\home\Desktop\shortsgenie\final_shorts_stable.mp4" # 보정된 완성본
SMOOTHING_WINDOW = 30                 # 보정 강도 (크면 더 부드러워짐, 보통 30)
# =========================================

def merge_audio(video_path, audio_source, output_path):
    """소리가 없는 보정 영상에 원본 오디오를 합치는 함수"""
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    
    cmd = [
        ffmpeg_exe, "-y",
        "-i", video_path,   # 보정된 영상 (소리 없음)
        "-i", audio_source, # 원본 영상 (소리 있음)
        "-c:v", "copy",     # 영상은 그대로 복사
        "-c:a", "aac",      # 오디오 코덱
        "-map", "0:v:0",    # 첫 번째 입력의 비디오 사용
        "-map", "1:a:0",    # 두 번째 입력의 오디오 사용
        "-shortest",        # 둘 중 짧은 길이에 맞춤
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_stabilization():
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ 파일이 없습니다: {INPUT_VIDEO}")
        return

    print(f"🌊 [VidStab] 흔들림 보정 시작... (강도: {SMOOTHING_WINDOW})")
    print("   (시간이 조금 걸릴 수 있습니다. 잠시만 기다려주세요...)")

    # 1. 안정화기 초기화
    stabilizer = VidStab()

    # 2. 흔들림 잡기 (소리 없는 임시 파일 생성)
    temp_output = "temp_stable.mp4"
    
    # stabilize 메서드 실행 (보더 타입을 'reflect'로 해서 검은 테두리 방지)
    stabilizer.stabilize(
        input_path=INPUT_VIDEO, 
        output_path=temp_output, 
        smoothing_window=SMOOTHING_WINDOW,
        border_type='reflect' 
    )

    print("✅ 영상 안정화 완료! 이제 소리를 합칩니다...")

    # 3. 오디오 합치기
    merge_audio(temp_output, INPUT_VIDEO, OUTPUT_VIDEO)

    # 4. 임시 파일 삭제
    if os.path.exists(temp_output):
        os.remove(temp_output)

    print("-" * 50)
    print(f"🎉 작업 끝! 결과 파일: {OUTPUT_VIDEO}")
    print("   이제 영상이 훨씬 부드러워졌을 겁니다! 😎")

if __name__ == "__main__":
    run_stabilization()