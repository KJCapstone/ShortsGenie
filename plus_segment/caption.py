import os
import subprocess
import imageio_ffmpeg as ffmpeg
from faster_whisper import WhisperModel
import json
import re # 정규표현식 (글자 다듬기용)

# ================= [설정] =================
INPUT_VIDEO = r"C:\Users\home\Desktop\shortsgenie\final_shorts_best.mp4"    # 자막 달고 싶은 영상
OUTPUT_VIDEO = r"C:\Users\home\Desktop\shortsgenie\final_shorts_captioned.mp4" # 완성된 영상
MODEL_SIZE = "large-v3-turbo"           # 정확도를 위해 large 추천
# =========================================

def sec_to_srt_time(seconds):
    """초 단위를 SRT 자막 시간 포맷(00:00:00,000)으로 변환"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    ms = int((s - int(s)) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"

def clean_text(text):
    """
    [핵심 기능] 자막을 자연스럽게 다듬는 함수
    """
    # 1. 의미 없는 추임새 제거 (필요하면 단어 추가)
    fillers = ["어", "음", "그", "저", "아", "막"]
    words = text.split()
    # 단어가 2개 이상일 때만 추임새 제거 (한 글자 감탄사는 살림 "아!")
    if len(words) > 1:
        words = [w for w in words if w not in fillers]
    
    cleaned = " ".join(words)

    # 2. 반복되는 특수문자 정리 (!! -> !)
    cleaned = re.sub(r'[!]{2,}', '!', cleaned)
    cleaned = re.sub(r'[?]{2,}', '?', cleaned)
    cleaned = re.sub(r'[.]{2,}', '...', cleaned)
    
    return cleaned.strip()

def create_srt(segments, srt_path):
    """Whisper 결과를 SRT 자막 파일로 변환"""
    print(f"📝 자막 파일(.srt) 생성 중...")
    
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            # 텍스트 다듬기 (자연스럽게)
            natural_text = clean_text(seg.text)
            
            # 내용이 없으면 건너뜀
            if not natural_text: continue

            start = sec_to_srt_time(seg.start)
            end = sec_to_srt_time(seg.end)
            
            f.write(f"{i+1}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{natural_text}\n\n")

def burn_subtitles(video_path, srt_path, output_path):
    """영상에 자막을 예쁘게 입히기 (Hardsub)"""
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    
    # 윈도우 경로 에러 방지 (역슬래시 -> 슬래시)
    video_path_fixed = video_path.replace("\\", "/")
    srt_path_fixed = srt_path.replace("\\", "/")
    
    # [쇼츠 스타일 자막 디자인]
    # Fontname: 맑은 고딕 (윈도우 기본 한글 폰트)
    # PrimaryColour: 노란색 (&H00FFFF - BGR 순서)
    # Outline: 검은색 테두리 두께 2
    # MarginV: 바닥에서 50만큼 띄움
    style = (
        "Fontname=Malgun Gothic,Fontsize=16,PrimaryColour=&H00FFFF,"
        "BackColour=&H80000000,BorderStyle=3,Outline=2,Shadow=0,"
        "Alignment=2,MarginV=50,Bold=1"
    )

    print(f"🎨 영상에 자막을 굽는 중... (스타일: 옐로우 볼드)")
    
    cmd = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-vf", f"subtitles='{srt_path_fixed}':force_style='{style}'",
        "-c:a", "copy",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        output_path
    ]
    
    try:
        # 윈도우 팝업 숨김
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        subprocess.run(cmd, check=True, startupinfo=startupinfo)
        print(f"✅ 자막 합성 완료! -> {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 자막 합성 실패: {e}")

def run_auto_caption():
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ 영상이 없습니다: {INPUT_VIDEO}")
        return

    print("🎧 오디오 분석 및 자막 생성 시작...")
    
    # 1. 모델 로드
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    
    # 2. 텍스트 추출 (Word-level timestamps 사용 추천하지만 여기선 심플하게)
    segments, _ = model.transcribe(
        INPUT_VIDEO, # 영상 파일을 바로 넣으면 알아서 오디오만 읽습니다
        language="ko",
        initial_prompt="자연스러운 자막, 축구 중계, 반말하지 않음.",
        vad_filter=True
    )
    
    # 3. SRT 만들기 (자연스럽게 다듬기 포함)
    srt_filename = "temp_caption.srt"
    create_srt(segments, srt_filename)
    
    # 4. 영상에 굽기
    burn_subtitles(INPUT_VIDEO, srt_filename, OUTPUT_VIDEO)
    
    # 5. 임시 파일 삭제
    if os.path.exists(srt_filename):
        os.remove(srt_filename)
        
    print("-" * 50)
    print(f"🎉 완성되었습니다! 파일을 확인하세요: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    run_auto_caption()