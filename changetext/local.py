import os
import json
import time
import imageio_ffmpeg as ffmpeg
import subprocess
from faster_whisper import WhisperModel

# 1. 분석할 영상 파일 이름
VIDEO_PATH = "test4.mp4"

# 2. 결과 파일 이름들
AUDIO_OUTPUT_PATH = "extracted_audio5.mp3"
JSON_OUTPUT_PATH = "match_transcript_local5.json"

# 3. 모델 크기 (CPU에서는 'medium' 추천. 더 빠르길 원하면 'small')
MODEL_SIZE = "medium" 

# 4. 축구 용어 및 문맥 잡아주는 힌트
SOCCER_PROMPT = (
    "축구 경기 중계, 캐스터와 해설위원의 대화. "
    "프리미어리그, 챔피언스리그, 월드컵, K리그, 라리가, 분데스리가. "
    "골, 득점, 슈팅, 패스, 드리블, 태클, 헤더, 크로스, 빌드업, 압박, 탈압박, "
    "프리킥, 코너킥, 페널티킥, 오프사이드, VAR, 파울, 경고, 퇴장, 옐로카드, 레드카드, "
    "전반전, 후반전, 추가시간, 교체, 감독, 주심, 부심, 홈팀, 원정팀, 승리, 패배, 무승부"
)

def sec_to_time(seconds):
    """초 단위를 '00분 00초' 형식으로 변환"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}분 {s:02d}초"

def extract_audio(video_path, audio_path):
    """영상에서 오디오 추출 (ffmpeg 사용)"""
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe, "-i", video_path, "-q:a", "0", "-map", "a", "-y", audio_path]
    
    print(f"오디오 추출 중... ({video_path})")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("오디오 추출 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"오디오 추출 실패: {e}")
        return False

def refine_segments(raw_segments, max_gap=0.8, max_chars=50):
    """뚝뚝 끊긴 텍스트를 자연스러운 문장으로 합치는 기능"""
    if not raw_segments: return []

    refined = []
    # Generator를 리스트 딕셔너리로 변환
    current_group = {
        "start": raw_segments[0].start,
        "end": raw_segments[0].end,
        "text": raw_segments[0].text.strip()
    }

    for next_seg in raw_segments[1:]:
        seg_dict = {
            "start": next_seg.start,
            "end": next_seg.end,
            "text": next_seg.text.strip()
        }
        
        time_gap = seg_dict['start'] - current_group['end']
        is_sentence_end = current_group['text'].endswith(('.', '?', '!'))
        
        if time_gap < max_gap and not is_sentence_end and len(current_group['text']) + len(seg_dict['text']) < max_chars:
            current_group['text'] += " " + seg_dict['text']
            current_group['end'] = seg_dict['end']
        else:
            refined.append(current_group)
            current_group = seg_dict
            
    refined.append(current_group)
    return refined

def run_local_whisper():
    # 스톱워치 시작
    total_start = time.time()

    if not os.path.exists(VIDEO_PATH):
        print(f"영상 파일이 없습니다: {VIDEO_PATH}")
        return

    print(f"로컬 모드 분석 (모델: {MODEL_SIZE})")
    print("CPU 모드")
    
    # 1. 오디오 추출
    if extract_audio(VIDEO_PATH, AUDIO_OUTPUT_PATH):
        
        # 2. 모델 로드
        print(f"모델 로딩 중... (내 컴퓨터 메모리 사용)")
        # CPU 최적화 설정 (int8)
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

        # 3. 변환 실행
        print("분석 중...")
        segments, info = model.transcribe(
            AUDIO_OUTPUT_PATH,
            beam_size=5,
            language="ko",
            initial_prompt=SOCCER_PROMPT,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False
        )

        # 리스트로 변환 (CPU 모드용)
        segments = list(segments)
        print(f"원본 조각 개수: {len(segments)}개")

        # 4. 문장 다듬기 (Refine)
        refined_data = refine_segments(segments)
        print(f"다듬어진 문장 개수: {len(refined_data)}개")

        # 5. 결과 저장
        final_output = []
        print("-" * 50)
        for seg in refined_data:
            item = {
                "label": "commentary",
                "text": seg['text'],
                "start_time": sec_to_time(seg['start']),
                "end_time": sec_to_time(seg['end']),
                "duration": round(seg['end'] - seg['start'], 2),
                "start_seconds": round(seg['start'], 2),
                "end_seconds": round(seg['end'], 2)
            }
            final_output.append(item)
            # 화면에는 일부만 출력
            print(f"[{item['start_time']}~{item['end_time']}] {item['text']}")

        with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)

        total_end = time.time()
        elapsed = total_end - total_start
        m = int(elapsed // 60)
        s = int(elapsed % 60)

        print("-" * 50)
        print(f"변환 끝! 결과 파일: {JSON_OUTPUT_PATH}")
        print(f"총 소요 시간: {m}분 {s}초")

if __name__ == "__main__":
    run_local_whisper()