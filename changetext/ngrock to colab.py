import requests
import os
import json
import imageio_ffmpeg as ffmpeg
import subprocess
import urllib3
import time

# ================= [설정값: 여기만 수정하세요] =================
# 1. 코랩 실행 후 나온 ngrok 주소를 여기에 붙여넣으세요
COLAB_SERVER_URL = " "

# 2. 분석할 영상 파일 이름
VIDEO_PATH = "test4.mp4"
EXTRA_MATCH_INFO = "" 


# 결과 파일 이름
AUDIO_OUTPUT_PATH = "extracted_audio3.mp3"
JSON_OUTPUT_PATH = "match_transcript3.json"

def sec_to_time(seconds):
    """초 단위를 '00분 00초' 형식으로 변환"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}분 {s:02d}초"

def extract_audio(video_path, audio_path):
    """영상에서 고음질 오디오 추출"""
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

def request_transcription(audio_path, server_url, match_info):
    """서버에 분석 요청 (파일 + 추가 정보 전송)"""
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    server_url = server_url.rstrip("/").replace("/transcribe", "")
    api_url = f"{server_url}/transcribe"
    headers = {"ngrok-skip-browser-warning": "69420"}
    
    print(f"서버로 전송 중... (추가 정보: '{match_info}')")
    
    # 추가 정보를 data에 담아서 보냅니다 (비어있으면 빈 값 전송됨)
    data = {"match_info": match_info}

    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
            response = requests.post(api_url, files=files, data=data, headers=headers, verify=False)
        
        if response.status_code == 200:
            print("서버 응답 성공!")
            return response.json()
        else:
            print(f"서버 오류: {response.status_code}")
            return None
    except Exception as e:
        print(f"연결 실패: {e}")
        return None

def refine_segments(raw_segments, max_gap=0.8, max_chars=50):
    """뚝뚝 끊긴 텍스트를 자연스러운 문장으로 합치는 기능"""
    if not raw_segments: return []

    refined = []
    current_group = raw_segments[0]

    for next_seg in raw_segments[1:]:
        time_gap = next_seg['start'] - current_group['end']
        is_sentence_end = current_group['text'].strip().endswith(('.', '?', '!'))
        
        if time_gap < max_gap and not is_sentence_end and len(current_group['text']) + len(next_seg['text']) < max_chars:
            current_group['text'] += " " + next_seg['text']
            current_group['end'] = next_seg['end']
        else:
            refined.append(current_group)
            current_group = next_seg
            
    refined.append(current_group)
    return refined

# 메인 실행
if __name__ == "__main__":
    total_start = time.time()
    
    if not os.path.exists(VIDEO_PATH):
        print(f"영상 파일이 없습니다: {VIDEO_PATH}")
    else:
        # 1. 오디오 추출
        if extract_audio(VIDEO_PATH, AUDIO_OUTPUT_PATH):
            
            # 2. 서버 요청 (EXTRA_MATCH_INFO 함께 전송)
            server_response = request_transcription(AUDIO_OUTPUT_PATH, COLAB_SERVER_URL, EXTRA_MATCH_INFO)
            
            if server_response:
                raw_segments = server_response.get("segments", [])
                print(f"원본 조각 개수: {len(raw_segments)}개")

                # 3. 문장 다듬기
                refined_data = refine_segments(raw_segments)
                print(f"다듬어진 문장 개수: {len(refined_data)}개")
                
                # 4. 결과 저장
                final_output = []
                print("-" * 50)
                for seg in refined_data:
                    item = {
                        "label": "commentary",
                        "text": seg['text'].strip(),
                        "start_time": sec_to_time(seg['start']),
                        "end_time": sec_to_time(seg['end']),
                        "duration": round(seg['end'] - seg['start'], 2),
                        "start_seconds": round(seg['start'], 2),
                        "end_seconds": round(seg['end'], 2)
                    }
                    final_output.append(item)
                    print(f"[{item['start_time']}~{item['end_time']}] {item['text']}")
                
                with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(final_output, f, indent=4, ensure_ascii=False)
                
                elapsed = time.time() - total_start
                m = int(elapsed // 60)
                s = int(elapsed % 60)

                print("-" * 50)
                print(f"변환 끝! 결과 파일: {JSON_OUTPUT_PATH}")
                print(f"총 소요 시간: {m}분 {s}초")