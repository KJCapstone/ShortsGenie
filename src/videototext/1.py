import os
import io
import time
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment, silence
from google.cloud import speech_v1p1beta1 as speech
# 💡 Silero VAD를 위한 라이브러리 추가 
import torch
import torchaudio
from scipy.ndimage.measurements import label as scipy_label 


# =====================================================
# ✅ 설정 (수정됨)
# =====================================================
INPUT_AUDIO_FILE = "match_audio.wav"
OUTPUT_DIR = "highlight_v2"

# ❌ 보안 이슈로 인해 하드코딩된 GOOGLE_CREDENTIALS 경로는 삭제합니다.
# GOOGLE_CREDENTIALS = r"C:\Users\home\Desktop\shortsgenie\ShortsGenie\src\videototext\diesel-channel-477619-u6-db0de75fbe60.json" 

# ✅ 코드는 이제 실행 전에 사용자가 환경 변수 'GOOGLE_APPLICATION_CREDENTIALS'를
# ✅ 직접 설정했음을 가정하고 진행합니다.

# 현재 디렉토리 구조에서 환경 변수가 설정되지 않았다면 다음 코드는 오류를 발생시키므로,
# 임시적으로 코드를 다음과 같이 변경하여 환경 변수가 설정되어 있는지 확인합니다.
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    # 🚨🚨🚨 이 줄은 임시 코드이며, 실제로는 환경 변수를 쉘에서 설정해야 합니다. 🚨🚨🚨
    print("\n⚠️ 경고: GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않았습니다.")
    # 개발 편의를 위해 설정 파일 경로를 환경 변수에 설정해야 한다면, 이 부분을 주석 해제하고 사용하세요.
    # 단, 이 파일은 반드시 .gitignore에 추가되어야 합니다.
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\home\Desktop\shortsgenie\ShortsGenie\src\videototext\diesel-channel-477619-u6-db0de75fbe60.json" 
    pass


os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# ✅ 공용 유틸: 처리시간 측정 & 타임 포맷
# (나머지 코드는 이전과 동일)
# =====================================================
def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"✅ {func.__name__} 완료 (걸린 시간: {end - start:.2f}초)\n")
        return result
    return wrapper

# 초를 HH:MM:SS.msms 포맷으로 변환
def format_time(seconds):
    """초를 HH:MM:SS.msms 포맷으로 변환"""
    # 소수점 둘째 자리까지 표시 (10ms 단위)
    ms = int((seconds - int(seconds)) * 100)
    s = int(seconds)
    m = s // 60
    h = m // 60
    return f"{h:02d}:{m % 60:02d}:{s % 60:02d}.{ms:02d}"

# =====================================================
# 💡 Silero VAD 유틸리티
# =====================================================
@timed
def load_silero_vad_model():
    """Silero VAD 모델 로드"""
    print("🧠 Silero VAD 모델 로드 중...")
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        return model, get_speech_timestamps, read_audio
    except Exception as e:
        print(f"❌ Silero VAD 모델 로드 실패. 필요한 라이브러리 (torch, torchaudio) 설치 및 인터넷 연결을 확인하세요. 오류: {e}")
        return None, None, None


# =====================================================
# [1/5] 오디오 전처리
# =====================================================
@timed
def preprocess_audio(input_path):
    print("🎧 [1/5] 오디오 전처리 중...")
    audio = AudioSegment.from_file(input_path)
    original_len = len(audio) / 1000

    # ⚠️ VAD는 음성만을 검출하므로, 전처리에서 과도한 Silence 제거는 신중해야 함
    # 원본 오디오를 16kHz, 모노로 변환 및 정규화만 수행
    cleaned = audio.set_frame_rate(16000).set_channels(1)
    cleaned = cleaned.apply_gain(-20 - cleaned.dBFS) # 대략 -20dBFS로 정규화

    output_path = os.path.join(OUTPUT_DIR, "cleaned_audio.wav")
    cleaned.export(output_path, format="wav", codec="pcm_s16le")

    print(f"📦 원본길이: {original_len:.1f}s → 전처리 후: {len(cleaned)/1000:.1f}s\n")
    return output_path


# =====================================================
# [2/5] Silero VAD 기반 하이라이트 후보 탐지 
# =====================================================
@timed
def detect_highlight_candidates_vad(audio_path, model, get_speech_timestamps, read_audio, 
                                    vad_threshold=0.8, min_clip_duration=2.0):
    print("🗣️ [2/5] Silero VAD 기반 하이라이트 후보 탐지 중...")
    
    # 1. 오디오 로드 및 텐서 변환 (Silero VAD 요구사항)
    SAMPLING_RATE = 16000 # 모델이 16000Hz를 요구
    try:
        # VAD 모델이 요구하는 텐서 형태로 로드
        audio_tensor = read_audio(audio_path, sampling_rate=SAMPLING_RATE)
    except Exception as e:
        print(f"❌ 오디오 텐서 변환 실패: {e}")
        return []

    # 2. 음성 활동 타임스탬프 추출
    speech_timestamps = get_speech_timestamps(
        audio_tensor, 
        model, 
        sampling_rate=SAMPLING_RATE,
        threshold=vad_threshold,
        min_speech_duration_ms=int(min_clip_duration * 1000), 
        min_silence_duration_ms=400 # 짧은 침묵은 연결
    )

    if not speech_timestamps:
        print("⚠️ VAD로 탐지된 음성 구간이 없습니다.")
        return []

    # 3. 샘플 인덱스를 초 단위로 변환
    candidates = []
    for ts in speech_timestamps:
        start_s = ts['start'] / SAMPLING_RATE
        end_s = ts['end'] / SAMPLING_RATE
        
        if end_s - start_s >= min_clip_duration:
             candidates.append((start_s, end_s))
    
    print(f"⚽ 탐지된 후보 구간 수: {len(candidates)}개 (VAD 임계값: {vad_threshold}, 최소 {min_clip_duration}s 이상 필터링)\n")
    return candidates


# =====================================================
# [3/5] 후보 구간 오디오 추출
# =====================================================
@timed
def extract_highlight_clips(audio_path, candidates):
    print("✂️ [3/5] 후보 구간 오디오 추출 중...")
    audio = AudioSegment.from_file(audio_path)
    clip_paths = []

    for i, (start, end) in enumerate(candidates):
        clip = audio[start * 1000 : end * 1000]
        clip_path = os.path.join(OUTPUT_DIR, f"highlight_{i}.wav")
        clip.export(clip_path, format="wav", codec="pcm_s16le", parameters=["-ac", "1", "-ar", "16000"])
        clip_paths.append(clip_path)
        print(f"🗂️ 구간 {i+1}: {start:.1f}s ~ {end:.1f}s ({(end-start):.1f}s)")

    print(f"✅ 총 {len(clip_paths)}개 하이라이트 클립 생성 완료\n")
    return clip_paths


# =====================================================
# [4/5] Google STT (후보 구간만)
# =====================================================
def stt_google_sdk(audio_path):
    """Google STT (10MB/60초 이하만)"""
    client = speech.SpeechClient()

    with io.open(audio_path, "rb") as f:
        audio_content = f.read()

    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        speech_contexts=[
            speech.SpeechContext(
                phrases=[
                    "골", "슛", "득점", "세이브", "찬스", "패스", "드리블",
                    "프리킥", "코너킥", "오프사이드", "패널티킥", "VAR",
                    "키퍼", "수비", "공격", "크로스", "심판", "헤딩"
                ],
                boost=15.0 
            )
        ]
    )

    response = client.recognize(config=config, audio=audio)
    
    sentence_data = []
    
    for result in response.results:
        words = result.alternatives[0].words
        
        if not words:
            continue
            
        start_time_s = words[0].start_time.total_seconds() 
        end_time_s = words[-1].end_time.total_seconds()
        
        text = result.alternatives[0].transcript
        
        sentence_data.append((start_time_s, end_time_s, text))
        
    return sentence_data


@timed
def transcribe_candidates(clip_paths, candidates):
    print("🗣️ [4/5] 후보 구간 STT 변환 중 (STT 60초 제한 대응 자동 분할)...")
    
    final_transcripts = [] 

    for i, (path, (clip_start_original, _)) in enumerate(zip(clip_paths, candidates)):
        print(f"\n🎧 ({i+1}/{len(clip_paths)}) 변환 중: {os.path.basename(path)}")

        audio = AudioSegment.from_file(path)
        duration_ms = len(audio)

        MAX_DURATION_MS = 58 * 1000 
        
        temp_sentence_data = []

        if duration_ms > MAX_DURATION_MS:
            print(f"⚠️ 클립 길이 {duration_ms/1000:.1f}s (60초 초과) → STT 제한에 맞춰 {MAX_DURATION_MS/1000:.1f}초 단위로 자동 분할 실행")

            start_ms = 0
            part_num = 1
            while start_ms < duration_ms:
                end_ms = min(start_ms + MAX_DURATION_MS, duration_ms)
                part = audio[start_ms:end_ms]
                temp_path = f"{path[:-4]}_part{part_num}.wav"
                
                part.export(temp_path, format="wav", codec="pcm_s16le", parameters=["-ac", "1", "-ar", "16000"])
                print(f"  ↳ 파트 {part_num}: {start_ms/1000:.1f}s~{end_ms/1000:.1f}s ({(end_ms-start_ms)/1000:.1f}s)")
                
                try:
                    part_sentences = stt_google_sdk(temp_path)
                    
                    offset_s = start_ms / 1000.0
                    for s_start, s_end, s_text in part_sentences:
                        temp_sentence_data.append((s_start + offset_s, s_end + offset_s, s_text))
                        
                except Exception as e:
                    print(f"  ❌ 파트 {part_num} 실패: {e}")
                os.remove(temp_path)
                start_ms = end_ms
                part_num += 1
                
        else:
            try:
                temp_sentence_data = stt_google_sdk(path)
            except Exception as e:
                print(f"  ❌ 변환 실패: {e}")
        
        offset_original_s = clip_start_original
        for s_start, s_end, s_text in temp_sentence_data:
            final_transcripts.append((s_start + offset_original_s, s_end + offset_original_s, s_text))


    full_text = "\n".join([t[2] for t in final_transcripts])
    with open(os.path.join(OUTPUT_DIR, "transcript_candidates.txt"), "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"✅ 후보 구간 텍스트 저장 완료\n")
    return final_transcripts 


# =====================================================
# [5/5] 키워드 기반 하이라이트 텍스트 추출
# =====================================================
@timed
def extract_keyword_highlights(final_transcripts, keywords=["골", "슛", "득점", "세이브", "찬스"]):
    print("⚽ [5/5] 텍스트 기반 하이라이트 추출 중...")

    highlights = []
    
    for start, end, text in final_transcripts:
        if not text:
            continue
            
        if any(k in text for k in keywords):
            highlights.append(f"[{format_time(start)} ~ {format_time(end)}] {text.strip()}")

    with open(os.path.join(OUTPUT_DIR, "highlight_result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(highlights))

    print(f"✅ 키워드 하이라이트 {len(highlights)}개 추출 완료\n")
    return highlights


# =====================================================
# 🚀 전체 파이프라인 실행
# =====================================================
if __name__ == "__main__":
    total_start = time.time()
    
    # 💡 VAD 모델 로드
    vad_model, get_speech_timestamps, read_audio = load_silero_vad_model()
    
    if vad_model is None or not os.path.exists(INPUT_AUDIO_FILE):
        if not os.path.exists(INPUT_AUDIO_FILE):
             print(f"❌ 오류: 입력 오디오 파일 '{INPUT_AUDIO_FILE}'을 찾을 수 없습니다. 경로를 확인해주세요.")
        # VAD 모델 로드 실패 시에도 종료
        print("\n🚫 Silero VAD 로드 또는 파일 오류로 파이프라인을 실행할 수 없습니다.")
        
    # 환경 변수가 설정되어 있지 않으면 경고 후 종료 (선택 사항: 실행을 막을 수도 있습니다.)
    elif "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("\n🚨🚨🚨 오류: GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않아 Google STT를 실행할 수 없습니다. 🚨🚨🚨")

    else:
        cleaned = preprocess_audio(INPUT_AUDIO_FILE)
        
        # 💡 RMS 대신 VAD 함수 호출
        candidates = detect_highlight_candidates_vad(
            cleaned, 
            vad_model, 
            get_speech_timestamps, 
            read_audio, 
            vad_threshold=0.8, # VAD 임계값 (0.1 ~ 0.99)
            min_clip_duration=2.0
        )
        
        if not candidates:
            print("\n🚨 VAD 탐지 결과, 유효한 하이라이트 후보가 없어 종료합니다.")
        else:
            clips = extract_highlight_clips(cleaned, candidates)
            
            final_transcripts = transcribe_candidates(clips, candidates)
            
            extract_keyword_highlights(final_transcripts)

            print(f"\n🎬 전체 완료! 총 소요 시간: {time.time() - total_start:.2f}초 🎉")