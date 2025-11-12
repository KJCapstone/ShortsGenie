import os
import io
import time
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage # 💡 GCS 라이브러리 추가
import torch
import torchaudio
from scipy.ndimage.measurements import label as scipy_label 


# =====================================================
# ✅ 설정 (수정됨)
# =====================================================
INPUT_AUDIO_FILE = "match_audio.wav"
OUTPUT_DIR = "highlight_v2"

# ⚠️ GCS 버킷 이름: 실제 사용자의 버킷 이름으로 변경하세요!
GCS_BUCKET_NAME = "shortsgenie-voice" 

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# ✅ 공용 유틸: 처리시간 측정 & 타임 포맷
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
    ms = int((seconds - int(seconds)) * 100)
    s = int(seconds)
    m = s // 60
    h = m // 60
    return f"{h:02d}:{m % 60:02d}:{s % 60:02d}.{ms:02d}"


# =====================================================
# 💡 GCS 유틸리티 (비동기 STT 필수)
# =====================================================
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """로컬 파일을 GCS에 업로드"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # 로컬 파일을 GCS로 업로드
    blob.upload_from_filename(source_file_name)
    
    # GCS URI 형식으로 반환
    return f"gs://{bucket_name}/{destination_blob_name}"


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
        print(f"❌ Silero VAD 모델 로드 실패. 오류: {e}")
        return None, None, None


# =====================================================
# [1/5] 오디오 전처리
# =====================================================
@timed
def preprocess_audio(input_path):
    print("🎧 [1/5] 오디오 전처리 중...")
    audio = AudioSegment.from_file(input_path)
    original_len = len(audio) / 1000

    # 원본 오디오를 16kHz, 모노로 변환 및 정규화
    cleaned = audio.set_frame_rate(16000).set_channels(1)
    cleaned = cleaned.apply_gain(-20 - cleaned.dBFS) # 대략 -20dBFS로 정규화

    output_path = os.path.join(OUTPUT_DIR, "cleaned_audio.wav")
    cleaned.export(output_path, format="wav", codec="pcm_s16le")

    print(f"📦 원본길이: {original_len:.1f}s → 전처리 후: {len(cleaned)/1000:.1f}s\n")
    return output_path


# =====================================================
# [2/5] Silero VAD 기반 하이라이트 후보 탐지 (최적화 적용)
# =====================================================
@timed
def detect_highlight_candidates_vad(audio_path, model, get_speech_timestamps, read_audio, 
                                    vad_threshold=0.9, min_clip_duration=3.0): # 💡 임계값 상향
    print("🗣️ [2/5] Silero VAD 기반 하이라이트 후보 탐지 중...")
    
    SAMPLING_RATE = 16000 
    try:
        audio_tensor = read_audio(audio_path, sampling_rate=SAMPLING_RATE)
    except Exception as e:
        print(f"❌ 오디오 텐서 변환 실패: {e}")
        return []

    # 음성 활동 타임스탬프 추출
    speech_timestamps = get_speech_timestamps(
        audio_tensor, 
        model, 
        sampling_rate=SAMPLING_RATE,
        threshold=vad_threshold,
        min_speech_duration_ms=int(min_clip_duration * 1000), 
        min_silence_duration_ms=500 # 💡 짧은 침묵은 연결
    )

    if not speech_timestamps:
        print("⚠️ VAD로 탐지된 음성 구간이 없습니다.")
        return []

    # 샘플 인덱스를 초 단위로 변환 및 필터링
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
        # pydub는 ms 단위
        clip = audio[start * 1000 : end * 1000] 
        clip_path = os.path.join(OUTPUT_DIR, f"highlight_{i}.wav")
        # STT에 적합하도록 포맷 설정
        clip.export(clip_path, format="wav", codec="pcm_s16le", parameters=["-ac", "1", "-ar", "16000"]) 
        clip_paths.append(clip_path)
        print(f"🗂️ 구간 {i+1}: {start:.1f}s ~ {end:.1f}s ({(end-start):.1f}s)")

    print(f"✅ 총 {len(clip_paths)}개 하이라이트 클립 생성 완료\n")
    return clip_paths


# =====================================================
# [4/5] Google STT 비동기 처리 (속도 개선)
# =====================================================
@timed
def transcribe_candidates_async(clip_paths, candidates, bucket_name):
    print("🗣️ [4/5] 비동기 STT 변환 중 (다중 클립 동시 병렬 처리로 속도 개선)...")

    speech_client = speech.SpeechClient()
    
    # 60초 제한을 신경 쓸 필요가 없어짐
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        speech_contexts=[
            speech.SpeechContext(
                phrases=["골", "슛", "득점", "세이브", "찬스", "패스", "드리블", "프리킥", "코너킥", "오프사이드", "패널티킥", "VAR"], 
                boost=15.0
            )
        ]
    )
    
    # 1. 모든 클립을 GCS에 업로드하고 비동기 요청 시작
    operations = []
    gcs_files_info = []
    
    for i, (path, (clip_start_original, _)) in enumerate(zip(clip_paths, candidates)):
        # GCS 내 객체 이름 설정 (예: highlight_v2/highlight_0.wav)
        gcs_blob_name = os.path.join(OUTPUT_DIR, os.path.basename(path))
        
        # 1-1. GCS에 업로드
        try:
            gcs_uri = upload_blob(bucket_name, path, gcs_blob_name)
            gcs_files_info.append((gcs_uri, clip_start_original)) 
            
            # 1-2. 비동기 STT 요청
            audio = speech.RecognitionAudio(uri=gcs_uri)
            operation = speech_client.long_running_recognize(config=config, audio=audio)
            operations.append(operation)
            print(f"   ▶️ ({i+1}/{len(clip_paths)}) GCS 업로드 및 비동기 요청 시작 완료: {os.path.basename(path)}")
        except Exception as e:
             print(f"   ❌ GCS 업로드 또는 비동기 요청 실패 ({os.path.basename(path)}): {e}")

    # 2. 모든 비동기 작업 결과 수집
    final_transcripts = []
    
    for i, (operation, (gcs_uri, clip_start_original)) in enumerate(zip(operations, gcs_files_info)):
        print(f"\n   ⏳ ({i+1}/{len(operations)}) 결과 대기 중: {os.path.basename(gcs_uri)}")
        
        try:
            # 병렬 처리된 결과가 도착할 때까지 대기
            response = operation.result(timeout=1000) 
        except Exception as e:
            print(f"   ❌ STT 작업 실패 ({os.path.basename(gcs_uri)}): {e}")
            continue

        # 3. 결과 파싱 및 원본 시간 오프셋 적용
        for result in response.results:
            words = result.alternatives[0].words
            
            if not words:
                continue
                
            start_time_s = words[0].start_time.total_seconds() 
            end_time_s = words[-1].end_time.total_seconds()
            text = result.alternatives[0].transcript
            
            # 타임라인을 위해 원본 오디오 기준으로 시간 변환
            final_transcripts.append((start_time_s + clip_start_original, end_time_s + clip_start_original, text))

    full_text = "\n".join([t[2] for t in final_transcripts])
    with open(os.path.join(OUTPUT_DIR, "transcript_candidates_async.txt"), "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"\n✅ 비동기 STT 및 텍스트 저장 완료")
    return final_transcripts


# =====================================================
# [5/5] 키워드 기반 하이라이트 텍스트 추출
# =====================================================
@timed
def extract_keyword_highlights(final_transcripts, keywords=["골", "슛", "득점", "세이브", "찬스", "대박", "환상적인", "미쳤다"]):
    print("⚽ [5/5] 텍스트 기반 하이라이트 추출 중...")

    highlights = []
    
    for start, end, text in final_transcripts:
        if not text:
            continue
            
        # 키워드 탐지
        if any(k in text for k in keywords):
            # 문장 길이가 너무 짧은 것은 제외 (단어 수 3개 미만)
            if len(text.strip().split()) >= 3:
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
    
    # 환경 변수 설정 확인 (STT 및 GCS 접근 필수)
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("\n🚨🚨🚨 오류: GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않아 Google STT/GCS를 실행할 수 없습니다. 🚨🚨🚨")
    elif not os.path.exists(INPUT_AUDIO_FILE):
        print(f"❌ 오류: 입력 오디오 파일 '{INPUT_AUDIO_FILE}'을 찾을 수 없습니다. 경로를 확인해주세요.")
    else:
        # 💡 VAD 모델 로드
        vad_model, get_speech_timestamps, read_audio = load_silero_vad_model()
        
        if vad_model is None:
             print("\n🚫 Silero VAD 로드 실패로 파이프라인을 실행할 수 없습니다.")
        else:
            cleaned = preprocess_audio(INPUT_AUDIO_FILE)
            
            # VAD 최적화 설정으로 후보 탐지
            candidates = detect_highlight_candidates_vad(
                cleaned, 
                vad_model, 
                get_speech_timestamps, 
                read_audio, 
                vad_threshold=0.9, 
                min_clip_duration=3.0
            )
            
            if not candidates:
                print("\n🚨 VAD 탐지 결과, 유효한 하이라이트 후보가 없어 종료합니다.")
            else:
                clips = extract_highlight_clips(cleaned, candidates)
                
                # 💡 비동기 STT 호출 (속도 개선 핵심)
                final_transcripts = transcribe_candidates_async(clips, candidates, bucket_name=GCS_BUCKET_NAME) 
                
                extract_keyword_highlights(final_transcripts)

                print(f"\n🎬 전체 완료! 총 소요 시간: {time.time() - total_start:.2f}초 🎉")