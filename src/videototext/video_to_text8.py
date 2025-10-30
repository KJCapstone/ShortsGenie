import os
import ffmpeg
import librosa
import numpy as np
from faster_whisper import WhisperModel
import torch
import soundfile as sf
import time

# ---------- 사용자 설정 ----------
VIDEO_PATH = "match.mp4"
AUDIO_FILE_WAV = "temp_audio.wav"
CHUNK_DIR = "librosa_chunks"
OUTPUT_TXT = "script.txt"

MODEL_FAST = "tiny"   # 빠른 스캔용
MODEL_FULL = "base"   # 최종 변환용
LANG = "ko"

RMS_THRESHOLD = 0.02       # 발화 판단 기준
MIN_CHUNK_LEN = 1.0         # 최소 청크 길이 (초)
MAX_CHUNK_LEN = 30.0        # 최대 청크 길이 (초)
KEYWORDS = ["골", "축구", "득점"]  # 관심 키워드

# ---------- 함수 정의 ----------

# 1️⃣ 오디오 추출
def extract_audio(video_path, audio_path):
    print(f"🎧 오디오 추출: {video_path}")
    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # 절대경로 지정

    (
        ffmpeg
        .input(video_path)
        .output(audio_path, ac=1, ar=16000, acodec="pcm_s16le")
        .overwrite_output()
        .run(cmd=ffmpeg_path, quiet=True)
    )
    print(f"✅ 오디오 추출 완료 → {audio_path}")

# 2️⃣ RMS 기반 발화 구간 추출
def extract_speech_chunks(audio_path, chunk_dir,
                          rms_thresh=RMS_THRESHOLD,
                          min_len=MIN_CHUNK_LEN,
                          max_len=MAX_CHUNK_LEN):
    print(f"\n🔇 RMS 기반 발화 구간 추출: {audio_path}")
    y, sr = librosa.load(audio_path, sr=16000)
    hop_length = 512
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    speech_mask = rms > rms_thresh

    # 연속 구간 병합
    chunks = []
    start = None
    for t, is_speech in zip(times, speech_mask):
        if is_speech and start is None:
            start = t
        elif not is_speech and start is not None:
            end = t
            if end - start >= min_len:
                chunks.append((start, min(end, start+max_len)))
            start = None
    if start is not None:
        chunks.append((start, min(times[-1], start+max_len)))

    # 파일로 저장
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_paths = []
    for i, (start, end) in enumerate(chunks):
        path = os.path.join(chunk_dir, f"chunk_{i}.wav")
        sf.write(path, y[int(start*sr):int(end*sr)], sr)
        chunk_paths.append((path, start, end))

    print(f"✅ {len(chunks)}개 청크 생성 → {chunk_dir}")
    return chunk_paths

# 3️⃣ 빠른 모델로 키워드 포함 청크 선택
def filter_chunks_by_keyword(chunk_paths, model_size=MODEL_FAST, lang=LANG, keywords=KEYWORDS):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device=="cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    selected_chunks = []
    for path, start_s, end_s in chunk_paths:
        segments, _ = model.transcribe(path, language=lang)
        text = " ".join(seg.text for seg in segments).lower()
        if any(k.lower() in text for k in keywords):
            selected_chunks.append((path, start_s, end_s))
    print(f"✅ {len(selected_chunks)}/{len(chunk_paths)}개 청크가 키워드 포함")
    return selected_chunks

# 4️⃣ 최종 변환 + 타임스탬프
def transcribe_final(chunks, output_txt, model_size=MODEL_FULL, lang=LANG):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device=="cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    start_all = time.perf_counter()
    with open(output_txt, "w", encoding="utf-8") as f_out:
        f_out.write("===== [Transcript with Timestamps] =====\n")
        for i, (path, start_s, _) in enumerate(chunks):
            chunk_start = time.perf_counter()
            print(f"▶️ 변환 중: 청크 {i+1}/{len(chunks)} → {path}")
            segments, _ = model.transcribe(path, language=lang)
            for seg in segments:
                seg_start = start_s + seg.start
                seg_end = start_s + seg.end
                line = f"[{seg_start:.1f}~{seg_end:.1f}] {seg.text.strip()}\n"
                f_out.write(line)
            elapsed = time.perf_counter() - chunk_start
            print(f"✅ 청크 {i+1} 완료 ({elapsed:.1f}s)")
    print(f"\n✅ 전체 변환 완료 → {os.path.abspath(output_txt)}")
    print(f"총 소요 시간: {time.perf_counter()-start_all:.1f}s")

# ----------------- 실행 -----------------
if __name__ == "__main__":
    extract_audio(VIDEO_PATH, AUDIO_FILE_WAV)
    chunks = extract_speech_chunks(AUDIO_FILE_WAV, CHUNK_DIR)
    selected_chunks = filter_chunks_by_keyword(chunks)
    transcribe_final(selected_chunks, OUTPUT_TXT)

    # 임시 오디오 제거
    if os.path.exists(AUDIO_FILE_WAV):
        os.remove(AUDIO_FILE_WAV)
