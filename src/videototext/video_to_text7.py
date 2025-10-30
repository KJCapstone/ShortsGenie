import os
import ffmpeg
from faster_whisper import WhisperModel
import torch
import time
import librosa
import soundfile as sf
import shutil # 임시 폴더 삭제를 위해 추가

# ---------- 사용자 설정 ----------
VIDEO_PATH = "match.mp4"
AUDIO_FILE_WAV = "temp_audio.wav"
CLEAN_AUDIO_DIR = "clean_chunks"
OUTPUT_TXT = "script.txt"

MODEL_SIZE = "tiny"   # 사용 가능한 모델: "tiny", "base", "small", "medium", "large-v3"
LANG = "ko"           # 언어 코드 (e.g., "ko", "en", "ja")

# librosa.effects.split을 위한 설정값
# 음량이 가장 큰 부분을 기준으로 top_db 데시벨 만큼 낮은 소리까지를 유효한 소리로 간주
# 값이 작을수록 침묵을 더 엄격하게 판단하여 더 잘게 쪼갬
TOP_DB = 35
# ----------------------------------


def extract_audio(video_path, audio_path):
    """ffmpeg로 영상에서 오디오 추출 (16kHz, 모노, PCM)"""
    print(f"🎧 오디오 추출 시작: {video_path}")
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, ac=1, ar=16000, acodec="pcm_s16le")
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        print("ffmpeg 오류 발생:")
        print(e.stderr.decode())
        raise


def fast_split_audio_with_librosa(audio_path, output_dir, top_db):
    """librosa를 사용하여 오디오를 빠르게 비음성 구간 기준으로 분할"""
    print(f"🔇 librosa로 청크 분리 시작: {audio_path}")
    
    y, sr = librosa.load(audio_path, sr=16000)
    intervals = librosa.effects.split(y, top_db=top_db)

    if len(intervals) == 0:
        raise Exception("발화 구간을 찾을 수 없습니다. TOP_DB 값을 조절해보세요.")

    os.makedirs(output_dir, exist_ok=True)
    chunk_paths = []
    chunk_times_in_audio = []

    for i, (start_frame, end_frame) in enumerate(intervals):
        chunk = y[start_frame:end_frame]
        path = os.path.join(output_dir, f"chunk_{i}.wav")
        sf.write(path, chunk, sr)
        chunk_paths.append(path)

        start_s = librosa.frames_to_time(start_frame, sr=sr)
        end_s = librosa.frames_to_time(end_frame, sr=sr)
        chunk_times_in_audio.append((start_s, end_s))

    print(f"✅ 총 {len(intervals)}개 청크 생성 완료 → {output_dir}")
    return chunk_paths, chunk_times_in_audio


def transcribe_chunks_merge_text(chunk_paths, chunk_times, output_txt, model_size, lang):
    """청크별 Whisper 변환 후 텍스트 병합 및 타임스탬프 기록"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"\n🎤 Whisper 모델 로드 시작: {model_size} (Device: {device}, Compute Type: {compute_type})")
    model_load_start = time.perf_counter()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    model_load_elapsed = time.perf_counter() - model_load_start
    print(f"✅ Whisper 모델 로드 완료 (소요 시간: {model_load_elapsed:.2f}초)")
    
    transcribe_start_time = time.perf_counter()

    with open(output_txt, "w", encoding="utf-8") as f_out:
        f_out.write(f"===== [음성인식 스크립트] 원본: {VIDEO_PATH} =====\n\n")

        for i, path in enumerate(chunk_paths):
            print(f"▶️ 변환 중: 청크 {i + 1}/{len(chunk_paths)} ({path})")
            
            segments, _ = model.transcribe(path, beam_size=5, language=lang)
            start_s_in_video, _ = chunk_times[i]

            for seg in segments:
                text = seg.text.strip()
                if text:
                    seg_start = start_s_in_video + seg.start
                    seg_end = start_s_in_video + seg.end
                    line = f"[{seg_start:0>7.2f} ~ {seg_end:0>7.2f}] {text}\n"
                    f_out.write(line)

    transcribe_elapsed = time.perf_counter() - transcribe_start_time
    print(f"✅ 모든 청크 변환 완료 (음성 인식 소요 시간: {transcribe_elapsed:.2f}초)")
    print(f"결과 파일이 다음 경로에 저장되었습니다: {os.path.abspath(output_txt)}")


if __name__ == "__main__":
    total_start_time = time.perf_counter()
    
    try:
        # --- 1단계: 영상에서 오디오 추출 ---
        step1_start_time = time.perf_counter()
        extract_audio(VIDEO_PATH, AUDIO_FILE_WAV)
        step1_elapsed = time.perf_counter() - step1_start_time
        print(f"✅ 1단계: 오디오 추출 완료 (소요 시간: {step1_elapsed:.2f}초)")

        # --- 2단계: 오디오를 청크 단위로 분리 ---
        step2_start_time = time.perf_counter()
        chunk_paths, chunk_times = fast_split_audio_with_librosa(AUDIO_FILE_WAV, CLEAN_AUDIO_DIR, top_db=TOP_DB)
        step2_elapsed = time.perf_counter() - step2_start_time
        print(f"✅ 2단계: 오디오 청크 분리 완료 (소요 시간: {step2_elapsed:.2f}초)")

        # --- 3단계: 청크별 음성 인식 및 결과 병합 ---
        step3_start_time = time.perf_counter()
        transcribe_chunks_merge_text(chunk_paths, chunk_times, OUTPUT_TXT, MODEL_SIZE, LANG)
        step3_elapsed = time.perf_counter() - step3_start_time
        print(f"✅ 3단계: 음성 인식 및 병합 완료 (소요 시간: {step3_elapsed:.2f}초)")

    except Exception as e:
        print(f"\n❗️오류 발생: {e}")

    finally:
        # --- 4단계: 임시 파일 및 폴더 정리 ---
        print("\n🧹 임시 파일 정리 중...")
        try:
            if os.path.exists(AUDIO_FILE_WAV):
                os.remove(AUDIO_FILE_WAV)
            if os.path.exists(CLEAN_AUDIO_DIR):
                shutil.rmtree(CLEAN_AUDIO_DIR)
            print("✅ 임시 파일 정리 완료.")
        except OSError as e:
            print(f"❗️임시 파일 삭제 중 오류 발생: {e}")
            
    total_elapsed = time.perf_counter() - total_start_time
    print(f"\n🎉 모든 작업이 완료되었습니다. (총 소요 시간: {total_elapsed:.2f}초)")