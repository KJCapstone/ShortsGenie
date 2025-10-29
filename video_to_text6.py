import os
import ffmpeg
from faster_whisper import WhisperModel
import torch
import time
from pydub import AudioSegment, effects
from pydub.silence import split_on_silence

# ---------- 사용자 설정 ----------
VIDEO_PATH = "match.mp4"
AUDIO_FILE_WAV = "temp_audio.wav"
CLEAN_AUDIO_DIR = "clean_chunks"
OUTPUT_TXT = "match_transcript.txt"

MODEL_SIZE = "tiny"   # tiny/base
LANG = "ko"

MIN_SILENCE_LEN = 1000
SILENCE_THRESH = -35
KEEP_SILENCE = 500
# ----------------------------------


def extract_audio(video_path, audio_path):
    """ffmpeg로 영상에서 오디오 추출 (16kHz, 모노, PCM)"""
    print(f"🎧 오디오 추출: {video_path}")
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, ac=1, ar=16000, acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=False)
    )
    print(f"✅ 오디오 추출 완료 → {audio_path}")


def split_and_clean_audio(audio_path, output_dir,
                          min_silence_len, silence_thresh, keep_silence):
    """오디오를 침묵 기준으로 청크 분리 후 normalize"""
    print(f"\n🔇 침묵 기준 청크 분리: {audio_path}")
    audio = AudioSegment.from_wav(audio_path)
    audio = effects.normalize(audio)

    avg_db = audio.dBFS
    adjusted_thresh = silence_thresh if avg_db < silence_thresh else avg_db - 10
    print(f"평균 볼륨: {avg_db:.1f} dBFS → 임계값: {adjusted_thresh:.1f} dBFS")

    chunks = split_on_silence(audio,
                              min_silence_len=min_silence_len,
                              silence_thresh=adjusted_thresh,
                              keep_silence=keep_silence)

    if not chunks:
        raise Exception("❌ 발화 구간을 찾을 수 없습니다. SILENCE_THRESH 조정 필요")

    os.makedirs(output_dir, exist_ok=True)
    chunk_paths = []
    chunk_times_in_audio = []

    start_ms = 0
    for i, chunk in enumerate(chunks):
        path = os.path.join(output_dir, f"chunk_{i}.wav")
        chunk.export(path, format="wav")
        chunk_paths.append(path)
        # 청크 시작/끝 시간 기록 (ms → s)
        end_ms = start_ms + len(chunk)
        chunk_times_in_audio.append((start_ms/1000.0, end_ms/1000.0))
        start_ms = end_ms

    print(f"✅ 총 {len(chunks)}개 청크 생성 → {output_dir}")
    return chunk_paths, chunk_times_in_audio


def transcribe_chunks_merge_text(chunk_paths, chunk_times, output_txt, model_size, lang):
    """청크별 Whisper 변환 후 텍스트 병합 + 영상 타임스탬프 + 청크 처리 시간 기록"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"\n⚙️ Whisper 모델 로드: {model_size}, {device}, {compute_type}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    all_text = []
    start_time_all = time.perf_counter()

    with open(output_txt, "w", encoding="utf-8") as f_out:
        f_out.write(f"===== [Transcript with Timestamps] Source: {os.path.basename(chunk_paths[0])} =====\n")

        for i, path in enumerate(chunk_paths):
            chunk_start_time = time.perf_counter()
            start_s, end_s = chunk_times[i]
            print(f"▶️ 변환 중: 청크 {i + 1}/{len(chunk_paths)} → {path}")
            
            segments, _ = model.transcribe(path, beam_size=5, language=lang)

            chunk_elapsed = time.perf_counter() - chunk_start_time
            print(f"✅ 청크 {i + 1} 완료 (소요: {chunk_elapsed:.1f}s)")

            for seg in segments:
                text = seg.text.strip()
                if text:
                    # 영상 기준 타임스탬프 계산
                    seg_start = start_s + seg.start
                    seg_end = start_s + seg.end
                    line = f"[{seg_start:.1f}~{seg_end:.1f}] {text}\n"
                    f_out.write(line)
                    all_text.append(text)

    total_elapsed = time.perf_counter() - start_time_all
    print(f"\n✅ 전체 변환 완료 → {os.path.abspath(output_txt)}")
    print(f"총 소요 시간: {total_elapsed:.1f}s")


if __name__ == "__main__":
    try:
        # 1️⃣ 오디오 추출
        extract_audio(VIDEO_PATH, AUDIO_FILE_WAV)

        # 2️⃣ 청크 단위 분리
        chunk_paths, chunk_times = split_and_clean_audio(AUDIO_FILE_WAV, CLEAN_AUDIO_DIR,
                                                         MIN_SILENCE_LEN, SILENCE_THRESH, KEEP_SILENCE)

        # 3️⃣ 청크별 변환 + 영상 타임스탬프 + 청크 처리 시간
        transcribe_chunks_merge_text(chunk_paths, chunk_times, OUTPUT_TXT, MODEL_SIZE, LANG)

    except Exception as e:
        print(f"\n🚨 오류 발생: {e}")

    finally:
        # 4️⃣ 임시 파일 정리
        try:
            if os.path.exists(AUDIO_FILE_WAV):
                os.remove(AUDIO_FILE_WAV)
        except OSError as e:
            print(f"임시 파일 삭제 오류: {e}")
