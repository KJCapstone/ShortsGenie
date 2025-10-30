import ffmpeg
from faster_whisper import WhisperModel
import torch

# 🎬 1️⃣ 변환할 축구 영상 경로
video_path = "match.mp4"
audio_path = "match_audio.wav"

# 🎧 2️⃣ 영상 → 오디오 추출 (16kHz, 모노)
print("🎧 영상에서 오디오 추출 중...")
ffmpeg.input(video_path).output(
    audio_path, ac=1, ar=16000, format="wav"
).run(overwrite_output=True)

# ⚙️ 3️⃣ Whisper 모델 불러오기
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Whisper 모델 로드 중... (device={device})")

model = WhisperModel("small", device=device)  # tiny/base/small/medium/large 선택 가능

# 🗣️ 4️⃣ 음성 → 텍스트 변환
print("🗣️ 음성 인식 시작...")
segments, info = model.transcribe(audio_path, beam_size=5, language="ko")

# 📝 5️⃣ 결과 저장
output_txt = "match_transcript.txt"
with open(output_txt, "w", encoding="utf-8") as f:
    for seg in segments:
        line = f"[{seg.start:.1f}~{seg.end:.1f}] {seg.text.strip()}\n"
        print(line, end="")
        f.write(line)

print(f"\n✅ 변환 완료! 결과 저장: {output_txt}")
