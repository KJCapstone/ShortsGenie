import os
import subprocess
import json

# ---------- 사용자 설정 -----------
INPUT_TRANSCRIPT = "match_transcript_1.txt"  # Whisper 변환 결과 텍스트
OUTPUT_JSON = "match_scene.json"
LLAMA_MODEL = "llama3"  # 로컬 Llama 모델 이름
# ----------------------------------

def load_transcript(file_path):
    """Whisper 텍스트 불러오기"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"입력 파일이 없습니다: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def llama3_json_summarize(transcript_text, output_json, model_name):
    """Llama3 로컬 모델로 JSON 장면 요약"""
    prompt = f"""
다음은 축구 경기 텍스트입니다.
각 발화를 분석해서 JSON 형식으로 장면 요약을 만들어 주세요.

JSON 형식:
[
  {{
    "start": "시작 시간(초)",
    "end": "종료 시간(초)",
    "description": "장면 설명 (누가 골 넣는지, 패스 등)"
  }}
]

텍스트:
\"\"\"
{transcript_text}
\"\"\"
"""

    # 임시 프롬프트 파일 생성
    temp_prompt_file = "temp_prompt.txt"
    with open(temp_prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    # Llama3 로컬 모델 실행
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, "--prompt-file", temp_prompt_file],
            capture_output=True,
            text=True,
            check=True
        )
        llm_output = result.stdout.strip()

        # JSON 저장
        with open(output_json, "w", encoding="utf-8") as f:
            f.write(llm_output)

        print(f"✅ Llama3 장면 요약 JSON 저장 → {os.path.abspath(output_json)}")
        print("출력 예시 (앞부분):")
        print(llm_output[:500], "...")

    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_prompt_file):
            os.remove(temp_prompt_file)

if __name__ == "__main__":
    try:
        transcript_text = load_transcript(INPUT_TRANSCRIPT)
        llama3_json_summarize(transcript_text, OUTPUT_JSON, LLAMA_MODEL)
    except Exception as e:
        print(f"\n🚨 오류 발생: {e}")
