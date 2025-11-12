import os
import time
from google import genai
from google.genai import types

# =====================================================
# ✅ 설정
# =====================================================
# STT 파이프라인에서 생성된 타임라인 텍스트 파일 경로
TRANSCRIPT_FILE = "highlight_v4/transcript_for_llm.txt"
OUTPUT_HIGHLIGHT_FILE = "final_highlights_from_llm.txt"

# LLM 모델 설정
MODEL_NAME = 'gemini-2.5-flash' # 빠르고 강력한 Gemini 모델 사용

# =====================================================
# [1/3] LLM 프롬프트 생성
# =====================================================
def create_llm_prompt(transcript_content):
    """
    LLM에게 전달할 구체적인 지시사항과 텍스트를 결합하여 프롬프트를 생성합니다.
    """
    
    # 💡 LLM에게 명확한 역할과 형식(JSON)을 지시하여 오류를 최소화합니다.
    system_instruction = (
        "당신은 스포츠 중계 전문 분석가입니다. 주어진 텍스트는 축구 경기 중계의 핵심 발화 구간이며, "
        "각 문장 앞에는 [시작 시간 ~ 종료 시간] 형태의 타임라인이 포함되어 있습니다. "
        "사용자의 목표는 편집을 위한 최종 하이라이트 목록을 얻는 것입니다."
    )
    
    user_prompt = (
        "--- 분석 대상 텍스트 시작 ---\n"
        f"{transcript_content}\n"
        "--- 분석 대상 텍스트 종료 ---\n\n"
        
        "위 텍스트를 분석하여, 경기 중 가장 중요하고 극적이며 흥분도가 높은 **5개**의 하이라이트 구간을 추출해 주세요. "
        "하이라이트의 기준은 '골, 슛, 득점, 세이브, 찬스' 등 결정적인 플레이여야 하며, 경기 시작/종료 등의 일반적인 멘트는 제외해야 합니다. "
        
        "출력 형식은 다음과 같은 **Python 리스트 형태의 JSON**이어야 합니다. 다른 설명이나 추가 텍스트 없이 JSON만 출력해 주세요."
        
        "[\n"
        "  {\n"
        '    "시간": "[HH:MM:SS.msms ~ HH:MM:SS.msms]",\n'
        '    "내용": "해설 내용"\n'
        "  },\n"
        "  ...\n"
        "]"
    )
    
    return system_instruction, user_prompt


# =====================================================
# [2/3] LLM API 호출
# =====================================================
def get_highlights_from_llm(system_instruction, user_prompt):
    """
    Gemini API를 호출하여 하이라이트를 추출합니다.
    """
    print(f"🧠 [2/3] {MODEL_NAME} 모델로 텍스트 분석 및 하이라이트 추출 중...")
    
    try:
        # 💡 API 키가 환경 변수에 설정되어 있는지 확인
        if 'GEMINI_API_KEY' not in os.environ:
             raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
             
        client = genai.Client()

        # 프롬프트 구성
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        )
        
        # API 호출 (JSON 형식 요청 포함)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[user_prompt],
            config=config
        )
        
        # LLM의 응답에서 JSON 문자열만 추출하여 반환
        return response.text.strip()
        
    except Exception as e:
        print(f"❌ Gemini API 호출 중 오류 발생: {e}")
        return None

# =====================================================
# [3/3] 최종 결과 저장 및 출력
# =====================================================
def save_and_print_highlights(json_string):
    """
    LLM이 추출한 JSON 형태의 하이라이트를 파일로 저장하고 출력합니다.
    """
    try:
        # LLM 응답을 파일로 저장
        with open(OUTPUT_HIGHLIGHT_FILE, "w", encoding="utf-8") as f:
            f.write(json_string)

        print("\n✅ [3/3] LLM 분석 완료!")
        print(f"🎉 최종 하이라이트 목록이 '{OUTPUT_HIGHLIGHT_FILE}'에 저장되었습니다.")
        
        # JSON 문자열을 보기 좋게 출력
        import json
        highlights = json.loads(json_string)
        print("\n--- 🏆 추출된 최종 하이라이트 5개 🏆 ---")
        for item in highlights:
            print(f"시간: {item['시간']} | 내용: {item['내용']}")
        print("---------------------------------------")

    except json.JSONDecodeError:
        print("\n❌ LLM이 유효한 JSON 형식을 반환하지 못했습니다. 응답을 직접 확인하세요.")
        print("LLM 응답:\n", json_string)
    except Exception as e:
        print(f"❌ 결과 저장 중 오류 발생: {e}")


# =====================================================
# 🚀 메인 실행
# =====================================================
if __name__ == "__main__":
    
    if not os.path.exists(TRANSCRIPT_FILE):
        print(f"🚨 오류: STT 결과 파일 '{TRANSCRIPT_FILE}'을 찾을 수 없습니다. STT 파이프라인을 먼저 실행하거나 경로를 확인하세요.")
    else:
        # 1. 텍스트 파일 읽기
        with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
            transcript_content = f.read()

        # 2. 프롬프트 생성
        system_inst, user_p = create_llm_prompt(transcript_content)
        
        # 3. LLM 호출
        json_result = get_highlights_from_llm(system_inst, user_p)
        
        if json_result:
            # 4. 결과 저장 및 출력
            save_and_print_highlights(json_result)