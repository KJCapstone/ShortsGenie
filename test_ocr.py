"""
PaddleOCR 결과 형식 테스트
"""
import cv2
from paddleocr import PaddleOCR

# 비디오에서 프레임 추출
cap = cv2.VideoCapture("input/korea_vs_brazil.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 5000)  # 5000번째 프레임
ret, frame = cap.read()
cap.release()

if not ret:
    print("프레임 읽기 실패")
    exit(1)

# 상단 20% 영역
height, width = frame.shape[:2]
top_region = frame[0:int(height * 0.2), :]

print(f"프레임 크기: {width}x{height}")
print(f"상단 영역 크기: {top_region.shape}")

# PaddleOCR 초기화
print("\nPaddleOCR 초기화 중...")
ocr = PaddleOCR(lang='en')

# OCR 실행
print("OCR 실행 중...")
result = ocr.predict(top_region)

# 결과 분석
print(f"\n결과 타입: {type(result)}")
print(f"결과 길이: {len(result) if hasattr(result, '__len__') else 'N/A'}")

if result and len(result) > 0:
    print(f"\n첫 번째 항목 타입: {type(result[0])}")
    print(f"첫 번째 항목 속성: {[attr for attr in dir(result[0]) if not attr.startswith('_')][:30]}")

    ocr_result = result[0]

    # OCRResult는 딕셔너리처럼 작동
    print(f"\nOCRResult keys: {list(ocr_result.keys())}")

    # 모든 키의 값 출력
    for key in ocr_result.keys():
        value = ocr_result[key]
        print(f"\n✅ '{key}': {type(value)}")
        if hasattr(value, '__len__') and not isinstance(value, (str, dict)):
            print(f"   길이: {len(value)}")
            if len(value) > 0 and len(value) < 50:
                print(f"   값: {value}")
            elif len(value) > 0:
                print(f"   처음 3개: {value[:3]}")
else:
    print("결과가 비어있습니다!")
