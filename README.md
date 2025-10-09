# SP-AI-LIGHTS (Sports AI Highlights)

AI 스포츠 하이라이트 자동 편집 시스템 - Phase 1 Prototype

## 개요

16:9 가로 영상을 9:16 세로 영상으로 자동 리프레이밍하는 AI 기반 시스템입니다.

### 주요 기능 (Phase 1)
- ✅ YOLOv8 기반 객체 탐지 (사람, 공)
- ✅ ByteTrack 기반 객체 추적
- ✅ ROI(관심 영역) 자동 계산
- ✅ Kalman Filter 기반 부드러운 카메라 워크
- ✅ 9:16 비율 자동 크롭
- ✅ PySide6 GUI 인터페이스

## 설치 방법

### 1. Python 3.8+ 설치

**macOS:**
```bash
# Homebrew 설치 (아직 없는 경우)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 3 설치
brew install python@3.11
```

**Windows:**
1. [Python 공식 사이트](https://www.python.org/downloads/)에서 Python 3.11 다운로드
2. 설치 시 **"Add Python to PATH"** 체크 필수
3. 설치 확인: `python --version`

### 2. FFmpeg 설치

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
1. [FFmpeg 공식 사이트](https://ffmpeg.org/download.html)에서 다운로드
2. 압축 해제 후 `bin` 폴더를 시스템 PATH에 추가
3. 설치 확인: `ffmpeg -version`

또는 [Chocolatey](https://chocolatey.org/) 사용:
```powershell
choco install ffmpeg
```

### 3. 프로젝트 클론 및 가상환경 생성

```bash
# 저장소 클론
git clone <repository-url>
cd shortsgenie/app

# 가상환경 생성
python3 -m venv venv  # macOS/Linux
python -m venv venv   # Windows
```

### 4. 가상환경 활성화

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 5. 의존성 설치

```bash
pip install -r requirements.txt
```

설치 확인:
```bash
pip list
```

### 6. YOLO 모델 다운로드

처음 실행 시 YOLOv8 모델이 자동으로 다운로드됩니다.
수동으로 다운로드하려면:

```bash
# 가상환경 활성화 후
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## 사용 방법

### GUI 모드 (권장)

```bash
python main.py
```

1. **Select Video** 버튼을 클릭하여 입력 비디오 선택
2. 출력 해상도 및 설정 조정 (기본값: 1080x1920, Kalman Filter ON)
3. **Start Processing** 버튼 클릭
4. 저장 위치 선택
5. 처리 완료 대기

### CLI 모드 (고급 사용자)

```python
from pathlib import Path
from src.pipeline.reframing_pipeline import process_video_simple

# 간단한 사용
stats = process_video_simple(
    input_path="input.mp4",
    output_path="output.mp4",
    output_width=1080,
    output_height=1920
)

print(stats)
```

## 프로젝트 구조

```
app/
├── src/
│   ├── core/              # 핵심 모듈
│   │   ├── detector.py    # YOLO 객체 탐지
│   │   ├── roi_calculator.py  # ROI 계산
│   │   ├── smoother.py    # Kalman Filter
│   │   └── cropper.py     # 크롭 처리
│   ├── pipeline/          # 파이프라인
│   │   └── reframing_pipeline.py
│   ├── gui/               # GUI
│   │   ├── main_window.py
│   │   ├── progress_dialog.py
│   │   └── video_player.py
│   ├── utils/             # 유틸리티
│   └── models/            # 데이터 모델
├── resources/             # 리소스 파일
├── output/                # 출력 파일
└── main.py                # 진입점
```

## 동작 원리

### Pipeline 단계

1. **객체 탐지 & 추적**
   - YOLOv8로 각 프레임에서 사람과 공 탐지
   - ByteTrack으로 객체 ID 추적

2. **ROI 계산**
   - 탐지된 객체들의 위치 기반 가중 평균 계산
   - 공(ball)에 더 높은 가중치 부여

3. **스무딩**
   - Kalman Filter로 ROI 중심점 경로 부드럽게 처리
   - 자연스러운 카메라 워크 생성

4. **크롭 & 저장**
   - 계산된 ROI 중심으로 9:16 비율 크롭
   - FFmpeg로 최종 영상 인코딩

## 설정

`src/utils/config.py`에서 다양한 파라미터 조정 가능:

```python
# 탐지 설정
confidence_threshold: float = 0.25  # 탐지 신뢰도
person_class_id: int = 0            # COCO 클래스 ID
sports_ball_class_id: int = 32

# ROI 가중치
ball_weight: float = 3.0            # 공 가중치
person_weight: float = 1.0          # 사람 가중치

# Kalman Filter
process_variance: float = 1e-3      # 프로세스 노이즈
measurement_variance: float = 1e-1  # 측정 노이즈
```

## 시스템 요구사항

- **OS:** macOS, Windows, Linux
- **Python:** 3.8+
- **GPU:** CUDA (NVIDIA) 또는 MPS (Apple Silicon) 권장 (CPU도 가능하나 느림)
- **메모리:** 최소 8GB RAM
- **디스크:** 최소 2GB 여유 공간

## 성능

- **GPU (Apple M1/M2):** ~30-60 FPS 처리
- **GPU (NVIDIA RTX):** ~60-120 FPS 처리
- **CPU:** ~5-15 FPS 처리

## 문제 해결

### YOLO 모델 로딩 실패
```bash
pip install --upgrade ultralytics
```

### GUI가 실행되지 않음
```bash
pip install --upgrade PySide6
```

### MPS 가속 사용 불가 (macOS)
- macOS 12.3+ 필요
- PyTorch 1.12+ 필요

### CUDA 사용 불가 (Windows/Linux)
```bash
# CUDA 11.8 기준
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
