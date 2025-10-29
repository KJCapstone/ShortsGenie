# Detection Visualization Tests

테스트 스크립트들을 사용하여 축구 경기 영상에서 공과 선수 탐지, 중요 선수 식별, 바운딩 박스 시각화를 수행할 수 있습니다.

## 출력 결과

3개의 영상이 생성되었습니다:

- **test_best_segment.mp4** (52MB) - 548-558초 구간, 20명 선수 + 중요 선수 탐지 ✅ 추천!
- **test_detection.mp4** (24MB) - 초반 5초 구간
- **test_detection_v2.mp4** (23MB) - 초반 10초 구간 (frame skip=2)
- **detection_comparison.mp4** (26MB) - 여러 구간 비교 영상

## 시각화 설명

### 바운딩 박스 색상
- 🔴 **빨간색** (두꺼운 선): **공** (sports ball)
- 🟡 **노란색/주황색** (두꺼운 선): **중요 선수** (Key Player)
  - 공 근처에 있음
  - 빠르게 움직임
  - 화면 중앙 근처
  - 높은 detection confidence
- 🟢 **초록색**: **일반 선수** (Player)

### 화면 정보
좌상단 오버레이:
- Frame: 프레임 번호
- Time: 타임스탬프 (초)
- Balls: 감지된 공 개수
- Players: 감지된 선수 총 개수
- Key Players: 중요 선수 개수
- Top Score: 가장 높은 importance score

각 박스 위:
- 라벨 (BALL, KEY #ID, #ID)
- Confidence score
- [S:X.X] Importance score (중요 선수만)

## 테스트 스크립트

### 1. test_detection_visualization.py (기본 버전)

기본 탐지 및 시각화:

```bash
cd app
python tests/test_detection_visualization.py
```

설정:
- 처음 300 프레임만 처리 (10초 @ 30fps)
- confidence_threshold = 0.3
- importance_threshold = 4.0

### 2. test_detection_visualization_v2.py (개선 버전) ✅ 추천

더 나은 탐지와 유연한 설정:

```bash
# 특정 구간 처리
python tests/test_detection_visualization_v2.py --start 548 --duration 10

# 파라미터
--start: 시작 시간 (초)
--duration: 처리할 길이 (초)
--skip: 프레임 스킵 (1=모든 프레임, 2=2프레임마다 1개)
--output: 출력 파일 경로
```

예시:
```bash
# 548초부터 10초 처리 (좋은 구간!)
python tests/test_detection_visualization_v2.py --start 548 --duration 10

# 274초부터 10초 처리 (공 있음)
python tests/test_detection_visualization_v2.py --start 274 --duration 10

# 빠른 테스트 (frame skip=3)
python tests/test_detection_visualization_v2.py --start 0 --duration 30 --skip 3
```

개선사항:
- Lower confidence threshold (0.25) → 더 많은 탐지
- Enhanced key player scoring → 공 없어도 작동
- 이동 속도 history 추적 (10 프레임)
- 크기 기반 importance (큰 바운딩 박스 = 카메라 가까움)
- 화면 하단 bias (액션 구역)

### 3. quick_preview.py (구간 탐색 도구)

영상에서 좋은 구간을 자동으로 찾아줍니다:

```bash
python tests/quick_preview.py
```

기능:
- 영상을 10개 구간으로 샘플링
- 각 구간의 선수/공 개수 분석
- Top 3 구간 출력
- 비교 영상 생성 (detection_comparison.mp4)

출력 예시:
```
Top 3 Segments (Best for Testing):
1. Time: 548.5s - 19 players, 1 balls
2. Time: 274.2s - 18 players, 1 balls
3. Time: 617.0s - 20 players, 0 balls
```

## 중요 선수 식별 기준

`EnhancedKeyPlayerAnalyzer`가 다음 기준으로 importance score 계산:

1. **공과의 거리** (+3.0 점)
   - 300px 이내면 점수 부여
   - 가까울수록 높은 점수

2. **이동 속도** (+2.0 점)
   - 최근 3프레임의 이동 거리 계산
   - 50px/frame 이상이면 만점
   - 25-50px/frame이면 절반 점수

3. **화면 중앙 proximity** (+1.5 점)
   - 중앙에서 300px 이내
   - 가까울수록 높은 점수

4. **Detection confidence** (+1.0 점)
   - 0.7 이상이면 점수 부여

5. **바운딩 박스 크기** (+1.0 점)
   - 평균보다 크면 점수 부여 (카메라에 가까움)

6. **화면 하단 위치** (+0.5 점)
   - 화면 하단 60% 영역이면 점수 부여

**총점 3.0 이상이면 "중요 선수"로 분류**

## 성능

- **Detection FPS**: 7-30 FPS (MPS 가속 사용)
- **처리 속도**: 10초 영상 = 약 1분 소요
- **GPU 메모리**: ~2GB (Apple Silicon MPS)

최적화 팁:
- `--skip 2` 또는 `--skip 3` 사용하여 빠르게 테스트
- 짧은 구간(5-10초)으로 먼저 테스트
- confidence_threshold 조정으로 탐지 수 제어

## 다음 단계

현재 검증된 내용:
- ✅ YOLOv8이 축구 선수를 잘 탐지함 (15-20명)
- ✅ ByteTrack이 선수 추적 가능
- ✅ 중요 선수 식별 로직 작동
- ⚠️ 공 탐지는 가끔씩만 성공 (개선 필요)

개선 방향:
1. **공 탐지 개선**
   - Sports ball dataset으로 fine-tuning
   - 더 작은 객체도 탐지하도록 설정 조정
   - 공 추적 전용 알고리즘 추가

2. **중요 선수 로직 개선**
   - 팀 색상 기반 구분 (OpenCV color clustering)
   - 골키퍼 vs 필드 플레이어 구분
   - 공격/수비 역할 추론

3. **Phase 2 준비**
   - 오디오 분석 모듈과 통합
   - ROI calculator 구현
   - Kalman filter smoother 구현
   - 최종 크롭 경로 생성

## 문제 해결

### "Cannot open video" 에러
- input/korea_vs_brazil.mp4 파일이 있는지 확인
- 파일 경로 확인

### 느린 처리 속도
- `--skip 2` 사용하여 프레임 건너뛰기
- 더 짧은 구간 처리 (--duration 5)

### Tracking 에러
- detector.reset_tracking() 수정됨
- 새 비디오 처리 시 자동으로 reset

### 공이 거의 안 보임
- confidence_threshold 낮추기 (0.2로)
- 548초, 274초 구간 테스트 (공이 있는 구간)
- 향후 fine-tuning 필요
