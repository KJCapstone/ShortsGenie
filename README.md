# ShortsGenie 🎬⚽

AI-powered sports video auto-editing system that transforms horizontal (16:9) soccer footage into vertical (9:16) short-form content optimized for social media platforms (TikTok, Reels, Shorts).

## Features

- **Hybrid Highlight Detection**: Combines Whisper + Gemini AI analysis for intelligent highlight extraction
- **Dynamic Reframing**: YOLOv8-based ball and player tracking with multiple detection backends
- **Scene-Aware Processing**: ResNet18 scene classification for intelligent ROI calculation
- **Smooth Camera Movement**: Kalman filter, EMA, and adaptive EMA smoothing options
- **Temporal Ball Filtering**: Savitzky-Golay filter for robust ball trajectory smoothing
- **Parallel Processing**: CPU-based parallel clip generation for faster processing
- **Audio Preservation**: Automatic audio extraction and merging
- **PySide6 GUI**: Full-featured interface with Korean language support

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd shortsgenie

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

Models are **not included** in repository due to file size. Download them to `resources/models/`:

**Required Models:**
- `best.pt` - Fine-tuned soccer detection model (default, ~6MB)
- `yolov8n.pt` - YOLOv8 nano (~6MB) [Auto-download]
- `scene_classifier/soccer_model_ver2.pth` - ResNet18 scene classifier (~44MB)

**Auto-download (YOLO models):**
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 3. Run the Application

```bash
# GUI Mode (recommended)
python main.py
```

## Project Structure

```
shortsgenie/
├── main.py                      # GUI entry point
├── src/
│   ├── core/                   # Detection, ROI, smoothing, cropping
│   │   ├── detector.py          # Generic YOLO detector
│   │   ├── soccernet_detector.py # SoccerNet fine-tuned detector
│   │   ├── temporal_filter.py   # Savitzky-Golay ball filtering
│   │   ├── roi_calculator.py   # ROI calculation with hysteresis
│   │   ├── smoother.py          # Kalman/EMA smoothing
│   │   └── cropper.py          # Video cropping with audio
│   ├── pipeline/               # Processing pipelines
│   │   ├── reframing_pipeline.py    # Core reframing (PHASE 2)
│   │   ├── highlight_pipeline.py     # Full highlight generation
│   │   └── pipeline_config.py       # Configuration system
│   ├── audio/                  # Audio analysis and transcription
│   │   ├── whisper_transcriber.py     # Local Whisper STT
│   │   ├── groq_transcriber.py        # Groq API transcription
│   │   ├── highlight_filter.py         # Audio highlight detection
│   │   └── scoreboard_ocr_detector.py   # PaddleOCR goal detection
│   ├── ai/                     # AI integration
│   │   └── transcript_analyzer.py  # Gemini AI analysis
│   ├── scene/                  # Scene processing
│   │   ├── scene_classifier.py    # ResNet18 scene classification
│   │   └── scene_manager.py      # Scene management
│   ├── gui/                    # PySide6 interface
│   │   ├── main_window.py       # Main window
│   │   ├── progress_page.py    # Processing progress
│   │   ├── highlight_selector.py # Highlight selection
│   │   ├── preview_page.py     # Video preview
│   │   └── output_page.py      # Output settings
│   ├── utils/                  # Utilities
│   │   ├── config.py           # Centralized configuration
│   │   ├── video_utils.py      # Video I/O
│   │   └── quality_presets.py  # Quality presets
│   └── models/                 # Data models
│       └── detection_result.py  # Detection structures
├── resources/models/            # ⚠️ Model files (NOT in git)
├── input/                      # Place input videos here
├── output/                     # Processed videos save here
├── test_*.py                  # Test scripts
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Detection Backends

The system supports multiple detection backends for ball and player detection:

### 1. Generic YOLO (`detector_backend="yolo"`)
- **Model**: YOLOv8 (n/s/m variants)
- **Training**: COCO dataset (universal object detection)
- **Use case**: General purpose, non-soccer content

### 2. SoccerNet YOLO (`detector_backend="soccernet"`) [Default]
- **Model**: YOLOv8 fine-tuned on SoccerNet dataset
- **Training**: Soccer-specific footage
- **Use case**: Soccer footage with standard camera angles

## Configuration

### Environment Variables (Optional)

Copy `.env.example` to `.env` and configure if needed:

```bash
cp .env.example .env
```

**Available variables:**
- `GOOGLE_API_KEY` - For Gemini AI integration (recommended for transcript analysis)
- `GROQ_API_KEY` - For Groq API (alternative to local Whisper)
- `FFMPEG_PATH` - Custom ffmpeg path (optional)

### Python Configuration

All settings in `src/utils/config.py` and `src/pipeline/pipeline_config.py`.

**Example usage:**
```python
from src.utils.config import AppConfig
from src.pipeline.reframing_pipeline import ReframingPipeline

config = AppConfig()
config.detection.detector_backend = "soccernet"
config.detection.confidence_threshold = 0.05

pipeline = ReframingPipeline(config)
stats = pipeline.process_goal_clip(
    clip_path="input/goal_clip.mp4",
    output_path="output/goal_clip_vertical.mp4"
)
```

## Processing Pipeline

The hybrid pipeline consists of following modules:

### 1. Audio Analysis (Optional)
- Whisper STT (local) or Groq API (cloud) for speech-to-text
- Audio excitement detection for highlight segments

### 2. Transcript Analysis (Enabled by default)
- Gemini AI analysis of transcripts for intelligent highlight extraction
- Context-aware highlight generation with descriptions

### 3. Scoreboard OCR (Optional)
- PaddleOCR-based goal detection from scoreboard
- Audio boost mode for high accuracy during exciting moments

### 4. Scene Classification (Per-clip)
- ResNet18-based scene type classification
- Detects: wide, close, audience, replay scenes

### 5. Dynamic Reframing
- Ball and player detection with YOLO/SoccerNet
- Temporal filtering (Savitzky-Golay)
- ROI calculation with hysteresis and scene locking
- Smoothing (Kalman/EMA/Adaptive EMA)

### 6. Video Generation
- Parallel clip generation (3+ clips)
- Video cropping and encoding
- Audio preservation and merging

## Testing

```bash
# Test SoccerNet reframing pipeline
python test_soccernet_pipeline.py input/test_clip.mp4

# Test scene-aware pipeline
python test_scene_aware_pipeline.py

# Test parallel processing performance
python test_parallel_pipeline.py input/test_video.mp4

# Test OCR scoreboard detection
python test_ocr.py
```

## Requirements

- Python 3.8+
- FFmpeg (system installation required)
- 8GB+ RAM recommended
- GPU optional but recommended (CUDA or Apple Silicon MPS)

## Dependencies

Key dependencies include:
- **PyTorch**: Deep learning framework
- **Ultralytics YOLO**: Object detection
- **OpenCV**: Computer vision
- **Whisper**: Speech-to-text
- **PaddleOCR**: OCR for goal detection
- **PySide6**: GUI framework
- **Librosa**: Audio analysis
- **scenedetect**: Scene boundary detection

## Performance

- **Processing Speed**: ~15-30 FPS on modern CPU/GPU
- **Parallel Speedup**: 2-4x faster for 3+ clips
- **Memory Usage**: ~2-4GB per GPU worker
- **Ball Detection**: 30-50%+ detection rate with temporal filtering
