"""Hybrid highlight generation pipeline with flexible module orchestration.

This module implements a Strategy pattern-based pipeline that can:
- Automatically select modules based on mode (goal/match/meme)
- Enable/disable modules via configuration
- Handle graceful fallbacks when modules fail
- Report detailed progress for GUI integration
- Parallel clip generation for faster processing
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import time
from datetime import datetime
import cv2
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch

from src.pipeline.pipeline_config import PipelineConfig, create_config_from_mode
from src.audio.scoreboard_ocr_detector import ScoreboardOCRDetector, GoalEvent
from src.audio.highlight_filter import AudioHighlightFilter
from src.audio.whisper_transcriber import WhisperTranscriber
from src.audio.groq_transcriber import GroqTranscriber
from src.ai.transcript_analyzer import TranscriptAnalyzer
from src.scene.scene_classifier import SceneClassifier
from src.core.video_editor import VideoEditor
from src.pipeline.reframing_pipeline import ReframingPipeline
from src.utils.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class Highlight:
    """Unified highlight data structure."""
    title: str
    description: str
    start_time: float  # seconds
    end_time: float    # seconds
    score: float = 1.0  # Confidence/quality score (0-1)
    video_path: Optional[str] = None  # Path to generated clip
    source: str = "unknown"  # Which module generated this
    metadata: Dict[str, Any] = None  # Additional info

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def duration(self) -> float:
        """Get highlight duration in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for GUI consumption."""
        return {
            "title": self.title,
            "description": self.description,
            "start_time": self._format_time(self.start_time),
            "end_time": self._format_time(self.end_time),
            "duration": f"{self.duration:.1f}s",
            "score": self.score,
            "video_path": self.video_path,
            "source": self.source,
            **self.metadata
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"


class HighlightPipeline:
    """Main pipeline orchestrator for hybrid highlight generation.

    This class coordinates multiple processing modules based on configuration
    and provides a unified interface for the GUI.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        progress_callback: Optional[Callable[[str, int, str], None]] = None
    ):
        """Initialize highlight pipeline.

        Args:
            config: Pipeline configuration. If None, uses default for goal mode.
            progress_callback: Optional callback for progress updates.
                Signature: callback(stage: str, progress: int, message: str)
        """
        self.config = config or PipelineConfig.for_goal_mode()
        self.progress_callback = progress_callback

        # Module instances (lazy initialization)
        self._ocr_detector = None
        self._audio_filter = None
        self._whisper = None
        self._transcript_analyzer = None
        self._scene_detector = None
        self._scene_classifier = None
        self._video_editor = None
        self._reframing_pipeline = None

        # Processing state
        self.video_path: Optional[Path] = None
        self.highlights: List[Highlight] = []
        self.processing_errors: List[str] = []

        # Performance tracking
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.performance_log_path = Path("output/performance_log.txt")

        logger.info(f"HighlightPipeline initialized in {self.config.mode} mode")
        logger.info(f"Enabled modules: {[name for name, _ in self.config.get_enabled_modules()]}")

    def process(self, video_path: str) -> List[Highlight]:
        """Process video and generate highlights.

        This is the main entry point that orchestrates all modules.

        Args:
            video_path: Path to input video file

        Returns:
            List of Highlight objects sorted by score (best first)
        """
        # Start timing
        self.start_time = time.time()

        self.video_path = Path(video_path)
        self.highlights = []
        self.processing_errors = []

        logger.info(f"="*60)
        logger.info(f"Starting {self.config.mode} mode pipeline")
        logger.info(f"Input: {self.video_path}")
        logger.info(f"="*60)

        self._report_progress("초기화", 0, "파이프라인 준비 중...")

        # Ensure output directories exist
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)

        # Process with enabled modules in priority order
        # Progress allocation based on ACTUAL enabled modules
        # Note: OCR and Audio are disabled by default in goal mode
        # Total: 60% for transcript analysis (Whisper + Gemini)
        module_weights = {
            "ocr_detection": 20,      # Disabled in goal mode
            "audio_analysis": 15,     # Disabled in goal mode
            "transcript_analysis": 55, # 5-60% (Whisper 5-35%, Gemini 35-60%)
            "scene_detection": 10,    # Disabled in goal mode
            "scene_classification": 0, # Runs per-clip, not in main loop
            "reframing": 0,           # Handled separately during clip generation
            "video_editing": 0        # Handled separately
        }

        # Stage-specific progress points for better UX
        # These match the actual execution flow in goal mode
        self.PROGRESS_INIT = 0           # Initialization start
        self.PROGRESS_WHISPER_START = 5  # Whisper transcription start
        self.PROGRESS_WHISPER_END = 35   # Whisper transcription end
        self.PROGRESS_GEMINI_START = 35  # Gemini analysis start
        self.PROGRESS_GEMINI_END = 60    # Gemini analysis end
        self.PROGRESS_POSTPROCESS_START = 60  # Post-processing start
        self.PROGRESS_POSTPROCESS_END = 65    # Post-processing end
        self.PROGRESS_GENERATION_START = 65   # Video generation start
        self.PROGRESS_GENERATION_END = 95     # Video generation end
        self.PROGRESS_COMPLETE = 100     # Final completion

        enabled_modules = self.config.get_enabled_modules()

        # Calculate total weight and progress per module
        current_progress = 0

        for idx, (module_name, module_config) in enumerate(enabled_modules):
            module_weight = module_weights.get(module_name, 10)

            try:
                self._process_module(module_name, current_progress, module_weight)
                current_progress += module_weight
            except Exception as e:
                error_msg = f"{module_name} failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.processing_errors.append(error_msg)

                # Continue with other modules (graceful degradation)
                self._report_progress(
                    module_name,
                    current_progress,
                    f"⚠️ {module_name} 실패, 다른 모듈로 계속 진행..."
                )
                current_progress += module_weight

        # Post-processing: merge, rank, and filter highlights
        self._report_progress("후처리", self.PROGRESS_POSTPROCESS_START, "하이라이트 병합 중...")

        # If no highlights were generated (all modules disabled), create dummy highlights for testing
        if not self.highlights:
            logger.warning("No highlights generated from modules. Creating dummy highlights for testing.")
            self._create_dummy_highlights()

        self._post_process_highlights()
        self._report_progress("후처리", self.PROGRESS_POSTPROCESS_END, "하이라이트 필터링 완료!")

        # Generate final clips if needed
        if self.config.reframing.enabled and self.highlights:
            self._report_progress("영상 생성", self.PROGRESS_GENERATION_START, "클립 생성 시작...")
            self._generate_final_clips()  # Will report 65-95% internally

        self._report_progress("완료", self.PROGRESS_COMPLETE, f"총 {len(self.highlights)}개 하이라이트 생성!")

        # End timing and log performance
        self.end_time = time.time()
        self._log_performance()

        logger.info(f"Pipeline completed: {len(self.highlights)} highlights generated")
        if self.processing_errors:
            logger.warning(f"Errors encountered: {self.processing_errors}")

        return self.highlights

    def _process_module(self, module_name: str, base_progress: int, module_weight: int) -> None:
        """Process a single module and update highlights.

        Args:
            module_name: Name of module to process
            base_progress: Base progress percentage for this module
            module_weight: Weight (percentage points) allocated to this module
        """
        logger.info(f"\n--- Processing module: {module_name} ---")

        if module_name == "ocr_detection":
            self._process_ocr_detection(base_progress, module_weight)
        elif module_name == "audio_analysis":
            self._process_audio_analysis(base_progress, module_weight)
        elif module_name == "transcript_analysis":
            self._process_transcript_analysis(base_progress, module_weight)
        elif module_name == "scene_detection":
            # Old PySceneDetect module - now handled per-clip with SceneClassifier
            logger.info("Scene detection now handled per-clip during reframing")
        elif module_name == "scene_classification":
            # Scene classification happens per-clip in _generate_final_clips()
            logger.info("Scene classification will be applied during clip generation")
        elif module_name == "reframing":
            # Reframing happens in post-processing after clips are selected
            logger.info("Reframing will be applied during clip generation")
        elif module_name == "video_editing":
            # Video editing happens in post-processing
            logger.info("Video editing will be applied during clip generation")
        else:
            logger.warning(f"Unknown module: {module_name}")

    def _process_ocr_detection(self, base_progress: int, module_weight: int) -> None:
        """Process OCR-based goal detection."""
        self._report_progress("OCR 골 감지", base_progress, "스코어보드 분석 중...")

        if self._ocr_detector is None:
            self._ocr_detector = ScoreboardOCRDetector(
                confidence_threshold=self.config.ocr_detection.confidence_threshold
            )

        goal_events = self._ocr_detector.detect_goals(
            str(self.video_path),
            frame_skip=self.config.ocr_detection.frame_skip
        )

        logger.info(f"OCR detected {len(goal_events)} goals")

        # Convert GoalEvent to Highlight
        for i, event in enumerate(goal_events):
            highlight = Highlight(
                title=f"⚽ 골 #{i+1}",
                description=f"스코어: {event.home_score}-{event.away_score}",
                start_time=max(0, event.timestamp - self.config.video_editing.clip_padding_before),
                end_time=event.timestamp + self.config.video_editing.clip_padding_after,
                score=event.confidence,
                source="ocr",
                metadata={
                    "home_score": event.home_score,
                    "away_score": event.away_score,
                    "exact_timestamp": event.timestamp
                }
            )
            self.highlights.append(highlight)

        self._report_progress("OCR 골 감지", base_progress + module_weight, f"{len(goal_events)}개 골 발견!")

    def _process_audio_analysis(self, base_progress: int, module_weight: int) -> None:
        """Process audio-based highlight detection."""
        self._report_progress("오디오 분석", base_progress, "흥분도 분석 중...")

        if self._audio_filter is None:
            # AudioHighlightFilter uses different parameter names
            self._audio_filter = AudioHighlightFilter(
                threshold_percentile=70,  # Top 30% as highlights
                merge_gap=2.0,
                segment_duration=self.config.audio_analysis.window_duration,
                verbose=False
            )

        # Extract audio and analyze
        audio_highlights = self._audio_filter.filter_audio(
            str(self.video_path)
        )

        logger.info(f"Audio analysis found {len(audio_highlights)} highlights")

        # If we already have OCR highlights, use audio to refine boundaries
        # Otherwise, create new highlights from audio
        if self.highlights and any(h.source == "ocr" for h in self.highlights):
            self._refine_highlights_with_audio(audio_highlights)
        else:
            # Create highlights from audio peaks
            for i, (start, end) in enumerate(audio_highlights):
                highlight = Highlight(
                    title=f"🎯 주요 장면 #{i+1}",
                    description="관중 반응이 높은 순간",
                    start_time=start,
                    end_time=end,
                    score=0.8,  # Default score for audio-based highlights
                    source="audio"
                )
                self.highlights.append(highlight)

        self._report_progress("오디오 분석", base_progress + module_weight, f"{len(audio_highlights)}개 구간 발견!")

    def _process_transcript_analysis(self, base_progress: int, module_weight: int) -> None:
        """Process transcript and AI analysis to generate highlights.

        Args:
            base_progress: Starting progress percentage (should be 5)
            module_weight: Total weight allocated to this module (55 = covers 5-60%)
        """
        # Use predefined progress points for better UX
        # Whisper: 5-35% (30% range)
        # Gemini: 35-60% (25% range)

        # Step 1: Initialize transcriber based on backend selection
        backend = self.config.transcript_analysis.backend

        if self._whisper is None:
            if backend == "groq":
                # Use Groq API (cloud-based, very fast)
                self._report_progress("음성 인식", self.PROGRESS_WHISPER_START, "Groq API 초기화 중...")
                self._whisper = GroqTranscriber(
                    api_key=self.config.transcript_analysis.groq_api_key,
                    model=self.config.transcript_analysis.groq_model,
                    language=self.config.transcript_analysis.language,
                    verbose=False
                )
                progress_msg = "Groq API로 변환 중... (수 초 소요)"
            else:
                # Use local Whisper (default)
                self._report_progress("음성 인식", self.PROGRESS_WHISPER_START, "Whisper 모델 초기화 중...")
                self._whisper = WhisperTranscriber(
                    model_size=self.config.transcript_analysis.model_size,
                    language=self.config.transcript_analysis.language,
                    verbose=False
                )
                progress_msg = "오디오 텍스트 변환 중... (약 10-30초 소요)"

        else:
            # Transcriber already initialized
            if backend == "groq":
                progress_msg = "Groq API로 변환 중... (수 초 소요)"
            else:
                progress_msg = "오디오 텍스트 변환 중... (약 10-30초 소요)"

        self._report_progress("음성 인식", self.PROGRESS_WHISPER_START + 5, progress_msg)

        whisper_result = self._whisper.transcribe(str(self.video_path))

        self._report_progress("음성 인식", self.PROGRESS_WHISPER_END, "음성 인식 완료!")

        # Whisper returns dict with 'segments', 'text', 'language'
        if isinstance(whisper_result, dict) and 'segments' in whisper_result:
            transcript_segments = whisper_result['segments']
            logger.info(f"Transcribed {len(transcript_segments)} segments")
        else:
            logger.error(f"Unexpected Whisper result format: {type(whisper_result)}")
            raise ValueError("Whisper transcription failed")

        # Save transcript to temp file for TranscriptAnalyzer
        temp_transcript_path = self.config.temp_dir / "transcript.txt"
        self._save_transcript_to_file(transcript_segments, temp_transcript_path)

        # Step 2: AI analysis to extract highlights
        if self.config.transcript_analysis.use_gemini:
            self._report_progress("AI 분석", self.PROGRESS_GEMINI_START, "Gemini AI 초기화 중...")

            if self._transcript_analyzer is None:
                self._transcript_analyzer = TranscriptAnalyzer(verbose=True)  # Enable debug output

            self._report_progress("AI 분석", self.PROGRESS_GEMINI_START + 5, "Gemini로 하이라이트 추출 중... (약 10-60초 소요)")

            # Extract highlights from transcript using Gemini
            try:
                logger.info(f"Starting Gemini analysis with transcript: {temp_transcript_path}")
                ai_highlights = self._transcript_analyzer.analyze_transcript(
                    transcript_path=str(temp_transcript_path)
                )

                logger.info(f"✅ AI extracted {len(ai_highlights)} highlights")
                print(f"\n🎯 DEBUG: Gemini returned {len(ai_highlights)} highlights")
                for i, h in enumerate(ai_highlights[:3], 1):
                    print(f"  {i}. [{h.get('start', 0):.1f}s - {h.get('end', 0):.1f}s] {h.get('type', 'unknown')}")

                # Check if no highlights were extracted
                if len(ai_highlights) == 0:
                    error_msg = "이 경기는 매우 지루하거나 오디오에 문제가 있을 수 있습니다. 하이라이트를 추출할 수 없습니다."
                    logger.warning(error_msg)
                    self.processing_errors.append(error_msg)
                    # Continue to create dummy highlights as fallback

                # Sort highlights by start time to ensure chronological order
                ai_highlights = sorted(ai_highlights, key=lambda x: x.get('start', 0))
                logger.info(f"Sorted {len(ai_highlights)} highlights chronologically")

                # Debug: Print highlight order
                for i, ai_hl in enumerate(ai_highlights):
                    logger.debug(f"  Highlight {i+1}: {ai_hl.get('start', 0):.1f}s - {ai_hl.get('end', 0):.1f}s ({ai_hl.get('type', 'unknown')})")

                # Convert to Highlight objects
                for i, ai_hl in enumerate(ai_highlights):
                    # TranscriptAnalyzer returns: {start, end, type, description}
                    highlight_type = ai_hl.get("type", "highlight")

                    # Create title based on type
                    type_emoji = {
                        "goal": "⚽",
                        "chance": "🎯",
                        "save": "🧤",
                        "foul": "🟨",
                        "moment": "✨"
                    }
                    emoji = type_emoji.get(highlight_type, "✨")
                    title = f"{emoji} {highlight_type.title()} #{i+1}"

                    highlight = Highlight(
                        title=title,
                        description=ai_hl.get("description", ""),
                        start_time=float(ai_hl.get("start", 0)),  # ✅ 'start' not 'start_time'
                        end_time=float(ai_hl.get("end", 30)),     # ✅ 'end' not 'end_time'
                        score=0.9 if highlight_type == "goal" else 0.7,
                        source="gemini",
                        metadata={
                            "type": highlight_type,
                            "raw_data": ai_hl
                        }
                    )
                    self.highlights.append(highlight)
                    logger.info(f"Added highlight {i+1}: {title} [{highlight.start_time:.1f}s - {highlight.end_time:.1f}s]")

                logger.info(f"✅ Total highlights in list after Gemini: {len(self.highlights)}")

            except Exception as e:
                logger.error(f"Gemini analysis failed: {e}")
                logger.error(f"Traceback: ", exc_info=True)
                self.processing_errors.append(f"AI analysis failed: {str(e)}")

                # Fallback: enhance existing highlights if any
                if self.highlights:
                    logger.info(f"Enhancing {len(self.highlights)} existing highlights with transcript")
                    for highlight in self.highlights:
                        relevant_transcript = self._get_transcript_for_time(
                            transcript_segments,
                            highlight.start_time,
                            highlight.end_time
                        )
                        if relevant_transcript:
                            highlight.metadata["transcript"] = relevant_transcript
                else:
                    logger.warning("No existing highlights to enhance - Gemini failed and no fallback highlights")

        self._report_progress("AI 분석", self.PROGRESS_GEMINI_END, f"{len(self.highlights)}개 하이라이트 추출 완료!")

    def _post_process_highlights(self) -> None:
        """Post-process highlights: merge, filter, rank."""
        logger.info(f"🔍 Starting post-process with {len(self.highlights)} highlights")
        print(f"\n🔍 DEBUG: Post-processing {len(self.highlights)} highlights")

        if not self.highlights:
            logger.warning("No highlights to post-process")
            return

        # Step 1: Remove duplicates and overlaps
        before_merge = len(self.highlights)
        self.highlights = self._merge_overlapping_highlights(self.highlights)
        logger.info(f"After merge: {before_merge} → {len(self.highlights)} highlights")

        # Step 2: Filter by duration (with detailed logging)
        logger.info(f"Filtering highlights by duration ({self.config.min_highlight_duration}s - {self.config.max_highlight_duration}s)")
        print(f"🔍 Duration filter: {self.config.min_highlight_duration}s - {self.config.max_highlight_duration}s")

        filtered_highlights = []
        for h in self.highlights:
            if h.duration < self.config.min_highlight_duration:
                logger.warning(
                    f"⚠️  Highlight too short: '{h.title}' ({h.duration:.1f}s < {self.config.min_highlight_duration}s) "
                    f"[{h.start_time:.1f}s - {h.end_time:.1f}s] - FILTERED OUT"
                )
            elif h.duration > self.config.max_highlight_duration:
                logger.warning(
                    f"⚠️  Highlight too long: '{h.title}' ({h.duration:.1f}s > {self.config.max_highlight_duration}s) "
                    f"[{h.start_time:.1f}s - {h.end_time:.1f}s] - FILTERED OUT"
                )
                print(f"\n{'='*60}")
                print(f"⚠️  DEBUG: Highlight exceeded max duration")
                print(f"{'='*60}")
                print(f"Title: {h.title}")
                print(f"Duration: {h.duration:.1f}s (max allowed: {self.config.max_highlight_duration}s)")
                print(f"Time range: {h.start_time:.1f}s - {h.end_time:.1f}s")
                print(f"Description: {h.description[:100]}...")
                print(f"Source: {h.source}")
                print(f"{'='*60}\n")
            else:
                filtered_highlights.append(h)

        self.highlights = filtered_highlights
        logger.info(f"After duration filter: {len(self.highlights)} highlights remain")
        print(f"🔍 After duration filter: {len(self.highlights)} highlights remain")

        # Step 3: Select highlights - goals only, unless 6+ goals
        logger.info(f"Selecting highlights (goals prioritized, max 5 unless 6+ goals)")

        # Separate goals from chances based on title/score
        goals = [h for h in self.highlights if "Goal" in h.title or h.score >= 0.85]
        chances = [h for h in self.highlights if h not in goals]

        logger.info(f"Found {len(goals)} goals and {len(chances)} chances")

        # NEW LOGIC: Prioritize goals, but include chances if no goals
        selected_highlights = []

        if len(goals) >= 6:
            # 6+ goals: select all goals (up to 6)
            goals.sort(key=lambda h: h.score, reverse=True)
            selected_highlights = goals[:6]
            logger.info(f"Selected 6 goals (6+ goals found)")
        elif len(goals) > 0:
            # 1-5 goals: select all goals only
            selected_highlights = goals
            logger.info(f"Selected {len(goals)} goals only (no chances added)")
        else:
            # No goals: select best chances (up to 5)
            chances.sort(key=lambda h: h.score, reverse=True)
            selected_highlights = chances[:5]
            logger.info(f"No goals found - selected {len(selected_highlights)} best chances instead")

        # Step 4: Sort ALL selected highlights chronologically
        selected_highlights.sort(key=lambda h: h.start_time)
        self.highlights = selected_highlights

        logger.info(f"✅ Post-processing complete: {len(self.highlights)} highlights sorted chronologically")
        print(f"✅ Final highlight count: {len(self.highlights)}")
        for i, h in enumerate(self.highlights, 1):
            print(f"  {i}. {h.title} [{h.start_time:.1f}s - {h.end_time:.1f}s] ({h.duration:.1f}s)")

    def _generate_final_clips(self) -> None:
        """Generate final video clips with reframing if enabled.

        This method automatically chooses between parallel and sequential processing
        based on configuration and the number of clips.
        """
        # Debug logging to stdout (always visible)
        print(f"\n{'='*60}")
        print(f"DEBUG: _generate_final_clips() called")
        print(f"DEBUG: enable_parallel = {self.config.enable_parallel_clip_generation}")
        print(f"DEBUG: highlights count = {len(self.highlights)}")
        print(f"{'='*60}\n")

        # Decide whether to use parallel processing
        use_parallel = (
            self.config.enable_parallel_clip_generation and
            len(self.highlights) >= 3  # Only use parallel for 3+ clips
        )

        if use_parallel:
            print(f"\n🔄 PARALLEL clip generation ({len(self.highlights)} clips)\n")
            logger.info(f"Using PARALLEL clip generation ({len(self.highlights)} clips)")
            self._generate_final_clips_parallel()
        else:
            print(f"\n⏩ SEQUENTIAL clip generation ({len(self.highlights)} clips)\n")
            logger.info(f"Using SEQUENTIAL clip generation ({len(self.highlights)} clips)")
            self._generate_final_clips_sequential()

    def _generate_final_clips_parallel(self) -> None:
        """Generate final clips using parallel processing with ProcessPoolExecutor."""
        try:
            # Determine number of workers
            # For cross-platform stability (Windows/Mac/Linux), use CPU-based parallel processing
            # Each worker runs detection/reframing independently on CPU
            max_workers = min(self.config.max_parallel_workers, os.cpu_count() or 2)

            # Detect available hardware for info only (workers will use CPU)
            device_info = "CPU"
            if torch.cuda.is_available():
                device_info = "CUDA (available but using CPU workers for stability)"
            elif torch.backends.mps.is_available():
                device_info = "MPS (available but using CPU workers for stability)"

            logger.info(f"🔄 Parallel processing with {max_workers} CPU worker(s)")
            logger.info(f"   Device info: {device_info}")

            # Prepare configuration dictionary (must be picklable)
            config_dict = {
                'temp_dir': str(self.config.temp_dir),
                'output_dir': str(self.config.output_dir),
                'scene_classification_enabled': self.config.scene_classification.enabled,
                'reframing_enabled': self.config.reframing.enabled,
                'confidence_threshold': self.config.reframing.confidence_threshold,
                'detector_backend': getattr(self.config.reframing, 'detector_backend', 'yolo'),
                'use_soccernet_model': self.config.reframing.use_soccernet_model,
                'use_temporal_filter': self.config.reframing.use_temporal_filter,
                'use_kalman_smoothing': self.config.reframing.use_kalman_smoothing,
                'scene_model_path': self.config.scene_classification.model_path,
                'scene_threshold': self.config.scene_classification.threshold,
                'scene_min_len': self.config.scene_classification.min_scene_len,
            }

            # Prepare highlight data (must be picklable)
            highlight_dicts = [
                {
                    'start_time': h.start_time,
                    'end_time': h.end_time,
                    'title': h.title,
                    'description': h.description
                }
                for h in self.highlights
            ]

            logger.info(f"Starting parallel clip generation with {max_workers} workers")

            # Submit all tasks
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit jobs
                futures = {
                    executor.submit(
                        _process_single_clip_worker,
                        i,
                        highlight_dicts[i],
                        str(self.video_path),
                        config_dict
                    ): i for i in range(len(self.highlights))
                }

                # Collect results as they complete
                completed_count = 0
                for future in as_completed(futures):
                    clip_index = futures[future]

                    try:
                        idx, output_clip_path, error_msg = future.result()

                        if error_msg:
                            logger.error(f"Clip {idx+1} failed: {error_msg}")
                            self.processing_errors.append(error_msg)
                        elif output_clip_path:
                            self.highlights[idx].video_path = output_clip_path
                            logger.info(f"✅ Clip {idx+1}/{len(self.highlights)} completed: {output_clip_path}")

                        completed_count += 1
                        progress = 95 + int(completed_count / len(self.highlights) * 4)
                        self._report_progress(
                            "영상 생성",
                            progress,
                            f"클립 {completed_count}/{len(self.highlights)} 완료"
                        )

                    except Exception as e:
                        logger.error(f"Clip {clip_index+1} processing failed: {e}")
                        self.processing_errors.append(f"Clip {clip_index+1} generation failed")

            # Check if any clips were successfully generated
            if completed_count == 0:
                logger.error("Parallel processing failed for all clips, falling back to sequential")
                self._generate_final_clips_sequential()
                return

            logger.info(f"Parallel clip generation complete: {completed_count}/{len(self.highlights)} successful")

            # Step 4: Merge all clips into final_shorts.mp4
            if self.config.reframing.enabled and len(self.highlights) > 0:
                # Collect successful clip paths in correct order (as Path objects)
                successful_clips = []
                for i, highlight in enumerate(self.highlights):
                    if highlight.video_path and Path(highlight.video_path).exists():
                        successful_clips.append(Path(highlight.video_path))  # Convert to Path object
                        logger.debug(f"Clip {i+1} ready for merge: {highlight.video_path}")

                # Validate clip existence
                if len(successful_clips) == 0:
                    logger.error("No clips available for merge after parallel processing")
                    return

                missing_count = len(self.highlights) - len(successful_clips)
                if missing_count > 0:
                    logger.warning(f"Missing {missing_count} clips, proceeding with {len(successful_clips)} available clips")

                # Report merge progress
                self._report_progress("영상 병합", 98, f"{len(successful_clips)}개 클립 병합 중...")
                logger.info(f"Merging {len(successful_clips)} clips into final_shorts.mp4")

                try:
                    final_shorts = self._merge_clips_to_final_shorts(
                        successful_clips,
                        self.video_path
                    )

                    # Update first highlight to point to merged video
                    self.highlights[0].video_path = str(final_shorts)

                    # Clear other highlights' video paths (they're now part of merged video)
                    for i in range(1, len(self.highlights)):
                        self.highlights[i].video_path = None

                    logger.info(f"✅ Final shorts video created: {final_shorts}")

                except Exception as e:
                    logger.error(f"Failed to merge clips: {e}", exc_info=True)
                    self.processing_errors.append(f"Video merge failed: {str(e)}")
                    # Keep individual clip paths if merge fails
                    logger.warning("Merge failed, keeping individual clip paths")

        except Exception as e:
            logger.error(f"Parallel processing failed with exception: {e}", exc_info=True)
            logger.info("Falling back to sequential processing")
            self._generate_final_clips_sequential()

    def _generate_final_clips_sequential(self) -> None:
        """Generate final clips sequentially with granular progress tracking."""
        if self._video_editor is None:
            self._video_editor = VideoEditor()

        if self._reframing_pipeline is None and self.config.reframing.enabled:
            # Create AppConfig with current pipeline settings
            app_config = AppConfig()
            app_config.detection.confidence_threshold = self.config.reframing.confidence_threshold

            # Pass through detector backend from reframing config
            if hasattr(self.config.reframing, 'detector_backend'):
                app_config.detection.detector_backend = self.config.reframing.detector_backend

            self._reframing_pipeline = ReframingPipeline(app_config)

        # Initialize scene classifier if enabled
        if self.config.scene_classification.enabled and self._scene_classifier is None:
            try:
                self._scene_classifier = SceneClassifier(self.config.scene_classification)
                logger.info("Scene classifier initialized successfully")
            except FileNotFoundError as e:
                logger.error(f"Scene classifier model not found: {e}")
                self.config.scene_classification.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize scene classifier: {e}")
                self.config.scene_classification.enabled = False

        # Calculate progress per clip (65-95% range = 30% total)
        total_clips = len(self.highlights)
        progress_per_clip = (self.PROGRESS_GENERATION_END - self.PROGRESS_GENERATION_START) / total_clips if total_clips > 0 else 0

        for i, highlight in enumerate(self.highlights):
            try:
                # Calculate current progress
                current_progress = self.PROGRESS_GENERATION_START + int(i * progress_per_clip)

                logger.info(f"Generating clip {i+1}/{len(self.highlights)}")
                self._report_progress(
                    "영상 생성",
                    current_progress,
                    f"클립 {i+1}/{total_clips} 생성 중..."
                )

                # Step 1: Extract clip
                temp_clip = self.config.temp_dir / f"clip_{i}_temp.mp4"
                self._video_editor.cut_segment(
                    str(self.video_path),
                    str(temp_clip),
                    highlight.start_time,
                    highlight.end_time
                )

                # Step 2: Run scene classification on extracted clip
                scene_json_path = None
                if self.config.scene_classification.enabled:
                    self._report_progress(
                        "장면 분류",
                        95 + int(i / len(self.highlights) * 2.5),
                        f"클립 {i+1} 장면 분석 중..."
                    )

                    scene_json_path = self.config.temp_dir / f"clip_{i}_scenes.json"
                    try:
                        scene_metadata = self._scene_classifier.classify_clip(
                            str(temp_clip),
                            output_json_path=str(scene_json_path)
                        )
                        logger.info(f"Classified {len(scene_metadata.segments)} scenes in clip {i+1}")
                    except Exception as e:
                        logger.warning(f"Scene classification failed for clip {i+1}: {e}")
                        scene_json_path = None  # Fallback to dynamic ROI

                # Step 3: Apply reframing with scene metadata
                if self.config.reframing.enabled:
                    output_clip = self.config.output_dir / f"highlight_{i+1}_vertical.mp4"

                    self._report_progress(
                        "영상 생성",
                        97 + int(i / len(self.highlights) * 3),
                        f"클립 {i+1} 리프레이밍 중..."
                    )

                    self._reframing_pipeline.process_goal_clip(
                        clip_path=str(temp_clip),
                        output_path=str(output_clip),
                        use_soccernet_model=self.config.reframing.use_soccernet_model,
                        use_temporal_filter=self.config.reframing.use_temporal_filter,
                        use_kalman_smoothing=self.config.reframing.use_kalman_smoothing,
                        scene_metadata_path=str(scene_json_path) if scene_json_path else None
                    )

                    highlight.video_path = str(output_clip)
                    temp_clip.unlink()  # Clean up temp file
                else:
                    # No reframing, just use extracted clip
                    highlight.video_path = str(temp_clip)

            except Exception as e:
                logger.error(f"Failed to generate clip {i+1}: {e}")
                self.processing_errors.append(f"Clip {i+1} generation failed")

        # Step 4: Merge all clips into single shorts video with audio
        if self.config.reframing.enabled and len(self.highlights) > 0:
            self._report_progress(
                "최종 병합",
                99,
                f"{len(self.highlights)}개 클립을 하나로 병합 중..."
            )

            try:
                # Collect all generated clip paths
                clip_paths = []
                for highlight in self.highlights:
                    if hasattr(highlight, 'video_path') and highlight.video_path:
                        clip_path = Path(highlight.video_path)
                        # Check if file actually exists before adding
                        if clip_path.exists():
                            clip_paths.append(clip_path)
                        else:
                            logger.warning(f"Clip file not found: {clip_path}")

                logger.info(f"Found {len(clip_paths)} valid clip files out of {len(self.highlights)} highlights")

                if clip_paths:
                    # Merge clips with original audio
                    self._report_progress("영상 생성", self.PROGRESS_GENERATION_END, f"{len(clip_paths)}개 클립 생성 완료!")

                    final_shorts = self._merge_clips_to_final_shorts(
                        clip_paths,
                        self.video_path
                    )

                    logger.info(f"✅ Final shorts video created: {final_shorts}")

                    # Update first highlight to point to final merged video
                    # (for backward compatibility with existing code)
                    self.highlights[0].video_path = str(final_shorts)

                    # Clean up individual clips
                    for clip_path in clip_paths:
                        try:
                            clip_path.unlink()
                            logger.info(f"Cleaned up temporary clip: {clip_path}")
                        except Exception as e:
                            logger.warning(f"Failed to delete temp clip {clip_path}: {e}")

            except Exception as e:
                logger.error(f"Failed to merge clips: {e}")
                self.processing_errors.append("Clip merging failed")

    # Helper methods

    def _report_progress(self, stage: str, progress: int, message: str) -> None:
        """Report progress to callback if provided."""
        if self.progress_callback:
            self.progress_callback(stage, progress, message)
        logger.info(f"[{progress}%] {stage}: {message}")

    def _refine_highlights_with_audio(self, audio_highlights: List[Tuple[float, float]]) -> None:
        """Refine OCR highlights using audio excitement peaks.

        Args:
            audio_highlights: List of (start, end) tuples from AudioHighlightFilter
        """
        for highlight in self.highlights:
            if highlight.source != "ocr":
                continue

            # Find overlapping audio highlight
            for start, end in audio_highlights:
                if self._ranges_overlap(highlight.start_time, highlight.end_time, start, end):
                    # Expand highlight to include audio excitement
                    highlight.start_time = min(highlight.start_time, start)
                    highlight.end_time = max(highlight.end_time, end)
                    highlight.score = min(1.0, highlight.score + 0.1)  # Boost score slightly
                    highlight.metadata["audio_refined"] = True
                    break

    def _get_transcript_for_time(
        self,
        transcript: List[Dict],
        start_time: float,
        end_time: float
    ) -> str:
        """Extract transcript segments within time range."""
        relevant = [
            seg["text"] for seg in transcript
            if start_time <= seg["start"] <= end_time or start_time <= seg["end"] <= end_time
        ]
        return " ".join(relevant)

    def _merge_overlapping_highlights(self, highlights: List[Highlight]) -> List[Highlight]:
        """Merge overlapping highlights."""
        if not highlights:
            return []

        # Sort by start time
        sorted_highlights = sorted(highlights, key=lambda h: h.start_time)
        merged = [sorted_highlights[0]]

        for current in sorted_highlights[1:]:
            last = merged[-1]

            # Check for overlap
            if current.start_time <= last.end_time:
                # Merge: extend end time and average scores
                last.end_time = max(last.end_time, current.end_time)
                last.score = (last.score + current.score) / 2
                last.description += f" / {current.description}"
            else:
                merged.append(current)

        return merged

    @staticmethod
    def _ranges_overlap(start1: float, end1: float, start2: float, end2: float) -> bool:
        """Check if two time ranges overlap."""
        return start1 <= end2 and start2 <= end1

    def _create_dummy_highlights(self) -> None:
        """Create single dummy highlight when no modules extracted highlights.

        This generates 1 sample highlight from the video.
        Used as fallback when all extraction modules fail.
        """
        logger.info("Creating 1 dummy highlight as fallback")

        # Get video duration to create appropriate clip length
        try:
            import cv2
            cap = cv2.VideoCapture(str(self.video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 60.0
            cap.release()
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")
            duration = 60.0

        # Create single highlight (max 60s)
        end_time = min(duration, 60.0)

        dummy_highlights = [
            Highlight(
                title="⚽ 전체 영상 하이라이트",
                description=f"영상 전체 (0-{int(end_time)}초)",
                start_time=0.0,
                end_time=end_time,
                score=0.5,
                source="dummy"
            ),
        ]

        self.highlights.extend(dummy_highlights)
        logger.info(f"Added {len(dummy_highlights)} dummy highlights")

    def _save_transcript_to_file(self, transcript_segments: List[Dict], output_path: Path) -> None:
        """Save Whisper transcript segments to text file.

        Args:
            transcript_segments: List of dicts with 'start', 'end', 'text' keys
            output_path: Path to save transcript
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in transcript_segments:
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('text', '')
                f.write(f"[{start:.2f}s - {end:.2f}s] {text}\n")

    def _merge_clips_to_final_shorts(self, clip_paths: List[Path], original_video_path: Path) -> Path:
        """Merge multiple highlight clips into single shorts video with original audio.

        Args:
            clip_paths: List of individual vertical clips (no audio)
            original_video_path: Original video path for audio extraction

        Returns:
            Path to final merged shorts video (1-3 minutes) with audio
        """
        if not clip_paths:
            raise ValueError("No clips to merge")

        logger.info(f"Merging {len(clip_paths)} clips into final shorts video")

        # Create concat file list for FFmpeg
        concat_file = self.config.temp_dir / "concat_list.txt"
        with open(concat_file, 'w', encoding='utf-8') as f:
            for clip in clip_paths:
                # FFmpeg concat requires absolute paths
                f.write(f"file '{clip.absolute()}'\n")

        # Step 1: Concat video clips (no audio yet)
        temp_video = self.config.temp_dir / "merged_video_only.mp4"
        concat_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',  # No re-encoding for speed
            str(temp_video)
        ]

        logger.info("Concatenating video clips...")
        subprocess.run(concat_cmd, check=True, capture_output=True)

        # Step 2: Extract audio from original video matching highlight timestamps
        # For simplicity, we'll extract audio from the same time ranges and concat
        temp_audios = []
        for i, highlight in enumerate(self.highlights[:len(clip_paths)]):
            audio_file = self.config.temp_dir / f"audio_{i}.aac"
            extract_audio_cmd = [
                'ffmpeg', '-y',
                '-i', str(original_video_path),
                '-ss', str(highlight.start_time),
                '-t', str(highlight.end_time - highlight.start_time),
                '-vn',  # No video
                '-acodec', 'copy',
                str(audio_file)
            ]
            subprocess.run(extract_audio_cmd, check=True, capture_output=True)
            temp_audios.append(audio_file)

        # Step 3: Concat audio files
        audio_concat_file = self.config.temp_dir / "audio_concat_list.txt"
        with open(audio_concat_file, 'w', encoding='utf-8') as f:
            for audio in temp_audios:
                f.write(f"file '{audio.absolute()}'\n")

        temp_audio = self.config.temp_dir / "merged_audio.aac"
        audio_concat_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(audio_concat_file),
            '-c', 'copy',
            str(temp_audio)
        ]
        subprocess.run(audio_concat_cmd, check=True, capture_output=True)

        # Step 4: Combine video + audio with H.264 compression
        output_path = self.config.output_dir / "final_shorts.mp4"
        combine_cmd = [
            'ffmpeg', '-y',
            '-i', str(temp_video),
            '-i', str(temp_audio),
            '-c:v', 'libx264',      # H.264 codec for better compression
            '-preset', 'medium',     # Encoding speed (faster, fast, medium, slow, slower)
            '-crf', '28',            # Quality (18-28 recommended, higher=smaller file)
            '-maxrate', '3M',        # Max bitrate 3 Mbps (good for shorts)
            '-bufsize', '6M',        # Buffer size
            '-pix_fmt', 'yuv420p',   # Pixel format for compatibility
            '-c:a', 'aac',           # AAC audio codec
            '-b:a', '128k',          # Audio bitrate
            '-shortest',             # Match shortest stream
            str(output_path)
        ]

        logger.info("Combining video and audio...")
        subprocess.run(combine_cmd, check=True, capture_output=True)

        # Cleanup temp files
        temp_video.unlink(missing_ok=True)
        temp_audio.unlink(missing_ok=True)
        for audio in temp_audios:
            audio.unlink(missing_ok=True)

        logger.info(f"✅ Final shorts created: {output_path}")
        return output_path

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds using OpenCV.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video to get duration: {video_path}")
            return 0.0

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps > 0:
            return frame_count / fps
        return 0.0

    def _log_performance(self) -> None:
        """Log pipeline performance metrics to console and file."""
        if self.start_time is None or self.end_time is None:
            logger.warning("Cannot log performance - timing not recorded")
            return

        # Calculate metrics
        processing_time = self.end_time - self.start_time
        input_duration = self._get_video_duration(self.video_path)

        # Calculate total highlight duration
        total_highlight_duration = sum(h.duration for h in self.highlights)

        # Format time as HH:MM:SS
        def format_duration(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            return f"{minutes:02d}:{secs:02d}"

        # Get backend info
        backend = self.config.transcript_analysis.backend
        backend_display = "Groq API" if backend == "groq" else "Local Whisper"

        # Log to console
        logger.info("=" * 70)
        logger.info("📊 PIPELINE PERFORMANCE REPORT")
        logger.info("=" * 70)
        logger.info(f"📁 Input Video: {self.video_path.name}")
        logger.info(f"⏱️  Input Duration: {format_duration(input_duration)} ({input_duration:.1f}s)")
        logger.info(f"🎬 Highlights Generated: {len(self.highlights)}")
        logger.info(f"⏱️  Total Highlight Duration: {format_duration(total_highlight_duration)} ({total_highlight_duration:.1f}s)")
        logger.info(f"⚡ Processing Time: {format_duration(processing_time)} ({processing_time:.1f}s)")
        logger.info(f"🚀 Speed: {input_duration / processing_time:.2f}x realtime" if processing_time > 0 else "🚀 Speed: N/A")
        logger.info(f"🎙️  Transcription Backend: {backend_display}")
        logger.info("=" * 70)

        # Log to file
        try:
            # Ensure output directory exists
            self.performance_log_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare log entry
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"""
{'=' * 80}
Timestamp: {timestamp}
Video: {self.video_path.name}
Input Duration: {format_duration(input_duration)} ({input_duration:.1f}s)
Highlights: {len(self.highlights)}
Highlight Duration: {format_duration(total_highlight_duration)} ({total_highlight_duration:.1f}s)
Processing Time: {format_duration(processing_time)} ({processing_time:.1f}s)
Speed: {input_duration / processing_time:.2f}x realtime
Mode: {self.config.mode}
Transcription Backend: {backend_display}
{'=' * 80}
"""

            # Append to file
            with open(self.performance_log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)

            logger.info(f"📝 Performance log saved to: {self.performance_log_path}")

        except Exception as e:
            logger.error(f"Failed to write performance log: {e}")


# ============================================================================
# PARALLEL PROCESSING WORKER FUNCTION
# ============================================================================
# This must be at module level (not inside class) for multiprocessing to work

def _process_single_clip_worker(
    clip_index: int,
    highlight_dict: Dict[str, Any],
    video_path: str,
    config_dict: Dict[str, Any]
) -> Tuple[int, Optional[str], Optional[str]]:
    """Worker function for parallel clip processing.

    This function is executed in a separate process and must:
    1. Be picklable (top-level function, not class method)
    2. Create its own model instances (not shared across processes)
    3. Handle all errors gracefully

    Args:
        clip_index: Index of highlight in the list
        highlight_dict: Highlight data as dictionary (start_time, end_time, etc.)
        video_path: Path to original video
        config_dict: Pipeline configuration as dictionary

    Returns:
        Tuple of (clip_index, output_clip_path, error_message)
        - clip_index: Index of processed clip
        - output_clip_path: Path to generated clip (None if failed)
        - error_message: Error description (None if successful)
    """
    import sys
    from pathlib import Path

    # Re-add app to path for imports (needed in subprocess)
    app_dir = Path(__file__).parent.parent.parent
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    try:
        import torch
        from src.scene.scene_classifier import SceneClassifier
        from src.pipeline.reframing_pipeline import ReframingPipeline
        from src.core.video_editor import VideoEditor
        from src.utils.config import AppConfig
        from src.pipeline.pipeline_config import SceneClassificationConfig, ReframingConfig

        # Force CPU mode for cross-platform stability and true parallel processing
        # This ensures workers don't compete for GPU resources and works on Windows/Mac/Linux
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Disable MPS

        logger = logging.getLogger(__name__)
        logger.info(f"[Worker {clip_index}] Starting clip processing in subprocess (CPU mode)")

        # Reconstruct configuration from dict
        temp_dir = Path(config_dict['temp_dir'])
        output_dir = Path(config_dict['output_dir'])
        scene_classification_enabled = config_dict.get('scene_classification_enabled', True)
        reframing_enabled = config_dict.get('reframing_enabled', True)

        # Reconstruct highlight data
        start_time = highlight_dict['start_time']
        end_time = highlight_dict['end_time']

        # Initialize modules (each worker gets its own instances)
        video_editor = VideoEditor()

        # Step 1: Extract clip
        temp_clip = temp_dir / f"clip_{clip_index}_temp.mp4"
        logger.info(f"[Worker {clip_index}] Extracting clip: {start_time:.1f}s - {end_time:.1f}s")

        video_editor.cut_segment(
            str(video_path),
            str(temp_clip),
            start_time,
            end_time
        )

        # Step 2: Scene classification (if enabled)
        scene_json_path = None
        if scene_classification_enabled:
            try:
                # Reconstruct SceneClassificationConfig
                scene_config = SceneClassificationConfig(
                    enabled=True,
                    model_path=config_dict.get('scene_model_path', 'resources/models/scene_classifier/soccer_model_ver2.pth'),
                    threshold=config_dict.get('scene_threshold', 27.0),
                    min_scene_len=config_dict.get('scene_min_len', 15)
                )

                scene_classifier = SceneClassifier(scene_config)
                scene_json_path = temp_dir / f"clip_{clip_index}_scenes.json"

                logger.info(f"[Worker {clip_index}] Classifying scenes")
                scene_metadata = scene_classifier.classify_clip(
                    str(temp_clip),
                    output_json_path=str(scene_json_path)
                )
                logger.info(f"[Worker {clip_index}] Classified {len(scene_metadata.segments)} scenes")

            except Exception as e:
                logger.warning(f"[Worker {clip_index}] Scene classification failed: {e}")
                scene_json_path = None

        # Step 3: Reframing (if enabled)
        if reframing_enabled:
            output_clip = output_dir / f"highlight_{clip_index+1}_vertical.mp4"

            # Create AppConfig for reframing pipeline
            app_config = AppConfig()
            app_config.detection.confidence_threshold = config_dict.get('confidence_threshold', 0.05)
            app_config.detection.detector_backend = config_dict.get('detector_backend', 'yolo')

            reframing_pipeline = ReframingPipeline(app_config)

            logger.info(f"[Worker {clip_index}] Reframing clip")
            reframing_pipeline.process_goal_clip(
                clip_path=str(temp_clip),
                output_path=str(output_clip),
                use_soccernet_model=config_dict.get('use_soccernet_model', True),
                use_temporal_filter=config_dict.get('use_temporal_filter', True),
                use_kalman_smoothing=config_dict.get('use_kalman_smoothing', True),
                scene_metadata_path=str(scene_json_path) if scene_json_path else None
            )

            # Clean up temp file
            temp_clip.unlink(missing_ok=True)

            logger.info(f"[Worker {clip_index}] Clip generation complete: {output_clip}")
            return (clip_index, str(output_clip), None)
        else:
            # No reframing, just return extracted clip
            logger.info(f"[Worker {clip_index}] Clip extraction complete (no reframing): {temp_clip}")
            return (clip_index, str(temp_clip), None)

    except Exception as e:
        error_msg = f"Clip {clip_index+1} processing failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return (clip_index, None, error_msg)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_highlights(
    video_path: str,
    mode: str = "골",
    progress_callback: Optional[Callable] = None
) -> List[Dict]:
    """Generate highlights from video with automatic mode selection.

    Args:
        video_path: Path to input video
        mode: One of "골", "경기", "밈"
        progress_callback: Optional progress callback

    Returns:
        List of highlight dictionaries ready for GUI display
    """
    config = create_config_from_mode(mode)
    pipeline = HighlightPipeline(config, progress_callback)
    highlights = pipeline.process(video_path)

    return [h.to_dict() for h in highlights]
