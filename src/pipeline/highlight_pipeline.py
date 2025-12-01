"""Hybrid highlight generation pipeline with flexible module orchestration.

This module implements a Strategy pattern-based pipeline that can:
- Automatically select modules based on mode (goal/match/meme)
- Enable/disable modules via configuration
- Handle graceful fallbacks when modules fail
- Report detailed progress for GUI integration
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import time

from src.pipeline.pipeline_config import PipelineConfig, create_config_from_mode
from src.audio.scoreboard_ocr_detector import ScoreboardOCRDetector, GoalEvent
from src.audio.highlight_filter import AudioHighlightFilter
from src.audio.whisper_transcriber import WhisperTranscriber
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
        # Define progress weights for each module (total = 85%)
        module_weights = {
            "ocr_detection": 20,      # Heavy module
            "audio_analysis": 15,     # Medium module
            "transcript_analysis": 50, # Heaviest (Whisper 30 + Gemini 20)
            "scene_detection": 10,
            "scene_classification": 5,
            "reframing": 0,           # Handled separately
            "video_editing": 0        # Handled separately
        }

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
        self._report_progress("후처리", 90, "하이라이트 정리 및 순위 결정...")

        # If no highlights were generated (all modules disabled), create dummy highlights for testing
        if not self.highlights:
            logger.warning("No highlights generated from modules. Creating dummy highlights for testing.")
            self._create_dummy_highlights()

        self._post_process_highlights()

        # Generate final clips if needed
        if self.config.reframing.enabled and self.highlights:
            self._report_progress("영상 생성", 95, "최종 클립 생성 중...")
            self._generate_final_clips()

        self._report_progress("완료", 100, f"총 {len(self.highlights)}개 하이라이트 생성!")

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
            base_progress: Starting progress percentage
            module_weight: Total weight allocated to this module (e.g., 50)
        """
        # Whisper takes ~60% of this module's time, Gemini takes ~40%
        whisper_weight = int(module_weight * 0.6)  # e.g., 30 out of 50
        gemini_weight = int(module_weight * 0.4)   # e.g., 20 out of 50

        self._report_progress("음성 인식", base_progress, "Whisper 모델 초기화 중...")

        # Step 1: Whisper transcription (optimized with beam_size=1)
        if self._whisper is None:
            self._whisper = WhisperTranscriber(
                model_size=self.config.transcript_analysis.model_size,
                language=self.config.transcript_analysis.language
            )

        self._report_progress("음성 인식", base_progress + 5, "오디오 텍스트 변환 중... (약 10-30초 소요)")

        whisper_result = self._whisper.transcribe(str(self.video_path))

        self._report_progress("음성 인식", base_progress + whisper_weight, "음성 인식 완료!")

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
            self._report_progress("AI 분석", base_progress + whisper_weight + 2, "Gemini로 하이라이트 추출 중... (약 10-60초 소요)")

            if self._transcript_analyzer is None:
                self._transcript_analyzer = TranscriptAnalyzer(verbose=False)

            # Extract highlights from transcript using Gemini
            try:
                ai_highlights = self._transcript_analyzer.analyze_transcript(
                    transcript_path=str(temp_transcript_path)
                )

                logger.info(f"AI extracted {len(ai_highlights)} highlights")

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
                        "foul": "🟨"
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

        self._report_progress("AI 분석", base_progress + module_weight, f"{len(self.highlights)}개 하이라이트 추출 완료!")

    def _post_process_highlights(self) -> None:
        """Post-process highlights: merge, filter, rank."""
        if not self.highlights:
            logger.warning("No highlights to post-process")
            return

        # Step 1: Remove duplicates and overlaps
        self.highlights = self._merge_overlapping_highlights(self.highlights)

        # Step 2: Filter by duration (with detailed logging)
        logger.info(f"Filtering highlights by duration ({self.config.min_highlight_duration}s - {self.config.max_highlight_duration}s)")

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

        # Step 3: Sort by score (best first)
        self.highlights.sort(key=lambda h: h.score, reverse=True)

        # Step 4: Limit to max_highlights
        if len(self.highlights) > self.config.max_highlights:
            self.highlights = self.highlights[:self.config.max_highlights]

        logger.info(f"Post-processing complete: {len(self.highlights)} highlights remain")

    def _generate_final_clips(self) -> None:
        """Generate final video clips with reframing if enabled."""
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

        for i, highlight in enumerate(self.highlights):
            try:
                logger.info(f"Generating clip {i+1}/{len(self.highlights)}")

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


# Convenience function for simple usage
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
