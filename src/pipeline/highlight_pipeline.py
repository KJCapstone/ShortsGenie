"""Hybrid highlight generation pipeline with flexible module orchestration.

This module implements a Strategy pattern-based pipeline that can:
- Automatically select modules based on mode (goal/match/meme)
- Enable/disable modules via configuration
- Handle graceful fallbacks when modules fail
- Report detailed progress for GUI integration
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
import time

from src.pipeline.pipeline_config import PipelineConfig, create_config_from_mode
from src.audio.scoreboard_ocr_detector import ScoreboardOCRDetector, GoalEvent
from src.audio.highlight_filter import AudioHighlightFilter
from src.audio.whisper_transcriber import WhisperTranscriber
from src.ai.transcript_analyzer import TranscriptAnalyzer
from src.core.scene_detector import SceneDetector
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
        enabled_modules = self.config.get_enabled_modules()
        total_stages = len(enabled_modules)

        for idx, (module_name, module_config) in enumerate(enabled_modules):
            stage_progress = int((idx / total_stages) * 90)  # Reserve 90-100% for post-processing

            try:
                self._process_module(module_name, stage_progress)
            except Exception as e:
                error_msg = f"{module_name} failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.processing_errors.append(error_msg)

                # Continue with other modules (graceful degradation)
                self._report_progress(
                    module_name,
                    stage_progress,
                    f"⚠️ {module_name} 실패, 다른 모듈로 계속 진행..."
                )

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

    def _process_module(self, module_name: str, base_progress: int) -> None:
        """Process a single module and update highlights.

        Args:
            module_name: Name of module to process
            base_progress: Base progress percentage for this module
        """
        logger.info(f"\n--- Processing module: {module_name} ---")

        if module_name == "ocr_detection":
            self._process_ocr_detection(base_progress)
        elif module_name == "audio_analysis":
            self._process_audio_analysis(base_progress)
        elif module_name == "transcript_analysis":
            self._process_transcript_analysis(base_progress)
        elif module_name == "scene_detection":
            self._process_scene_detection(base_progress)
        elif module_name == "reframing":
            # Reframing happens in post-processing after clips are selected
            logger.info("Reframing will be applied during clip generation")
        elif module_name == "video_editing":
            # Video editing happens in post-processing
            logger.info("Video editing will be applied during clip generation")
        else:
            logger.warning(f"Unknown module: {module_name}")

    def _process_ocr_detection(self, base_progress: int) -> None:
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

        self._report_progress("OCR 골 감지", base_progress + 5, f"{len(goal_events)}개 골 발견!")

    def _process_audio_analysis(self, base_progress: int) -> None:
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

        self._report_progress("오디오 분석", base_progress + 5, f"{len(audio_highlights)}개 구간 발견!")

    def _process_transcript_analysis(self, base_progress: int) -> None:
        """Process transcript and AI analysis to generate highlights."""
        self._report_progress("해설 분석", base_progress, "음성 인식 중...")

        # Step 1: Whisper transcription
        if self._whisper is None:
            self._whisper = WhisperTranscriber(
                model_size=self.config.transcript_analysis.model_size,
                language=self.config.transcript_analysis.language
            )

        transcript_segments = self._whisper.transcribe(str(self.video_path))
        logger.info(f"Transcribed {len(transcript_segments)} segments")

        # Save transcript to temp file for TranscriptAnalyzer
        temp_transcript_path = self.config.temp_dir / "transcript.txt"
        self._save_transcript_to_file(transcript_segments, temp_transcript_path)

        # Step 2: AI analysis to extract highlights
        if self.config.transcript_analysis.use_gemini:
            self._report_progress("AI 분석", base_progress + 3, "Gemini로 하이라이트 추출 중...")

            if self._transcript_analyzer is None:
                self._transcript_analyzer = TranscriptAnalyzer(verbose=False)

            # Extract highlights from transcript using Gemini
            try:
                ai_highlights = self._transcript_analyzer.analyze_transcript(
                    transcript_path=str(temp_transcript_path)
                )

                logger.info(f"AI extracted {len(ai_highlights)} highlights")

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
                # Fallback: enhance existing highlights if any
                if self.highlights:
                    for highlight in self.highlights:
                        relevant_transcript = self._get_transcript_for_time(
                            transcript_segments,
                            highlight.start_time,
                            highlight.end_time
                        )
                        if relevant_transcript:
                            highlight.metadata["transcript"] = relevant_transcript

        self._report_progress("해설 분석", base_progress + 5, f"{len(self.highlights)}개 하이라이트 추출 완료!")

    def _process_scene_detection(self, base_progress: int) -> None:
        """Process scene boundary detection."""
        self._report_progress("장면 전환 감지", base_progress, "장면 경계 분석 중...")

        if self._scene_detector is None:
            self._scene_detector = SceneDetector(
                method=self.config.scene_detection.method,
                threshold=self.config.scene_detection.threshold
            )

        scenes = self._scene_detector.detect_scenes(str(self.video_path))

        logger.info(f"Detected {len(scenes)} scene transitions")

        # Use scenes to refine highlight boundaries
        for highlight in self.highlights:
            self._align_to_scene_boundaries(highlight, scenes)

        self._report_progress("장면 전환 감지", base_progress + 5, f"{len(scenes)}개 장면 감지!")

    def _post_process_highlights(self) -> None:
        """Post-process highlights: merge, filter, rank."""
        if not self.highlights:
            logger.warning("No highlights to post-process")
            return

        # Step 1: Remove duplicates and overlaps
        self.highlights = self._merge_overlapping_highlights(self.highlights)

        # Step 2: Filter by duration
        self.highlights = [
            h for h in self.highlights
            if self.config.min_highlight_duration <= h.duration <= self.config.max_highlight_duration
        ]

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
            app_config = AppConfig()
            app_config.detection.confidence_threshold = self.config.reframing.confidence_threshold
            self._reframing_pipeline = ReframingPipeline(app_config)

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

                # Step 2: Apply reframing if enabled
                if self.config.reframing.enabled:
                    output_clip = self.config.output_dir / f"highlight_{i+1}_vertical.mp4"

                    self._reframing_pipeline.process_goal_clip(
                        clip_path=str(temp_clip),
                        output_path=str(output_clip),
                        use_soccernet_model=self.config.reframing.use_soccernet_model,
                        use_temporal_filter=self.config.reframing.use_temporal_filter,
                        use_kalman_smoothing=self.config.reframing.use_kalman_smoothing
                    )

                    highlight.video_path = str(output_clip)
                    temp_clip.unlink()  # Clean up temp file
                else:
                    # No reframing, just use extracted clip
                    highlight.video_path = str(temp_clip)

            except Exception as e:
                logger.error(f"Failed to generate clip {i+1}: {e}")
                self.processing_errors.append(f"Clip {i+1} generation failed")

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

    def _align_to_scene_boundaries(self, highlight: Highlight, scenes: List[tuple]) -> None:
        """Adjust highlight boundaries to align with scene transitions."""
        for scene_start, scene_end in scenes:
            # If highlight falls within a scene, align to scene boundaries
            if scene_start <= highlight.start_time <= scene_end:
                highlight.start_time = scene_start
            if scene_start <= highlight.end_time <= scene_end:
                highlight.end_time = scene_end

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
        """Create dummy highlights for testing when no modules are enabled.

        This generates 3 sample highlights at different time points in the video.
        Used for testing the pipeline flow without actual detection modules.
        """
        logger.info("Creating 3 dummy highlights for testing")

        dummy_highlights = [
            Highlight(
                title="🎯 테스트 하이라이트 #1",
                description="영상 초반부 (10-40초)",
                start_time=10.0,
                end_time=40.0,
                score=0.9,
                source="dummy"
            ),
            Highlight(
                title="🎯 테스트 하이라이트 #2",
                description="영상 중반부 (60-90초)",
                start_time=60.0,
                end_time=90.0,
                score=0.85,
                source="dummy"
            ),
            Highlight(
                title="🎯 테스트 하이라이트 #3",
                description="영상 후반부 (120-150초)",
                start_time=120.0,
                end_time=150.0,
                score=0.8,
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
