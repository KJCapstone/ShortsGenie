"""QThread worker for non-blocking pipeline execution in GUI.

This module provides a PySide6 QThread wrapper around the HighlightPipeline
to prevent GUI freezing during long processing operations.
"""

import logging
from typing import List, Dict, Optional
from PySide6.QtCore import QThread, Signal

from src.pipeline.highlight_pipeline import HighlightPipeline, Highlight
from src.pipeline.pipeline_config import create_config_from_mode, PipelineConfig

logger = logging.getLogger(__name__)


class PipelineWorker(QThread):
    """Background worker for running the highlight generation pipeline.

    Signals:
        progress_updated: Emitted during processing with (stage, progress, message)
        stage_changed: Emitted when entering a new processing stage (stage_name)
        processing_completed: Emitted when done with list of highlight dicts
        processing_failed: Emitted on error with error message
    """

    # Signals for GUI communication
    progress_updated = Signal(str, int, str)  # stage, progress, message
    stage_changed = Signal(str)  # stage_name
    processing_completed = Signal(list)  # List[Dict] of highlights
    processing_failed = Signal(str)  # error_message

    def __init__(
        self,
        video_path: str,
        mode: str,
        config: Optional[PipelineConfig] = None,
        backend: str = "whisper",
        groq_api_key: str = "",
        parent=None
    ):
        """Initialize pipeline worker.

        Args:
            video_path: Path to input video file
            mode: Processing mode ("골", "경기", "밈")
            config: Optional custom configuration. If None, uses mode default.
            backend: Transcription backend ("whisper" or "groq")
            groq_api_key: Groq API key (if backend is "groq")
            parent: Parent QObject
        """
        super().__init__(parent)
        self.video_path = video_path
        self.mode = mode
        self.config = config or create_config_from_mode(mode)
        self.pipeline: Optional[HighlightPipeline] = None
        self._is_cancelled = False

        # Apply backend settings to config
        self.config.transcript_analysis.backend = backend
        if backend == "groq" and groq_api_key:
            self.config.transcript_analysis.groq_api_key = groq_api_key

    def run(self):
        """Run the pipeline in background thread.

        This method is called automatically when thread.start() is invoked.
        """
        try:
            logger.info(f"PipelineWorker started for {self.video_path}")
            logger.info(f"Mode: {self.mode}")

            # Create pipeline with progress callback
            self.pipeline = HighlightPipeline(
                config=self.config,
                progress_callback=self._progress_callback
            )

            # Run processing
            self.stage_changed.emit("초기화")
            highlights = self.pipeline.process(self.video_path)

            # Check if cancelled before emitting results
            if self._is_cancelled:
                logger.info("Processing was cancelled")
                self.processing_failed.emit("사용자가 취소했습니다")
                return

            # Convert to dictionaries for GUI
            highlight_dicts = [h.to_dict() for h in highlights]

            logger.info(f"PipelineWorker completed: {len(highlights)} highlights")
            self.processing_completed.emit(highlight_dicts)

        except Exception as e:
            error_msg = f"파이프라인 오류: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.processing_failed.emit(error_msg)

    def cancel(self):
        """Request cancellation of processing.

        Note: Cancellation is cooperative - the pipeline will stop at the
        next checkpoint. Current module may finish before stopping.
        """
        logger.info("Cancellation requested")
        self._is_cancelled = True

    def _progress_callback(self, stage: str, progress: int, message: str):
        """Internal callback for pipeline progress updates.

        Emits signals to update GUI in a thread-safe manner.

        Args:
            stage: Current processing stage name
            progress: Progress percentage (0-100)
            message: Detailed progress message
        """
        # Check for cancellation
        if self._is_cancelled:
            logger.debug(f"Skipping progress update (cancelled): {stage}")
            return

        # Emit signals
        self.progress_updated.emit(stage, progress, message)

        # Emit stage change if different from last
        if not hasattr(self, "_last_stage") or self._last_stage != stage:
            self.stage_changed.emit(stage)
            self._last_stage = stage


class BatchPipelineWorker(QThread):
    """Worker for processing multiple videos in batch mode.

    Signals:
        batch_progress_updated: Emitted with (current, total, video_name)
        video_completed: Emitted when one video finishes with (index, highlights)
        batch_completed: Emitted when all videos are done
        batch_failed: Emitted on error with (video_index, error_message)
    """

    # Batch signals
    batch_progress_updated = Signal(int, int, str)  # current, total, video_name
    video_completed = Signal(int, list)  # index, highlights
    batch_completed = Signal()
    batch_failed = Signal(int, str)  # index, error_message

    # Forward individual video progress
    progress_updated = Signal(str, int, str)  # stage, progress, message

    def __init__(
        self,
        video_paths: List[str],
        mode: str,
        config: Optional[PipelineConfig] = None,
        parent=None
    ):
        """Initialize batch worker.

        Args:
            video_paths: List of video file paths to process
            mode: Processing mode for all videos
            config: Optional custom configuration
            parent: Parent QObject
        """
        super().__init__(parent)
        self.video_paths = video_paths
        self.mode = mode
        self.config = config or create_config_from_mode(mode)
        self._is_cancelled = False

    def run(self):
        """Process all videos in sequence."""
        total = len(self.video_paths)
        logger.info(f"Starting batch processing: {total} videos")

        for idx, video_path in enumerate(self.video_paths):
            if self._is_cancelled:
                logger.info("Batch processing cancelled")
                return

            try:
                # Use Path for cross-platform compatibility
                from pathlib import Path
                video_name = Path(video_path).name
                logger.info(f"Processing video {idx+1}/{total}: {video_name}")

                # Update batch progress
                self.batch_progress_updated.emit(idx + 1, total, video_name)

                # Process single video
                pipeline = HighlightPipeline(
                    config=self.config,
                    progress_callback=lambda s, p, m: self.progress_updated.emit(s, p, m)
                )

                highlights = pipeline.process(video_path)
                highlight_dicts = [h.to_dict() for h in highlights]

                # Emit completion for this video
                self.video_completed.emit(idx, highlight_dicts)

            except Exception as e:
                error_msg = f"Failed: {str(e)}"
                logger.error(f"Video {idx} failed: {error_msg}", exc_info=True)
                self.batch_failed.emit(idx, error_msg)

                # Continue with next video (don't stop batch on single failure)

        # All done
        logger.info("Batch processing completed")
        self.batch_completed.emit()

    def cancel(self):
        """Cancel batch processing."""
        logger.info("Batch cancellation requested")
        self._is_cancelled = True


# Example usage for testing
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    def on_progress(stage: str, progress: int, message: str):
        print(f"[{progress}%] {stage}: {message}")

    def on_completed(highlights: List[Dict]):
        print(f"\n✓ Completed! Generated {len(highlights)} highlights:")
        for i, h in enumerate(highlights, 1):
            print(f"  {i}. {h['title']} ({h['start_time']} - {h['end_time']})")

    def on_failed(error: str):
        print(f"\n✗ Failed: {error}")

    # Test with a video file
    worker = PipelineWorker(
        video_path="input/test_video.mp4",
        mode="골"
    )

    worker.progress_updated.connect(on_progress)
    worker.processing_completed.connect(on_completed)
    worker.processing_failed.connect(on_failed)
    worker.finished.connect(app.quit)

    worker.start()
    sys.exit(app.exec())
