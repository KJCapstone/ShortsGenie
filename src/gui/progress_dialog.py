"""Progress dialog for video processing."""

from pathlib import Path
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QTextEdit, QPushButton  # type: ignore[import-not-found]
from PySide6.QtCore import QThread, Signal, Qt  # type: ignore[import-not-found]


class ProcessingThread(QThread):
    """Background thread for video processing."""

    # Signals
    status_updated = Signal(str)
    progress_updated = Signal(int)
    log_message = Signal(str)
    processing_finished = Signal(bool, dict)

    def __init__(self, pipeline, input_path: str, output_path: str):
        """
        Initialize processing thread.

        Args:
            pipeline: ReframingPipeline instance
            input_path: Path to input video
            output_path: Path to output video
        """
        super().__init__()
        self.pipeline = pipeline
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def run(self):
        """Run the processing pipeline."""
        try:
            self.status_updated.emit("Starting video processing...")
            self.log_message.emit(f"Input: {self.input_path}")
            self.log_message.emit(f"Output: {self.output_path}")

            # Process video
            stats = self.pipeline.process_video(
                input_path=self.input_path,
                output_path=self.output_path,
                progress_callback=self._progress_callback,
            )

            self.status_updated.emit("Processing complete!")
            self.processing_finished.emit(True, stats)

        except Exception as e:
            self.log_message.emit(f"ERROR: {str(e)}")
            self.status_updated.emit(f"Processing failed: {str(e)}")
            self.processing_finished.emit(False, {})

    def _progress_callback(self, current: int, total: int, message: str = ""):
        """
        Callback for progress updates.

        Args:
            current: Current frame number
            total: Total number of frames
            message: Optional status message
        """
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_updated.emit(progress)

        if message:
            self.log_message.emit(message)


class ProgressDialog(QDialog):
    """Dialog showing video processing progress."""

    def __init__(self, parent=None):
        """Initialize progress dialog."""
        super().__init__(parent)
        self.setWindowTitle("Processing Video")
        self.setMinimumSize(600, 400)
        self.setModal(True)

        # Setup UI
        layout = QVBoxLayout()

        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Log text area
        log_label = QLabel("Processing Log:")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Close button (initially disabled)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.close_button.setEnabled(False)
        layout.addWidget(self.close_button)

        self.setLayout(layout)

    def set_status(self, status: str):
        """
        Update status label.

        Args:
            status: Status message
        """
        self.status_label.setText(status)

    def set_progress(self, progress: int):
        """
        Update progress bar.

        Args:
            progress: Progress percentage (0-100)
        """
        self.progress_bar.setValue(progress)

    def log(self, message: str):
        """
        Add message to log.

        Args:
            message: Log message
        """
        self.log_text.append(message)

    def processing_finished(self, success: bool):
        """
        Handle processing completion.

        Args:
            success: Whether processing was successful
        """
        self.close_button.setEnabled(True)
        if success:
            self.set_status("✅ Processing Complete!")
        else:
            self.set_status("❌ Processing Failed")
