"""Main page for file selection and editing option selection."""

import os
import logging
from typing import List, Dict, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QProgressBar, QPushButton
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QPainter, QPen, QColor

from src.pipeline.pipeline_worker import PipelineWorker

logger = logging.getLogger(__name__)


# Constants for styling and configuration
HEADER_HEIGHT_MIN = 60
HEADER_HEIGHT_MAX = 80
HEADER_BG_COLOR = "#7B68BE"
CONTENT_BG_COLOR = "#F4F2FB"
FRAME_MAX_WIDTH = 1100
FRAME_BG_COLOR = "#F4F2FB"
FRAME_BORDER_COLOR = "#CCCCCC"
DASHED_FRAME_BG_COLOR = "#F4F2FB"
DASHED_LINE_COLOR = "#CCCCCC"


# Processing steps
PROCESSING_STEPS = [
    "중계 내용 텍스트 변환",
    "객체 인식",
    "이벤트 태깅"
]


class DashedFrame(QFrame):
    """Custom QFrame with dashed border."""
    
    def __init__(self, parent: QWidget = None) -> None:
        """Initialize the dashed frame."""
        super().__init__(parent)
        
    def paintEvent(self, event) -> None:
        """Paint the dashed border."""
        super().paintEvent(event)
        
        # Set up painter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Configure pen for dashed line
        pen = QPen(QColor(DASHED_LINE_COLOR))
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setDashPattern([1, 2])  # [선 길이, 간격 길이]
        painter.setPen(pen)
        
        # Draw rounded rectangle with dash border
        rect = self.rect().adjusted(1, 1, -1, -1)  # Add padding to prevent clipping
        painter.drawRoundedRect(rect, 10, 10)


class ProgressPage(QWidget):
    """
    Process page widget showing video processing progress (waiting screen).
    
    This page displays real-time progress of video editing with:
    - Step-by-step status indicators
    - Percentage completion for each step
    - Overall progress bar
    - Visual feedback for completed/in-progress/pending steps
    
    This is a waiting screen with no user interaction during processing.
    Once complete, it can automatically transition to the next page.
    
    Signals:
        back_requested: Emitted when processing is complete or cancelled
    
    Attributes:
        file_path (str): Path to the selected video file
        option (str): Selected editing option
        progress (int): Overall progress (0-100)
        step_labels (list): Labels for each processing step
        step_progress (list): Progress values for each step
    """
    # AI 처리 완료 시 emit할 signal
    processing_completed = Signal(list)
    back_requested = Signal()  # 뒤로 가기 요청 시 emit
    
    def __init__(self) -> None:
        """Initialize the main page and set up UI components."""
        super().__init__()
        self.file_path: str = ""
        self.option: str = ""
        self.progress: int = 0
        self.step_labels: list = []
        self.step_progress: list = [0, 0, 0, 0]  # Progress for each step
        self.current_step: int = 0
        self.timer: QTimer = None

        # Pipeline worker
        self.worker: Optional[PipelineWorker] = None
        self.current_stage: str = ""

        self._setup_ui()
        
    def _setup_ui(self) -> None:
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create and add header
        header = self._create_header()
        main_layout.addWidget(header)
        
        # Create and add content
        content = self._create_content()
        main_layout.addWidget(content)
        
    def _create_header(self) -> QWidget:
        """Create the header bar."""
        header = QWidget()
        header.setMinimumHeight(HEADER_HEIGHT_MIN)
        header.setMaximumHeight(HEADER_HEIGHT_MAX)
        header.setStyleSheet(f"background-color: {HEADER_BG_COLOR};")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(10, 0, 20, 0)

        # Back button (left side)
        self.back_button = QPushButton("← 뒤로")
        self.back_button.setFixedHeight(35)
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14pt;
                font-family: Arial;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """)
        self.back_button.setCursor(Qt.PointingHandCursor)
        self.back_button.clicked.connect(self.on_back_button_clicked)
        layout.addWidget(self.back_button)

        # Spacer
        layout.addStretch()

        # Icon
        icon_label = QLabel("🎬")
        icon_label.setFont(QFont("Arial", 20))
        icon_label.setStyleSheet("border: none; background-color: transparent;")

        # Title
        title_label = QLabel("Shorts Genie")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: white; border: none; background-color: transparent;")

        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addStretch()

        return header
    
    def _create_content(self) -> QWidget:
        """Create the main content area."""
        content = QWidget()
        content.setStyleSheet(f"background-color: {CONTENT_BG_COLOR};")
        
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 20, 30, 20)
        
        # Create outer frame
        outer_frame = self._create_outer_frame()
        
        frame_container = QHBoxLayout()    
        frame_container.addWidget(outer_frame)
  
        layout.addLayout(frame_container)
        
        return content
    
    def _create_outer_frame(self) -> QFrame:
        """Create the outer frame containing all content."""
        outer_frame = QFrame()
        outer_frame.setMaximumWidth(FRAME_MAX_WIDTH)
        outer_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {FRAME_BG_COLOR};
                border: 2px solid {FRAME_BORDER_COLOR};
                border-radius: 15px;
            }}
        """)
        
        outer_layout = QVBoxLayout(outer_frame)
        outer_layout.setContentsMargins(90, 50, 90, 50)
        
        # Add dashed frame with file input and options
        dashed_frame = self._create_dashed_frame()
        outer_layout.addWidget(dashed_frame)
        outer_layout.addSpacing(20)
        
        return outer_frame
    
    def _create_dashed_frame(self) -> DashedFrame:
        """Create the dashed frame containing file input and option selection."""
        frame = DashedFrame()
        frame.setStyleSheet(f"""
            DashedFrame {{
                background-color: {DASHED_FRAME_BG_COLOR};
                border: none;
                border-radius: 10px;
            }}
        """)

        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(40, 40, 40, 40)
        frame_layout.setSpacing(15)
        
        # Processing steps
        self.step_labels = []
        for i, step_text in enumerate(PROCESSING_STEPS):
            step_layout = self._create_step_item(step_text, i)
            frame_layout.addLayout(step_layout)
        
        frame_layout.addSpacing(20)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #DDDDDD;
                border-radius: 5px;
                text-align: center;
                height: 25px;
                background-color: #F5F5F5;
            }
            QProgressBar::chunk {
                background-color: #7B68BE;
                border-radius: 4px;
            }
        """)
        frame_layout.addWidget(self.progress_bar)

        # Status message with loading animation
        self.status_label = QLabel("영상 분석 중입니다")
        self.status_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.status_label.setStyleSheet("color: #666666; border: none; background-color: transparent;")
        self.status_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(self.status_label)

        # Loading animation state
        self.loading_dots = 0
        self.loading_timer = QTimer()
        self.loading_timer.timeout.connect(self._update_loading_animation)
        self.loading_timer.start(500)  # Update every 500ms

        return frame
    
    def _create_step_item(self, text: str, index: int) -> QHBoxLayout:
        """
        Create a single step item with text and progress indicator.
        
        Args:
            text: Step description text
            index: Step index (0-3)
            
        Returns:
            QHBoxLayout: Layout containing step information
        """
        layout = QHBoxLayout()
        
        # Bullet point
        bullet = QLabel("•")
        bullet.setFont(QFont("Arial", 12))
        bullet.setStyleSheet("color: #333333; border: none; background-color: transparent;")
        
        # Step text
        step_label = QLabel(text)
        step_label.setFont(QFont("Arial", 11, QFont.Bold))
        step_label.setStyleSheet("color: #666666; border: none; background-color: transparent;")
        
        # Progress percentage label (hidden initially)
        progress_label = QLabel("")
        progress_label.setFont(QFont("Arial", 10, QFont.Bold))
        progress_label.setStyleSheet("color: #7B68BE; border: none; background-color: transparent;")
        progress_label.setVisible(False)
        
        layout.addWidget(bullet)
        layout.addWidget(step_label)
        layout.addStretch()
        layout.addWidget(progress_label)
        
        # Store labels for later updates
        self.step_labels.append({
            'bullet': bullet,
            'text': step_label,
            'progress': progress_label
        })
        
        return layout

    @Slot(str, str)
    def set_data(self, file_path: str, option: str) -> None:
        """
        Set the file path and option to display.

        Updates the labels with the provided information and logs
        to console for debugging purposes.

        Args:
            file_path: Path to the selected video file
            option: Selected editing option
        """
        self.file_path = file_path
        self.option = option

        # Log for debugging
        print(f"\n{'=' * 60}")
        print("Process Page Data Received")
        print(f"{'=' * 60}")
        print(f"File Path: {file_path}")
        print(f"Selected Option: {option}")
        print(f"{'=' * 60}\n")

        # Start real pipeline processing
        self.start_real_processing()

    def start_real_processing(self) -> None:
        """
        Start real pipeline processing with PipelineWorker.

        This replaces the fake simulation with actual AI processing.
        """
        logger.info(f"Starting real processing: {self.file_path} ({self.option} mode)")

        # Reset progress
        self.progress = 0
        self.current_step = 0
        self.step_progress = [0, 0, 0]
        self.update_progress_display()

        # Create worker
        self.worker = PipelineWorker(
            video_path=self.file_path,
            mode=self.option
        )

        # Connect signals
        self.worker.progress_updated.connect(self._on_progress_updated)
        self.worker.stage_changed.connect(self._on_stage_changed)
        self.worker.processing_completed.connect(self._on_processing_completed)
        self.worker.processing_failed.connect(self._on_processing_failed)

        # Start processing
        self.worker.start()

    def start_progress_simulation(self) -> None:
        """
        Start fake progress simulation with QTimer.

        This simulates the AI processing with automatic progress updates.
        DEPRECATED: Use start_real_processing() instead.
        """
        self.progress = 0
        self.current_step = 0
        self.step_progress = [0, 0, 0]

        # Create and start timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_fake_progress)
        self.timer.start(50)  # Update every 50ms
    
    @Slot()
    def update_fake_progress(self) -> None:
        """
        Update fake progress (simulation).
        
        This will be replaced with real progress updates from AI.
        """
        # Increment progress
        self.progress += 1
        
        # Update step progress based on overall progress
        if self.progress <= 33:
            # Step 1: 0-33%
            self.current_step = 0
            self.step_progress[0] = self.progress * 3.03
        elif self.progress <= 66:
            # Step 2: 33-66%
            self.current_step = 1
            self.step_progress[0] = 100
            self.step_progress[1] = (self.progress - 33) * 3.03
        else:
            # Step 3: 66-100%
            self.current_step = 2
            self.step_progress[0] = 100
            self.step_progress[1] = 100
            self.step_progress[2] = (self.progress - 66) * 2.97
        
        # Update UI
        self.update_progress_display()
        
        # Stop when complete
        if self.progress >= 100:
            self.timer.stop()
            self.on_processing_complete()
    
    def update_progress_display(self) -> None:
        """
        Update the progress display (progress bar and step indicators).
        """
        # Update progress bar
        self.progress_bar.setValue(self.progress)
        
        # Update each step
        for i, labels in enumerate(self.step_labels):
            step_prog = int(self.step_progress[i])
            
            if step_prog > 0:
                # Show progress percentage
                labels['progress'].setText(f"{step_prog}% 완료")
                labels['progress'].setVisible(True)
                
                # Change text color to black for active steps
                labels['text'].setStyleSheet("color: #333333; border: none; background-color: transparent;")
                labels['bullet'].setStyleSheet("color: #7B68BE; border: none; background-color: transparent;")
                
                if step_prog >= 100:
                    # Completed step - show checkmark
                    labels['bullet'].setText("✓")
                    labels['bullet'].setStyleSheet("color: #4CAF50; border: none; background-color: transparent;")
            else:
                # Pending step - gray text
                labels['text'].setStyleSheet("color: #AAAAAA; border: none; background-color: transparent;")
                labels['bullet'].setStyleSheet("color: #CCCCCC; border: none; background-color: transparent;")
    
    def on_processing_complete(self) -> None:
        """
        Handle processing completion.
        
        This will be called when progress reaches 100%.
        """
        print("\n" + "=" * 60)
        print("Processing Complete!")
        print("=" * 60 + "\n")
        
        # AI 결과 가져오기
        ai_results = self._get_ai_results()

        # SelectPage로 결과 전달
        self.processing_completed.emit(ai_results)

    def _get_ai_results(self) -> List:
        """
        Get AI processing results.

        Returns:
        List of highlight dictionaries with keys:
        - title: 하이라이트 제목
        - description: 하이라이트 설명
        - start_time: 시작 시간
        - end_time: 종료 시간
        - video_path: 생성된 영상 파일 경로
        """
        # DEPRECATED: This is now replaced by real pipeline results
        # Kept for backwards compatibility

        return [
            {
                'title': '후보 1) 골 모음 영상',
                'description': '선수들의 골을 모아서 보여줌',
                'start_time': '00:30',
                'end_time': '01:15'
            },
            {
                'title': '후보 2) 경기 주요 영상',
                'description': '골 넣은걸 제외하고 볼만할 모음',
                'start_time': '02:45',
                'end_time': '03:20'
            },
            {
                'title': '후보 3) 밈 영상',
                'description': '음악이 추가된 하이라이트',
                'start_time': '05:10',
                'end_time': '06:00'
            }
        ]

    # Real pipeline signal handlers

    @Slot(str, int, str)
    def _on_progress_updated(self, stage: str, progress: int, message: str) -> None:
        """Handle progress update from pipeline worker.

        Args:
            stage: Current processing stage name
            progress: Overall progress (0-100)
            message: Detailed progress message
        """
        logger.debug(f"Progress: [{progress}%] {stage}: {message}")

        # Update overall progress
        self.progress = progress

        # Map stage to step index
        stage_to_step = {
            "초기화": 0,
            "OCR 골 감지": 0,
            "오디오 분석": 1,
            "해설 분석": 1,
            "AI 분석": 1,
            "장면 전환 감지": 2,
            "후처리": 2,
            "영상 생성": 2,
            "완료": 2
        }

        step_idx = stage_to_step.get(stage, 0)
        self.current_step = step_idx

        # Update step progress (distribute 0-100 across 3 steps)
        if progress <= 33:
            self.step_progress[0] = progress * 3.03
        elif progress <= 66:
            self.step_progress[0] = 100
            self.step_progress[1] = (progress - 33) * 3.03
        else:
            self.step_progress[0] = 100
            self.step_progress[1] = 100
            self.step_progress[2] = (progress - 66) * 2.97

        # Update UI
        self.update_progress_display()

    @Slot(str)
    def _on_stage_changed(self, stage: str) -> None:
        """Handle stage change from pipeline worker.

        Args:
            stage: New stage name
        """
        logger.info(f"Stage changed: {stage}")
        self.current_stage = stage
        # Update status label immediately
        self.status_label.setText(f"{stage}...")

    @Slot(list)
    def _on_processing_completed(self, highlights: List[Dict]) -> None:
        """Handle successful completion from pipeline worker.

        Args:
            highlights: List of highlight dictionaries from pipeline
        """
        logger.info(f"Processing completed with {len(highlights)} highlights")

        # Ensure progress is at 100%
        self.progress = 100
        self.step_progress = [100, 100, 100]
        self.update_progress_display()

        # Emit results to SelectPage
        print("\n" + "=" * 60)
        print(f"✓ Real Processing Complete! ({len(highlights)} highlights)")
        print("=" * 60 + "\n")

        self.processing_completed.emit(highlights)

    @Slot(str)
    def _on_processing_failed(self, error: str) -> None:
        """Handle processing failure from pipeline worker.

        Args:
            error: Error message
        """
        logger.error(f"Processing failed: {error}")

        print("\n" + "=" * 60)
        print(f"✗ Processing Failed: {error}")
        print("=" * 60 + "\n")

        # For now, emit empty results (could show error page instead)
        self.processing_completed.emit([])

    def _update_loading_animation(self) -> None:
        """Update loading animation (dots cycling)."""
        if self.progress >= 100:
            # Stop animation when complete
            self.loading_timer.stop()
            self.status_label.setText("처리 완료!")
            return

        # Cycle through 0, 1, 2, 3 dots
        self.loading_dots = (self.loading_dots + 1) % 4
        dots = "." * self.loading_dots
        spaces = " " * (3 - self.loading_dots)  # Keep text width consistent

        # Update status label with animated dots
        base_text = "영상 분석 중입니다"
        if self.current_stage:
            # Use current stage name if available
            base_text = self.current_stage

        self.status_label.setText(f"{base_text}{dots}{spaces}")

    @Slot()
    def on_back_button_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()