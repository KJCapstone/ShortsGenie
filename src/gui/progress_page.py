"""Process page showing video processing progress."""

import logging
from typing import List, Dict, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QProgressBar, QPushButton, QSizePolicy, QMessageBox
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


# Processing steps (updated to match actual pipeline execution)
PROCESSING_STEPS = [
    "음성 인식 (Whisper)",      # Step 1: Whisper transcription (5-35%)
    "AI 하이라이트 분석",        # Step 2: Gemini AI analysis (35-60%)
    "영상 생성 및 편집"          # Step 3: Post-processing + Video generation (60-100%)
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
        
        # [수정] 점선 패턴 통일 [4, 4]
        pen.setDashPattern([4, 4]) 
        painter.setPen(pen)
        
        # Draw rounded rectangle with dash border
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 10, 10)


class ProgressPage(QWidget):
    """Process page widget showing video processing progress."""
    
    # AI 처리 완료 시 emit할 signal
    processing_completed = Signal(list)
    back_requested = Signal()
    
    def __init__(self) -> None:
        """Initialize the main page and set up UI components."""
        super().__init__()
        self.file_path: str = ""
        self.option: str = ""
        self.progress: int = 0
        self.step_labels: list = []
        self.step_progress: list = [0, 0, 0, 0]
        self.current_step: int = 0
        self.timer: QTimer = None

        # Pipeline worker
        self.worker: Optional[PipelineWorker] = None
        self.current_stage: str = ""
        
        # Backend settings
        self.backend: str = "whisper"
        self.groq_api_key: str = ""

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
        layout.setContentsMargins(10, 0, 10, 0) # 좌우 여백 동일하게

        # [1] 왼쪽: 뒤로 버튼
        self.back_button = QPushButton("← 뒤로")
        self.back_button.setFixedHeight(35)
        # 버튼 스타일 (폰트 통일)
        button_style = """
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """
        self.back_button.setStyleSheet(button_style)
        self.back_button.setCursor(Qt.PointingHandCursor)
        self.back_button.clicked.connect(self.on_back_button_clicked)
        layout.addWidget(self.back_button)

        # [2] 왼쪽 스페이서
        layout.addStretch()

        # [3] 가운데: 제목 (아이콘 + 텍스트 합침)
        title_label = QLabel("🎬ShortsGenie")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: white; border: none; background-color: transparent;")
        title_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(title_label)

        # [4] 오른쪽 스페이서
        layout.addStretch()

        # [5] 오른쪽: 투명한 더미 버튼 (제목 중앙 정렬을 위한 핵심!)
        dummy_button = QPushButton("← 뒤로") # 왼쪽 버튼과 내용 동일하게
        dummy_button.setFixedHeight(35)
        dummy_button.setStyleSheet(button_style) 
        dummy_button.setFlat(True) 
        dummy_button.setEnabled(False) 
        # 글자색 투명하게 (안 보이게)
        dummy_button.setStyleSheet(button_style + "QPushButton { color: transparent; }") 
        
        # 공간만 차지하도록 설정
        dummy_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(dummy_button)

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
        outer_layout.setContentsMargins(40, 30, 40, 30)
        
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
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #7B68BE;
                border-radius: 4px;
            }
        """)
        frame_layout.addWidget(self.progress_bar)

        # Status message with loading animation
        self.status_label = QLabel("영상 분석 중입니다")
        
        # [수정] 기본 폰트 사용
        status_font = QFont()
        status_font.setPointSize(10)
        status_font.setBold(True)
        self.status_label.setFont(status_font)
        
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
        """Create a single step item with text and progress indicator."""
        layout = QHBoxLayout()
        
        # [수정] 기본 폰트 사용
        # Bullet point
        bullet = QLabel("•")
        bullet_font = QFont()
        bullet_font.setPointSize(12)
        bullet.setFont(bullet_font)
        bullet.setStyleSheet("color: #333333; border: none; background-color: transparent;")
        
        # Step text
        step_label = QLabel(text)
        step_font = QFont()
        step_font.setPointSize(11)
        step_font.setBold(True)
        step_label.setFont(step_font)
        step_label.setStyleSheet("color: #666666; border: none; background-color: transparent;")
        
        # Progress percentage label (hidden initially)
        progress_label = QLabel("")
        prog_font = QFont()
        prog_font.setPointSize(10)
        prog_font.setBold(True)
        progress_label.setFont(prog_font)
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
    def set_data(self, file_path: str, option: str, backend: str = "whisper", groq_api_key: str = "") -> None:
        """
        Set the file path and option to display.

        Updates the labels with the provided information and logs
        to console for debugging purposes.

        Args:
            file_path: Path to the selected video file
            option: Selected editing option
            backend: Transcription backend ("whisper" or "groq")
            groq_api_key: Groq API key (if backend is "groq")
        """
        self.file_path = file_path
        self.option = option
        self.backend = backend
        self.groq_api_key = groq_api_key

        # Log for debugging
        print(f"\n{'=' * 60}")
        print("Process Page Data Received")
        print(f"{'=' * 60}")
        print(f"File Path: {file_path}")
        print(f"Selected Option: {option}")
        print(f"Backend: {backend}")
        print(f"{'=' * 60}\n")

        # Start real pipeline processing
        self.start_real_processing()

    def start_real_processing(self) -> None:
        """
        Start real pipeline processing with PipelineWorker.

        This replaces the fake simulation with actual AI processing.
        """
        logger.info(f"Starting real processing: {self.file_path} ({self.option} mode, backend: {self.backend})")

        # Reset progress
        self.progress = 0
        self.current_step = 0
        self.step_progress = [0, 0, 0]
        self.update_progress_display()

        # Create worker
        self.worker = PipelineWorker(
            video_path=self.file_path,
            mode=self.option,
            backend=self.backend,
            groq_api_key=self.groq_api_key
        )

        # Connect signals
        self.worker.progress_updated.connect(self._on_progress_updated)
        self.worker.stage_changed.connect(self._on_stage_changed)
        self.worker.processing_completed.connect(self._on_processing_completed)
        self.worker.processing_failed.connect(self._on_processing_failed)

        # Start processing
        self.worker.start()

    def update_progress_display(self) -> None:
        """Update the progress display (progress bar and step indicators)."""
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
    
    # Real pipeline signal handlers

    @Slot(str, int, str)
    def _on_progress_updated(self, stage: str, progress: int, message: str) -> None:
        """Handle progress update from pipeline worker."""
        logger.debug(f"Progress: [{progress}%] {stage}: {message}")

        # Update overall progress
        self.progress = progress

        # Map stage to step index (updated to match actual execution flow)
        stage_to_step = {
            "초기화": 0,
            "음성 인식": 0,       # Step 1: Whisper transcription (5-35%)
            "AI 분석": 1,         # Step 2: Gemini analysis (35-60%)
            "해설 분석": 1,       # Legacy fallback
            "후처리": 2,          # Step 3: Post-processing + Video gen (60-100%)
            "영상 생성": 2,
            "영상 병합": 2,
            "완료": 2
        }

        step_idx = stage_to_step.get(stage, 0)
        self.current_step = step_idx

        # Update step progress based on actual progress ranges
        # Step 1: 0-35% (Whisper transcription)
        # Step 2: 35-60% (Gemini AI analysis)
        # Step 3: 60-100% (Post-processing + Video generation)
        if progress <= 35:
            self.step_progress[0] = (progress / 35) * 100
        elif progress <= 60:
            self.step_progress[0] = 100
            self.step_progress[1] = ((progress - 35) / 25) * 100
        else:
            self.step_progress[0] = 100
            self.step_progress[1] = 100
            self.step_progress[2] = ((progress - 60) / 40) * 100

        # Update UI
        self.update_progress_display()

    @Slot(str)
    def _on_stage_changed(self, stage: str) -> None:
        """Handle stage change from pipeline worker."""
        logger.info(f"Stage changed: {stage}")
        self.current_stage = stage
        # Update status label immediately
        self.status_label.setText(f"{stage}...")

    @Slot(list)
    def _on_processing_completed(self, highlights: List[Dict]) -> None:
        """Handle successful completion from pipeline worker."""
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
        """Handle processing failure from pipeline worker."""
        logger.error(f"Processing failed: {error}")

        print("\n" + "=" * 60)
        print(f"✗ Processing Failed: {error}")
        print("=" * 60 + "\n")

        # Show error dialog to user
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("처리 실패")

        # Customize message based on error type
        if "지루하거나 오디오에 문제" in error:
            msg_box.setText("하이라이트 추출 실패")
            msg_box.setInformativeText(
                "이 경기는 매우 지루하거나 오디오에 문제가 있을 수 있습니다.\n\n"
                "가능한 원인:\n"
                "• 골이나 결정적 찬스가 없는 경기 (0-0 무승부 등)\n"
                "• 음성 인식 실패 또는 중계 음성 없음\n"
                "• 오디오 품질이 매우 낮음\n\n"
                "다른 경기 영상으로 시도해보세요."
            )
        else:
            msg_box.setText("처리 중 오류가 발생했습니다")
            msg_box.setInformativeText(error)

        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()

        # Emit empty results to return to main page
        self.processing_completed.emit([])

    def _update_loading_animation(self) -> None:
        """Update loading animation (dots cycling)."""
        if self.progress >= 100:
            self.loading_timer.stop()
            self.status_label.setText("처리 완료!")
            return

        self.loading_dots = (self.loading_dots + 1) % 4
        dots = "." * self.loading_dots
        spaces = " " * (3 - self.loading_dots)

        base_text = "영상 분석 중입니다"
        if self.current_stage:
            base_text = self.current_stage

        self.status_label.setText(f"{base_text}{dots}{spaces}")

    @Slot()
    def on_back_button_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()