"""Main page for file selection and editing option selection."""

import os
from typing import List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QFrame, QProgressBar
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QPainter, QPen, QColor


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


class ProcessPage(QWidget):
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
    # Signal: emitted when editing is requested with file path and selected option
    back_requested = Signal()
    
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
        layout.setContentsMargins(15, 0, 0, 0)
        
        # Icon
        icon_label = QLabel("🎬")
        icon_label.setFont(QFont("Arial", 20))
        
        # Title
        title_label = QLabel("Shorts Genie")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        
        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addStretch()
        
        return header
    
    def _create_content(self) -> QWidget:
        """Create the main content area."""
        content = QWidget()
        content.setStyleSheet(f"background-color: {CONTENT_BG_COLOR};")
        
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Create outer frame
        outer_frame = self._create_outer_frame()
        
        # Center the outer frame
        frame_container = QHBoxLayout()
        frame_container.addStretch()
        frame_container.addWidget(outer_frame)
        frame_container.addStretch()
        
        layout.addLayout(frame_container)
        layout.addStretch()
        
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

        # Status message
        self.status_label = QLabel("영상 분석 중입니다...")
        self.status_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.status_label.setStyleSheet("color: #666666; border: none; background-color: transparent;")
        self.status_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(self.status_label)

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
        
        # Start fake progress simulation
        self.start_progress_simulation()

    def start_progress_simulation(self) -> None:
        """
        Start fake progress simulation with QTimer.
        
        This simulates the AI processing with automatic progress updates.
        Later, this will be replaced with real AI progress.
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
        
        # Here you can:
        # - Show completion message
        # - Enable "View Result" button
        # - Automatically go to result page
        # etc.
    
    @Slot()
    def go_back(self) -> None:
        """
        Request navigation back to main page.
        
        Emits the back_requested signal to notify the parent window
        to switch to the main page.
        """
        self.back_requested.emit()