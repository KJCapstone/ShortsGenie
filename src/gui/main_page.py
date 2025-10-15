"""Main page for file selection and editing option selection."""

import os
from typing import List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QFileDialog, QFrame, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot
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
INPUT_HEIGHT_MIN = 35
INPUT_HEIGHT_MAX = 45
BUTTON_HEIGHT = 35
EDIT_BUTTON_SIZE = (150, 40)

# Editing options
EDITING_OPTIONS = ["⚽ 골 모음 영상", "⚡ 경기 주요 영상", "🎵 밈 영상"]


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


class MainPage(QWidget):
    """
    Main page widget for video file and option selection.
    
    이 페이지는 사용자가 다음을 수행할 수 있게 합니다:
    1. 비디오 파일 탐색 및 선택
    2. 사전 정의된 조건에서 편집 옵션 선택
    3. 편집 프로세스 시작
    
    Signals:
        edit_requested: 사용자가 편집을 시작할 때 발생 (file_path, option)
    
    Attributes:
        file_path_edit (QLineEdit): 파일 경로 입력 필드
        option_buttons (List[QPushButton]): 옵션 선택 버튼 리스트
        selected_option_index (int): 현재 선택된 옵션의 인덱스
    """
    
    # Signal: emitted when editing is requested with file path and selected option
    edit_requested = Signal(str, str)
    
    def __init__(self) -> None:
        """Initialize the main page and set up UI components."""
        super().__init__()
        self.option_buttons: List[QPushButton] = []
        self.selected_option_index: int = 0
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
        
        # Add edit button
        edit_button = self._create_edit_button()
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(edit_button)
        button_layout.addStretch()
        
        outer_layout.addLayout(button_layout)
        
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
        
        inner_layout = QVBoxLayout(frame)
        inner_layout.setContentsMargins(90, 50, 90, 50)
        inner_layout.setSpacing(20)
        
        # File input section
        self._add_file_input_section(inner_layout)
        inner_layout.addSpacing(20)
        
        # Option selection section
        self._add_option_selection_section(inner_layout)
        
        return frame
    
    def _add_file_input_section(self, layout: QVBoxLayout) -> None:
        """Add file input section to the given layout."""
        # Label
        file_label = QLabel("편집할 영상을 넣어주세요.")
        file_label_font = QFont()
        file_label_font.setPointSize(12)
        file_label_font.setBold(True)
        file_label.setFont(file_label_font)
        file_label.setStyleSheet("border: none; background-color: transparent;")
        layout.addWidget(file_label)
        
        # File input row
        file_input_layout = QHBoxLayout()
        
        # Path input field
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("/path/input.mp4")
        self.file_path_edit.setMinimumHeight(INPUT_HEIGHT_MIN)
        self.file_path_edit.setMaximumHeight(INPUT_HEIGHT_MAX)
        self.file_path_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #DDDDDD;
                border-radius: 5px;
                padding: 5px 10px;
                background-color: #FAFAFA;
            }
        """)
        
        # Browse button
        browse_btn = QPushButton("찾아보기")
        browse_btn.setMinimumSize(80, BUTTON_HEIGHT)
        browse_btn.setMaximumWidth(100)
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                border: none;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #D0D0D0;
            }
        """)
        browse_btn.clicked.connect(self.browse_file)
        
        file_input_layout.addWidget(self.file_path_edit)
        file_input_layout.addWidget(browse_btn)
        layout.addLayout(file_input_layout)
    
    def _add_option_selection_section(self, layout: QVBoxLayout) -> None:
        """Add option selection section to the given layout."""

        # Label
        condition_label = QLabel("원하는 영상 조건을 골라주세요.")
        condition_label_font = QFont()
        condition_label_font.setPointSize(12)
        condition_label_font.setBold(True)
        condition_label.setFont(condition_label_font)
        condition_label.setStyleSheet("border: none; background-color: transparent;")
        layout.addWidget(condition_label)
        
        # Option buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        for i, text in enumerate(EDITING_OPTIONS):
            btn = self._create_option_button(text)
            self.option_buttons.append(btn)
            button_layout.addWidget(btn)
        
        # Set default selection
        self.option_buttons[0].setChecked(True)
        self.selected_option_index = 0
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def _create_option_button(self, text: str) -> QPushButton:
        """Create an option selection button."""

        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setMinimumHeight(BUTTON_HEIGHT)
        btn.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                background-color: white;
                border: 1px solid #DDDDDD;
                border-radius: 10px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #F0F0F0;
                border: 1px solid #BBBBBB;
            }
            QPushButton:hover {
                background-color: #FAFAFA;
            }
        """)
        btn.clicked.connect(lambda checked, b=btn: self.on_option_clicked(b))
        return btn
    
    def _create_edit_button(self) -> QPushButton:
        """Create the main edit button."""

        edit_button = QPushButton("영상 편집하기")
        edit_button.setMinimumSize(*EDIT_BUTTON_SIZE)
        edit_button.setMaximumSize(200, 50)
        edit_button.setStyleSheet(f"""
            QPushButton {{
                background-color: white;
                border: 1px solid #DDDDDD;
                border-radius: 10px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #7B68BE;
                color: white;
            }}
        """)
        edit_button.clicked.connect(self.start_editing)
        return edit_button
    
    @Slot()
    def browse_file(self) -> None:
        """Open file dialog for video file selection."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "영상 파일 선택",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)
    
    @Slot(object)
    def on_option_clicked(self, clicked_button: QPushButton) -> None:
        """Handle option button click."""

        # Uncheck all other buttons
        for btn in self.option_buttons:
            if btn != clicked_button:
                btn.setChecked(False)
        
        # Update selected index
        self.selected_option_index = self.option_buttons.index(clicked_button)
    
    @Slot()
    def start_editing(self) -> None:
        """Start the editing process."""

        file_path = self.file_path_edit.text().strip()
        
        # Validate file path
        if not file_path:
            QMessageBox.warning(
                self,
                "경고",
                "영상 파일을 선택해주세요."
            )
            return
        
        # Validate existence
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "경고", "파일이 존재하지 않습니다.")
            return
        
        # Validate extension
        if not file_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            QMessageBox.warning(self, "경고", "지원하지 않는 영상 형식입니다.")
            return
        
        # Get selected option
        selected_option = EDITING_OPTIONS[self.selected_option_index]
        
        # Emit signal to request page transition
        self.edit_requested.emit(file_path, selected_option)