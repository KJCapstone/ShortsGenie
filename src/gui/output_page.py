"""Output settings page for video export configuration."""

from typing import Optional, Dict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QComboBox, QFrame,
    QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont, QPainter, QPen, QColor


# Color scheme
HEADER_BG_COLOR = "#7B68BE"
CONTENT_BG_COLOR = "#F4F2FB"
FRAME_BG_COLOR = "#F4F2FB"
FRAME_BORDER_COLOR = "#CCCCCC"
PRIMARY_COLOR = "#7B68BE"
TEXT_COLOR = "#374151"
INPUT_BORDER_COLOR = "#CCCCCC"
DASHED_FRAME_BG_COLOR = "#F9F8FC"
DASHED_LINE_COLOR = "#CCCCCC"

# Size constants
HEADER_HEIGHT_MIN = 60
HEADER_HEIGHT_MAX = 80
INPUT_HEIGHT = 35
BUTTON_HEIGHT = 35
EXPORT_BUTTON_SIZE = (150, 40)

class DashedFrame(QFrame):
    """
    Custom QFrame with dashed border.
    Provides consistent dashed border styling across all pages.
    """
    
    def __init__(self, parent=None):
        """Initialize the dashed frame."""
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {DASHED_FRAME_BG_COLOR};")
        
    def paintEvent(self, event):
        """Paint the dashed border."""
        super().paintEvent(event)
        
        # Set up painter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Configure pen for dashed line
        pen = QPen(QColor(DASHED_LINE_COLOR))
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setDashPattern([1, 2])
        painter.setPen(pen)
        
        # Draw rounded rectangle with dash border
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 10, 10)


class OutputSettingsPage(QWidget):
    """
    Output settings page for video export configuration.
    
    Provides interface for:
    - Selecting save directory
    - Setting filename
    - Choosing output quality
    
    Signals:
        export_requested: Emitted with (save_path, filename, quality)
    """
    
    # Signals
    export_requested = Signal(str, str, str)
    back_requested = Signal()  # 뒤로 가기 요청 시 emit
    
    def __init__(self, parent=None):
        """Initialize the output settings page."""
        super().__init__(parent)
        self.video_info = None
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = self._setup_header()
        main_layout.addWidget(header)
        
        # Content area
        content = self._setup_content()
        main_layout.addWidget(content)
        
    def _setup_header(self) -> QWidget:
        """Create the header bar with app branding."""
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

        # App icon
        icon_label = QLabel("🎬")
        icon_label.setFont(QFont("Arial", 20))
        icon_label.setStyleSheet("border: none; background-color: transparent;")

        # App title
        title_label = QLabel("Shorts Genie")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: white; border: none; background-color: transparent;")

        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addStretch()

        return header
    
    def _setup_content(self) -> QWidget:
        """Create the main content area."""
        content = QWidget()
        content.setStyleSheet(f"background-color: {CONTENT_BG_COLOR};")
        
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(30, 20, 30, 20)

        # Container frame
        container = self._setup_container()
        content_layout.addWidget(container)

        return content
    
    def _setup_container(self) -> QFrame:
        """Create the main container frame."""
        container = QFrame()
        container.setObjectName("container")
        container.setStyleSheet(f"""
            #container {{
                background-color: {FRAME_BG_COLOR};
                border: 2px solid {FRAME_BORDER_COLOR};
                border-radius: 20px;
            }}
        """)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Settings area with dashed border
        settings_frame = self._setup_settings_frame()
        layout.addWidget(settings_frame)
        
        # Export button
        export_button = self._setup_export_button()
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(export_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)

        return container
    
    def _setup_settings_frame(self) -> DashedFrame:
        """Create the settings frame with input fields."""
        frame = DashedFrame()
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Row 1: Save path
        path_section = self._setup_path_section()
        layout.addLayout(path_section)
        
        # Row 2: Filename and quality
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(20)
        
        filename_section = self._setup_filename_section()
        quality_section = self._setup_quality_section()
        
        row2_layout.addLayout(filename_section, 1)
        row2_layout.addLayout(quality_section, 1)
        
        layout.addLayout(row2_layout)
        
        return frame
    
    def _setup_path_section(self) -> QVBoxLayout:
        """Create the save path selection section."""
        layout = QVBoxLayout()
        
        # Label
        label = QLabel("저장경로 선택")
        label.setFont(QFont("맑은 고딕", 12, QFont.Bold))
        label.setStyleSheet(f"color: {TEXT_COLOR};")
        
        # Input row
        input_layout = QHBoxLayout()
        
        # Path input field
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("/path/output.mp4")
        self.path_input.setFixedHeight(INPUT_HEIGHT)
        self.path_input.setStyleSheet(self._get_input_style())
        
        # Folder button
        folder_button = QPushButton("📁")
        folder_button.setFixedSize(BUTTON_HEIGHT, BUTTON_HEIGHT)
        folder_button.setCursor(Qt.PointingHandCursor)
        folder_button.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e5e7eb;
            }
        """)
        folder_button.clicked.connect(self._select_folder)
        
        input_layout.addWidget(self.path_input)
        input_layout.addWidget(folder_button)
        
        layout.addWidget(label)
        layout.addLayout(input_layout)
        
        return layout
    
    def _setup_filename_section(self) -> QVBoxLayout:
        """Create the filename input section."""
        layout = QVBoxLayout()
        
        # Label
        label = QLabel("파일이름")
        label.setFont(QFont("맑은 고딕", 12, QFont.Bold))
        label.setStyleSheet(f"color: {TEXT_COLOR};")
        
        # Input field
        self.filename_input = QLineEdit()
        self.filename_input.setPlaceholderText("손흥민골모음")
        self.filename_input.setFixedHeight(INPUT_HEIGHT)
        self.filename_input.setStyleSheet(self._get_input_style())
        
        layout.addWidget(label)
        layout.addWidget(self.filename_input)
        
        return layout
    
    def _setup_quality_section(self) -> QVBoxLayout:
        """Create the quality selection section."""
        layout = QVBoxLayout()
        
        # Label
        label = QLabel("화질 선택")
        label.setFont(QFont("맑은 고딕", 12, QFont.Bold))
        label.setStyleSheet(f"color: {TEXT_COLOR};")
        
        # Quality dropdown
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["720 p", "1080 p", "1440 p", "2160 p (4K)"])
        self.quality_combo.setCurrentText("1080 p")
        self.quality_combo.setFixedSize(150, INPUT_HEIGHT)
        self.quality_combo.setCursor(Qt.PointingHandCursor)
        self.quality_combo.setStyleSheet(f"""
            QComboBox {{
                border: 1px solid {INPUT_BORDER_COLOR};
                border-radius: 8px;
                padding: 0 10px;
                font-size: 11px;
                color: #6b7280;
                background-color: white;
            }}
            QComboBox:focus {{
                border: 2px solid {PRIMARY_COLOR};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #6b7280;
                margin-right: 5px;
            }}
        """)
        
        layout.addWidget(label)
        layout.addWidget(self.quality_combo)
        
        return layout
    
    def _setup_export_button(self) -> QPushButton:
        """Create the export button."""
        button = QPushButton("출력하기")
        button.setMinimumSize(*EXPORT_BUTTON_SIZE)
        button.setMaximumSize(200, 50)
        button.setCursor(Qt.PointingHandCursor)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: white;
                border: 1px solid #DDDDDD;
                border-radius: 10px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #6A5AAE;
            }}
            QPushButton:pressed {{
                background-color: #5A4A9E;
            }}
        """)
        button.clicked.connect(self._on_export_clicked)
        
        return button
    
    def _get_input_style(self) -> str:
        """Get common input field stylesheet."""
        return f"""
            QLineEdit {{
                border: 1px solid {INPUT_BORDER_COLOR};
                border-radius: 8px;
                padding: 0 10px;
                font-size: 11px;
                color: #6b7280;
                background-color: white;
            }}
            QLineEdit:focus {{
                border: 2px solid {PRIMARY_COLOR};
            }}
            QLineEdit::placeholder {{
                color: #9ca3af;
            }}
        """
    
    @Slot()
    def _select_folder(self):
        """Open folder selection dialog."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "저장 경로 선택",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder:
            self.path_input.setText(folder)
    
    @Slot()
    def _on_export_clicked(self):
        """Handle export button click with validation."""
        save_path = self.path_input.text()
        filename = self.filename_input.text()
        quality = self.quality_combo.currentText()
        
        # Validate inputs
        if not save_path:
            QMessageBox.warning(
                self,
                "경로 선택 필요",
                "저장 경로를 선택해주세요."
            )
            return
        
        if not filename:
            QMessageBox.warning(
                self,
                "파일명 입력 필요",
                "파일 이름을 입력해주세요."
            )
            return
        
        # Log settings
        print(f"출력 설정:")
        print(f"저장경로: {save_path}")
        print(f"파일이름: {filename}")
        print(f"화질: {quality}")
        
        # Emit export signal
        self.export_requested.emit(save_path, filename, quality)
    
    @Slot(str, str, str)
    def set_output_settings(
        self, 
        save_path: Optional[str] = None, 
        filename: Optional[str] = None, 
        quality: Optional[str] = None
    ):
        """
        Set output settings programmatically.
        
        Args:
            save_path: Default save path
            filename: Default filename
            quality: Default quality (e.g., "1080 p")
        """
        if save_path:
            self.path_input.setText(save_path)
        if filename:
            self.filename_input.setText(filename)
        if quality and quality in ["720 p", "1080 p", "1440 p", "2160 p (4K)"]:
            self.quality_combo.setCurrentText(quality)
    
    def get_output_settings(self) -> Dict[str, str]:
        """
        Get current output settings.
        
        Returns:
            Dictionary with save_path, filename, and quality
        """
        return {
            'save_path': self.path_input.text(),
            'filename': self.filename_input.text(),
            'quality': self.quality_combo.currentText()
        }

    @Slot()
    def on_back_button_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()
