"""Main page for file selection and editing option selection."""

import os
from typing import List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QFileDialog, QFrame, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont, QPainter, QPen, QColor, QDragEnterEvent, QDropEvent


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

# 높이 및 크기 설정
INPUT_HEIGHT_MIN = 45 
INPUT_HEIGHT_MAX = 45
BUTTON_HEIGHT = 45
EDIT_BUTTON_SIZE = (160, 45) 

# Editing options
EDITING_OPTIONS = ["⚽ 골 모음 영상", "⚡ 경기 주요 영상", "🎵 밈 영상"]


class DashedFrame(QFrame):
    """Custom QFrame with dashed border."""
    
    # Signal: emitted when a file is dropped with its path
    file_dropped = Signal(str)

    def __init__(self, parent: QWidget = None) -> None:
        """Initialize the dashed frame."""
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._is_dragging = False
        
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._is_dragging = True
            self.update()
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event) -> None:
        """Handle drag leave event."""
        self._is_dragging = False
        self.update()
            
    def dropEvent(self, event: QDropEvent) -> None:
        """Handle file drop event."""
        self._is_dragging = False
        self.update()
        
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
        
        for file_path in files:
            if file_path.lower().endswith(video_extensions):
                self.file_dropped.emit(file_path)
                event.acceptProposedAction()
                return
        
        event.ignore()
        
    def paintEvent(self, event) -> None:
        """Paint the dashed border."""
        if self._is_dragging:
            painter_bg = QPainter(self)
            painter_bg.setRenderHint(QPainter.Antialiasing)
            painter_bg.fillRect(self.rect(), QColor("#F0F0FF"))
        
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen = QPen(QColor(DASHED_LINE_COLOR))
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        # [수정] 점선 패턴 통일 [4, 4]
        pen.setDashPattern([4, 4])
        painter.setPen(pen)
        
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 10, 10)


class MainPage(QWidget):
    """Main page widget for video file and option selection."""
    
    edit_requested = Signal(str, str)
    
    def __init__(self) -> None:
        super().__init__()
        self.option_buttons: List[QPushButton] = []
        self.selected_option_index: int = 0
        self._setup_ui()
        
    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        header = self._create_header()
        main_layout.addWidget(header)
        
        content = self._create_content()
        main_layout.addWidget(content)
        
    def _create_header(self) -> QWidget:
        """Create the header bar with unified title."""
        header = QWidget()
        header.setMinimumHeight(HEADER_HEIGHT_MIN)
        header.setMaximumHeight(HEADER_HEIGHT_MAX)
        header.setStyleSheet(f"background-color: {HEADER_BG_COLOR};")
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(15, 0, 15, 0)
        
        # [수정] Arial 제거 -> 기본 폰트 사용 (디자인 통일)
        # 아이콘과 제목
        title_label = QLabel("🎬ShortsGenie")
        title_font = QFont()
        title_font.setPointSize(20) # 20포인트
        title_font.setBold(True)    # 굵게
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: white; border: none; background-color: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        
        # 메인 페이지는 뒤로가기 버튼이 없으므로, 양쪽 Stretch만으로 완벽한 중앙 정렬이 됩니다.
        layout.addStretch(1)      # 왼쪽 여백
        layout.addWidget(title_label) # 가운데 제목
        layout.addStretch(1)      # 오른쪽 여백
        
        return header
    
    def _create_content(self) -> QWidget:
        content = QWidget()
        content.setStyleSheet(f"background-color: {CONTENT_BG_COLOR};")
        
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 20, 30, 20)
        
        outer_frame = self._create_outer_frame()
        
        frame_container = QHBoxLayout()
        frame_container.addWidget(outer_frame)
        
        layout.addLayout(frame_container)
        
        return content
    
    def _create_outer_frame(self) -> QFrame:
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
        
        dashed_frame = self._create_dashed_frame()
        outer_layout.addWidget(dashed_frame)
        outer_layout.addSpacing(20)
        
        edit_button = self._create_edit_button()
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(edit_button)
        button_layout.addStretch()
        
        outer_layout.addLayout(button_layout)
        
        return outer_frame
    
    def _create_dashed_frame(self) -> DashedFrame:
        frame = DashedFrame()
        frame.setStyleSheet(f"""
            DashedFrame {{
                background-color: {DASHED_FRAME_BG_COLOR};
                border: none;
                border-radius: 10px;
            }}
        """)

        frame.file_dropped.connect(self.on_file_dropped)
        
        inner_layout = QVBoxLayout(frame)
        inner_layout.setContentsMargins(20, 20, 20, 20)
        inner_layout.setSpacing(20)
        
        inner_layout.addStretch(1) 
        self._add_file_input_section(inner_layout)
        inner_layout.addStretch(1) 
        
        return frame
    
    def _add_file_input_section(self, layout: QVBoxLayout) -> None:
        group_widget = QWidget()
        group_layout = QVBoxLayout(group_widget)
        group_layout.setContentsMargins(0, 0, 0, 0)
        
        # 글자("편집할 영상...")와 입력창 사이 간격
        group_layout.setSpacing(40) 
        
        # [수정] Arial 제거
        file_label = QLabel("편집할 영상을 넣어주세요.")
        file_label_font = QFont()
        file_label_font.setPointSize(14)
        file_label_font.setBold(True)
        file_label.setFont(file_label_font)
        file_label.setAlignment(Qt.AlignCenter)
        file_label.setStyleSheet("border: none; background-color: transparent; color: #333;")
        
        group_layout.addWidget(file_label)
        
        # File input row
        file_input_layout = QHBoxLayout()
        file_input_layout.setSpacing(5) 
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("/path/input.mp4")
        self.file_path_edit.setMinimumHeight(INPUT_HEIGHT_MIN)
        self.file_path_edit.setMaximumHeight(INPUT_HEIGHT_MAX)
        self.file_path_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #DDDDDD;
                border-radius: 8px;
                padding: 5px 15px;
                background-color: #FAFAFA;
                color: #333;
                font-size: 14px; 
            }
        """)
        
        browse_btn = QPushButton("찾아보기")
        browse_btn.setMinimumSize(100, BUTTON_HEIGHT)
        browse_btn.setMaximumWidth(120)
        browse_btn.setCursor(Qt.PointingHandCursor)
        # [수정] 폰트 스타일 통일
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                border: none;
                border-radius: 8px;
                padding: 5px;
                font-weight: bold;
                font-size: 13px;
                color: #333;
            }
            QPushButton:hover {
                background-color: #D0D0D0;
            }
        """)
        browse_btn.clicked.connect(self.browse_file)
        
        file_input_layout.addWidget(self.file_path_edit)
        file_input_layout.addWidget(browse_btn)
        
        group_layout.addLayout(file_input_layout)
        
        layout.addWidget(group_widget)
    
    def _create_edit_button(self) -> QPushButton:
        edit_button = QPushButton("영상 편집하기")
        edit_button.setMinimumSize(*EDIT_BUTTON_SIZE)
        edit_button.setMaximumSize(200, 50)
        edit_button.setCursor(Qt.PointingHandCursor)
        
        # [수정] 폰트 스타일 통일
        edit_button.setStyleSheet(f"""
            QPushButton {{
                background-color: white;
                border: 1px solid #DDDDDD;
                border-radius: 10px;
                font-size: 15px;
                font-weight: bold;
                color: #333;
            }}
            QPushButton:hover {{
                background-color: #7B68BE;
                color: white;
                border: none;
            }}
        """)
        edit_button.clicked.connect(self.start_editing)
        return edit_button
    
    @Slot()
    def browse_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "영상 파일 선택",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)
    
    @Slot(str)
    def on_file_dropped(self, file_path: str) -> None:
        self.file_path_edit.setText(file_path)

    @Slot()
    def start_editing(self) -> None:
        file_path = self.file_path_edit.text().strip()
        
        if not file_path:
            QMessageBox.warning(self, "경고", "영상 파일을 선택해주세요.")
            return
        
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "경고", "파일이 존재하지 않습니다.")
            return
        
        if not os.access(file_path, os.R_OK):
            QMessageBox.critical(self, "권한 오류", "파일 읽기 권한이 없습니다.")
            return
        
        if not file_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            QMessageBox.warning(self, "경고", "지원하지 않는 영상 형식입니다.")
            return
        
        selected_option = EDITING_OPTIONS[0] 
        self.edit_requested.emit(file_path, selected_option)