"""Dialog for showing detailed information about merged highlights."""

from typing import List, Dict
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QWidget, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont


class SceneCard(QFrame):
    """Card widget showing individual scene information."""

    def __init__(self, scene_num: int, title: str, description: str, start_time, end_time, parent=None):
        """Initialize scene card.

        Args:
            scene_num: Scene number (1-indexed for display)
            title: Scene title
            description: Scene description
            start_time: Start time in seconds
            end_time: End time in seconds
        """
        super().__init__(parent)
        self.scene_num = scene_num
        self.title = title
        self.description = description
        self.start_time = start_time
        self.end_time = end_time
        self._setup_ui()

    def _setup_ui(self):
        """Setup the card UI."""
        self.setStyleSheet("""
            SceneCard {
                background-color: white;
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                padding: 12px;
            }
            SceneCard:hover {
                border-color: #7B68BE;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Scene number and time range
        header_layout = QHBoxLayout()

        scene_label = QLabel(f"장면 {self.scene_num}")
        scene_label.setFont(QFont("Arial", 11, QFont.Bold))
        scene_label.setStyleSheet("color: #7B68BE;")
        header_layout.addWidget(scene_label)

        header_layout.addStretch()

        time_label = QLabel(self._format_time_range(self.start_time, self.end_time))
        time_label.setFont(QFont("Arial", 9))
        time_label.setStyleSheet("color: #666666;")
        header_layout.addWidget(time_label)

        layout.addLayout(header_layout)

        # Title
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        title_label.setStyleSheet("color: #333333;")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(self.description)
        desc_label.setFont(QFont("Arial", 9))
        desc_label.setStyleSheet("color: #666666;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

    def _format_time_range(self, start, end) -> str:
        """Format time range as MM:SS - MM:SS."""
        # Convert to float if string
        try:
            start = float(start) if start else 0.0
            end = float(end) if end else 0.0
        except (ValueError, TypeError):
            return "00:00 - 00:00"

        start_min = int(start // 60)
        start_sec = int(start % 60)
        end_min = int(end // 60)
        end_sec = int(end % 60)
        return f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"


class HighlightDetailDialog(QDialog):
    """Dialog showing all scenes included in the merged highlight video."""

    # Signal emitted when user confirms and wants to proceed to preview
    preview_requested = Signal()

    def __init__(self, highlights: List[Dict], parent=None):
        """Initialize the dialog.

        Args:
            highlights: List of all highlight dictionaries (including merged ones)
        """
        super().__init__(parent)
        self.highlights = highlights
        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("포함된 장면들")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        self.resize(650, 600)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Header
        header_label = QLabel("🎬 하이라이트 영상에 포함된 장면들")
        header_font = QFont("Arial", 16, QFont.Bold)
        header_label.setFont(header_font)
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("color: #7B68BE; padding: 10px;")
        layout.addWidget(header_label)

        # Subtitle
        subtitle_label = QLabel(f"총 {len(self.highlights)}개의 장면이 병합되었습니다")
        subtitle_label.setFont(QFont("Arial", 11))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #666666; padding-bottom: 10px;")
        layout.addWidget(subtitle_label)

        # Scroll area for scene cards
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #F4F2FB;
            }
        """)

        # Container for cards
        cards_container = QWidget()
        cards_layout = QVBoxLayout(cards_container)
        cards_layout.setContentsMargins(10, 10, 10, 10)
        cards_layout.setSpacing(12)

        # Add scene cards
        for idx, highlight in enumerate(self.highlights, start=1):
            card = SceneCard(
                scene_num=idx,
                title=highlight.get('title', f'장면 #{idx}'),
                description=highlight.get('description', '설명 없음'),
                start_time=highlight.get('start_time', 0.0),
                end_time=highlight.get('end_time', 0.0)
            )
            cards_layout.addWidget(card)

        cards_layout.addStretch()
        scroll_area.setWidget(cards_container)
        layout.addWidget(scroll_area)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # Preview button
        preview_button = QPushButton("미리보기로 이동")
        preview_button.setFont(QFont("Arial", 11, QFont.Bold))
        preview_button.setStyleSheet("""
            QPushButton {
                background-color: #7B68BE;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 32px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #6A5AAD;
            }
            QPushButton:pressed {
                background-color: #594A9C;
            }
        """)
        preview_button.clicked.connect(self._on_preview_clicked)
        button_layout.addWidget(preview_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

    def _on_preview_clicked(self):
        """Handle preview button click."""
        self.preview_requested.emit()
        self.accept()
