"""Result page for displaying detected video highlights."""

from typing import List, Dict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QFrame, QGridLayout
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QPainter, QPen, QColor, QPixmap


# Constants for styling and configuration
HEADER_HEIGHT_MIN = 60
HEADER_HEIGHT_MAX = 80
HEADER_BG_COLOR = "#7B68BE"
CONTENT_BG_COLOR = "#F4F2FB"
FRAME_MAX_WIDTH = 740
FRAME_BG_COLOR = "#FFFFFF"
FRAME_BORDER_COLOR = "#CCCCCC"
DASHED_FRAME_BG_COLOR = "#F9F8FC"
DASHED_LINE_COLOR = "#CCCCCC"
HIGHLIGHT_CARD_WIDTH = 190
HIGHLIGHT_CARD_HEIGHT = 110


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
        pen.setDashPattern([1, 2])
        painter.setPen(pen)
        
        # Draw rounded rectangle with dash border
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 10, 10)


class HighlightCard(QWidget):
    """Widget representing a single highlight video card."""
    
    def __init__(self, index: int, title: str, description: str, thumbnail_path: str = None, parent: QWidget = None) -> None:
        """
        Initialize a highlight card.
        
        Args:
            index: Highlight number (0, 1, 2, ...)
            title: Highlight title
            description: Highlight description
            thumbnail_path: Path to thumbnail image (optional)
            parent: Parent widget
        """
        super().__init__(parent)
        self.index = index
        self.title = title
        self.description = description
        self.thumbnail_path = thumbnail_path
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup the highlight card UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Container for thumbnail
        thumb_container = QWidget()
        thumb_container_layout = QVBoxLayout(thumb_container)  # ← 레이아웃 추가!
        thumb_container_layout.setContentsMargins(0, 0, 0, 0)
        thumb_container_layout.setSpacing(0)
        
        # Dashed frame (video thumbnail placeholder)
        thumbnail_frame = DashedFrame()
        thumbnail_frame.setFixedSize(HIGHLIGHT_CARD_WIDTH, HIGHLIGHT_CARD_HEIGHT)
        thumbnail_frame.setStyleSheet(f"""
            DashedFrame {{
                background-color: {DASHED_FRAME_BG_COLOR};
                border: none;
                border-radius: 12px;
            }}
        """)
        
        # Thumbnail content
        thumb_content_layout = QVBoxLayout(thumbnail_frame)
        thumb_content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Thumbnail image label
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet("border: none; background-color: transparent;")
        
        # Load thumbnail if provided, otherwise show icon
        if self.thumbnail_path:
            self.set_thumbnail(self.thumbnail_path)
        else:
            # Default video icon
            self.thumbnail_label.setFont(QFont("Arial", 32))
            self.thumbnail_label.setText("🎬")
            self.thumbnail_label.setStyleSheet("border: none; background-color: transparent; color: #D0D0D0;")
        
        thumb_content_layout.addWidget(self.thumbnail_label)

        # Add thumbnail frame to container layout
        thumb_container_layout.addWidget(thumbnail_frame)
        
        # Title
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Arial", 9, QFont.Bold))
        title_label.setStyleSheet("color: #333333; border: none; background-color: transparent;")
        title_label.setWordWrap(True)
        title_label.setAlignment(Qt.AlignLeft)
        title_label.setMaximumHeight(32)
        
        # Description
        desc_label = QLabel(self.description)
        desc_label.setFont(QFont("Arial", 8))
        desc_label.setStyleSheet("color: #666666; border: none; background-color: transparent;")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignLeft)
        desc_label.setFixedHeight(36)

        # Add widgets to layout
        layout.addWidget(thumb_container)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addStretch()
    
    def set_thumbnail(self, thumbnail_path: str) -> None:
        """
        Set the thumbnail image from a file path.
        
        Args:
            thumbnail_path: Path to the thumbnail image file
        """
        pixmap = QPixmap(thumbnail_path)
        if not pixmap.isNull():
            # Scale pixmap to fit the frame while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                HIGHLIGHT_CARD_WIDTH - 4,
                HIGHLIGHT_CARD_HEIGHT - 4,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.thumbnail_label.setPixmap(scaled_pixmap)
            self.thumbnail_label.setStyleSheet("""
                border: none; 
                background-color: transparent;
                border-radius: 10px;
            """)
        else:
            # If image loading fails, show icon
            self.thumbnail_label.setFont(QFont("Arial", 32))
            self.thumbnail_label.setText("🎬")
            self.thumbnail_label.setStyleSheet("border: none; background-color: transparent; color: #D0D0D0;")


class ResultPage(QWidget):
    """
    Result page widget showing detected highlights in grid layout.
    
    This page displays the highlights detected from video analysis:
    - Grid layout of highlight cards (3 columns, 1 row)
    - Each card shows thumbnail, title, and description
    
    Attributes:
        highlights (list): List of highlight data dictionaries
        highlight_cards (list): List of HighlightCard widgets
    """
    
    def __init__(self) -> None:
        """Initialize the result page and set up UI components."""
        super().__init__()
        self.highlights: List[Dict] = []
        self.highlight_cards: List[HighlightCard] = []
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
        
        # Center the outer frame
        frame_container = QHBoxLayout()
        frame_container.addStretch()
        frame_container.addWidget(outer_frame)
        frame_container.addStretch()
        
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
                border-radius: 20px;
            }}
        """)
        
        outer_layout = QVBoxLayout(outer_frame)
        outer_layout.setContentsMargins(20, 18, 20, 18)
        outer_layout.setSpacing(12)

        # Container for grid
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(18)  # Spacing between cards
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        
        outer_layout.addWidget(self.grid_container)
        
        return outer_frame
    
    @Slot(list)
    def set_highlights(self, highlights: List[Dict]) -> None:
        """
        Set and display the highlights.
        
        Args:
            highlights: List of highlight dictionaries with keys:
                - title: Highlight title
                - description: Highlight description
                - start_time: Start time
                - end_time: End time
                - thumbnail_path: (optional) Path to thumbnail image
        """
        self.highlights = highlights
        self.highlight_cards = []
        
        # Clear existing cards
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add highlight cards in 3-column grid (1 row only)
        for i, highlight in enumerate(highlights):
            if i >= 3:  # Only show first 3 highlights
                break
                
            col = i  # Column 0, 1, 2
            
            card = HighlightCard(
                index=i,
                title=highlight.get('title', f'하이라이트 #{i+1}'),
                description=highlight.get('description', '하이라이트 설명'),
                thumbnail_path=highlight.get('thumbnail_path', None)
            )
            
            self.grid_layout.addWidget(card, 0, col)
            self.highlight_cards.append(card)
        
        # Log for debugging
        print(f"\n{'=' * 60}")
        print(f"Result Page: Displaying {min(len(highlights), 3)} highlights")
        print(f"{'=' * 60}\n")
    
    @Slot()
    def load_sample_data(self) -> None:
        """Load sample highlight data for testing."""
        sample_highlights = [
            {
                'title': '후보 1) 골 모음 영상',
                'description': '선수들의 골을 모아서 보여줌',
                'start_time': '00:30',
                'end_time': '01:15',
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
        
        self.set_highlights(sample_highlights)