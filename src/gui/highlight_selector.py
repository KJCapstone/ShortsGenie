"""Result page for displaying detected video highlights."""

from typing import List, Dict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QGridLayout, QSizePolicy, QPushButton
)
from PySide6.QtCore import Qt, Slot, Signal, QUrl
from PySide6.QtGui import QFont, QPainter, QPen, QColor
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

# Constants for styling and configuration
HEADER_HEIGHT_MIN = 60
HEADER_HEIGHT_MAX = 80
HEADER_BG_COLOR = "#7B68BE"
CONTENT_BG_COLOR = "#F4F2FB"
FRAME_MAX_WIDTH = 740
FRAME_BG_COLOR = "#F4F2FB"
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

    # Signal: emitted when card is clicked
    video_clicked = Signal(int, str)  # (index, video_path)
    
    def __init__(self, index: int, title: str, description: str, video_path: str = None, parent: QWidget = None) -> None:
        """Initialize a highlight card."""
        super().__init__(parent)
        self.index = index
        self.title = title
        self.description = description
        self.video_path = video_path
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup the highlight card UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Container for video/thumbnail with layout
        video_container = QWidget()
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        video_container_layout.setSpacing(0)
        
        # Dashed frame (video container)
        video_frame = DashedFrame()
        video_frame.setFixedSize(HIGHLIGHT_CARD_WIDTH, HIGHLIGHT_CARD_HEIGHT)
        video_frame.setStyleSheet(f"""
            DashedFrame {{
                background-color: {DASHED_FRAME_BG_COLOR};
                border: none;
                border-radius: 12px;
            }}
        """)
        
        # Video content
        video_content_layout = QVBoxLayout(video_frame)
        video_content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Video widget or placeholder
        if self.video_path:
            # Use video widget for video files
            self.video_widget = QVideoWidget()
            self.video_widget.setStyleSheet("border: none; background-color: transparent;")
            
            # Setup media player
            self.media_player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.audio_output.setMuted(True)  # Mute audio
            self.media_player.setAudioOutput(self.audio_output)
            self.media_player.setVideoOutput(self.video_widget)
            
            # Load video
            self.media_player.setSource(QUrl.fromLocalFile(self.video_path))
            
            # Pause at first frame (don't play)
            self.media_player.pause()
            
            video_content_layout.addWidget(self.video_widget)
        else:
            # Default video icon
            self.placeholder_label = QLabel()
            self.placeholder_label.setAlignment(Qt.AlignCenter)
            self.placeholder_label.setFont(QFont("Arial", 32))
            self.placeholder_label.setText("🎬")
            self.placeholder_label.setStyleSheet("border: none; background-color: transparent; color: #D0D0D0;")
            video_content_layout.addWidget(self.placeholder_label)

        # Add video frame to container layout
        video_container_layout.addWidget(video_frame, alignment=Qt.AlignCenter)
        
        # Title
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Arial", 9, QFont.Bold))
        title_label.setStyleSheet("color: #333333; border: none; background-color: transparent;")
        title_label.setWordWrap(True)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setMaximumHeight(32)
        
        # Description
        desc_label = QLabel(self.description)
        desc_label.setFont(QFont("Arial", 8))
        desc_label.setStyleSheet("color: #666666; border: none; background-color: transparent;")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setFixedHeight(36)

        # Add widgets to layout
        layout.addWidget(video_container)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addStretch()
    
    def mousePressEvent(self, event) -> None:
        """Handle mouse click event."""
        if event.button() == Qt.LeftButton and self.video_path:
            # Emit signal when card is clicked
            self.video_clicked.emit(self.index, self.video_path)
    
class SelectPage(QWidget):
    """Result page widget showing detected highlights in grid layout."""
    
    # Signal: emitted when video preview is requested
    video_preview_requested = Signal(list, int)  # (highlights, selected_index)
    back_requested = Signal()  # 뒤로 가기 요청 시 emit

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
        layout.setContentsMargins(10, 0, 10, 0) # 좌우 여백 10으로 통일

        # [1] 왼쪽: 뒤로 버튼
        self.back_button = QPushButton("← 뒤로")
        self.back_button.setFixedHeight(35)
        
        # 버튼 스타일 정의 (재사용을 위해 변수로)
        button_style = """
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14pt;
                font-family: pretendard;
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

        # [3] 가운데: 아이콘 + 제목 (중앙 정렬을 위해 컨테이너 사용 안 함)
        # 제목
        title_label = QLabel("🎬ShortsGenie")
        title_label.setFont(QFont("pretendard", 20, QFont.Bold))
        title_label.setStyleSheet("color: white; border: none; background-color: transparent;")

        layout.addWidget(title_label)
        
        # [4] 오른쪽 스페이서
        layout.addStretch()

        # [5] 오른쪽: 투명한 더미 버튼 (제목 중앙 정렬을 위한 핵심!)
        # 왼쪽 버튼과 완전히 동일한 크기와 스타일을 가지지만, 내용은 없고 클릭 불가능하게 만듭니다.
        dummy_button = QPushButton("← 뒤로") 
        dummy_button.setFixedHeight(35)
        dummy_button.setStyleSheet(button_style) # 스타일 복사
        dummy_button.setFlat(True) 
        dummy_button.setEnabled(False) # 클릭 불가
        # 글자색을 투명하게 해서 안 보이게 만듭니다.
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

        # Prevent unnecessary vertical expansion
        self.grid_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        
        # Center grid_container within outer_frame
        outer_layout.addStretch()
        outer_layout.addWidget(self.grid_container)
        outer_layout.addStretch()

        return outer_frame
    
    @Slot(list)
    def set_highlights(self, highlights: List[Dict]) -> None:
        """Set and display the highlights."""
        self.highlights = highlights
        self.highlight_cards = []
        
        # Clear existing cards
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add only the first highlight card (centered)
        if highlights:
            highlight = highlights[0]  # 첫 번째 하이라이트만 사용
        
            card = HighlightCard(
                index=0,
                title='하이라이트',
                description=highlight.get('description', '하이라이트 설명'),
                video_path=highlight.get('video_path', None)
            )

            # Connect card click signal to result page signal
            card.video_clicked.connect(self._on_card_clicked)

            # Add card to center (row 0, column 1 with column span)
            # Use column stretch to center the card
            self.grid_layout.addWidget(card, 0, 1, Qt.AlignCenter)
            self.grid_layout.setColumnStretch(0, 1)  # Left stretch
            self.grid_layout.setColumnStretch(1, 0)  # Center (no stretch)
            self.grid_layout.setColumnStretch(2, 1)  # Right stretch

            self.highlight_cards.append(card)

        # Log for debugging
        print(f"\n{'=' * 60}")
        print(f"Result Page: Displaying {min(len(highlights), 3)} highlights")
        print(f"{'=' * 60}\n")
    
    @Slot(int, str)
    def _on_card_clicked(self, index: int, video_path: str) -> None:
        """Handle card click event - go directly to preview page."""
        print(f"\n{'=' * 60}")
        print(f"카드 클릭됨 - 미리보기로 이동")
        print(f"Index: {index}")
        print(f"Video Path: {video_path}")
        print(f"{'=' * 60}\n")

        # Emit signal to navigate to preview page directly
        self.video_preview_requested.emit(self.highlights, index)

    @Slot()
    def on_back_button_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()