"""Video preview page for playing highlight videos."""

from typing import List, Dict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QSlider, QStyle, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot, QUrl
from PySide6.QtGui import QFont, QPainter, QPen, QColor
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget


# Constants for styling
HEADER_HEIGHT_MIN = 60
HEADER_HEIGHT_MAX = 80
HEADER_BG_COLOR = "#7B68BE"
CONTENT_BG_COLOR = "#F4F2FB"
FRAME_BG_COLOR = "#F4F2FB"
FRAME_BORDER_COLOR = "#CCCCCC"
DASHED_LINE_COLOR = "#CCCCCC"


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


class VideoListItem(QWidget):
    """Widget for a single video item in the list."""
    
    # Signal emitted when item is clicked
    clicked = Signal(int)
    
    def __init__(self, index: int, title: str, video_path: str = None, is_selected: bool = False, parent: QWidget = None) -> None:
        """Initialize video list item."""
        super().__init__(parent)
        self.index = index
        self.title = title
        self.video_path = video_path
        self.is_selected = is_selected
        self._setup_ui()
        
    def _setup_ui(self) -> None:
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Video widget or placeholder
        if self.video_path:
            # Video exists - show video only (no frame)
            self.video_widget = QVideoWidget()
            # 고정 크기 제거 - 비율을 유지하며 유연하게 크기 조정
            self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.video_widget.setStyleSheet(f"""
                QVideoWidget {{
                    background-color: {'#EDE7F6' if self.is_selected else '#000000'};
                    border: none;
                    border-radius: 12px;
                }}
            """)
            self.video_widget.setCursor(Qt.PointingHandCursor)
            
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
            
            layout.addWidget(self.video_widget)
        else:
            # No video - show dashed frame with icon
            self.frame = DashedFrame()
            # 고정 크기 제거 - 비율을 유지하며 유연하게 크기 조정
            self.frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.frame.setStyleSheet(f"""
                DashedFrame {{
                    background-color: {'#EDE7F6' if self.is_selected else '#F9F8FC'};
                    border: none;
                    border-radius: 12px;
                }}
            """)
            self.frame.setCursor(Qt.PointingHandCursor)
            
            # Frame content
            frame_layout = QVBoxLayout(self.frame)
            frame_layout.setContentsMargins(5, 5, 5, 5)
            
            # Default video icon placeholder
            self.placeholder_label = QLabel()
            self.placeholder_label.setAlignment(Qt.AlignCenter)
            self.placeholder_label.setFont(QFont("Arial", 24))
            self.placeholder_label.setText("🎬")
            self.placeholder_label.setStyleSheet("border: none; background-color: transparent; color: #D0D0D0;")
            
            frame_layout.addWidget(self.placeholder_label)
            
            layout.addWidget(self.frame)
        
        # Title label (below the video/frame)
        self.title_label = QLabel(self.title)
        self.title_label.setFont(QFont("Arial", 12, QFont.Bold if self.is_selected else QFont.Normal))  # 9 → 12
        self.title_label.setStyleSheet("color: #333333; border: none; background-color: transparent;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setMaximumHeight(30)
        
        layout.addWidget(self.title_label)
    
    def mousePressEvent(self, event) -> None:
        """Handle mouse click."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)
    
    def set_selected(self, selected: bool) -> None:
        """Set selection state."""
        self.is_selected = selected
        
        # 타이틀 레이블 업데이트
        self.title_label.setFont(QFont("Arial", 12, QFont.Bold if selected else QFont.Normal))  # 9 → 12
        
        if self.video_path:
            # Video exists - update video widget style
            self.video_widget.setStyleSheet(f"""
                QVideoWidget {{
                    background-color: {'#EDE7F6' if selected else '#000000'};
                    border: none;
                    border-radius: 12px;
                }}
            """)
        else:
            # No video - update dashed frame style
            self.frame.setStyleSheet(f"""
                DashedFrame {{
                    background-color: {'#EDE7F6' if selected else '#F9F8FC'};
                    border: none;
                    border-radius: 12px;
                }}
            """)


class VideoPreviewPage(QWidget):
    """
    Video preview page with list and player layout.
    
    Layout:
    - Left: Video list (골 모음 영상)
    - Right: Video player with controls
    
    Signals:
        output_settings_requested: Emitted when user clicks edit button
    """
    
    # Signal
    output_settings_requested = Signal(dict) # 출력하기 버튼 클릭 시 발생
    back_requested = Signal()  # 뒤로 가기 요청 시 emit

    def __init__(self) -> None:
        """Initialize the video preview page."""
        super().__init__()
        self.highlights: List[Dict] = []
        self.current_index = 0
        self.video_items: List[VideoListItem] = []
        self._setup_ui()
        self._setup_media_player()
        
    def _setup_ui(self) -> None:
        """Setup the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create header
        header = self._create_header()
        main_layout.addWidget(header)
        
        # Create content area
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
        """Create the content area."""
        content = QWidget()
        content.setStyleSheet(f"background-color: {CONTENT_BG_COLOR};")
        
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 20, 30, 20)
        
        # Main container frame
        container_frame = QFrame()
        container_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {FRAME_BG_COLOR};
                border: 2px solid {FRAME_BORDER_COLOR};
                border-radius: 20px;
            }}
        """)
        
        container_layout = QHBoxLayout(container_frame)
        container_layout.setContentsMargins(20, 20, 20, 20)
        container_layout.setSpacing(20)
        
        # Left side: Video list
        #left_panel = self._create_left_panel()
        
        # Right side: Video player
        right_panel = self._create_right_panel()
        
        #container_layout.addWidget(left_panel, stretch=35)
        container_layout.addWidget(right_panel, stretch=100)
        
        layout.addWidget(container_frame)
        
        return content
    
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with video list."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel("⚽ 골 모음 영상")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))  # 12 → 16
        title_label.setStyleSheet("color: #333333; border: none; background-color: transparent;")
        
        # Container for video items (no scroll)
        self.video_list_container = QWidget()
        self.video_list_layout = QVBoxLayout(self.video_list_container)
        self.video_list_layout.setContentsMargins(0, 0, 0, 0)
        self.video_list_layout.setSpacing(15)
        
        layout.addWidget(title_label)
        layout.addWidget(self.video_list_container)
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with video player."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        
        # Video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(350)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_widget.setStyleSheet("""
            QVideoWidget {
                background-color: #E8E8E8;
                border: 2px solid #CCCCCC;
                border-radius: 12px;
            }
        """)
        
        # Controls
        controls = self._create_controls()
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.edit_button = QPushButton("✏️ 수정하기")
        self.edit_button.setMinimumHeight(40)
        self.edit_button.setStyleSheet("""
            QPushButton {
                background-color: #FFF9C4;
                color: #333333;
                border: none;
                border-radius: 8px;
                font-size: 14pt;
                font-weight: bold;
                padding: 0 20px;
            }
            QPushButton:hover {
                background-color: #FFF59D;
            }
        """)
        
        self.complete_button = QPushButton("✅ 출력하기")
        self.complete_button.setMinimumHeight(40)
        self.complete_button.setStyleSheet("""
            QPushButton {
                background-color: #C5E1A5;
                color: #333333;
                border: none;
                border-radius: 8px;
                font-size: 14pt;
                font-weight: bold;
                padding: 0 20px;
            }
            QPushButton:hover {
                background-color: #AED581;
            }
        """)

        # Connect button signals
        self.complete_button.clicked.connect(self._on_complete_button_clicked)
        
        button_layout.addStretch()
        #button_layout.addWidget(self.edit_button)
        button_layout.addWidget(self.complete_button)
        
        layout.addWidget(self.video_widget)
        layout.addWidget(controls)
        layout.addLayout(button_layout)
        
        return panel
    
    def _create_controls(self) -> QWidget:
        """Create playback controls."""
        controls = QWidget()
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Play button
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.setFixedSize(35, 35)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #7B68BE;
                border: none;
                border-radius: 17px;
            }
            QPushButton:hover {
                background-color: #6A57AD;
            }
        """)
        
        # Progress slider
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 6px;
                background: #E0E0E0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #7B68BE;
                border: 2px solid #7B68BE;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #7B68BE;
                border-radius: 3px;
            }
        """)

        # Time label (current time / total time)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFont(QFont("Arial", 12))  # 9 → 12
        self.time_label.setStyleSheet("color: #666666; border: none; background-color: transparent;")
        self.time_label.setMinimumWidth(80)
        self.time_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.play_button)
        layout.addWidget(self.progress_slider)
        layout.addWidget(self.time_label)
        
        return controls
    
    def _setup_media_player(self) -> None:
        """Setup the media player."""
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        
        # Connect signals
        self.play_button.clicked.connect(self._on_play_pause_clicked)
        self.media_player.positionChanged.connect(self._on_position_changed)
        self.media_player.durationChanged.connect(self._on_duration_changed)
        self.media_player.playbackStateChanged.connect(self._on_playback_state_changed)
        self.progress_slider.sliderMoved.connect(self._on_slider_moved)

    def _format_time(self, milliseconds: int) -> str:
        """
        Convert milliseconds to MM:SS format.
        
        Args:
            milliseconds: Time in milliseconds
            
        Returns:
            Formatted time string (MM:SS)
        """
        seconds = milliseconds // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    @Slot()
    def _on_play_pause_clicked(self) -> None:
        """Handle play/pause button click."""
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()
    
    @Slot(int)
    def _on_position_changed(self, position: int) -> None:
        """Update progress slider and time label."""
        self.progress_slider.setValue(position)

        # Update time label
        duration = self.media_player.duration()
        current_time = self._format_time(position)
        total_time = self._format_time(duration)
        self.time_label.setText(f"{current_time} / {total_time}")
    
    @Slot(int)
    def _on_duration_changed(self, duration: int) -> None:
        """Update slider range and time label."""
        self.progress_slider.setRange(0, duration)

        # Initialize time label
        total_time = self._format_time(duration)
        self.time_label.setText(f"00:00 / {total_time}")
    
    @Slot(int)
    def _on_slider_moved(self, position: int) -> None:
        """Seek to position."""
        self.media_player.setPosition(position)
    
    @Slot()
    def _on_playback_state_changed(self, state) -> None:
        """Update play button icon."""
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
    
    @Slot()
    def _on_complete_button_clicked(self) -> None:
        """Handle complete button click - emit selected highlight info."""
        if self.highlights and self.current_index < len(self.highlights):
            # 1. 현재 선택된 영상 정보 가져오기
            selected_highlight = self.highlights[self.current_index]
            # 2. 시그널 발생 - 출력 설정 페이지로 전환 요청
            self.output_settings_requested.emit(selected_highlight)
            
            print(f"\n{'=' * 60}")
            print(f"편집하기 버튼 클릭!")
            print(f"선택된 영상: {selected_highlight.get('title', 'N/A')}")
            print(f"영상 경로: {selected_highlight.get('video_path', 'N/A')}")
            print(f"{'=' * 60}\n")
        else:
            print("선택된 영상이 없습니다.")

    @Slot(int)
    def _on_video_item_clicked(self, index: int) -> None:
        """Handle video list item click."""
        self.current_index = index
        
        # Update selection
        for i, item in enumerate(self.video_items):
            item.set_selected(i == index)
        
        # Load and play selected video
        if index < len(self.highlights):
            video_path = self.highlights[index].get('video_path')
            if video_path:
                self.media_player.setSource(QUrl.fromLocalFile(video_path))
                self.media_player.play()
    
    @Slot(list, int)
    def load_highlights(self, highlights: List[Dict], selected_index: int = 0) -> None:
        """
        Load highlights and create video list.
        
        Args:
            highlights: List of highlight dictionaries
        """
        self.highlights = highlights
        self.current_index = selected_index # 선택된 인덱스로 설정
        
        # Clear existing items
        #self.video_items.clear()
        #while self.video_list_layout.count() > 1:  # Keep the stretch
        #    item = self.video_list_layout.takeAt(0)
        #    if item.widget():
        #        item.widget().deleteLater()
        
        # Create video items (only for highlights with valid video_path)
        #valid_index = 0
        #for i, highlight in enumerate(highlights):
        #    video_path = highlight.get('video_path', None)

            # Skip highlights without valid video files
        #    if not video_path:
        #        continue

        #    title = highlight.get('title', f'후보 {valid_index+1} 영상')

        #    item = VideoListItem(
        #        index=valid_index,
        #        title=title,
        #        video_path=video_path,
        #        is_selected=(valid_index == selected_index)
        #    )
        #    item.clicked.connect(self._on_video_item_clicked)

        #    self.video_list_layout.insertWidget(valid_index + 1, item)
        #    self.video_items.append(item)
        #    valid_index += 1
        
        # Load first video
        if highlights and selected_index < len(highlights):
            video_path = highlights[selected_index].get('video_path')
            if video_path:
                self.media_player.setSource(QUrl.fromLocalFile(video_path))
                self.media_player.play()

    def hideEvent(self, event) -> None:
        """Stop video playback when page is hidden."""
        # 오른쪽 패널의 영상 재생 중지
        if self.media_player:
            self.media_player.stop()

        super().hideEvent(event)

    @Slot()
    def on_back_button_clicked(self) -> None:
        """Handle back button click."""
        self.back_requested.emit()