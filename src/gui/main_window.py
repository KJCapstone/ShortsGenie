"""Main window with page navigation."""

from typing import Dict
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QMessageBox
from PySide6.QtCore import Slot
from .main_page import MainPage
from .progress_page import ProgressPage
from .highlight_selector import SelectPage
from .preview_page import VideoPreviewPage
from .output_page import OutputSettingsPage

# Constrants
WINDOW_TITLE = "Shorts Genie"
WINDOW_MIN_WIDTH = 800
WINDOW_MIN_HEIGHT = 600
WINDOW_INIT_WIDTH = 800
WINDOW_INIT_HEIGHT = 600

class MainWindow(QMainWindow):
    """Main application window managing page navigation."""
    OPTION_MAP = {

        "⚽ 골 모음 영상": "골",
        "⚡ 경기 주요 영상": "경기",
        "🎵 밈 영상": "밈"     

    }
    
    def __init__(self):
        """Initialize the main window and set up UI components."""
        super().__init__()
        self.selected_video_info = None # 선택된 영상 정보 저장

        # Cache for back navigation
        self.cached_highlights = None
        self.cached_file_path = None
        self.cached_option = None

        self._setup_ui()
        
    def _setup_ui(self):
        """Initialize the user interface."""

        # Configure window properties
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(WINDOW_INIT_WIDTH, WINDOW_INIT_HEIGHT)
        
        # Create central stacked widget for page management
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Initialize pages
        self.main_page = MainPage()
        self.progress_page = ProgressPage()
        self.select_page = SelectPage()
        self.preview_page = VideoPreviewPage()
        self.output_page = OutputSettingsPage()
        
        # Add pages to stack (index order matters - first is default)
        self.stacked_widget.addWidget(self.main_page)     # index 0
        self.stacked_widget.addWidget(self.progress_page)    # index 1
        self.stacked_widget.addWidget(self.select_page)   # index 2
        self.stacked_widget.addWidget(self.preview_page)  # index 3
        self.stacked_widget.addWidget(self.output_page)   # index 4

        # Connect signals to slots
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Connect signals from child pages to appropriate slots."""
        # Forward navigation
        self.main_page.edit_requested.connect(self.show_progress_page)
        self.progress_page.processing_completed.connect(self.show_select_page)
        self.select_page.video_preview_requested.connect(self.show_preview_page)
        self.preview_page.output_settings_requested.connect(self.show_output_page)

        # Back navigation
        self.progress_page.back_requested.connect(self.handle_progress_page_back)
        self.select_page.back_requested.connect(self.handle_select_page_back)
        self.preview_page.back_requested.connect(self.handle_preview_page_back)
        self.output_page.back_requested.connect(self.handle_output_page_back)

    @Slot(str, str)
    def show_progress_page(self, file_path: str, option: str) -> None:
        """
        Switch to editing page and pass data to it.

        Args:
            file_path: Path to the selected video file
            option: Selected editing option/condition
        """
        # Cache for potential back navigation
        self.cached_file_path = file_path
        self.cached_option = option

        selected_option = self.OPTION_MAP.get(option)

        self.progress_page.set_data(file_path, selected_option)
        self.stacked_widget.setCurrentIndex(1)

    @Slot(list)
    def show_select_page(self, highlights) -> None:
        """Show the select page."""
        # Cache highlights for potential back navigation
        self.cached_highlights = highlights

        self.select_page.set_highlights(highlights)
        self.stacked_widget.setCurrentIndex(2)

    @Slot(list, int)
    def show_preview_page(self, highlights, selected_index):
        """
        Show preview page with highlights.
        
        Args:
            highlights: List of highlight dictionaries
            selected_index: Index of initially selected video
        """
        self.preview_page.load_highlights(highlights, selected_index)
        self.stacked_widget.setCurrentIndex(3)

    @Slot(dict)
    def show_output_page(self, video_info: Dict) -> None:
        """
        Show output page with selected video info.
        
        Args:
            video_info: Dictionary containing selected video information
        """
        # 선택된 영상 정보 저장
        self.selected_video_info = video_info
        self.output_page.video_info = video_info

        # 페이지 전환
        self.stacked_widget.setCurrentIndex(4)

    # Back navigation handlers

    @Slot()
    def handle_progress_page_back(self) -> None:
        """Handle back navigation from progress page."""
        # Cancel processing if active
        if hasattr(self.progress_page, 'worker') and self.progress_page.worker:
            if self.progress_page.worker.isRunning():
                # Show confirmation dialog
                reply = QMessageBox.question(
                    self,
                    "처리 취소",
                    "영상 분석이 진행 중입니다.\n취소하고 돌아가시겠습니까?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

                # Cancel worker
                self.progress_page.worker.cancel()

        # Clear cached data
        self.cached_file_path = None
        self.cached_option = None
        self.cached_highlights = None

        # Navigate back
        self.stacked_widget.setCurrentIndex(0)

    @Slot()
    def handle_select_page_back(self) -> None:
        """Handle back navigation from select page."""
        # User wants to select different video - clear cache
        self.cached_highlights = None
        self.cached_file_path = None
        self.cached_option = None

        # Navigate back to main page
        self.stacked_widget.setCurrentIndex(0)

    @Slot()
    def handle_preview_page_back(self) -> None:
        """Handle back navigation from preview page."""
        # Stop video playback before navigating
        if hasattr(self.preview_page, 'media_player'):
            self.preview_page.media_player.stop()

        # Stop preview videos in left panel
        if hasattr(self.preview_page, 'video_items'):
            for item in self.preview_page.video_items:
                if hasattr(item, 'media_player'):
                    item.media_player.stop()

        # Restore select page with cached highlights
        if self.cached_highlights:
            self.select_page.set_highlights(self.cached_highlights)

        # Navigate back to select page
        self.stacked_widget.setCurrentIndex(2)

    @Slot()
    def handle_output_page_back(self) -> None:
        """Handle back navigation from output page."""
        # Navigate back to preview page
        # Video info is already loaded in preview page
        self.stacked_widget.setCurrentIndex(3)