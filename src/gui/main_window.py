"""Main window with page navigation."""

from typing import Dict
from PySide6.QtWidgets import QMainWindow, QStackedWidget
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
    
    def __init__(self):
        """Initialize the main window and set up UI components."""
        super().__init__()
        self.selected_video_info = None # 선택된 영상 정보 저장
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
        self.main_page.edit_requested.connect(self.show_progress_page)
        self.select_page.video_preview_requested.connect(self.show_preview_page)
        self.preview_page.output_settings_requested.connect(self.show_output_page)

    @Slot()   
    def show_main_page(self) -> None:
        """Switch back to main page."""
        self.stacked_widget.setCurrentIndex(0)

    @Slot(str, str)
    def show_progress_page(self, file_path: str, option: str) -> None:
        """
        Switch to editing page and pass data to it.
        
        Args:
            file_path: Path to the selected video file
            option: Selected editing option/condition
        """
        self.progress_page.set_data(file_path, option)
        self.stacked_widget.setCurrentIndex(1)

    @Slot()
    def show_select_page(self) -> None:
        """Show the select page."""
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
        # 선택된 영상 정보만 저장 (UI는 건드리지 않음 - 사용자가 직접 입력)
        self.selected_video_info = video_info
        self.output_page.video_info = video_info
        
        # 페이지 전환
        self.stacked_widget.setCurrentIndex(4)