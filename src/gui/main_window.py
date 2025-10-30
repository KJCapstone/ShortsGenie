"""Main window with page navigation."""

from PySide6.QtWidgets import QMainWindow, QStackedWidget
from PySide6.QtCore import Slot
from .main_page import MainPage
from .progress_page import ProcessPage

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
        self.processing = ProcessPage()
        
        # Add pages to stack (index order matters - first is default)
        self.stacked_widget.addWidget(self.main_page)     # index 0
        self.stacked_widget.addWidget(self.processing)    # index 1
        
        # Connect signals to slots
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Connect signals from child pages to appropriate slots."""
        self.main_page.edit_requested.connect(self.show_process_page)
        self.processing.back_requested.connect(self.show_main_page)

    @Slot(str, str)
    def show_process_page(self, file_path: str, option: str) -> None:
        """
        Switch to editing page and pass data to it.
        
        Args:
            file_path: Path to the selected video file
            option: Selected editing option/condition
        """
        self.processing.set_data(file_path, option)
        self.stacked_widget.setCurrentIndex(1)

    @Slot()   
    def show_main_page(self):
        """Switch back to main page."""
        self.stacked_widget.setCurrentIndex(0)