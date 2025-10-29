"""Main window with page navigation."""

from PySide6.QtWidgets import QMainWindow, QStackedWidget
from PySide6.QtCore import Slot
from .main_page import MainPage

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
        
        # Add pages to stack (index order matters - first is default)
        self.stacked_widget.addWidget(self.main_page)     # index 0
        
        # Connect signals to slots
        self.main_page.edit_requested.connect(self.show_main_page)

    @Slot()   
    def show_main_page(self):
        """Switch back to main page."""
        self.stacked_widget.setCurrentIndex(0)