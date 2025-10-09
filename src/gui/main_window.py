"""Main window for the application."""

from PySide6.QtWidgets import (  # type: ignore[import-not-found]
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
)
from PySide6.QtCore import Qt  # type: ignore[import-not-found]
from PySide6.QtGui import QFont  # type: ignore[import-not-found]


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        """Initialize main window."""
        super().__init__()
        self.setWindowTitle("SP-AI-LIGHTS - Sports Highlights Auto Editor")
        self.setMinimumSize(800, 600)

        # Setup UI
        self._setup_ui()
        self.statusBar().showMessage("Ready")

    def _setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Title
        title = QLabel("SP-AI-LIGHTS")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Sports AI Highlights - Auto Editor")
        subtitle.setStyleSheet("color: gray; font-size: 14pt;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Add stretch to center content
        layout.addStretch()
