"""Video player widget (stub implementation)."""

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout  # type: ignore[import-not-found]
from PySide6.QtCore import Qt  # type: ignore[import-not-found]


class VideoPlayerWidget(QWidget):
    """Placeholder video player widget."""

    def __init__(self):
        """Initialize video player widget."""
        super().__init__()
        layout = QVBoxLayout()

        label = QLabel("Video Preview (Not Implemented)")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: gray; font-style: italic;")

        layout.addWidget(label)
        self.setLayout(layout)
        self.setMinimumHeight(300)
