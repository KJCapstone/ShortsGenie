"""Main entry point for SP-AI-LIGHTS application."""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import QApplication
from src.gui.main_window import MainWindow


def main():
    """Main application entry point."""
    print("\n" + "=" * 60)
    print("SP-AI-LIGHTS - Sports Highlights Auto Editor")
    print("AI-Powered Dynamic Reframing: 16:9 → 9:16")
    print("=" * 60 + "\n")

    # Print device info
    print()

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("SP-AI-LIGHTS")
    app.setOrganizationName("SportsAI")

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
